"""
train.py - Complete training harness with all classes
"""
from pathlib import Path
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_pipeline import CorpusStream, TextChunk

import cProfile
import pstats
from memory_profiler import profile

@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    # Model parameters
    vocab_size: int = 8192
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 10
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # System parameters
    device: str = "cpu"
    dtype: torch.dtype = torch.bfloat16
    num_workers: int = 2
    buffer_size: int = 1000
    
    # Paths
    output_dir: Path = Path("models")
    checkpoint_dir: Path = Path("checkpoints")
    
    # Optimization
    gradient_accumulation_steps: int = 8
    mixed_precision: bool = True
    max_chunks: Optional[int] = None

class TrainingStats:
    """Tracks training metrics"""
    def __init__(self):
        self.epoch_loss = 0.0
        self.steps = 0
        self.tokens_seen = 0
        self.best_loss = float('inf')
        self.start_time = time.time()
        
    def update(self, loss: float, num_tokens: int):
        self.epoch_loss += loss
        self.steps += 1
        self.tokens_seen += num_tokens
    
    def get_metrics(self) -> Dict[str, Any]:
        elapsed = time.time() - self.start_time
        return {
            'loss': self.epoch_loss / max(1, self.steps),
            'tokens_per_second': self.tokens_seen / max(1, elapsed),
            'steps': self.steps,
        }

def collate_chunks(batch: List[TextChunk]) -> List[TextChunk]:
    """Custom collate function for TextChunk objects"""
    return batch

class TrainingHarness:
    """Main training orchestrator"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        config: TrainingConfig,
        corpus_path: str,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.corpus_path = corpus_path
        self.stats = TrainingStats()
        
        # Set up directories
        self.config.output_dir.mkdir(exist_ok=True, parents=True)
        self.config.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_epochs
        )
        
        # Set up data pipeline
        self.corpus = CorpusStream(
            corpus_path=corpus_path,
            chunk_size=512,
            buffer_size=config.buffer_size,
            max_chunks=config.max_chunks
        )
        
        # Set up data loader with custom collate function
        self.dataloader = DataLoader(
            self.corpus,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=collate_chunks
        )
        
        # Set up logging
        self._setup_logging()
        
        # Initialize mixed precision if requested
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None

    def _setup_logging(self):
        self.logger = logging.getLogger('fakesmrt.training')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _prepare_batch(self, chunks: List[TextChunk]) -> Dict[str, torch.Tensor]:
        """Prepare a batch of chunks for training"""
        texts = [chunk.text for chunk in chunks]
        
        # Tokenize with explicit max length and truncation
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,  # Explicit max length
            return_tensors="pt"
        )
        
        # Ensure we don't exceed the model's maximum sequence length
        max_seq_length = 512
        input_ids = encodings.input_ids[:, :max_seq_length]
        attention_mask = encodings.attention_mask[:, :max_seq_length]
        
        # Create shifted input/labels for language modeling
        input_ids = input_ids[:, :-1]
        labels = encodings.input_ids[:, 1:max_seq_length]  # Ensure labels match input length
        attention_mask = attention_mask[:, :-1]
        
        return {
            "input_ids": input_ids.to(self.config.device),
            "labels": labels.to(self.config.device),
            "attention_mask": attention_mask.to(self.config.device)
        }
    
    @profile
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        self.stats = TrainingStats()
        
        pbar = tqdm(
            self.dataloader,
            desc=f"Epoch {epoch}",
            total=len(self.corpus) // self.config.batch_size
        )
        
        accumulated_loss = 0
        
        for step, batch_chunks in enumerate(pbar):
            # Prepare batch
            batch = self._prepare_batch(batch_chunks)
            
            # Forward pass with optional mixed precision
            with torch.autocast(
                device_type=self.config.device,
                dtype=self.config.dtype,
                enabled=self.config.mixed_precision
            ):
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps
            
            # Backward pass with gradient accumulation
            if self.config.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            accumulated_loss += loss.item()
            
            # Update weights if we've accumulated enough gradients
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                # Optimizer step
                if self.config.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update stats and progress bar
                self.stats.update(
                    accumulated_loss,
                    batch["input_ids"].numel()
                )
                accumulated_loss = 0
                
                metrics = self.stats.get_metrics()
                pbar.set_postfix(
                    loss=f"{metrics['loss']:.4f}",
                    tokens_sec=f"{metrics['tokens_per_second']:.0f}"
                )
            
            # Save checkpoint if loss improved
            if self.stats.epoch_loss < self.stats.best_loss:
                self.save_checkpoint(epoch, is_best=True)
                self.stats.best_loss = self.stats.epoch_loss
        
        pbar.close()
        self.scheduler.step()
        
        return self.stats.get_metrics()

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save a model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'stats': self.stats.get_metrics(),
            'config': self.config
        }
        
        checkpoint_path = self.config.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.config.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load a saved checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch']

    def train(self):
        """Full training loop"""
        self.logger.info("Starting training...")
        
        for epoch in range(self.config.max_epochs):
            self.logger.info(f"Starting epoch {epoch+1}/{self.config.max_epochs}")
            
            metrics = self.train_epoch(epoch)
            
            self.logger.info(
                f"Epoch {epoch+1} complete: "
                f"loss={metrics['loss']:.4f}, "
                f"tokens/sec={metrics['tokens_per_second']:.0f}"
            )
            
            self.save_checkpoint(epoch)
        
        self.logger.info("Training complete!")