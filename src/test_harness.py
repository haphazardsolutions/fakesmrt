"""
test_harness.py - Simple test of the training infrastructure
"""
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from pathlib import Path
import chardet
from typing import List
import logging
import os

from train import TrainingConfig, TrainingHarness

class TinyTransformer(nn.Module):
    """Minimal transformer for testing"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Set max sequence length
        self.max_seq_length = 512
        
        # Token embeddings
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(self.max_seq_length, config.hidden_size)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
        # Output layer
        self.output = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor = None,
        attention_mask: torch.Tensor = None
    ):
        # Ensure we don't exceed maximum sequence length
        seq_length = min(input_ids.size(1), self.max_seq_length)
        input_ids = input_ids[:, :seq_length]
        if labels is not None:
            labels = labels[:, :seq_length]
        if attention_mask is not None:
            attention_mask = attention_mask[:, :seq_length]
        
        # Create position ids
        position_ids = torch.arange(
            seq_length,
            device=input_ids.device,
            dtype=torch.long
        ).unsqueeze(0).expand_as(input_ids)
        
        # Get embeddings
        token_embeds = self.embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        embeds = token_embeds + position_embeds
        
        # Run through transformer
        if attention_mask is not None:
            # Convert boolean mask to float mask for transformer
            attention_mask = attention_mask.float()
            attention_mask = attention_mask.masked_fill(
                attention_mask == 0,
                float('-inf')
            ).masked_fill(
                attention_mask == 1,
                float(0.0)
            )
        
        hidden_states = self.transformer(
            embeds,
            src_key_padding_mask=attention_mask
        )
        
        # Get logits
        logits = self.output(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Reshape logits and labels for loss calculation
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.reshape(-1, vocab_size)
            labels_flat = labels.reshape(-1)
            
            # Compute loss
            loss = self.criterion(logits_flat, labels_flat)
        
        return type('ModelOutput', (), {'loss': loss, 'logits': logits})()

def is_valid_text_file(file_path: Path) -> bool:
    """Check if file is valid text and can be read"""
    try:
        # Read a sample of the file to detect encoding
        with open(file_path, 'rb') as f:
            raw = f.read(min(32768, file_path.stat().st_size))
        result = chardet.detect(raw)
        
        # Skip if no encoding detected or confidence is low
        if not result['encoding'] or result['confidence'] < 0.9:
            return False
            
        # Try reading the file with detected encoding
        with open(file_path, 'r', encoding=result['encoding']) as f:
            # Read first few lines as a test
            for _ in range(10):
                line = f.readline()
                if '\0' in line:  # Skip binary files
                    return False
            return True
            
    except Exception as e:
        logging.warning(f"Skipping {file_path}: {str(e)}")
        return False

def prepare_tokenizer_files(corpus_path: str, max_files: int = 10) -> List[str]:
    """Prepare a list of valid text files for tokenizer training"""
    valid_files = []
    files = Path(corpus_path).rglob("*.txt")
    
    print("Scanning for valid text files...")
    for file_path in files:
        if len(valid_files) >= max_files:
            break
            
        if is_valid_text_file(file_path):
            valid_files.append(str(file_path))
            print(f"Added {file_path.name} to tokenizer training set")
    
    if not valid_files:
        raise ValueError("No valid text files found for tokenizer training!")
    
    return valid_files

def create_test_tokenizer(corpus_path: str) -> PreTrainedTokenizerFast:
    """Create a basic tokenizer trained on a small sample of the corpus"""
    # Initialize a byte-level BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    
    # Basic normalization and pre-tokenization
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.Lowercase(),
        normalizers.StripAccents(),
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.Punctuation(),
    ])
    
    # Train tokenizer on a small sample
    trainer = trainers.BpeTrainer(
        vocab_size=8192,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    
    # Get validated files for tokenizer training
    print("Preparing files for tokenizer training...")
    training_files = prepare_tokenizer_files(corpus_path, max_files=10)
    print(f"Found {len(training_files)} valid files for tokenizer training")
    
    # Train the tokenizer
    print("Training tokenizer...")
    tokenizer.train(files=training_files, trainer=trainer)
    
    # Convert to PreTrainedTokenizerFast with max length set
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=512  # Set maximum sequence length
    )

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Configuration with reduced scope for testing
    config = TrainingConfig(
        vocab_size=8192,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        batch_size=16,         # Increased for better throughput
        max_epochs=1,          # Just one epoch for testing
        num_workers=2,         # Adjusted for your CPU
        mixed_precision=False,  # Disable for testing
        gradient_accumulation_steps=2,
        buffer_size=1000
    )
    
    # Paths
    corpus_path = "/mnt/alpha/media/library/"
    
    # Set tokenizer parallelism explicitly
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print("Creating tokenizer...")
    try:
        tokenizer = create_test_tokenizer(corpus_path)
        print("Tokenizer created successfully!")
    except Exception as e:
        print(f"Error creating tokenizer: {str(e)}")
        raise
    
    print("Initializing model...")
    model = TinyTransformer(config)
    
    print("Setting up training harness...")
    harness = TrainingHarness(
        model=model,
        tokenizer=tokenizer,
        config=config,
        corpus_path=corpus_path
    )
    
    print("Starting test training run...")
    try:
        # Limit the number of chunks for testing
        print("Limiting test to 1000 chunks...")
        harness.corpus.max_chunks = 1000  # Process only 1000 chunks
        harness.train()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()