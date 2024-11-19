"""
data_pipeline.py - Data pipeline with fixed multiprocessing
"""
import os
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
import logging
from dataclasses import dataclass
import multiprocessing as mp
from itertools import islice
import random
import chardet
import torch
from torch.utils.data import IterableDataset
import math
from queue import Queue
from threading import Thread

@dataclass
class TextChunk:
    """Represents a chunk of processed text with metadata"""
    text: str
    source: str
    length: int
    chunk_id: int

class ProcessingStats:
    """Track processing statistics"""
    def __init__(self):
        self.processed_files = 0
        self.failed_files = 0
        self.processed_chunks = 0
        self.total_tokens = 0
        
    def update(self, chunks: List[TextChunk], success: bool):
        if success:
            self.processed_files += 1
            self.processed_chunks += len(chunks)
            self.total_tokens += sum(chunk.length for chunk in chunks)
        else:
            self.failed_files += 1

class ChunkProcessor:
    """Handles text chunk processing without multiprocessing"""
    
    @staticmethod
    def detect_encoding(file_path: str) -> str:
        """Detect file encoding using chardet"""
        try:
            with open(file_path, 'rb') as f:
                raw = f.read(min(32768, os.path.getsize(file_path)))
            result = chardet.detect(raw)
            return result['encoding'] or 'utf-8'
        except Exception:
            return 'utf-8'

    @staticmethod
    def process_file(file_path: str, chunk_size: int, start_chunk_id: int) -> Tuple[List[TextChunk], bool]:
        """Process a single file into chunks"""
        chunks = []
        success = False
        
        try:
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252', None]
            
            for encoding in encodings_to_try:
                try:
                    if encoding is None:
                        encoding = ChunkProcessor.detect_encoding(file_path)
                    
                    with open(file_path, 'r', encoding=encoding) as f:
                        current_chunk = []
                        current_length = 0
                        chunk_id = start_chunk_id
                        
                        for line in f:
                            if '\0' in line:  # Skip binary content
                                continue
                                
                            line = " ".join(line.split())
                            if not line:
                                continue
                                
                            words = line.split()
                            current_chunk.extend(words)
                            current_length += len(words)
                            
                            while current_length >= chunk_size:
                                chunk_text = " ".join(current_chunk[:chunk_size])
                                chunks.append(TextChunk(
                                    text=chunk_text,
                                    source=str(file_path),
                                    length=chunk_size,
                                    chunk_id=chunk_id
                                ))
                                chunk_id += 1
                                
                                current_chunk = current_chunk[chunk_size:]
                                current_length = len(current_chunk)
                        
                        if current_length > chunk_size // 2:
                            chunk_text = " ".join(current_chunk)
                            chunks.append(TextChunk(
                                text=chunk_text,
                                source=str(file_path),
                                length=current_length,
                                chunk_id=chunk_id
                            ))
                        
                        success = True
                        break
                        
                except UnicodeDecodeError:
                    if encoding == encodings_to_try[-1]:
                        logging.warning(f"Failed to decode {file_path} with all encodings")
                    continue
                    
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            
        return chunks, success

class CorpusStream(IterableDataset):
    """Memory-efficient text corpus processor with DataLoader compatibility"""
    
    def __init__(
        self,
        corpus_path: str,
        chunk_size: int = 512,
        file_extensions: List[str] = ['.txt'],
        buffer_size: int = 1000,
        max_chunks: Optional[int] = None
    ):
        self.corpus_path = Path(corpus_path)
        self.chunk_size = chunk_size
        self.file_extensions = file_extensions
        self.buffer_size = buffer_size
        self.max_chunks = max_chunks
        self.stats = ProcessingStats()
        self._setup_logging()
        
        # Calculate approximate dataset size
        self.approx_size = self._estimate_size()
        
        # Initialize processor
        self.processor = ChunkProcessor()

    def _setup_logging(self):
        self.logger = logging.getLogger('fakesmrt.data')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)

    def _estimate_size(self) -> int:
        """Estimate the number of chunks in the dataset"""
        total_bytes = sum(
            f.stat().st_size
            for ext in self.file_extensions
            for f in self.corpus_path.rglob(f"*{ext}")
        )
        estimated_chunks = total_bytes / (4 * self.chunk_size)
        if self.max_chunks:
            return min(math.ceil(estimated_chunks), self.max_chunks)
        return math.ceil(estimated_chunks)

    def __len__(self) -> int:
        """Return estimated dataset size"""
        return self.approx_size

    def _find_files(self) -> List[Path]:
        """Recursively find all matching files in corpus directory"""
        files = []
        for ext in self.file_extensions:
            files.extend(list(self.corpus_path.rglob(f"*{ext}")))
        return files

    def __iter__(self) -> Iterator[TextChunk]:
        """Iterator implementation for DataLoader compatibility"""
        worker_info = torch.utils.data.get_worker_info()
        files = self._find_files()
        random.shuffle(files)
        
        if worker_info is not None:
            # Split files among workers
            per_worker = int(math.ceil(len(files) / worker_info.num_workers))
            start_idx = worker_info.id * per_worker
            end_idx = min(start_idx + per_worker, len(files))
            files = files[start_idx:end_idx]
        
        chunk_id_offset = 0
        chunks_yielded = 0
        buffer = []
        
        # Process files sequentially
        for file_path in files:
            if self.max_chunks and chunks_yielded >= self.max_chunks:
                break
                
            chunks, success = self.processor.process_file(
                str(file_path),
                self.chunk_size,
                chunk_id_offset
            )
            
            self.stats.update(chunks, success)
            buffer.extend(chunks)
            
            # Shuffle and yield from buffer when it's full
            while len(buffer) >= self.buffer_size:
                random.shuffle(buffer)
                while len(buffer) > self.buffer_size // 2:
                    if self.max_chunks and chunks_yielded >= self.max_chunks:
                        return
                    yield buffer.pop()
                    chunks_yielded += 1
            
            chunk_id_offset += 1000
        
        # Yield remaining chunks
        if buffer:
            random.shuffle(buffer)
            while buffer and (not self.max_chunks or chunks_yielded < self.max_chunks):
                yield buffer.pop()
                chunks_yielded += 1

    def get_sample(self, n_chunks: int = 5) -> List[TextChunk]:
        """Get a small sample of chunks for inspection"""
        return list(islice(self.__iter__(), n_chunks))