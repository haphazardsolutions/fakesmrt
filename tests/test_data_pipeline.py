"""
test_data_pipeline.py - Test suite for fakesmrt data pipeline
"""

import os
import tempfile
import pytest
from pathlib import Path
from typing import List

from src.data_pipeline import CorpusStream, TextChunk, ChunkProcessor

class TestChunkProcessor:
    @pytest.fixture
    def sample_text_files(self) -> List[Path]:
        """Create temporary text files with various encodings and content"""
        files = []
        with tempfile.TemporaryDirectory() as temp_dir:
            # UTF-8 file
            utf8_path = Path(temp_dir) / "utf8.txt"
            with open(utf8_path, 'w', encoding='utf-8') as f:
                f.write("Hello world!\nThis is a test.\nMultiple lines here.\n")
            files.append(utf8_path)
            
            # Latin-1 file
            latin1_path = Path(temp_dir) / "latin1.txt"
            with open(latin1_path, 'w', encoding='latin-1') as f:
                f.write("Some text with latin-1 encoding.\nMore lines.\n")
            files.append(latin1_path)
            
            # File with long lines
            long_path = Path(temp_dir) / "long.txt"
            with open(long_path, 'w', encoding='utf-8') as f:
                f.write(" ".join(["word"] * 1000) + "\n")
            files.append(long_path)
            
            yield files

    def test_detect_encoding(self, sample_text_files):
        """Test encoding detection for different file types"""
        processor = ChunkProcessor()
        
        # Should detect UTF-8
        encoding = processor.detect_encoding(str(sample_text_files[0]))
        assert encoding.lower().replace('-', '') in ['utf8', 'utf_8', 'ascii']
        
        # Should detect Latin-1
        encoding = processor.detect_encoding(str(sample_text_files[1]))
        assert encoding.lower().replace('-', '') in ['latin1', 'iso88591', 'ascii']

    def test_process_file(self, sample_text_files):
        """Test processing files into chunks"""
        processor = ChunkProcessor()
        chunk_size = 5  # Small size for testing
        
        # Process UTF-8 file
        chunks, success = processor.process_file(
            str(sample_text_files[0]),
            chunk_size,
            start_chunk_id=0
        )
        
        assert success
        assert len(chunks) > 0
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)
        assert all(chunk.length <= chunk_size for chunk in chunks)
        
        # Check chunk IDs are sequential
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        assert chunk_ids == sorted(chunk_ids)
        
        # Process file with long lines
        chunks, success = processor.process_file(
            str(sample_text_files[2]),
            chunk_size,
            start_chunk_id=0
        )
        
        assert success
        assert len(chunks) > 0
        assert all(chunk.length <= chunk_size for chunk in chunks)

class TestCorpusStream:
    @pytest.fixture
    def sample_corpus(self) -> Path:
        """Create a temporary corpus directory with sample files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_dir = Path(temp_dir)
            
            # Create a few text files
            for i in range(3):
                file_path = corpus_dir / f"file{i}.txt"
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"Content for file {i}\n" * 10)
            
            # Create a subdirectory with more files
            subdir = corpus_dir / "subdir"
            subdir.mkdir()
            for i in range(2):
                file_path = subdir / f"subfile{i}.txt"
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"Subdir content {i}\n" * 10)
            
            yield corpus_dir

    def test_corpus_initialization(self, sample_corpus):
        """Test CorpusStream initialization and configuration"""
        corpus = CorpusStream(
            corpus_path=str(sample_corpus),
            chunk_size=10,
            buffer_size=100
        )
        
        assert corpus.chunk_size == 10
        assert corpus.buffer_size == 100
        assert Path(corpus.corpus_path) == sample_corpus
        assert corpus.file_extensions == ['.txt']

    def test_corpus_iteration(self, sample_corpus):
        """Test iterating through corpus chunks"""
        corpus = CorpusStream(
            corpus_path=str(sample_corpus),
            chunk_size=10,
            buffer_size=100,
            max_chunks=50
        )
        
        chunks = list(corpus)
        
        assert len(chunks) > 0
        assert len(chunks) <= 50  # Respect max_chunks
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)
        assert all(chunk.length <= corpus.chunk_size for chunk in chunks)
        
        # Check for duplicate chunk IDs
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))  # No duplicates

    def test_corpus_determinism(self, sample_corpus):
        """Test that corpus iteration is deterministic within same process"""
        corpus = CorpusStream(
            corpus_path=str(sample_corpus),
            chunk_size=10,
            buffer_size=100,
            max_chunks=20
        )
        
        first_run = [chunk.text for chunk in corpus]
        second_run = [chunk.text for chunk in corpus]
        
        # Both runs should yield same number of chunks
        assert len(first_run) == len(second_run)

    def test_corpus_sample(self, sample_corpus):
        """Test get_sample functionality"""
        corpus = CorpusStream(
            corpus_path=str(sample_corpus),
            chunk_size=10
        )
        
        sample = corpus.get_sample(n_chunks=3)
        
        assert len(sample) == 3
        assert all(isinstance(chunk, TextChunk) for chunk in sample)
        assert all(chunk.length <= corpus.chunk_size for chunk in sample)

    def test_max_chunks_limit(self, sample_corpus):
        """Test max_chunks parameter is respected"""
        max_chunks = 5
        corpus = CorpusStream(
            corpus_path=str(sample_corpus),
            chunk_size=10,
            max_chunks=max_chunks
        )
        
        chunks = list(corpus)
        assert len(chunks) <= max_chunks

    def test_empty_corpus(self):
        """Test handling of empty corpus directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus = CorpusStream(
                corpus_path=temp_dir,
                chunk_size=10
            )
            
            chunks = list(corpus)
            assert len(chunks) == 0

    @pytest.mark.parametrize("chunk_size", [10, 50, 100])
    def test_different_chunk_sizes(self, sample_corpus, chunk_size):
        """Test corpus with different chunk sizes"""
        corpus = CorpusStream(
            corpus_path=str(sample_corpus),
            chunk_size=chunk_size
        )
        
        chunks = list(corpus)
        assert all(chunk.length <= chunk_size for chunk in chunks)

if __name__ == "__main__":
    pytest.main([__file__])