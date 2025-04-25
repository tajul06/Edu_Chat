import os
import torch
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
from torch.quantization import quantize_dynamic

# Set number of threads for CPU operations
cpu_threads = os.cpu_count() - 1  # Leave one thread for the OS
os.environ['OMP_NUM_THREADS'] = str(cpu_threads)
os.environ['MKL_NUM_THREADS'] = str(cpu_threads)
torch.set_num_threads(cpu_threads)

print(f"Using {cpu_threads} CPU threads for processing")

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Best choice for CPU-only systems:
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # Faster than L6, still good quality

# Replace the current quantization code with this safer version:
try:
    if torch.__version__ >= "1.8.0":  # Only attempt on newer PyTorch
        model.model = quantize_dynamic(model.model, {torch.nn.Linear}, dtype=torch.qint8)
        print("Model successfully quantized for faster CPU inference")
except Exception as e:
    print(f"Quantization not attempted or failed: {e}")

# Alternative lightweight model:
# model = SentenceTransformer('distiluse-base-multilingual-cased-v1')  # If you need multilingual

def chunk_text(text, threshold=0.7, max_chunk_size=1000):
    """Memory-optimized text chunking"""
    if not text:
        return []
    
    # Process in smaller batches if text is very large
    if len(text) > 100000:  # For very large documents
        print("Large document detected, processing in segments...")
        segments = [text[i:i+100000] for i in range(0, len(text), 100000)]
        all_chunks = []
        for i, segment in enumerate(segments):
            print(f"Processing segment {i+1}/{len(segments)}")
            sentences = sent_tokenize(segment)
            segment_chunks = chunk_text_semantically(sentences, threshold, max_chunk_size)
            all_chunks.extend(segment_chunks)
        return all_chunks
    
    # Normal processing for smaller documents
    sentences = sent_tokenize(text)
    return chunk_text_semantically(sentences, threshold, max_chunk_size)

def chunk_text_semantically(sentences, threshold=0.7, max_chunk_size=1000):
    """
    Chunks text based on semantic similarity using Sentence-BERT.
    Optimized for performance with batch encoding.
    
    Args:
        sentences: List of sentences
        threshold: Similarity threshold (0-1)
        max_chunk_size: Maximum characters in a chunk
        
    Returns:
        List of semantically coherent text chunks
    """
    if not sentences:
        return []
    
    print(f"Generating embeddings for {len(sentences)} sentences in one batch...")
    
    # Generate all embeddings in one batch - MAJOR PERFORMANCE IMPROVEMENT
    batch_size = get_optimal_batch_size()
    all_embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=True, batch_size=batch_size)
    
    print("Creating chunks based on embeddings...")
    chunks = []
    current_chunk = []
    current_chunk_size = 0
    current_idx = 0  # Track indices instead of re-encoding

    for i, sentence in enumerate(sentences):
        # Start a new chunk if this is the first sentence
        if not current_chunk:
            current_chunk.append(sentence)
            current_chunk_size = len(sentence)
            current_idx = i
            continue

        # Add size of current sentence (plus space)
        sentence_size = len(sentence) + 1
        
        # Check for max chunk size
        if current_chunk_size + sentence_size > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_chunk_size = len(sentence)
            current_idx = i
            continue

        # Use pre-computed embeddings - much faster
        last_embedding = all_embeddings[current_idx]
        current_embedding = all_embeddings[i]
        
        # Calculate cosine similarity
        similarity = np.dot(last_embedding, current_embedding) / (
            np.linalg.norm(last_embedding) * np.linalg.norm(current_embedding)
        )

        # Decide whether to add to current chunk or start a new one
        if similarity >= threshold:
            current_chunk.append(sentence)
            current_chunk_size += sentence_size
            # Don't update current_idx - we always compare to first sentence in chunk
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_chunk_size = len(sentence)
            current_idx = i

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    print(f"Created {len(chunks)} semantic chunks")
    return chunks

def generate_embeddings(chunks):
    """
    Generate embeddings for a list of text chunks.
    
    Args:
        chunks: List of text chunks
        
    Returns:
        List of embeddings as numpy arrays
    """
    if not chunks:
        return []
    
    # Use the sentence transformer model to encode each chunk
    embeddings = model.encode(chunks, convert_to_numpy=True)
    
    return embeddings

def get_embedding_model():
    """Returns the sentence transformer model used for embeddings."""
    return model

def get_optimal_batch_size():
    """Determine optimal batch size based on available system memory"""
    import psutil
    
    # Get available memory in GB
    available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)
    
    # Scale batch size based on available memory
    if available_memory > 8:  # More than 8GB available
        return 64
    elif available_memory > 4:  # 4-8GB available
        return 32
    else:  # Less than 4GB
        return 16

