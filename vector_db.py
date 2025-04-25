import os
import faiss
import numpy as np
import pickle
import json
from pathlib import Path

# Directory to store vector database files
VECTOR_DB_DIR = "vector_db_data"
INDEX_FILE = os.path.join(VECTOR_DB_DIR, "faiss_index.bin")
METADATA_FILE = os.path.join(VECTOR_DB_DIR, "chunks_metadata.pkl")

# Ensure directory exists
Path(VECTOR_DB_DIR).mkdir(parents=True, exist_ok=True)

# In-memory storage for document chunks
document_chunks = {}  # {doc_id: [chunks]}
chunk_to_doc_mapping = {}  # {chunk_idx: (doc_id, chunk_idx_in_doc)}

# Global index for FAISS
index = None

def create_document_store(dimension=384):  # MiniLM-L6 uses 384 dimensions
    """
    Initialize the vector database.
    
    Args:
        dimension: Dimensionality of the embeddings
    
    Returns:
        True if successful
    """
    global index
    
    # Check if we already have an index
    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        return load_document_store()
    
    # Create a new Faiss index - using L2 distance and flat index for accuracy
    index = faiss.IndexFlatL2(dimension)
    
    # Save empty metadata
    save_metadata()
    
    return True

def add_document(doc_id, chunks, embeddings):
    """
    Add document chunks and their embeddings to the vector database.
    
    Args:
        doc_id: Document identifier
        chunks: List of text chunks from the document
        embeddings: List of embeddings (numpy arrays) corresponding to chunks
        
    Returns:
        True if successful
    """
    global index, document_chunks, chunk_to_doc_mapping
    
    # Verify index exists
    if index is None:
        create_document_store(embeddings[0].shape[0])
    
    # Store document chunks
    document_chunks[doc_id] = chunks
    
    # Map global indices to document chunks
    start_idx = index.ntotal
    for i, embedding in enumerate(embeddings):
        chunk_to_doc_mapping[start_idx + i] = (doc_id, i)
    
    # Add embeddings to FAISS index
    embeddings_np = np.array(embeddings).astype('float32')
    index.add(embeddings_np)
    
    # Save the updated metadata
    save_metadata()
    
    return True

def query_similar_chunks(query_embedding, top_k=5, doc_id=None):
    """
    Retrieve similar chunks from the vector database.
    
    Args:
        query_embedding: Embedding vector of the query
        top_k: Number of most similar chunks to retrieve
        doc_id: Optional document ID to filter results
        
    Returns:
        List of tuples (chunk_text, doc_id, similarity_score)
    """
    global index, document_chunks, chunk_to_doc_mapping
    
    if index is None or index.ntotal == 0:
        return []
    
    # Debug print
    print(f"Query params: embedding shape {query_embedding.shape}, top_k={top_k}, doc_id={doc_id}")
    
    # Convert to numpy array with correct shape and dtype
    query_np = np.array([query_embedding]).astype('float32')
    
    # Search the index
    distances, indices = index.search(query_np, min(top_k, index.ntotal))
    
    # Prepare results
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:  # Valid index
            chunk_doc_id, chunk_idx = chunk_to_doc_mapping[int(idx)]
            
            # Filter by document ID if specified
            if doc_id is not None and chunk_doc_id != doc_id:
                continue
                
            chunk_text = document_chunks[chunk_doc_id][chunk_idx]
            similarity = 1 / (1 + distances[0][i])  # Convert distance to similarity score
            results.append((chunk_text, chunk_doc_id, similarity))
    
    return results

def get_vector_store():
    """
    Get the vector store instance.
    
    Returns:
        The FAISS index object
    """
    global index
    if index is None:
        create_document_store()
    return index

def save_metadata():
    """Save document chunks and mapping information."""
    global document_chunks, chunk_to_doc_mapping
    
    with open(METADATA_FILE, 'wb') as f:
        pickle.dump({
            'document_chunks': document_chunks,
            'chunk_to_doc_mapping': chunk_to_doc_mapping
        }, f)

def load_document_store():
    """
    Load existing vector DB from disk.
    
    Returns:
        True if successful
    """
    global index, document_chunks, chunk_to_doc_mapping
    
    try:
        # Load FAISS index
        index = faiss.read_index(INDEX_FILE)
        
        # Load metadata
        with open(METADATA_FILE, 'rb') as f:
            metadata = pickle.load(f)
            document_chunks = metadata['document_chunks']
            chunk_to_doc_mapping = metadata['chunk_to_doc_mapping']
        
        return True
    
    except Exception as e:
        print(f"Error loading vector database: {e}")
        # Initialize new if loading fails
        index = faiss.IndexFlatL2(384)  # Default to 384 dimensions
        document_chunks = {}
        chunk_to_doc_mapping = {}
        return False

def save_document_store():
    """
    Persist the vector database to disk.
    
    Returns:
        True if successful
    """
    global index
    
    try:
        if index is not None:
            faiss.write_index(index, INDEX_FILE)
            save_metadata()
        return True
    except Exception as e:
        print(f"Error saving vector database: {e}")
        return False

def get_document_chunks(doc_id):
    """
    Retrieve all chunks for a specific document.
    
    Args:
        doc_id: Document identifier
        
    Returns:
        List of text chunks
    """
    global document_chunks
    return document_chunks.get(doc_id, [])

# Create document store when module is imported
create_document_store()

# Save database when exiting
import atexit
atexit.register(save_document_store)

# For testing
if __name__ == "__main__":
    # Create dummy embeddings
    dim = 384
    test_embeddings = [np.random.rand(dim).astype('float32') for _ in range(5)]
    test_chunks = [f"This is test chunk {i}" for i in range(5)]
    
    # Test adding document
    add_document(1, test_chunks, test_embeddings)
    
    # Test retrieval
    query = np.random.rand(dim).astype('float32')
    results = query_similar_chunks(query, 3)
    
    print(f"Found {len(results)} similar chunks")
    print(f"Total documents: {len(document_chunks)}")
    print(f"Total chunks indexed: {index.ntotal}")