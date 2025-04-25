import os
import random
import numpy as np
from semantic_chunker import get_embedding_model
from vector_db import query_similar_chunks
from textwrap import dedent
from pathlib import Path

# Load embedding model once at module level
embedding_model = get_embedding_model()

# Mock LLM responses for demonstration
GENERIC_RESPONSES = [
    "Based on my knowledge, I can provide this answer...",
    "According to general information, the answer is...",
    "I believe the answer to your question is...",
    "From what I understand, I can tell you that..."
]

# Create directory for model files if it doesn't exist
os.makedirs("models", exist_ok=True)

def setup_cpu_llm():
    """Initialize a CPU-optimized language model"""
    try:
        # Option 1: Using llama-cpp-python (most efficient)
        from llama_cpp import Llama
        
        model_path = "models/llama-3.1-8b-instruct-q4_k_m.gguf"
        
        # Check if model exists
        if not Path(model_path).exists():
            print(f"Model file not found: {model_path}")
            print("Please download the model or use a different path")
            return None
        
        # Initialize with CPU-friendly settings
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,         # Larger context window
            n_batch=512,
            n_threads=os.cpu_count() - 1,  # Use more threads
            n_gpu_layers=0,     # CPU only
            f16_kv=True,        # Memory optimization
            verbose=False       # Less logging noise
        )
        
        print(f"CPU-optimized LLM loaded successfully")
        return llm
    
    except ImportError:
        print("llama-cpp-python not installed. Install with:")
        print("pip install llama-cpp-python")
        return None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

# Try to initialize the model
llm = setup_cpu_llm()

def process_query_with_rag(query, doc_id=None, session_id=None, conversation_memory=None):
    """
    Process a query using Retrieval Augmented Generation.
    
    Args:
        query: User's question
        doc_id: Optional document ID for context
        session_id: Unique identifier for the user session
        conversation_memory: Optional ConversationMemory instance
        
    Returns:
        Generated response string
    """
    print(f"\nProcessing query: '{query}'")
    
    # Simple response if no document context
    if not doc_id:
        print("No document context provided, generating generic response")
        response = generate_generic_response(query)
        
        # Add to conversation memory if available
        if conversation_memory and session_id:
            conversation_memory.add_message(session_id, query, True)
            conversation_memory.add_message(session_id, response, False)
            
        return response
    
    print(f"Using document ID: {doc_id}")
    
    # Get relevant chunks from vector DB
    context_chunks = retrieve_relevant_chunks(query, doc_id)
    
    if not context_chunks:
        print("No relevant chunks found in the document")
        response = "I couldn't find any relevant information in the document. Could you try rephrasing your question?"
        
        # Add to conversation memory if available
        if conversation_memory and session_id:
            conversation_memory.add_message(session_id, query, True)
            conversation_memory.add_message(session_id, response, False)
            
        return response
    
    # Generate response using context AND conversation history
    response = generate_response_with_context(
        query, 
        context_chunks,
        conversation_history=conversation_memory.format_for_prompt(session_id) if conversation_memory and session_id else None
    )
    
    # Add to conversation memory if available
    if conversation_memory and session_id:
        conversation_memory.add_message(session_id, query, True)
        conversation_memory.add_message(session_id, response, False)
    
    return response

def retrieve_relevant_chunks(query, doc_id, top_k=3):
    """
    Retrieve relevant chunks from vector database.
    
    Args:
        query: User's question
        doc_id: Document ID
        top_k: Number of chunks to retrieve
        
    Returns:
        List of relevant text chunks
    """
    try:
        print(f"Generating embedding for query: '{query}'")
        # Generate real embedding for the query using our model
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)[0]
        
        print(f"Querying vector database for top {top_k} chunks")
        # Query the vector DB using the real query embedding
        results = query_similar_chunks(query_embedding, top_k=top_k, doc_id=doc_id)
        
        if not results:
            print("No results returned from vector database")
            return []
        
        # Extract just the text from results (assuming query_similar_chunks returns tuples)
        chunks = [chunk_text for chunk_text, _, score in results]
        
        print(f"Retrieved {len(chunks)} chunks from vector database")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1} (length: {len(chunk)}): {chunk[:100]}...")
        
        return chunks
        
    except Exception as e:
        print(f"Error retrieving chunks: {str(e)}")
        return []

def generate_response_with_context(query, context_chunks, conversation_history=None):
    """
    Generate a response based on retrieved context and conversation history.
    
    Args:
        query: User's question
        context_chunks: Retrieved context
        conversation_history: Optional formatted conversation history
        
    Returns:
        Generated response
    """
    try:
        # Combine context chunks with reasonable formatting
        formatted_context = "\n\n---\n\n".join(context_chunks)
        
        # Create a prompt for the LLM, including conversation history if available
        if conversation_history and conversation_history.strip():
            print(f"Including conversation history ({len(conversation_history)} chars)")
            prompt = dedent(f"""
            [INST] <<SYS>>
            You are an educational assistant that provides informative, clear explanations.
            Answer questions based on the context provided. Include relevant details and brief explanations.
            Provide 2-3 sentences of explanation when answering, but remain focused on the facts.
            If appropriate, include a simple example to illustrate concepts.
            Format scientific notation properly (e.g., 10^(-24) not 10%).
            If unsure, say "I don't have enough information."
            <</SYS>>

            CONTEXT:
            {formatted_context}

            CONVERSATION HISTORY:
            {conversation_history}

            QUESTION:
            {query}  

            ANSWER: [/INST]
            """).strip()
        else:
            # Original prompt without conversation history
            prompt = dedent(f"""
            [INST] <<SYS>>
            You are an educational assistant that provides informative, clear explanations.
            Answer questions based on the context provided. Include relevant details and brief explanations.
            Provide 2-3 sentences of explanation when answering, but remain focused on the facts.
            If appropriate, include a simple example to illustrate concepts.
            Format scientific notation properly (e.g., 10^(-24) not 10%).
            If unsure, say "I don't have enough information."
            <</SYS>>

            CONTEXT:
            {formatted_context}

            QUESTION:
            {query}  

            ANSWER: [/INST]
            """).strip()
        
        print(f"Generated prompt of length {len(prompt)} characters")
        
        # Call LLM for response generation
        response = call_llm(prompt)
        
        return response
        
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return f"I'm sorry, I encountered an error while processing your question: {str(e)}"

def call_llm(prompt):
    """
    Call the language model to generate a response.
    
    Args:
        prompt: The prompt to send to the LLM
        
    Returns:
        Generated text response
    """
    # Check if we have a real LLM available
    if llm is None:
        print("Using mock LLM as real model is not available")
        return mock_llm_response(prompt)
    
    try:
        print("Generating response with local CPU-optimized LLM...")
        
        # Generate text with the model
        output = llm(
            prompt,
            max_tokens=768,       
            temperature=0.1,      
            top_p=0.9,            
            top_k=40,             
            repeat_penalty=1.2,   
            stop=["[/INST]", "</s>", "QUESTION:", "CONTEXT:", "[INST]", "[SYS]", "/SYS"],
            echo=False           
        )
        
        # Extract the generated text from the response
        if isinstance(output, dict) and 'choices' in output:
            response = clean_response(output['choices'][0]['text'].strip())
        else:
            # Direct string response from some models
            response = clean_response(output.strip())
            
        print(f"Generated {len(response)} characters of response")
        return response
        
    except Exception as e:
        print(f"Error calling LLM: {str(e)}")
        print("Falling back to mock response")
        return mock_llm_response(prompt)

def mock_llm_response(prompt):
    """Fallback response generator when real LLM fails"""
    # Existing mock implementation
    query = prompt.split("QUESTION:")[1].split("ANSWER:")[0].strip()
    context = prompt.split("CONTEXT:")[1].split("QUESTION:")[0].strip()
    keywords = extract_keywords(context, 5)
    
    response = (
        f"Based on the document, I found information related to your question about {query}. "
        f"The document discusses {', '.join(keywords)}. "
        f"According to the text, this topic involves important concepts and applications "
        f"that are relevant to your question."
    )
    
    return response

def extract_keywords(text, num_keywords=5):
    """Extract simple keywords from text for mock responses"""
    # Very simple keyword extraction
    common_words = {'the', 'and', 'a', 'to', 'of', 'in', 'is', 'that', 'it', 'for', 
                   'as', 'with', 'on', 'was', 'be', 'this', 'are'}
    
    words = text.lower().split()
    words = [word.strip('.,;:()[]{}""\'').lower() for word in words]
    words = [word for word in words if word and word not in common_words and len(word) > 3]
    
    # Count word frequency
    word_counts = {}
    for word in words:
        if word in word_counts:
            word_counts[word]+= 1
        else:
            word_counts[word] = 1
    
    # Get top keywords
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    keywords = [word for word, count in sorted_words[:num_keywords]]
    
    # If we don't have enough keywords, add some generic ones
    while len(keywords) < num_keywords:
        keywords.append("relevant concepts")
    
    return keywords

def generate_generic_response(query):
    """
    Generate a generic response for queries without document context.
    """
    print("Generating generic response (no document context)")
    
    response_templates = [
        "I can provide better answers if you upload a document and select it before asking questions.",
        "To get more specific information, please upload a document that contains the details you're looking for.",
        "I'm designed to answer questions based on document context. Please upload and select a document for more accurate answers.",
        "For more detailed answers, try uploading a PDF document related to your question."
    ]
    
    response = f"{np.random.choice(response_templates)} What would you like to know?"
    
    return response

def clean_response(response):
    """Clean the response for clear, direct answers"""
    # Remove system and instruction formatting markers
    system_markers = [
        "[INST]", "[/INST]", "[SYS]", "[/SYS]", 
        "/INST", "/SYS", "<<SYS>>", "<</SYS>>", "SYS ", " /SYS",
        "[ANS]", "[/ANS]", "ANS ", " /ANS",  # Answer markers
        "//", "/*", "*/", "##",  # Comment markers
        "</s>", "<s>"  # Model-specific markers
    ]
    
    for marker in system_markers:
        response = response.replace(marker, "")
    
    # Remove "ANSWER" prefix if present
    if response.strip().startswith("ANSWER"):
        response = response.strip()[6:].strip()
    
    # Remove angle brackets and other unwanted formatting characters
    response = response.replace("<>", "")
    response = response.replace("<", "")
    response = response.replace(">", "")
    
    # Handle lines with code comments
    lines = response.split('\n')
    cleaned_lines = []
    for line in lines:
        if line.strip().startswith("//") or line.strip().startswith("#"):
            continue  # Skip comment-only lines
        cleaned_lines.append(line)
    response = '\n'.join(cleaned_lines)
    
    # Remove thinking patterns completely
    thinking_patterns = [
        "Let me think", "I need to analyze", "Let's see", "To answer this",
        "Based on the information", "If I look at", "I'll analyze",
        "First, I'll", "Let me check", "I should consider"
    ]
    
    # Remove the entire thinking section
    for pattern in thinking_patterns:
        if pattern in response:
            pattern_index = response.find(pattern)
            if pattern_index > -1:
                end_of_thinking = response.find(". ", pattern_index)
                if end_of_thinking > -1:
                    # Only keep the answer part
                    response = response[end_of_thinking+2:].strip()
                    break
    
    # Remove other common prefixes that make responses less direct
    prefixes_to_remove = [
        "Based on the document, ", 
        "According to the context, ",
        "From the information provided, ",
        "The document states that ",
        "As mentioned in the context, ",
        "In response to your question, "
    ]
    
    for prefix in prefixes_to_remove:
        if response.lower().startswith(prefix.lower()):
            response = response[len(prefix):].strip()
    
    # Fix scientific notation
    response = response.replace("10%", "10^(-24)")
    response = response.replace("10^-", "10^(-")
    
    # Clean up sentence format
    response = response.strip()
    if response and not response[0].isupper():
        response = response[0].upper() + response[1:]
    
    print(f"Cleaned response (length: {len(response)})")
    
    return response

"""
MODEL USAGE INFORMATION:
-----------------------
This system uses a pre-trained language model (llama-3.1-8b-instruct-q4_k_m.gguf),
which doesn't require training from scratch. The steps to get better generation are:

1. Download the model: Make sure the model file exists in the models/ directory
   You can download it from Hugging Face: https://huggingface.co/TheBloke/Llama-3.1-8B-Instruct-GGUF/

2. Adjust parameters: You can modify the parameters in the call_llm() function:
   - Lower temperature (0.1) for more factual responses
   - Higher temperature (0.7-0.9) for more creative responses
   - Adjust top_p, top_k, and repeat_penalty for response quality

3. Fine-tuning (optional): For domain-specific improvements, you can fine-tune
   the model, but this requires more resources and technical knowledge.
"""

def fine_tune_model(training_data_path, output_model_path):
    """
    Fine-tune the language model for better domain-specific responses.
    
    Args:
        training_data_path: Path to training data
        output_model_path: Path to save the fine-tuned model
        
    Returns:
        Success status (boolean)
    """
    try:
        print("Fine-tuning is an advanced feature that requires:")
        print("1. Significant computational resources (CPU/GPU)")
        print("2. Properly formatted training data")
        print("3. Additional libraries (transformers, peft, etc.)")
        
        # Implementation would involve:advanced feature that requires:")
        # - Loading the base model
        # - Preparing the datasetraining data")
        # - Using LoRA or QLoRA for parameter-efficient fine-tuningprint("3. Additional libraries (transformers, peft, etc.)")
        # - Saving the resulting model
        
        print("This function is a placeholder. To implement fine-tuning:")
        print("1. Install required packages: pip install transformers peft datasets")
        print("2. Prepare training data in the correct format")
        print("3. Consider using Hugging Face's training scripts for easier fine-tuning")
        
        return False
    except Exception as e:
        print(f"Error in fine-tuning: {str(e)}")
        return False
