# EduChat - AI-Powered Educational Assistant

EduChat is a local, offline-capable educational assistant that helps students understand their study materials by answering questions based on uploaded documents.

![EduChat Interface](static/img/ai-assistant.png)

## Overview

EduChat combines advanced natural language processing with document understanding to create an AI-powered educational tool. Users can upload their study materials (PDFs) and ask questions about the content. The system processes the documents, understands the semantic meaning of the text, and generates accurate responses based on the uploaded materials.

### Key Features

- **Document Processing**: Upload PDFs like textbooks or notes for context-aware answers
- **Retrieval Augmented Generation (RAG)**: Combines document retrieval with language model generation
- **Conversational Memory**: Remembers previous exchanges for contextual interactions
- **Local Processing**: Runs entirely on your machine without sending data to external servers
- **User Management**: Multi-user support with personalized document libraries
- **Mobile-Friendly Interface**: Responsive design works across devices

## Technical Architecture

EduChat consists of several integrated components:

1. **Document Processing Pipeline**:
   - Text extraction from PDFs
   - Semantic chunking of content
   - Vector embedding generation 
   - Storage in FAISS vector database

2. **RAG System**:
   - Query embedding
   - Semantic search in vector DB
   - Context assembly with prompt engineering
   - Response generation with local LLM
  
3. **Web Interface**:
   - Flask-based web application
   - User authentication and session management
   - Document management
   - Real-time chat interface

## Requirements

- Python 3.9+
- 8GB+ RAM (16GB recommended)
- 10GB+ disk space
- CPU with 4+ cores

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/educhat.git
cd educhat
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download NLTK resources**

```bash
python nltksetup.py
```

5. **Download the LLM model**

Download the language model file and place it in the `models` directory:
- Recommended: `llama-3.1-8b-instruct-q4_k_m.gguf` (4.7GB)
- Download from [TheBloke/Llama-3.1-8B-Instruct-GGUF](https://huggingface.co/TheBloke/Llama-3.1-8B-Instruct-GGUF)

## Usage

1. **Start the application**

```bash
python app.py
```

2. **Access the web interface**

Open your browser and go to: `http://127.0.0.1:5000`

3. **Create an account and log in**

4. **Upload a document**
   - Click the "Upload" button in the documents sidebar
   - Select a PDF file
   - Wait for processing to complete

5. **Start chatting**
   - Select a document from the sidebar to provide context
   - Type your questions in the chat box
   - Receive AI-powered answers based on your documents

## Project Structure

- **`app.py`**: Main Flask application with routes and database models
- **`pdf_processor.py`**: PDF text extraction and processing
- **`semantic_chunker.py`**: Text chunking based on semantic similarity
- **`vector_db.py`**: FAISS vector database interface
- **`rag_system.py`**: Retrieval Augmented Generation implementation
- **`conversation_memory.py`**: Conversation history management
- **`templates/`**: HTML templates for web interface
- **`static/`**: CSS, JS, and image files
- **`models/`**: Directory for language model files
- **`uploads/`**: Directory for uploaded documents
- **`vector_db_data/`**: Vector database storage

## Customization

- **Model**: You can replace the default LLM with other compatible GGUF models
- **UI**: Modify templates and CSS to change the appearance
- **Parameters**: Adjust various parameters in `rag_system.py` for different response styles

## Limitations

- Processing large documents may be slow on less powerful hardware
- Math equations and complex diagrams may not be properly interpreted
- The quality of responses depends on the quality of the uploaded documents

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Llama 3.1 by Meta](https://ai.meta.com/resources/models-and-libraries/llama/)
- [FAISS by Facebook AI Research](https://github.com/facebookresearch/faiss)
- [Sentence-Transformers](https://www.sbert.net/)
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
- [Flask](https://flask.palletsprojects.com/)