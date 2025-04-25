from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
import os
import uuid
from datetime import datetime

# Import specialized components
from pdf_processor import extract_text_from_pdf , process_pdf
from semantic_chunker import chunk_text, generate_embeddings
from vector_db import create_document_store, add_document, get_vector_store
from rag_system import process_query_with_rag
from conversation_memory import ConversationMemory

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///educhat.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'pdf' ,  'PDF'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
db = SQLAlchemy(app)

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize conversation memory
conversation_memory = ConversationMemory(max_history=5)

# Database models
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    documents = db.relationship('Document', backref='owner', lazy=True)
    chat_messages = db.relationship('ChatMessage', backref='user', lazy=True)

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(255), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    processed = db.Column(db.Boolean, default=False)
    processing_error = db.Column(db.Text, nullable=True)  # Store error messages
    processing_attempts = db.Column(db.Integer, default=0)  # Track retry attempts
    
class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    is_user = db.Column(db.Boolean, default=True)  # True if from user, False if from bot
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    document_id = db.Column(db.Integer, db.ForeignKey('document.id'), nullable=True)  # Optional link to document

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Helper function to check allowed file extensions
def allowed_file(filename):
    """Validate file extension and basic content type"""
    # Check file extension
    if '.' not in filename:
        return False
    
    extension = filename.rsplit('.', 1)[1].lower()
    if extension not in app.config['ALLOWED_EXTENSIONS']:
        print(f"Invalid extension: {extension}")
        return False
    
    return True

# Routes for authentication
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Check if user exists
        user = User.query.filter_by(username=username).first()
        if user:
            flash('Username already exists')
            return redirect(url_for('signup'))
        
        email_exists = User.query.filter_by(email=email).first()
        if email_exists:
            flash('Email already in use')
            return redirect(url_for('signup'))
            
        # Create new user
        new_user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully, please login')
        return redirect(url_for('login'))
        
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if not user or not check_password_hash(user.password_hash, password):
            flash('Please check your login details and try again.')
            return redirect(url_for('login'))
            
        login_user(user)
        return redirect(url_for('chat'))
        
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# Main routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))
    return render_template('index.html')

@app.route('/chat')
@login_required
def chat():
    # Get user's documents
    documents = Document.query.filter_by(user_id=current_user.id).all()
    
    # Get recent chat messages
    messages = ChatMessage.query.filter_by(user_id=current_user.id).order_by(ChatMessage.timestamp).limit(50).all()
    
    return render_template('chat.html', documents=documents, messages=messages)

@app.route('/upload_document', methods=['POST'])
@login_required
def upload_document():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('chat'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('chat'))
    
    if file and allowed_file(file.filename):
        # Save file with unique name
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Create document record
        new_document = Document(
            filename=filename,
            filepath=filepath,
            user_id=current_user.id
        )
        db.session.add(new_document)
        db.session.commit()
        
        # Process the document asynchronously (in a real app, you'd use a task queue)
        try:
            # Start processing the document
            new_document.processing_attempts += 1
            db.session.commit()
            
            # Extract text from PDF
            text = process_pdf(filepath, skip_first=6, skip_last=2)
            
            # Chunk the text semantically
            chunks = chunk_text(text)
            
            # Generate embeddings for chunks
            embeddings = generate_embeddings(chunks)
            
            # Add to vector database
            add_document(new_document.id, chunks, embeddings)
            
            # Mark document as processed
            new_document.processed = True
            db.session.commit()
            
            flash('Document uploaded and processed successfully')
        except Exception as e:
            error_message = str(e)
            print(f"ERROR processing document: {error_message}")
            
            # Update document with error info
            new_document.processing_error = error_message
            if new_document.processing_attempts >= 2:
                # After two attempts, mark as failed permanently
                try:
                    # Remove the file
                    if os.path.exists(filepath):
                        os.remove(filepath)
                except Exception as file_error:
                    print(f"Error removing file: {str(file_error)}")
                
                # Delete the document record
                db.session.delete(new_document)
                db.session.commit()
                flash(f'Document processing failed: {error_message}. Please try a different file.')
            else:
                # Save the error but keep the document record
                db.session.commit()
                flash(f'Error processing document: {error_message}. You can retry or delete it.')
                
            return redirect(url_for('chat'))
    
    flash('Invalid file type')
    return redirect(url_for('chat'))

@app.route('/send_message', methods=['POST'])
@login_required
def send_message():
    """Process a message from the user and return a response"""
    message_content = request.form.get('message')
    document_id = request.form.get('document_id')  # Optional document context
    
    if not message_content:
        return jsonify({'status': 'error', 'message': 'No message content'})
    
    # Save user message
    user_message = ChatMessage(
        content=message_content,
        is_user=True,
        user_id=current_user.id,
        document_id=document_id if document_id else None
    )
    db.session.add(user_message)
    db.session.commit()
    
    # Get or create a persistent session ID
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    session_id = session['user_id']
    
    # Generate response
    try:
        if document_id:
            document_id = int(document_id)  # Ensure integer type
            print(f"Using document ID: {document_id} (type: {type(document_id)})")
            # Use RAG with the specified document
            document = Document.query.get(document_id)
            if document and document.processed:
                # Process with RAG
                response = process_query_with_rag(
                    query=message_content,
                    doc_id=document_id,
                    session_id=session_id,  # Make sure this is passed
                    conversation_memory=conversation_memory  # Make sure this is passed
                )
            else:
                response = "This document is still being processed or doesn't exist. Please try again later."
        else:
            # Standard LLM response without RAG
            response = process_query_with_rag(
                query=message_content,
                doc_id=None,
                session_id=session_id,  # Make sure this is passed
                conversation_memory=conversation_memory  # Make sure this is passed
            )
        
        # Save bot response
        bot_message = ChatMessage(
            content=response,
            is_user=False,
            user_id=current_user.id,
            document_id=document_id if document_id else None
        )
        db.session.add(bot_message)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': response,
            'timestamp': bot_message.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f"Error generating response: {str(e)}"
        })

@app.route('/get_documents', methods=['GET'])
@login_required
def get_documents():
    documents = Document.query.filter_by(user_id=current_user.id).all()
    docs_list = [{'id': doc.id, 'name': doc.filename, 'processed': doc.processed} for doc in documents]
    return jsonify({'documents': docs_list})

@app.route('/clear_chat', methods=['POST'])
@login_required
def clear_chat():
    """Clear the chat history"""
    # Delete all user's chat messages
    ChatMessage.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    
    # Clear conversation memory
    session_id = session.get('id', None)
    if session_id:
        conversation_memory.clear_history(session_id)
    
    return redirect(url_for('chat'))

@app.route('/remove_document/<int:doc_id>', methods=['POST'])
@login_required
def remove_document(doc_id):
    document = Document.query.get(doc_id)
    
    # Check if document exists and belongs to current user
    if not document or document.user_id != current_user.id:
        return jsonify({'status': 'error', 'message': 'Document not found'})
    
    # Try to remove the file
    try:
        if os.path.exists(document.filepath):
            os.remove(document.filepath)
    except Exception as e:
        print(f"Error removing file: {str(e)}")
    
    # Delete the document record
    db.session.delete(document)
    db.session.commit()
    
    return jsonify({'status': 'success', 'message': 'Document removed'})

@app.route('/retry_document/<int:doc_id>', methods=['POST'])
@login_required
def retry_document(doc_id):
    document = Document.query.get(doc_id)
    
    # Check if document exists and belongs to current user
    if not document or document.user_id != current_user.id:
        return jsonify({'status': 'error', 'message': 'Document not found'})
    
    if not os.path.exists(document.filepath):
        # File is missing, can't retry
        db.session.delete(document)
        db.session.commit()
        return jsonify({'status': 'error', 'message': 'Document file is missing. Please upload again.'})
    
    # Reset processing status
    document.processed = False
    document.processing_error = None
    document.processing_attempts += 1
    db.session.commit()
    
    # Start processing in a new thread
    # ... (similar to upload_document processing code)
    
    return jsonify({'status': 'success', 'message': 'Processing started'})

# Initialize database
with app.app_context():
    db.create_all()
    # Initialize vector DB on startup
    create_document_store()

if __name__ == '__main__':
    app.run(debug=True)
