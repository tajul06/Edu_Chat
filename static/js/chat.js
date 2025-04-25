$(document).ready(function() {
    // DOM elements
    const chatHistory = document.getElementById('chatHistory');
    const messageForm = $('#messageForm');
    const messageInput = $('#messageInput');
    const typingIndicator = $('#typingIndicator');
    const documentList = $('#documentList');
    const activeDocumentBadge = $('#activeDocument');
    
    // Initialize chat
    initializeChat();
    
    // Handle document selection
    $(document).on('click', '.document-item', function() {
        selectDocument($(this));
    });
    
    // Handle message submission
    messageForm.submit(function(e) {
        e.preventDefault();
        sendMessage();
    });
    
    // Initialize chat interface
    function initializeChat() {
        // Scroll chat to bottom on load
        scrollToBottom();
        
        // Check if there's a document to pre-select (e.g., from previous session)
        const storedDocId = localStorage.getItem('selectedDocumentId');
        if (storedDocId) {
            const storedDocElement = $(`.document-item[data-document-id="${storedDocId}"]`);
            if (storedDocElement.length > 0) {
                selectDocument(storedDocElement);
            }
        }
    }
    
    // Select a document for context
    function selectDocument(docElement) {
        // Update UI
        $('.document-item').removeClass('active');
        docElement.addClass('active');
        
        // Update form and badge
        const docId = docElement.data('document-id');
        const docName = docElement.data('document-name');
        
        $('#documentId').val(docId);
        activeDocumentBadge.text(docName).show();
        
        // Store selection in local storage
        localStorage.setItem('selectedDocumentId', docId);
        
        // Optional: Inform user their context has changed
        if (chatHistory.children.length > 0) {
            addSystemMessage('Document context changed to: ' + docName);
        }
    }
    
    // Send a message to the server
    function sendMessage() {
        const message = messageInput.val().trim();
        
        if (message === '') return;
        
        // Clear input
        messageInput.val('');
        messageInput.focus();
        
        // Add user message to chat
        addUserMessage(message);
        
        // Show typing indicator
        typingIndicator.show();
        
        // Send message to server
        $.post('/send_message', {
            message: message,
            document_id: $('#documentId').val()
        })
        .done(function(response) {
            // Hide typing indicator
            typingIndicator.hide();
            
            if (response.status === 'success') {
                // Add bot response to chat
                addBotMessage(response.message, response.timestamp);
            } else {
                // Show error
                addSystemMessage('Error: ' + response.message);
            }
        })
        .fail(function(xhr, status, error) {
            typingIndicator.hide();
            addSystemMessage('Error: Could not connect to server');
            console.error("AJAX Error:", status, error);
        });
    }
    
    // Add a user message to the chat
    function addUserMessage(message) {
        const messageHtml = `
            <div class="message user-message">
                <div class="message-content">${escapeHtml(message)}</div>
                <div class="message-time">${getCurrentTime()}</div>
            </div>
        `;
        $('#chatHistory').append(messageHtml);
        scrollToBottom();
    }
    
    // Add a bot message to the chat
    function addBotMessage(message, timestamp) {
        const time = timestamp || getCurrentTime();
        const messageHtml = `
            <div class="message bot-message">
                <div class="message-content">${message}</div>
                <div class="message-time">${time}</div>
            </div>
        `;
        $('#chatHistory').append(messageHtml);
        scrollToBottom();
        
        // Apply syntax highlighting to code blocks if needed
        if (typeof hljs !== 'undefined') {
            $('pre code').each(function(i, block) {
                hljs.highlightBlock(block);
            });
        }
    }
    
    // Add a system message (notifications, errors, etc)
    function addSystemMessage(message) {
        const messageHtml = `
            <div class="message system-message">
                <div class="message-content"><i class="fas fa-info-circle me-2"></i>${message}</div>
                <div class="message-time">${getCurrentTime()}</div>
            </div>
        `;
        $('#chatHistory').append(messageHtml);
        scrollToBottom();
    }
    
    // Get current time formatted as HH:MM
    function getCurrentTime() {
        const now = new Date();
        return now.getHours().toString().padStart(2, '0') + ':' + 
               now.getMinutes().toString().padStart(2, '0');
    }
    
    // Escape HTML to prevent XSS
    function escapeHtml(text) {
        return $('<div>').text(text).html();
    }
    
    // Scroll chat to bottom
    function scrollToBottom() {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
    
    // Refresh document list periodically to check for processing status
    function refreshDocumentList() {
        // Clear any existing messages
        $('#messages').empty();
        
        $.get('/get_documents')
            .done(function(data) {
                if (data.documents) {
                    updateDocumentList(data.documents);
                }
            });
    }
    
    // Update the document list with fresh data
    function updateDocumentList(documents) {
        // Save current selection
        const currentDocId = $('#documentId').val();
        
        // Clear and rebuild list
        documentList.empty();
        
        if (documents.length === 0) {
            documentList.html(`
                <div class="text-center text-muted py-3">
                    <i class="fas fa-folder-open mb-2" style="font-size: 2rem;"></i>
                    <p class="mb-0">No documents yet</p>
                </div>
            `);
            return;
        }
        
        documents.forEach(function(doc) {
            const isActive = doc.id.toString() === currentDocId;
            const isProcessed = doc.processed;
            const hasError = doc.processing_error;
            
            let statusDisplay = '';
            let actionButtons = '';
            
            if (isProcessed) {
                statusDisplay = '<span class="badge bg-success">Ready</span>';
            } else if (hasError) {
                statusDisplay = '<span class="badge bg-danger">Failed</span>';
                actionButtons = `
                    <button type="button" class="btn btn-sm btn-warning retry-document" 
                            data-document-id="${doc.id}" title="Try again">
                        <i class="fas fa-redo"></i>
                    </button>
                    <button type="button" class="btn btn-sm btn-danger remove-document" 
                            data-document-id="${doc.id}" title="Remove document">
                        <i class="fas fa-trash"></i>
                    </button>
                `;
            } else {
                statusDisplay = '<span class="badge bg-warning">Processing <i class="fas fa-spinner fa-spin"></i></span>';
                actionButtons = `
                    <button type="button" class="btn btn-sm btn-danger cancel-processing" 
                            data-document-id="${doc.id}" title="Cancel processing">
                        <i class="fas fa-times"></i>
                    </button>
                `;
            }
            
            documentList.append(`
                <div class="list-group-item document-item-container">
                    <button type="button" class="list-group-item-action document-item ${isActive ? 'active' : ''} ${!isProcessed ? 'disabled' : ''}"
                            data-document-id="${doc.id}" data-document-name="${doc.name}" style="border:none; background:none; width:75%; text-align:left;">
                        <i class="fas ${isProcessed ? 'fa-file-pdf' : hasError ? 'fa-exclamation-triangle' : 'fa-spinner fa-spin'} me-2"></i>
                        ${escapeHtml(doc.name)}
                    </button>
                    ${statusDisplay}
                    <div class="float-end">
                        ${actionButtons}
                    </div>
                </div>
            `);
        });
        
        // Attach event handlers for buttons
        $('.remove-document').click(function(e) {
            e.stopPropagation();
            const docId = $(this).data('document-id');
            removeDocument(docId);
        });
        
        $('.retry-document').click(function(e) {
            e.stopPropagation();
            const docId = $(this).data('document-id');
            retryDocument(docId);
        });
    }
    
    function removeDocument(docId) {
        if (confirm('Remove this document?')) {
            $.post(`/remove_document/${docId}`)
                .done(function(response) {
                    if (response.status === 'success') {
                        refreshDocumentList();
                        addSystemMessage('Document removed');
                    } else {
                        addSystemMessage('Error: ' + response.message);
                    }
                })
                .fail(function() {
                    addSystemMessage('Error: Could not remove document');
                });
        }
    }
    
    function retryDocument(docId) {
        $.post(`/retry_document/${docId}`)
            .done(function(response) {
                if (response.status === 'success') {
                    refreshDocumentList();
                    addSystemMessage('Processing document again...');
                } else {
                    addSystemMessage('Error: ' + response.message);
                }
            })
            .fail(function() {
                addSystemMessage('Error: Could not retry document processing');
            });
    }
    
    // Check for unprocessed documents and refresh their status
    function checkUnprocessedDocuments() {
        const hasUnprocessed = $('.document-item:has(.fa-spinner)').length > 0;
        if (hasUnprocessed) {
            refreshDocumentList();
            // Schedule next check in 5 seconds
            setTimeout(checkUnprocessedDocuments, 5000);
        }
    }
    
    // Start checking for unprocessed documents if any
    checkUnprocessedDocuments();
});