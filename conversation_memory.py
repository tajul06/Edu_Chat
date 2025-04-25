import time
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class Message:
    """Represents a single message in the conversation"""
    content: str
    is_user: bool
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class ConversationMemory:
    """Manages conversation history for each user session"""
    
    def __init__(self, max_history: int = 10):
        """
        Initialize conversation memory
        
        Args:
            max_history: Maximum number of exchanges to remember
        """
        self.conversations: Dict[str, List[Message]] = {}
        self.max_history = max_history
    
    def add_message(self, session_id: str, content: str, is_user: bool) -> None:
        """
        Add a message to the conversation history
        
        Args:
            session_id: Unique identifier for the user session
            content: Message content
            is_user: True if the message is from the user, False if from the bot
        """
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        self.conversations[session_id].append(Message(content, is_user))
        
        # Trim history if it gets too long
        if len(self.conversations[session_id]) > self.max_history * 2:  # *2 for pairs of messages
            # Keep the most recent messages
            self.conversations[session_id] = self.conversations[session_id][-self.max_history * 2:]
    
    def get_history(self, session_id: str, limit: Optional[int] = None) -> List[Message]:
        """
        Get conversation history for a session
        
        Args:
            session_id: Unique identifier for the user session
            limit: Max number of messages to return (most recent first)
            
        Returns:
            List of messages in the conversation
        """
        if session_id not in self.conversations:
            return []
        
        history = self.conversations[session_id]
        if limit is not None:
            history = history[-limit:]
        
        return history
    
    def clear_history(self, session_id: str) -> None:
        """
        Clear conversation history for a session
        
        Args:
            session_id: Unique identifier for the user session
        """
        if session_id in self.conversations:
            self.conversations[session_id] = []
    
    def format_for_prompt(self, session_id: str, max_tokens: int = 1000) -> str:
        """
        Format conversation history for inclusion in a prompt
        
        Args:
            session_id: Unique identifier for the user session
            max_tokens: Approximate max tokens to include (rough estimate)
            
        Returns:
            Formatted conversation history string
        """
        if session_id not in self.conversations:
            return ""
        
        history = self.conversations[session_id]
        
        # Format the conversation
        formatted = []
        for msg in history:
            role = "User" if msg.is_user else "Assistant"
            formatted.append(f"{role}: {msg.content}")
        
        # Join with newlines
        result = "\n".join(formatted)
        
        # Simple token estimation (4 chars ≈ 1 token)
        if len(result) > max_tokens * 4:
            # Keep truncating until we're under the limit
            while len(result) > max_tokens * 4 and len(formatted) > 2:
                formatted = formatted[2:]  # Remove oldest exchange
                result = "\n".join(formatted)
        
        return result
