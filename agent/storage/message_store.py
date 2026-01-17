"""
Persistent SQLite-based KV storage for raw messages and their IDs.
Stores message_uid -> {text, timestamp, original_message} mappings.
"""
import logging
import sqlite3
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)


class MessageStore:
    """Thread-safe SQLite-based persistent KV store for raw messages."""
    
    def __init__(self, db_path: str = "/tmp/messages.db"):
        self.db_path = db_path
        self._lock = Lock()
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """Create database and table if they don't exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        message_uid TEXT PRIMARY KEY,
                        raw_message TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
                logger.info(f"Message store initialized at {self.db_path}")
            finally:
                conn.close()
    
    def store(self, message_uid: str, raw_message: str, timestamp: datetime) -> None:
        """
        Store a raw message with its metadata.
        
        Args:
            message_uid: Unique message identifier
            raw_message: Raw message content
            timestamp: Message timestamp
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO messages (message_uid, raw_message, timestamp)
                    VALUES (?, ?, ?)
                    """,
                    (message_uid, raw_message, timestamp.isoformat())
                )
                conn.commit()
                logger.debug(f"Stored message {message_uid} in persistent store")
            finally:
                conn.close()
    
    def get(self, message_uid: str) -> Optional[Dict]:
        """
        Retrieve a message by its UID.
        
        Args:
            message_uid: Message identifier
            
        Returns:
            Dict with keys: text, timestamp (or None if not found)
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute(
                    """
                    SELECT raw_message, timestamp
                    FROM messages
                    WHERE message_uid = ?
                    """,
                    (message_uid,)
                )
                row = cursor.fetchone()
                
                if row:
                    return {
                        "text": row[0],
                        "timestamp": datetime.fromisoformat(row[1])
                    }
                return None
            finally:
                conn.close()
    
    def clear(self) -> None:
        """Clear all stored messages. Useful for testing."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("DELETE FROM messages")
                conn.commit()
                logger.info("Message store cleared")
            finally:
                conn.close()
    
    def size(self) -> int:
        """Return the number of stored messages."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute("SELECT COUNT(*) FROM messages")
                return cursor.fetchone()[0]
            finally:
                conn.close()


# Global singleton instance
_message_store = MessageStore()


def get_message_store() -> MessageStore:
    """
    Get the global message store instance.
    
    Returns:
        MessageStore singleton
    """
    return _message_store
