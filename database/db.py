"""
Database Management Module (src/database.py)

Stores and manages prediction logs and user feedback:
- id (Primary Key)
- message_text (Raw content)
- type (SMS, Email, Phishing, URL)
- prediction (Spam, Legitimate (Ham))
- probability (Spam probability float)
- model_used (e.g. TFIDF + svm_rbf)
- timestamp (ISO-8601 string)
- user_feedback (Optional user verification: 'Correct', 'Incorrect: Actually Spam', etc.)

Reads connection configuration dynamically from .env with fallback to SQLite.
"""

from __future__ import annotations

import datetime
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

DB_TYPE = os.getenv("DB_TYPE", "sqlite").lower()
SQLITE_DB_PATH = PROJECT_ROOT / os.getenv("SQLITE_DB_PATH", "data/predictions.db")
SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


class DatabaseManager:
    """
    Unified Database abstraction layer supporting SQLite by default,
    with extensible hooks for MySQL and MongoDB.
    """

    def __init__(self, db_type: str = DB_TYPE):
        self.db_type = db_type
        if self.db_type == "sqlite":
            self.db_path = str(SQLITE_DB_PATH)
            self._init_sqlite()
        elif self.db_type == "mysql":
            self._init_mysql()
        elif self.db_type == "mongodb":
            self._init_mongodb()
        else:
            self.db_path = str(SQLITE_DB_PATH)
            self._init_sqlite()

    def _get_sqlite_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sqlite(self):
        """Create tables if they do not exist."""
        with self._get_sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_text TEXT NOT NULL,
                    type TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    probability REAL NOT NULL,
                    model_used TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_feedback TEXT
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_type ON predictions(type);
            """)
            conn.commit()

    def _init_mysql(self):
        """Placeholder initialization for MySQL."""
        # Configured via .env: MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
        pass

    def _init_mongodb(self):
        """Placeholder initialization for MongoDB."""
        # Configured via .env: MONGODB_URI, MONGODB_DATABASE
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # Core Database CRUD Operations
    # ─────────────────────────────────────────────────────────────────────────

    def save_prediction(
        self,
        message_text: str,
        msg_type: str,
        prediction: str,
        probability: float,
        model_used: str,
        user_feedback: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> int:
        """
        Insert a prediction record into the database.
        Returns the new inserted record ID.
        """
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self._get_sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (
                    message_text,
                    type,
                    prediction,
                    probability,
                    model_used,
                    timestamp,
                    user_feedback
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                str(message_text),
                str(msg_type),
                str(prediction),
                float(probability),
                str(model_used),
                str(timestamp),
                user_feedback
            ))
            conn.commit()
            return int(cursor.lastrowid)

    def update_feedback(self, prediction_id: int, feedback: str) -> bool:
        """
        Attach user feedback to an existing prediction record.
        """
        with self._get_sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE predictions
                SET user_feedback = ?
                WHERE id = ?
            """, (str(feedback), int(prediction_id)))
            conn.commit()
            return cursor.rowcount > 0

    def get_history(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Fetch recent predictions in reverse chronological order.
        """
        with self._get_sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    id,
                    message_text,
                    type,
                    prediction,
                    probability,
                    model_used,
                    timestamp,
                    user_feedback
                FROM predictions
                ORDER BY id DESC
                LIMIT ? OFFSET ?
            """, (int(limit), int(offset)))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_stats(self) -> Dict[str, Any]:
        """
        Aggregate totals, spam rates, and breakdown by channel type.
        """
        with self._get_sqlite_connection() as conn:
            cursor = conn.cursor()

            # Total counts
            cursor.execute("SELECT COUNT(*) AS total FROM predictions")
            total = cursor.fetchone()["total"]

            # Spam count
            cursor.execute("SELECT COUNT(*) AS spam_count FROM predictions WHERE prediction = 'Spam'")
            spam_count = cursor.fetchone()["spam_count"]
            ham_count = total - spam_count
            spam_rate = round((spam_count / max(total, 1)) * 100.0, 2)

            # Breakdown by type
            cursor.execute("""
                SELECT type, COUNT(*) AS count
                FROM predictions
                GROUP BY type
            """)
            by_type = {row["type"]: row["count"] for row in cursor.fetchall()}

            # User feedback count
            cursor.execute("""
                SELECT COUNT(*) AS fb_count
                FROM predictions
                WHERE user_feedback IS NOT NULL AND user_feedback != ''
            """)
            feedback_count = cursor.fetchone()["fb_count"]

            return {
                "total_checked": total,
                "spam_count": spam_count,
                "ham_count": ham_count,
                "spam_percentage": spam_rate,
                "breakdown_by_type": by_type,
                "user_feedback_count": feedback_count,
                "db_backend": self.db_type
            }

    def get_prediction_by_id(self, prediction_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a specific prediction by primary key id."""
        with self._get_sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions WHERE id = ?", (int(prediction_id),))
            row = cursor.fetchone()
            return dict(row) if row else None


# Global database instance
db = DatabaseManager()

