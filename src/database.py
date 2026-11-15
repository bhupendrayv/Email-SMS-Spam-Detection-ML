"""
Database Management Module (src/database.py)

Stores and manages prediction logs and user feedback:
- id (Primary Key / Sequential ID)
- message_text (Raw content)
- type (SMS, Email, Phishing, URL)
- prediction (Spam, Legitimate (Ham))
- probability (Spam probability float)
- model_used (e.g. TFIDF + svm_rbf)
- timestamp (ISO-8601 string)
- user_feedback (Optional user verification: 'Correct', 'Incorrect: Actually Spam', etc.)

Reads connection configuration dynamically from .env with full MongoDB and SQLite support.
"""

from __future__ import annotations

import datetime
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger("spam_detection_db")

DB_TYPE = os.getenv("DB_TYPE", "sqlite").lower()
SQLITE_DB_PATH = PROJECT_ROOT / os.getenv("SQLITE_DB_PATH", "data/predictions.db")
SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "spam_detection_db")


class DatabaseManager:
    """
    Unified Database abstraction layer supporting MongoDB and SQLite.
    Dynamically switches based on DB_TYPE in .env, with automatic failover to SQLite.
    """

    def __init__(self, db_type: str = DB_TYPE):
        self.db_type = db_type
        if self.db_type == "mongodb":
            self._init_mongodb()
        elif self.db_type == "sqlite":
            self.db_path = str(SQLITE_DB_PATH)
            self._init_sqlite()
        else:
            self.db_type = "sqlite"
            self.db_path = str(SQLITE_DB_PATH)
            self._init_sqlite()

    # ─────────────────────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────────────────────

    def _init_sqlite(self):
        """Create SQLite tables and indexes if they do not exist."""
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
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_type ON predictions(type);")
            conn.commit()

    def _get_sqlite_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_mongodb(self):
        """Initialize MongoDB client, collection, and indexes."""
        try:
            from pymongo import MongoClient, ASCENDING, DESCENDING
            self.mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=2000)
            # Verify connectivity
            self.mongo_client.admin.command("ping")
            self.mongo_db = self.mongo_client[MONGODB_DATABASE]
            self.mongo_collection = self.mongo_db["predictions"]
            self.mongo_collection.create_index([("timestamp", DESCENDING)])
            self.mongo_collection.create_index([("type", ASCENDING)])
            self.mongo_collection.create_index([("id", DESCENDING)])
            logger.info(f"Connected to MongoDB at {MONGODB_URI} (Database: {MONGODB_DATABASE})")
        except Exception as e:
            logger.warning(f"MongoDB connection failed ({e}). Falling back to SQLite.")
            self.db_type = "sqlite"
            self.db_path = str(SQLITE_DB_PATH)
            self._init_sqlite()

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
        Returns the inserted record ID.
        """
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if self.db_type == "mongodb":
            try:
                last_doc = self.mongo_collection.find_one(sort=[("id", -1)])
                next_id = (int(last_doc["id"]) + 1) if (last_doc and "id" in last_doc) else 1
                doc = {
                    "id": next_id,
                    "message_text": str(message_text),
                    "type": str(msg_type),
                    "prediction": str(prediction),
                    "probability": float(probability),
                    "model_used": str(model_used),
                    "timestamp": str(timestamp),
                    "user_feedback": user_feedback
                }
                self.mongo_collection.insert_one(doc)
                return next_id
            except Exception as e:
                logger.error(f"Error saving to MongoDB: {e}. Writing to SQLite fallback.")

        # SQLite implementation (primary or fallback)
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
        if self.db_type == "mongodb":
            try:
                res = self.mongo_collection.update_one(
                    {"id": int(prediction_id)},
                    {"$set": {"user_feedback": str(feedback)}}
                )
                if res.matched_count > 0:
                    return True
            except Exception as e:
                logger.error(f"Error updating feedback in MongoDB: {e}")

        # SQLite implementation (primary or fallback)
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
        if self.db_type == "mongodb":
            try:
                cursor = (
                    self.mongo_collection.find({}, {"_id": 0})
                    .sort("id", -1)
                    .skip(int(offset))
                    .limit(int(limit))
                )
                return list(cursor)
            except Exception as e:
                logger.error(f"Error fetching history from MongoDB: {e}")

        # SQLite implementation (primary or fallback)
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
        if self.db_type == "mongodb":
            try:
                total = self.mongo_collection.count_documents({})
                spam_count = self.mongo_collection.count_documents({"prediction": "Spam"})
                ham_count = max(total - spam_count, 0)
                spam_rate = round((spam_count / max(total, 1)) * 100.0, 2)

                pipeline = [{"$group": {"_id": "$type", "count": {"$sum": 1}}}]
                by_type = {doc["_id"]: doc["count"] for doc in self.mongo_collection.aggregate(pipeline) if doc.get("_id")}

                fb_count = self.mongo_collection.count_documents({"user_feedback": {"$nin": [None, ""]}})

                return {
                    "total_checked": total,
                    "spam_count": spam_count,
                    "ham_count": ham_count,
                    "spam_percentage": spam_rate,
                    "breakdown_by_type": by_type,
                    "user_feedback_count": fb_count,
                    "db_backend": "mongodb"
                }
            except Exception as e:
                logger.error(f"Error calculating stats in MongoDB: {e}")

        # SQLite implementation (primary or fallback)
        with self._get_sqlite_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) AS total FROM predictions")
            total = cursor.fetchone()["total"]

            cursor.execute("SELECT COUNT(*) AS spam_count FROM predictions WHERE prediction = 'Spam'")
            spam_count = cursor.fetchone()["spam_count"]
            ham_count = max(total - spam_count, 0)
            spam_rate = round((spam_count / max(total, 1)) * 100.0, 2)

            cursor.execute("""
                SELECT type, COUNT(*) AS count
                FROM predictions
                GROUP BY type
            """)
            by_type = {row["type"]: row["count"] for row in cursor.fetchall()}

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
        if self.db_type == "mongodb":
            try:
                return self.mongo_collection.find_one({"id": int(prediction_id)}, {"_id": 0})
            except Exception as e:
                logger.error(f"Error fetching prediction by id from MongoDB: {e}")

        with self._get_sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions WHERE id = ?", (int(prediction_id),))
            row = cursor.fetchone()
            return dict(row) if row else None


# Global database instance
db = DatabaseManager()
