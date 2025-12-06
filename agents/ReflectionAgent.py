import csv
import os
import logging
from datetime import datetime
from typing import Dict, Any, List
from interfaces.agents.IReflectionAgent import IReflectionAgent

logger = logging.getLogger(__name__)

class ReflectionAgent(IReflectionAgent):
    """
    Implementation of the Reflection Agent.
    Stores feedback in a CSV file and provides analysis capabilities.
    """

    def __init__(self, storage_path: str = "data/feedback_log.csv"):
        self.storage_path = storage_path
        self._ensure_storage_exists()
        logger.info("[ReflectionAgent] Initialized.")

    def _ensure_storage_exists(self):
        """Ensures the CSV file and directory exist with headers."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        if not os.path.exists(self.storage_path):
            with open(self.storage_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Rating", "Message"])
            logger.info(f"[ReflectionAgent] Created storage file at {self.storage_path}")

    def log_feedback(self, rating: int, message: str) -> bool:
        """
        Logs user feedback to the CSV file.
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.storage_path, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, rating, message])
            
            logger.info(f"[ReflectionAgent] Feedback logged: {rating} stars")
            return True
        except Exception as e:
            logger.error(f"[ReflectionAgent] Failed to log feedback: {e}")
            return False

    def analyze_feedback(self) -> Dict[str, Any]:
        """
        Reads the CSV and calculates basic stats.
        This simulates the 'background job' analysis.
        """
        try:
            if not os.path.exists(self.storage_path):
                return {"status": "No data", "average_rating": 0, "total_reviews": 0}

            ratings = []
            messages = []
            
            with open(self.storage_path, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    try:
                        ratings.append(int(row["Rating"]))
                        messages.append(row["Message"])
                    except ValueError:
                        continue # Skip malformed rows

            total_reviews = len(ratings)
            average_rating = sum(ratings) / total_reviews if total_reviews > 0 else 0.0

            summary = {
                "status": "Success",
                "total_reviews": total_reviews,
                "average_rating": round(average_rating, 2),
                "recent_messages": messages[-5:] if messages else []
            }
            
            logger.info(f"[ReflectionAgent] Analysis complete. Avg Rating: {average_rating:.2f}")
            return summary

        except Exception as e:
            logger.error(f"[ReflectionAgent] Analysis failed: {e}")
            return {"status": "Error", "error": str(e)}
