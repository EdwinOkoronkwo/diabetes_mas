import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.ReflectionAgent import ReflectionAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_analysis_job():
    """
    Simulates the background job that runs every other week.
    It initializes the ReflectionAgent and triggers the analysis.
    """
    logger.info("Starting scheduled feedback analysis job...")
    
    agent = ReflectionAgent()
    summary = agent.analyze_feedback()
    
    print("\n=== Feedback Analysis Report ===")
    print(f"Status: {summary.get('status')}")
    print(f"Total Reviews: {summary.get('total_reviews')}")
    print(f"Average Rating: {summary.get('average_rating')}")
    print("Recent Feedback Messages:")
    for msg in summary.get('recent_messages', []):
        print(f" - {msg}")
    print("================================\n")
    
    logger.info("Feedback analysis job completed.")

if __name__ == "__main__":
    run_analysis_job()
