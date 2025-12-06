from agents.ReflectionAgent import ReflectionAgent
import os

def test_reflection_agent():
    agent = ReflectionAgent()
    
    # Test data
    user_id = "test_user"
    rating = 5
    feedback = "Great app! Very helpful."
    
    # Process feedback
    result = agent.process_feedback(user_id, rating, feedback)
    
    print(f"Result: {result}")
    
    # Check if file exists
    if os.path.exists("data/feedback_log.csv"):
        print("Feedback log file created.")
        with open("data/feedback_log.csv", "r") as f:
            content = f.read()
            print("File content:")
            print(content)
            if "test_user" in content and "Great app! Very helpful." in content:
                print("SUCCESS: Feedback recorded correctly.")
            else:
                print("FAILURE: Feedback not found in file.")
    else:
        print("FAILURE: Feedback log file not created.")

if __name__ == "__main__":
    test_reflection_agent()
