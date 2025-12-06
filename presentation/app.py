import streamlit as st
import sys
import os

# Add the project root to the python path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from orchestration.controller import LangGraphAgentController
from agents.IOManager import IOManager
from agents.RetrievalAgent import RetrievalAgent
from agents.PredictiveAgent import PredictiveAgent
from agents.ReflectionAgent import ReflectionAgent

# --- Page Config ---
st.set_page_config(
    page_title="Diabetes Multi-Agentic System",
    page_icon="🩺",
    layout="wide"
)

# --- Initialization (Cached) ---
# @st.cache_resource
def initialize_system():
    """
    Initializes the Multi-Agent System components once.
    """
    print("Initializing MAS System...")
    io_manager = IOManager()
    retrieval_agent = RetrievalAgent()
    predictive_agent = PredictiveAgent()
    reflection_agent = ReflectionAgent()
    
    controller = LangGraphAgentController(
        io_manager=io_manager,
        retrieval_agent=retrieval_agent,
        predictive_agent=predictive_agent
    )
    return controller, reflection_agent

try:
    controller, reflection_agent = initialize_system()
except Exception as e:
    st.error(f"Failed to initialize system: {e}")
    st.stop()

# --- UI Layout ---
st.title("🩺 Diabetes Risk Assessment System")
st.markdown("""
This system uses a Multi-Agent Architecture to assess diabetes risk.
1.  Retrieval Agent: Searches medical knowledge base for context.
2.  Predictive Agent: Uses ML & LLM to calculate risk and generate a plan.
""")

# --- Sidebar ---
with st.sidebar:
    st.header("System Status")
    st.success("Agents Online")
    st.info("Models Loaded")
    st.warning("For Educational Use Only")

# --- Main Input ---
user_input = st.text_area("Describe your symptoms and condition:", height=100, placeholder="e.g., I have been feeling very thirsty lately, frequent urination, and my vision is sometimes blurry.")

if st.button("Analyze Risk", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some symptoms first.")
    else:
        with st.spinner("Agents are working... (Retrieving -> Predicting -> Planning)"):
            try:
                # Run the pipeline
                result_state = controller.run(user_input)
                st.session_state["result_state"] = result_state
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")

# --- Display Results (if available) ---
if "result_state" in st.session_state:
    result_state = st.session_state["result_state"]
    
    # --- Display Results ---
    st.divider()
    
    # 1. Top Level Metrics
    col1, col2 = st.columns(2)
    
    risk_level = result_state.get("predicted_risk", {}).get("risk", "Unknown")
    risk_score = result_state.get("predicted_risk", {}).get("score", 0.0)
    
    with col1:
        st.subheader("Risk Assessment")
        if risk_level.lower() == "high":
            st.error(f"Risk Level: {risk_level.upper()}")
        elif risk_level.lower() == "moderate":
            st.warning(f"Risk Level: {risk_level.upper()}")
        else:
            st.success(f"Risk Level: {risk_level.upper()}")
        
        st.metric("ML Confidence Score", f"{risk_score:.2%}")

    with col2:
        st.subheader("Analysis Summary")
        st.write(result_state.get("rag_summary", "No summary available."))

    # 2. Health Plan
    st.subheader("== Recommended Health Plan ==")
    plan = result_state.get("recommended_plan", "No plan generated.")
    
    # Format the plan properly if it's a dictionary
    if isinstance(plan, dict):
        sections = [
            ("immediateActions", "Immediate Actions"),
            ("medicationManagement", "Medication Management"),
            ("lifestyleInterventions", "Lifestyle Interventions"),
            ("followUp", "Follow-Up")
        ]

        for key, title in sections:
            content = plan.get(key)
            if content:
                st.markdown(f"**{title}:**")
                if isinstance(content, list):
                    for item in content:
                        st.markdown(f"- {item}")
                elif isinstance(content, str):
                    st.markdown(f"- {content}")
        
        # If dict is empty or has no recognized keys
        if not any(plan.get(k) for k in ["immediateActions", "medicationManagement", "lifestyleInterventions", "followUp"]):
            st.info("No specific recommendations generated.")
    else:
        # It's a string
        st.markdown(plan)

    # 3. Debug / Transparency
    with st.expander("View Retrieved Medical Context (RAG)"):
        st.markdown(result_state.get("retrieved_context", "No context retrieved."))
    
    with st.expander("View Raw Agent State"):
        st.json(result_state)

    # --- Feedback Section ---
    st.divider()
    st.subheader("Feedback")
    st.write("Please rate the quality of the health plan and provide any feedback.")
    
    with st.form("feedback_form"):
        rating = st.slider("Rating Slider (1-5 Stars)", 1, 5, 5)
        feedback_msg = st.text_area("Feedback Message", placeholder="Was this helpful? Please share any suggestions?")
        submitted = st.form_submit_button("Submit Feedback")
        
        if submitted:
            if reflection_agent.log_feedback(rating, feedback_msg):
                st.success("Thank you! Your feedback has been recorded.")
            else:
                st.error("Failed to record feedback.")
