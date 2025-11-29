# Project Structure: Diabetes Multi-Agent System

This document outlines the modular, testable, and maintainable structure of the Diabetes Multi-Agent System.

## Directory Structure

```text
diabetes_mas/
├── agents/                 # Implementation of specific agents
│   ├── IOManager.py        # Handles input validation, augmentation, and output
│   ├── RetrievalAgent.py   # Handles interaction with the RAG system
│   ├── PredictiveAgent.py  # Generates risk predictions and health plans
│   └── ...
├── config/                 # Configuration files
│   ├── agents.yml          # Agent-specific settings
│   └── app.yml             # General application settings
├── controller/             # Orchestration logic
│   └── LangGraphAgentController.py # Main controller using LangGraph
├── interfaces/             # Abstract base classes (Contracts)
│   ├── agents/             # Interfaces for agents
│   ├── controllers/        # Interfaces for controllers
│   └── ...
├── pipeline/               # LangGraph pipeline definitions
│   ├── build_graph.py      # Constructs the state graph
│   └── nodes/              # Node implementations for the graph
├── rag/                    # Retrieval-Augmented Generation components
│   ├── chroma_db/          # Vector database storage
│   ├── generator/          # LLM generation logic
│   ├── retriever/          # Document retrieval logic
│   └── ...
├── system/                 # Core system components
│   ├── AgentState.py       # State definition for the graph
│   └── ...
└── tests/                  # Unit and integration tests
    ├── test_io_manager.py
    ├── test_pipeline.py
    └── ...
```

## Key Design Principles

### 1. Modularity
The project is divided into distinct modules based on functionality:
- **Agents:** Encapsulate specific logic (Input, Retrieval, Prediction).
- **RAG:** Self-contained module for knowledge retrieval and generation.
- **Pipeline:** Defines the flow of execution separately from the agent logic.
- **Interfaces:** Defines contracts that allow for easy swapping of implementations.

### 2. Testability
- **Interfaces:** The use of interfaces (e.g., `IInputManager`, `IRetrievalAgent`) allows for easy mocking of dependencies during unit testing.
- **Tests Directory:** A dedicated `tests/` folder contains tests mirroring the source structure, ensuring high code coverage.
- **Pure Functions:** Many components are designed as pure functions or stateless classes where possible, making them easier to test.

### 3. Maintainability
- **Configuration:** Hard-coded values are avoided by using YAML configuration files in the `config/` directory.
- **Type Hinting:** Python type hints are used throughout to improve code readability and catch errors early.
- **Clear Separation:** The separation of the `Controller` (orchestration) from the `Agents` (business logic) ensures that changes in workflow do not affect agent implementation and vice versa.
