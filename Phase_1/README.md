# LLM Planning Simulation Research

This repository contains code for ongoing research exploring LLM-based planning simulations using LangGraph. The simulation attempts to generate a sequence of steps to achieve a goal, considering initial knowledge (ground truths), potential problems (vulnerabilities), and resource constraints (budget, time). It uses Qdrant for semantic de-duplication of proposed steps and OpenAI models for reasoning.

## Project Structure

```
.
├── .git/
├── .gitignore
├── .pytest_cache/
├── plan_sim/           # Core simulation package
│   ├── __init__.py
│   ├── config.py       # Configuration definition (dataclass)
│   ├── env_tool.py     # Loads environment variables (.env)
│   ├── llm_utils.py    # LLM invocation utilities
│   ├── main.py         # Defines the LangGraph structure
│   ├── nodes.py        # Implements graph node logic & helpers
│   ├── prompts.py      # LLM prompt templates
│   ├── states.py       # Pydantic models for graph state
│   └── vector_store.py # Qdrant interaction (novelty check, storage)
├── qdrant_db/          # Local Qdrant data (ignored by git)
├── requirements.txt    # Python dependencies
├── results_comparison.md # Comparison of example simulation runs
├── tests/              # Pytest tests
│   ├── __init__.py
│   └── test_graph_interaction.py # Tests graph execution with scenarios
├── venv/               # Python virtual environment (ignored by git)
└── README.md           # This file
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    *   Create a file named `.env` in the project root directory (alongside `requirements.txt`).
    *   Add your OpenAI API key to the `.env` file:
        ```dotenv
        OPENAI_API_KEY='your_openai_api_key_here'
        ```

## Running the Simulation / Tests

The primary way to run the simulation is through the provided tests, which execute predefined scenarios.

1.  **Ensure your virtual environment is active.**
2.  **Run pytest from the project root directory:**
    ```bash
    pytest
    ```
    This will discover and run the tests in the `tests/` directory (e.g., `test_graph_interaction.py`). The output of the simulation steps for each scenario will be printed to the console.

    *Note: The tests make live calls to the OpenAI API and will incur costs.*

## Key Components

*   **LangGraph:** Used to define and execute the stateful planning simulation as a graph.
*   **OpenAI:** Language models (e.g., GPT-4o Mini) are used for generating ideas, predicting outcomes, making decisions, and summarizing state.
*   **Qdrant:** A vector database used locally to store generated ideas and check for semantic similarity to encourage novelty within simulation steps.
*   **Pydantic:** Used for defining the structure of the simulation state and ensuring type safety.

## Research Findings

See `results_comparison.md` for an analysis of example simulation runs comparing outcomes for different scenarios and highlighting the non-deterministic nature of the simulation for complex tasks. 