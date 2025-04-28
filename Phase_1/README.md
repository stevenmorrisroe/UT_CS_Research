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

1.  **Navigate to the Phase_1 directory:**
    ```bash
    cd Phase_1
    ```
    *(All subsequent commands assume you are in the `Phase_1` directory)*

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv venv  # Use python3 or python depending on your system
    source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`
    ```
    *You should see `(venv)` appear at the start of your terminal prompt.*

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure your virtual environment is active before running pip)*

4.  **Set up environment variables:**
    *   Create a file named `.env` **directly inside the `Phase_1` directory**.
    *   Add your OpenAI API key to the `.env` file like this:
        ```dotenv
        OPENAI_API_KEY='your_openai_api_key_here'
        ```
    *   Replace `'your_openai_api_key_here'` with your actual key.

## Running the Simulation / Tests

The primary way to run the simulation is through the provided pytest tests, which execute predefined scenarios.

1.  **Ensure your virtual environment (`venv`) is active.** (See Setup Step 2)
2.  **Run pytest from the `Phase_1` directory:**
    ```bash
    pytest
    ```
    *Alternatively, if the venv is not active, you can run: `venv/bin/pytest`*

    This command will discover and execute the tests located in the `tests/` directory (specifically `test_graph_interaction.py`). The simulation progress for each scenario will be printed to your console.

    *Note: The tests make live calls to the OpenAI API and will incur costs.*

    **Troubleshooting:**
    *   **`OpenAIError: api_key... not set`**: Double-check that your `.env` file is correctly named, located in the `Phase_1` directory, and contains the `OPENAI_API_KEY` variable with your key. Ensure the virtual environment was active when installing requirements and running `pytest`.
    *   **`Recursion limit of X reached`**: Complex scenarios (like framing a house) might exceed the default step limit. If a test fails with this error, you can increase the limit by modifying the `recursion_limit` parameter within the `RunnableConfig` object in `tests/test_graph_interaction.py` for the specific scenario.

## Key Components

*   **LangGraph:** Used to define and execute the stateful planning simulation as a graph.
*   **OpenAI:** Language models (e.g., GPT-4o Mini) are used for generating ideas, predicting outcomes, making decisions, and summarizing state.
*   **Qdrant:** A vector database used locally to store generated ideas and check for semantic similarity to encourage novelty within simulation steps.
*   **Pydantic:** Used for defining the structure of the simulation state and ensuring type safety.

## Research Findings

See `results_comparison.md` for an analysis of example simulation runs comparing outcomes for different scenarios and highlighting the non-deterministic nature of the simulation for complex tasks. 