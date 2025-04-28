# UT CS Research Repository

This repository contains code and findings related to ongoing AI and LLM research projects conducted at UT CS. The projects are organized into phases.

## Repository Structure

```
.
├── .gitignore
├── Phase_1/            # Research on LLM-based Planning Simulation
│   ├── plan_sim/       # Core planning simulation code (LangGraph, OpenAI, Qdrant)
│   ├── tests/
│   ├── README.md       # Detailed setup and explanation for Phase 1
│   └── ...             # Other config and results files
├── Phase_2/            # Research on AI Agent Interaction and Persona Modeling
│   ├── persona_clustering/ # Topic modeling (NMF/LDA) to generate buyer personas
│   │   ├── persona_generator_refactored/ # Code for persona generation
│   │   ├── tests/
│   │   ├── README.md   # Detailed setup and explanation for persona clustering
│   │   └── ...         # Output directories, config
│   └── eBay_Simulation/    # Simulates sales conversations (LangGraph, Gemini, eBay API)
│       ├── core/         # Core sales simulation logic
│       ├── data/         # Persona prompts and product indices (used by simulation)
│       ├── tests/
│       ├── tools/        # eBay API integration
│       ├── README.md     # Detailed setup and explanation for eBay simulation
│       └── ...           # DB setup, reporting scripts, config
└── README.md           # This file (repository overview)
```

## Project Summaries

### Phase 1: LLM Planning Simulation

*   **Location:** `Phase_1/`
*   **Purpose:** Explores LLM-based planning simulations using LangGraph. The simulation generates steps to achieve a goal, considering initial knowledge, potential problems, and resource constraints.
*   **Key Technologies:** LangGraph, OpenAI, Qdrant (for semantic de-duplication).
*   **Details:** See `Phase_1/README.md` for setup and usage instructions.

### Phase 2: Persona Modeling and Sales Simulation

This phase contains two interconnected projects:

1.  **Persona Clustering:**
    *   **Location:** `Phase_2/persona_clustering/`
    *   **Purpose:** Uses topic modeling (NMF/LDA) on customer data to identify underlying topics and generate distinct buyer personas.
    *   **Key Technologies:** Python, likely scikit-learn/pandas.
    *   **Details:** See `Phase_2/persona_clustering/README.md`. *Note: Requires external data.*

2.  **eBay Sales Conversation Simulation:**
    *   **Location:** `Phase_2/eBay_Simulation/`
    *   **Purpose:** Simulates sales conversations between an AI seller agent and persona-based buyer agents (generated from the clustering project). Analyzes sales strategies and agent interactions.
    *   **Key Technologies:** LangGraph, Google Gemini, eBay API, PostgreSQL.
    *   **Details:** See `Phase_2/eBay_Simulation/README.md` for detailed setup (API keys, database) and execution instructions.

## Getting Started

To work with a specific project, navigate to its directory (e.g., `Phase_1/`, `Phase_2/eBay_Simulation/`) and follow the instructions in its corresponding `README.md` file for environment setup, dependencies, and running the code.
