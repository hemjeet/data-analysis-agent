# Agentic Data Analysis Agent

An intelligent, agentic data analysis assistant built using Anthropic's Claude models (`claude-haiku` and `claude-sonnet`) and Python's `pandas` library. The agent can dynamically execute Python code in a sandboxed environment to interact with and extract insights from datasets (e.g., `salary_info.csv`).

## Features

- **Dynamic Code Execution**: The agent automatically writes and executes Python/Pandas code to precisely answer your questions about the dataset.
- **Model Routing**: Dynamically routes simple questions to faster/cheaper models (like Haiku) and complex analytical questions to more capable models (like Sonnet) to balance speed, intelligence, and cost (`cost_control.py`).
- **Cost & Token Tracking**: Keeps a running tab on token usage and API costs, warning you if the context window gets too large.
- **Sandboxed Execution**: `run_python` operates securely with restricted builtins, blocked imports (e.g., `os`, `requests`), and maximum execution limits to prevent arbitrary code execution (`guradrails.py`).
- **Robust Retry Logic**: Built-in exponential backoff for API limits, tool error budgets (avoids infinite error loops), and loop detection (`retry.py`).
- **Anti-Hedging Guardrails**: Validates that the agent provides data-backed answers rather than estimations or fabricated numbers.
- **Persistent Memory**: Chat context is automatically saved to `memory.json` so you can pick up where you left off.

## Requirements

- Python 3.9+
- `pandas`
- `anthropic`
- `python-dotenv`

## Setup & Usage

1. Create a `.env` file in the root directory and add your Anthropic API key:
   ```env
   ANTHROPIC_API_KEY=your_api_key_here
   ```

2. Make sure your dataset (e.g., `salary_info.csv`) is present in the working directory.

3. Run the interactive chat agent:
   ```bash
   python chat.py
   ```

4. Ask natural language questions about your dataset!
   - *"What is the average salary by department?"*
   - *"Who has the most experience?"*
   - *"Is there a correlation between years of experience and salary?"*

Type `reset` to clear the conversation memory, or `exit` to quit.
