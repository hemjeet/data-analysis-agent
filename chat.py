import anthropic
import json
import pathlib
from dotenv import load_dotenv
import pandas as pd
import io
from contextlib import redirect_stdout
from config import get_system_prompt
from cost_control import CostTracker, trim_messages, trim_tool_outputs, context_is_safe, route, make_cached_system
from guradrails import validate_input
from retry import run_agent_turn

# ── Load env ───────────────────────────────────────────────
load_dotenv()

client = anthropic.Anthropic()

# ── Load dataset ───────────────────────────────────────────
df = pd.read_csv("salary_info.csv")

print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}\n")

# ── Tools ──────────────────────────────────────────────────
tools = [
    {
        "name": "run_python",
        "description": """
Run Python and pandas code to analyze the dataframe.The dataframe is already loaded as `df`.
Always use print() to show your results.
""",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python/pandas code to run. Always use print()."
                }
            },
            "required": ["code"]
        }
    }
]

# ── Tool functions ─────────────────────────────────────────
def run_python(code: str) -> str:
    output = io.StringIO()

    try:
        with redirect_stdout(output):
            exec(code, {"df": df, "pd": pd})

        result = output.getvalue()

        return result if result else "Code ran but no output. Did you use print()?"

    except Exception as e:
        return f"Error: {e}"


def run_tool(tool_name, tool_input):
    if tool_name == "run_python":
        return run_python(tool_input["code"])

    return f"Unknown tool: {tool_name}"


# ── Memory persistence ─────────────────────────────────────
MEM_FILE = pathlib.Path("memory.json")


def load_memory() -> list:
    if MEM_FILE.exists():
        try:
            return json.loads(MEM_FILE.read_text())
        except Exception:
            pass

    return []


def save_memory(msgs: list):
    MEM_FILE.write_text(json.dumps(msgs[-30:], default=str))


# ── History trimming ───────────────────────────────────────
MAX_TURNS = 15


def trim_history(msgs: list) -> list:
    if len(msgs) > MAX_TURNS * 2:
        return msgs[-(MAX_TURNS * 2):]

    return msgs
# ── one-time setup ─────────────────────────────────────────
system_prompt = get_system_prompt(df)
cached_system = make_cached_system(system_prompt)  
tracker = CostTracker()

# ── Load persisted messages ────────────────────────────────
messages = load_memory()

# ── Chat function ──────────────────────────────────────────
def chat(user_input):
    print(f"\nYou: {user_input}")
    print("-" * 40)
    global messages
    
    user_input = validate_input(user_input)
    
    messages.append({
        "role": "user",
        "content": user_input
    })

    model, max_tok = route(user_input)
    print(f"  [→ {model.split('-')[1]} | max {max_tok} tokens]")

    trimmed = trim_tool_outputs(trim_messages(messages))

    if not context_is_safe(client, cached_system, tools, trimmed):
        print("Type 'reset' to start a new session.")
        return
    
    answer, messages = run_agent_turn(
        user_msg    = user_input,
        messages    = trimmed,
        client      = client,
        system      = cached_system,   
        tools       = tools,
        run_tool_fn = run_tool,
        model       = model,           
        max_tokens  = max_tok,        
    )

    # Note: run_agent_turn doesn't return usage in this iteration.
    # Cost tracking would require modification of run_agent_turn.
    tracker.record(model, None)  # track cost placeholder
    print(f"Assistant: {answer}")
    save_memory(messages)

    # handle reset command in main loop:
    if user_input.lower() == "reset":
        messages = []
        save_memory(messages)
        print("Session cleared.")
        return

    
# ── Run it ─────────────────────────────────────────────────
if __name__ == "__main__":

    print("Data Analyst Agent ready!")
    print("Ask me anything about the dataset.")
    print("Type 'exit' to quit.\n")

    while True:

        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break

        if user_input:
            chat(user_input)