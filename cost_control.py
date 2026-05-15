# cost_control.py
import anthropic

# ── Pricing table (update when Anthropic changes rates) ────
PRICES = {
    "claude-haiku-4-5": {
        "input": 0.00000080, "output": 0.000004,
        "cache_read": 0.00000008, "cache_write": 0.000001,
    },
    "claude-sonnet-4-20250514": {
        "input": 0.000003, "output": 0.000015,
        "cache_read": 0.0000003, "cache_write": 0.00000375,
    },
}

# ── 1. Prompt caching ──────────────────────────────────────
def make_cached_system(system_prompt: str) -> list:
    """
    Wraps your system prompt in a cache_control block.
    First call pays full price. Every subsequent call
    on the same prompt pays ~10% (cache hit).
    """
    return [{"type": "text", "text": system_prompt,
             "cache_control": {"type": "ephemeral"}}]


# ── 2. Model routing ───────────────────────────────────────
SIMPLE_SIGNALS  = ["average","mean","max","min","count",
                   "how many","total","sum","show","list",
                   "what is","top","bottom","highest","lowest"]

COMPLEX_SIGNALS = ["why","explain","compare","analyse","trend",
                   "predict","correlation","outlier","anomaly",
                   "pattern","relationship","suggest","recommend",
                   "across","between","over time","difference"]

MODELS = {
    "simple":  "claude-haiku-4-5",
    "complex": "claude-sonnet-4-20250514",
}
MAX_TOKENS = {
    "simple":  512,
    "complex": 1500,
}

def classify_query(query: str) -> str:
    low = query.lower()
    complex_hits = sum(1 for s in COMPLEX_SIGNALS if s in low)
    simple_hits  = sum(1 for s in SIMPLE_SIGNALS  if s in low)
    if complex_hits >= 2 or (complex_hits >= 1 and simple_hits == 0):
        return "complex"
    return "simple"

def route(query: str) -> tuple[str, int]:
    """Returns (model_name, max_tokens) for a query."""
    c = classify_query(query)
    return MODELS[c], MAX_TOKENS[c]


# ── 3. Context trimming ────────────────────────────────────
def trim_messages(messages: list, max_turns: int = 15) -> list:
    """Keep last max_turns user/assistant pairs."""
    boundaries = [i for i, m in enumerate(messages)
                  if m["role"] == "user"
                  and isinstance(m["content"], str)]
    if len(boundaries) <= max_turns:
        return messages
    return messages[boundaries[-max_turns]:]

def trim_tool_outputs(messages: list, max_len: int = 500) -> list:
    """Truncate long tool results in old turns."""
    out = []
    for m in messages:
        if m["role"] == "user" and isinstance(m["content"], list):
            content = []
            for b in m["content"]:
                if (b.get("type") == "tool_result"
                        and len(b.get("content", "")) > max_len):
                    b = {**b, "content": b["content"][:max_len] + "…[trimmed]"}
                content.append(b)
            m = {**m, "content": content}
        out.append(m)
    return out


# ── 4. Token counting + safety gate ───────────────────────
TOKEN_WARN  = 50_000
TOKEN_LIMIT = 150_000

def count_tokens(client: anthropic.Anthropic,
                 system, tools, messages) -> int:
    r = client.messages.count_tokens(
        model="claude-haiku-4-5",   # counting is model-agnostic
        system=system, tools=tools, messages=messages,
    )
    return r.input_tokens

def context_is_safe(client, system, tools, messages) -> bool:
    n = count_tokens(client, system, tools, messages)
    print(f"  [context: {n:,} tokens]")
    if n > TOKEN_LIMIT:
        print(f"Context too large ({n:,}). Clear history with 'reset'.")
        return False
    if n > TOKEN_WARN:
        print(f"Large context ({n:,} tokens). Consider 'reset' soon.")
    return True


# ── 5. Cost tracker ────────────────────────────────────────
class CostTracker:
    def __init__(self):
        self.calls = 0
        self.total_usd = 0.0
        self.by_model: dict[str, float] = {}

    def record(self, model: str, usage):
        p = PRICES.get(model, PRICES["claude-haiku-4-5"])
        cost = (
            getattr(usage, "input_tokens",  0) * p["input"]  +
            getattr(usage, "output_tokens", 0) * p["output"] +
            getattr(usage, "cache_read_input_tokens",    0) * p["cache_read"]  +
            getattr(usage, "cache_creation_input_tokens",0) * p["cache_write"]
        )
        self.calls += 1
        self.total_usd += cost
        self.by_model[model] = self.by_model.get(model, 0) + cost

    def summary(self) -> str:
        return (f"{self.calls} calls | "
                f"${self.total_usd:.5f} total | "
                f"cache savings visible in usage.cache_read_input_tokens")

    def print_breakdown(self):
        print(f"\n── Session cost ──────────────────")
        for model, cost in self.by_model.items():
            label = model.split("-")[1]          # haiku / sonnet
            print(f"  {label:10s} ${cost:.5f}")
        print(f"  {'TOTAL':10s} ${self.total_usd:.5f}")
        print(f"  {self.calls} API calls")