# retry.py
import time
import random
import logging
import anthropic

log = logging.getLogger(__name__)

# ── Error classification ───────────────────────────────────
RETRYABLE_ERRORS = (
    anthropic.RateLimitError,
    anthropic.APIStatusError,
    anthropic.APIConnectionError,
)

# ── Layer 1: API retry with exponential backoff ────────────
def api_call_with_retry(client: anthropic.Anthropic, max_retries: int = 4, **kwargs):
    """
    Wraps client.messages.create() with exponential backoff.
    Retries only on transient errors. Fails immediately on auth/bad-request.
    """
    for attempt in range(max_retries):
        try:
            return client.messages.create(**kwargs)

        except anthropic.AuthenticationError:
            log.error("Auth failed — check ANTHROPIC_API_KEY")
            raise  # never retry — won't fix itself

        except anthropic.BadRequestError as e:
            log.error(f"Bad request: {e}")
            raise  # never retry — fix the input

        except RETRYABLE_ERRORS as e:
            if attempt == max_retries - 1:
                log.error(f"Gave up after {max_retries} retries: {e}")
                raise
            wait = (2 ** attempt) + random.uniform(0, 1)  # jitter
            log.warning(f"Retry {attempt + 1}/{max_retries} in {wait:.1f}s — {type(e).__name__}")
            print(f"  [retry] API error, retrying in {wait:.1f}s...")
            time.sleep(wait)


# ── Layer 2: Tool error budget ─────────────────────────────
class ErrorBudget:
    """
    Tracks tool errors per agent turn.
    Raises if Claude keeps making the same mistake.
    """
    def __init__(self, max_errors: int = 2):
        self.max = max_errors
        self.count = 0
        self.last_error = ""

    def record(self, result: str) -> bool:
        """Returns True if the result is an error and budget is exceeded."""
        if ("Error:" in result) or ("Blocked:" in result):
            self.count += 1
            self.last_error = result
            if self.count > self.max:
                return True  # budget exceeded
        return False

    def exceeded(self) -> bool:
        return self.count > self.max


# ── Layer 3: Loop detection ────────────────────────────────
class LoopDetector:
    """
    Detects when the agent calls the same tool with the same
    input twice in a row — a sign it's stuck.
    """
    def __init__(self):
        self.history: list[str] = []

    def record(self, tool_name: str, tool_input: dict) -> bool:
        """Returns True if a loop is detected."""
        sig = f"{tool_name}:{sorted(tool_input.items())}"
        self.history.append(sig)
        if len(self.history) >= 2 and self.history[-1] == self.history[-2]:
            return True
        return False


# ── Layer 4: Answer quality check ─────────────────────────
HEDGE_WORDS = [
    "i think", "i believe", "approximately", "roughly",
    "around", "probably", "i'm not sure", "i assume",
    "it seems", "maybe", "perhaps",
]

def check_answer_quality(answer: str, tool_was_called: bool) -> str:
    """
    Appends a warning if the answer looks like a guess
    rather than being backed by actual tool output.
    """
    low = answer.lower()
    if not tool_was_called and any(h in low for h in HEDGE_WORDS):
        answer += (
            "\n\n⚠ This answer may be an estimate — "
            "I didn't run the calculation. Ask me to verify with the data."
        )
    return answer


# ── Full agent turn with all error handling ────────────────
def run_agent_turn(
    messages: list,
    client: anthropic.Anthropic,
    system: str,
    tools: list,
    run_tool_fn,        
    model: str = "claude-haiku-4-5",
    max_steps: int = 10,
    max_tokens: int = 4096,
) -> tuple[str, list]:
    """
    Runs one full agent turn with:
      - API retry on transient failures
      - Tool error budget (max 2 code errors per turn)
      - Loop detection
      - Answer quality check

    Returns (final_answer, updated_messages).
    """

    budget    = ErrorBudget(max_errors=2)
    detector  = LoopDetector()
    tool_called = False

    for step in range(max_steps):

        # ── API call with retry ───────────────────────────
        try:
            response = api_call_with_retry(
                client,
                model=model,
                max_tokens=max_tokens,
                system=system,
                tools=tools,
                messages=messages,
            )
        except Exception as e:
            return f"API failed after retries: {e}", messages

        content = response.content
        messages.append({"role": "assistant", "content": content})

        # ── Final answer ──────────────────────────────────
        if response.stop_reason != "tool_use":
            answer = " ".join(
                b.text for b in content if hasattr(b, "text")
            )
            if response.stop_reason == "max_tokens":
                answer += "\n\n[Warning: Response was truncated due to token limit]"
            answer = check_answer_quality(answer, tool_called)
            return answer, messages

        # ── Tool calls ────────────────────────────────────
        tool_results = []
        for block in content:
            if block.type != "tool_use":
                continue

            # loop detection
            if detector.record(block.name, block.input):
                return (
                    "I seem to be going in circles. "
                    "Could you rephrase the question?",
                    messages,
                )

            tool_called = True
            print(f"\n  -> {block.name}({list(block.input.keys())})")

            result = run_tool_fn(block.name, block.input)

            # error budget
            if budget.record(result):
                return (
                    f"I hit repeated errors trying to answer. "
                    f"Last error: {budget.last_error}",
                    messages,
                )

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        if not tool_results:
            return "Unexpected API response: no tools were called. Please try again.", messages

        messages.append({"role": "user", "content": tool_results})

    return (
        "Reached the step limit. "
        "Try asking a more specific question.",
        messages,
    )


# ── Streaming variant for Streamlit ───────────────────────
def run_agent_turn_streaming(
    messages: list,
    client: anthropic.Anthropic,
    system: str,
    tools: list,
    run_tool_fn,
    model: str = "claude-haiku-4-5",
    max_steps: int = 10,
    max_tokens: int = 4096,
):
    """
    Streaming version of run_agent_turn.

    Generator yielding event dicts:
      text_delta    — a chunk of streamed text
      thinking_done — signals that preceding text was intermediate (tool calls follow)
      tool_start    — a tool is about to execute
      tool_end      — tool finished with result
      end           — final answer + updated messages
      error         — unrecoverable error
    """
    budget = ErrorBudget(max_errors=2)
    detector = LoopDetector()
    tool_called = False

    for step in range(max_steps):

        # ── Streaming API call ────────────────────────────
        try:
            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                system=system,
                tools=tools,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    yield {"type": "text_delta", "text": text}
                response = stream.get_final_message()

        except anthropic.AuthenticationError:
            yield {"type": "error", "text": "Auth failed -- check ANTHROPIC_API_KEY"}
            return
        except anthropic.BadRequestError as e:
            yield {"type": "error", "text": f"Bad request: {e}"}
            return
        except Exception as e:
            yield {"type": "error", "text": f"API error: {e}"}
            return

        content = response.content
        messages.append({"role": "assistant", "content": content})

        # ── Final answer ──────────────────────────────────
        if response.stop_reason != "tool_use":
            answer = " ".join(
                b.text for b in content if hasattr(b, "text")
            )
            if response.stop_reason == "max_tokens":
                answer += "\n\n[Warning: Response was truncated due to token limit]"
            answer = check_answer_quality(answer, tool_called)
            yield {"type": "end", "answer": answer, "messages": messages}
            return

        # ── Tool-use response — preceding text was thinking ──
        yield {"type": "thinking_done"}

        tool_results = []
        for block in content:
            if block.type != "tool_use":
                continue

            if detector.record(block.name, block.input):
                yield {
                    "type": "end",
                    "answer": "I seem to be going in circles. Could you rephrase the question?",
                    "messages": messages,
                }
                return

            tool_called = True
            yield {"type": "tool_start", "name": block.name, "input": block.input}

            result = run_tool_fn(block.name, block.input)

            yield {"type": "tool_end", "name": block.name, "result": result}

            if budget.record(result):
                yield {
                    "type": "end",
                    "answer": f"I hit repeated errors. Last error: {budget.last_error}",
                    "messages": messages,
                }
                return

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        if not tool_results:
            yield {
                "type": "end",
                "answer": "Unexpected API response: no tools were called. Please try again.",
                "messages": messages,
            }
            return

        messages.append({"role": "user", "content": tool_results})

    yield {
        "type": "end",
        "answer": "Reached the step limit. Try asking a more specific question.",
        "messages": messages,
    }