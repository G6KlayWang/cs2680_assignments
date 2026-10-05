"""Model client: one chat-completions call with retries, trace logging and tool-call parsing.

Native OpenAI function calling is used when the course API accepts it. If a call with `tools` is
rejected (HTTP 400 mentioning tools/functions) the client switches to a text protocol: the tool
schemas go into the system prompt and the model writes <tool_call> blocks, which are parsed here.
Either way the solver sees the same ModelTurn / ToolCall objects.
"""

import json
import random
import re
import sys
import time

from . import config, prompts

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _debug(msg: str):
    print(f"[madsOpt] {msg}", file=sys.stderr, flush=True)


class ContextLengthError(Exception):
    """The API refused the request because the conversation is too long."""


class ToolCall:
    __slots__ = ("id", "name", "arguments", "error")

    def __init__(self, id: str, name: str, arguments=None, error=None):
        self.id, self.name, self.arguments, self.error = id, name, arguments, error


class ModelTurn:
    """One assistant reply: the message to append to the history, its text, its tool calls."""

    def __init__(self, message: dict, text: str, tool_calls: list):
        self.message, self.text, self.tool_calls = message, text, tool_calls


def _parse_args(raw):
    """Tool arguments -> (dict | None, error | None)."""
    if isinstance(raw, dict):
        return raw, None
    try:
        obj = json.loads(raw or "{}")
    except json.JSONDecodeError as e:
        return None, f"tool arguments are not valid JSON ({e}); resend the call with valid JSON"
    if not isinstance(obj, dict):
        return None, "tool arguments must be a JSON object"
    return obj, None


def to_text_protocol(messages: list, tools: list) -> list:
    """Rewrite an OpenAI tool-calling history for a model without native tools."""
    out = []
    for m in messages:
        role, content = m["role"], m.get("content") or ""
        if role == "system":
            specs = json.dumps([t["function"] for t in tools])
            m = {"role": "system", "content": content + "\n\n" + prompts.TEXT_PROTOCOL_APPENDIX.format(specs=specs)}
        elif role == "assistant" and m.get("tool_calls"):
            blocks = []
            for tc in m["tool_calls"]:
                args, _ = _parse_args(tc["function"].get("arguments"))
                blocks.append("<tool_call>\n" + json.dumps({"name": tc["function"]["name"], "arguments": args or {}}) + "\n</tool_call>")
            m = {"role": "assistant", "content": (content + "\n" + "\n".join(blocks)).strip()}
        elif role == "tool":
            m = {"role": "user", "content": f'<tool_result id="{m.get("tool_call_id", "")}">\n{content}\n</tool_result>'}
        else:
            m = {"role": role, "content": content}
        if out and out[-1]["role"] == "user" and m["role"] == "user":
            out[-1] = {"role": "user", "content": out[-1]["content"] + "\n\n" + m["content"]}
        else:
            out.append(m)
    return out


class ModelClient:
    def __init__(self, model_id: str, logger, tools: list, client=None):
        try:
            import openai
        except ImportError:                      # local dry runs with an injected fake client
            openai = None
        self.openai = openai
        if client is None:
            self.client = openai.OpenAI(base_url=config.get_base_url(), api_key=config.get_api_key(),
                                        timeout=config.API_TIMEOUT_S, max_retries=0)
        else:
            self.client = client
        self.model_id = model_id
        self.logger = logger
        self.tools = tools
        self.native_tools = True
        self.extra_body = {}
        if config.REASONING_EFFORT:
            self.extra_body["reasoning_effort"] = config.REASONING_EFFORT
        if config.MAX_COMPLETION_TOKENS:
            self.extra_body["max_completion_tokens"] = config.MAX_COMPLETION_TOKENS
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def set_model(self, model_id: str):
        self.model_id = model_id

    # -- error classification -------------------------------------------------------------
    def _is(self, e, name: str) -> bool:
        cls = getattr(self.openai, name, None) if self.openai else None
        return cls is not None and isinstance(e, cls)

    def _retryable(self, e) -> bool:
        if self._is(e, "APIConnectionError") or self._is(e, "RateLimitError") or self._is(e, "InternalServerError"):
            return True
        status = getattr(e, "status_code", None)
        return status in (408, 409, 425, 429) or (isinstance(status, int) and status >= 500)

    def _handle_bad_request(self, e) -> bool:
        """Adapt to a 400 if possible; True = retry immediately, otherwise raise."""
        msg = str(e).lower()
        if self.native_tools and ("tool" in msg or "function" in msg):
            _debug("the API rejected native tool calling; switching to the text tool protocol")
            self.native_tools = False
            return True
        for key in list(self.extra_body):
            if key in msg or key.replace("_", " ") in msg:
                _debug(f"the API rejected {key!r}; dropping it")
                self.extra_body.pop(key)
                return True
        if ("context" in msg and ("length" in msg or "token" in msg or "maximum" in msg)) \
                or "too long" in msg or "too many tokens" in msg or "maximum context" in msg:
            raise ContextLengthError(str(e)[:300])
        raise e

    # -- the call -------------------------------------------------------------------------
    def chat(self, messages: list, iteration: int) -> ModelTurn:
        attempt = 0
        while True:
            kwargs = {"model": self.model_id,
                      "messages": messages if self.native_tools else to_text_protocol(messages, self.tools)}
            if self.native_tools:
                kwargs["tools"] = self.tools
                kwargs["tool_choice"] = "auto"
            if self.extra_body:
                kwargs["extra_body"] = dict(self.extra_body)
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except Exception as e:                  # noqa: BLE001 — classified below
                if self._is(e, "BadRequestError") or getattr(e, "status_code", None) == 400:
                    if self._handle_bad_request(e):
                        continue
                if not self._retryable(e) or attempt >= config.API_MAX_RETRIES:
                    raise
                backoff = min(60.0, 2.0 * (2 ** attempt)) * (0.5 + random.random())
                self.logger.api_retry(iteration, f"{type(e).__name__}: {str(e)[:200]}", round(backoff, 1))
                _debug(f"model call failed ({type(e).__name__}); retry {attempt + 1} in {backoff:.0f}s")
                time.sleep(backoff)
                attempt += 1
        self.calls += 1
        usage = getattr(resp, "usage", None)
        pt = getattr(usage, "prompt_tokens", None)
        ct = getattr(usage, "completion_tokens", None)
        tt = getattr(usage, "total_tokens", None)
        if tt is None and pt is not None and ct is not None:
            tt = pt + ct
        self.prompt_tokens += pt or 0
        self.completion_tokens += ct or 0
        self.logger.api_request(iteration, pt, ct, tt, self.model_id)   # per-request model id (new trace format)
        return self._parse(resp, iteration)

    def _parse(self, resp, iteration: int) -> ModelTurn:
        choices = getattr(resp, "choices", None) or []
        msg = choices[0].message if choices else None
        text = (getattr(msg, "content", None) or "") if msg is not None else ""
        message = {"role": "assistant", "content": text}
        calls = []
        if self.native_tools:
            raw_calls = (getattr(msg, "tool_calls", None) or []) if msg is not None else []
            history_calls = []
            for i, tc in enumerate(raw_calls):
                fn = getattr(tc, "function", None)
                if fn is None:
                    continue
                cid = getattr(tc, "id", None) or f"call_{iteration}_{i}"
                raw_args = getattr(fn, "arguments", None)
                args, err = _parse_args(raw_args)
                history_calls.append({"id": cid, "type": "function",
                                      "function": {"name": fn.name, "arguments": raw_args if isinstance(raw_args, str) else json.dumps(args or {})}})
                calls.append(ToolCall(cid, fn.name, args, err))
            if history_calls:
                message["tool_calls"] = history_calls
        else:
            for i, m in enumerate(TOOL_CALL_RE.finditer(text)):
                cid = f"call_{iteration}_{i}"
                try:
                    obj = json.loads(m.group(1))
                    name = obj.get("name") if isinstance(obj, dict) else None
                    if not name:
                        raise ValueError("missing \"name\"")
                    args, err = _parse_args(obj.get("arguments", {}))
                    calls.append(ToolCall(cid, str(name), args, err))
                except (json.JSONDecodeError, ValueError) as e:
                    calls.append(ToolCall(cid, "invalid", None, f"malformed <tool_call> block: {e}"))
        return ModelTurn(message, text, calls)

    def tool_result_message(self, call: ToolCall, content: str) -> dict:
        if self.native_tools:
            return {"role": "tool", "tool_call_id": call.id, "content": content}
        return {"role": "user", "content": f'<tool_result id="{call.id}" name="{call.name}">\n{content}\n</tool_result>'}
