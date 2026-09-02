"""Model-provider abstraction.

The agent depends on `LLMBackend`, never on a specific provider. Swapping the
runtime model is a backend change, not an architecture change.

Cost policy: no paid API is used anywhere in this project. `ClaudeCLIBackend`
shells out to the Claude Code CLI, which authenticates against the user's existing
Claude subscription (one-time `claude setup-token`). If that is unavailable the
system still runs end to end on the deterministic policies.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    meta: dict = field(default_factory=dict)


class LLMBackend:
    name = "base"

    def available(self) -> Tuple[bool, str]:
        raise NotImplementedError

    def complete(self, system: str, user: str) -> LLMResponse:
        raise NotImplementedError


class ClaudeCLIBackend(LLMBackend):
    """Headless Claude Code CLI. Zero marginal cost on a Claude subscription.

    Tool use is disabled so the CLI behaves as a plain completion endpoint: the
    only tools in this system are ours, executed by our own runtime.
    """
    name = "claude_cli"

    def __init__(self, model: str = "sonnet", timeout: int = 240,
                 binary: Optional[str] = None):
        self.model = model
        self.timeout = timeout
        self.binary = binary or os.environ.get("ADOPS_CLAUDE_BIN") or shutil.which("claude")
        self.calls = 0

    def available(self) -> Tuple[bool, str]:
        if not self.binary:
            return False, ("`claude` CLI not found on PATH. Install it with "
                           "`npm install -g @anthropic-ai/claude-code` (free).")
        try:
            probe = subprocess.run([self.binary, "-p", "--output-format", "json",
                                    "--model", self.model],
                                   input="Reply with the single word: ready",
                                   capture_output=True, text=True, timeout=90)
        except Exception as exc:                      # noqa: BLE001
            return False, "could not run the claude CLI: %s" % exc
        # The CLI exits non-zero on an auth/API error but still prints a JSON
        # envelope, so parse stdout first and only fall back to the exit code.
        try:
            data = json.loads(probe.stdout)
        except ValueError:
            return False, ("claude CLI exited %d without JSON output: %s"
                           % (probe.returncode, (probe.stderr or probe.stdout or "")[:200]))
        if data.get("is_error"):
            msg = str(data.get("result", ""))
            if "not logged in" in msg.lower() or "login" in msg.lower():
                return False, ("claude CLI is installed but not authenticated. Run "
                               "`claude setup-token` once in a terminal (uses your "
                               "existing subscription; no API billing).")
            return False, "claude CLI error: %s" % msg[:200]
        return True, "ready (model=%s, cost reported by CLI: %s)" % (
            self.model, data.get("total_cost_usd"))

    def complete(self, system: str, user: str) -> LLMResponse:
        if not self.binary:
            raise LLMError("claude CLI not available")
        cmd = [self.binary, "-p", "--output-format", "json", "--model", self.model,
               "--system-prompt", system, "--allowedTools", ""]
        try:
            proc = subprocess.run(cmd, input=user, capture_output=True, text=True,
                                  timeout=self.timeout)
        except subprocess.TimeoutExpired:
            raise LLMError("claude CLI timed out after %ds" % self.timeout)
        try:
            data = json.loads(proc.stdout)
        except ValueError:
            raise LLMError("claude CLI exited %d without JSON output: %s"
                           % (proc.returncode, (proc.stderr or proc.stdout or "")[:300]))
        if data.get("is_error"):
            raise LLMError("claude CLI error: %s" % str(data.get("result"))[:300])
        self.calls += 1
        usage = data.get("usage", {}) or {}
        return LLMResponse(text=str(data.get("result", "")), meta={
            "model": self.model,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cost_usd": data.get("total_cost_usd"),
            "duration_ms": data.get("duration_ms"),
        })


class ScriptedBackend(LLMBackend):
    """Deterministic canned responses. Used by tests to exercise the full agent
    runtime -- prompt assembly, parsing, retries, guardrails -- with no model."""
    name = "scripted"

    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.calls = 0
        self.prompts: List[Tuple[str, str]] = []

    def available(self) -> Tuple[bool, str]:
        return True, "scripted backend with %d responses" % len(self.responses)

    def complete(self, system: str, user: str) -> LLMResponse:
        self.prompts.append((system, user))
        if not self.responses:
            raise LLMError("scripted backend exhausted")
        self.calls += 1
        return LLMResponse(text=self.responses.pop(0), meta={"scripted": True})


class UnavailableBackend(LLMBackend):
    name = "unavailable"

    def __init__(self, reason: str):
        self.reason = reason

    def available(self) -> Tuple[bool, str]:
        return False, self.reason

    def complete(self, system: str, user: str) -> LLMResponse:
        raise LLMError(self.reason)


def get_backend(name: str = "claude_cli", **kwargs) -> LLMBackend:
    if name in ("claude_cli", "claude"):
        return ClaudeCLIBackend(**kwargs)
    if name == "scripted":
        return ScriptedBackend(kwargs.get("responses", []))
    return UnavailableBackend("unknown backend %r" % name)
