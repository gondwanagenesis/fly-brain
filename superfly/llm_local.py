"""Local language model for the prosthesis, through an Ollama server.

    ollama pull qwen2.5:1.5b        # any small instruct model
    SUPERFLY_LLM=ollama SUPERFLY_OLLAMA_MODEL=qwen2.5:1.5b python -m superfly.chat

A small local model is a reasonable choice here: the prosthesis only has to
render a frame of a few dozen numbers into a sentence and map a sentence onto
a fixed menu of senses. The faithfulness check in superfly/language.py applies
to every backend equally, so a weaker model costs fluency, not fidelity.
"""
from __future__ import annotations

import json
import os
import urllib.request

from superfly.language import (PARSE_SYSTEM, SPEAK_SYSTEM, Plan, ThoughtFrame,
                             plan_from_json, plan_schema)


class OllamaBackend:
    name = "ollama"

    def __init__(self, model=None, host=None, timeout=60):
        self.model = model or os.environ.get("SUPERFLY_OLLAMA_MODEL", "qwen2.5:1.5b")
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
                     ).rstrip("/")
        self.timeout = timeout

    def _chat(self, system, user, fmt=None):
        body = {"model": self.model, "stream": False,
                "options": {"temperature": 0.2},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}]}
        if fmt is not None:
            body["format"] = fmt
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["message"]["content"]

    def speak(self, frame: ThoughtFrame) -> str:
        return self._chat(SPEAK_SYSTEM, json.dumps(frame.compact())).strip()

    def parse(self, text: str) -> Plan:
        return plan_from_json(json.loads(self._chat(PARSE_SYSTEM, text,
                                                    fmt=plan_schema())))
