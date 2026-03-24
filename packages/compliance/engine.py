# Starter Python file
from dataclasses import dataclass
from typing import List
from pathlib import Path
import yaml, re

@dataclass
class CheckOutcome:
    block: bool
    flags: List[str]
    redactions: List[str]

class ComplianceEngine:
    def __init__(self, policy_path: str):
        self.policy_path = policy_path
        self._policy = {"policies": []}
        p = Path(policy_path)
        if p.exists():
            try:
                self._policy = yaml.safe_load(p.read_text(encoding="utf-8")) or {"policies": []}
            except Exception:
                self._policy = {"policies": []}

    def check(self, text: str, *, phase: str) -> CheckOutcome:
        flags: List[str] = []
        redactions: List[str] = []
        block = False
        content = text or ""

        for pol in self._policy.get("policies", []):
            if phase not in pol.get("phase", ["pre", "post"]):
                continue

            for rx in pol.get("match_regex", []):
                try:
                    if re.search(rx, content, flags=re.IGNORECASE):
                        flags.append(pol.get("id", "policy"))
                        if pol.get("action", "WARN").upper() == "BLOCK":
                            block = True
                except re.error:
                    continue

            for kw in pol.get("match", []):
                if kw.lower() in content.lower():
                    flags.append(pol.get("id", "policy"))
                    if pol.get("action", "WARN").upper() == "BLOCK":
                        block = True

        flags = list(dict.fromkeys(flags))
        return CheckOutcome(block=block, flags=flags, redactions=redactions)
