from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Finding:
    """One violated criterion.

    criterion: C1 output, C2a snapshot state, C3 test/reward, C4 structure,
    A2 a stateless call changed the state.
    code: a finer label (for C4, which invariant).
    """

    criterion: str
    code: str
    detail: str
    node_id: Optional[str] = None
    path: Optional[tuple] = None
    tier: int = 4

    def __str__(self) -> str:
        where = f" node={self.node_id}" if self.node_id else ""
        return f"[{self.criterion}/{self.code}]{where} {self.detail}"


@dataclass
class ValidationReport:
    findings: List[Finding] = field(default_factory=list)
    checked: Dict[str, int] = field(default_factory=dict)

    def count(self, what: str, n: int = 1) -> None:
        self.checked[what] = self.checked.get(what, 0) + n

    @property
    def criteria_fired(self) -> set:
        return {f.criterion for f in self.findings}

    def by_criterion(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for f in self.findings:
            out[f.criterion] = out.get(f.criterion, 0) + 1
        return out
