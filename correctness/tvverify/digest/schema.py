"""Layered state digests.

A digest is a set of named components, each hashed separately, so a mismatch
says *what* differed (e.g. "env: PATH changed") instead of just "different".
Only `strict` components take part in equality; advisory components (tmp,
processes) are recorded to explain output differences but never decide it.

Sandboxes return digests as plain dicts (Sandbox.state_digest) so tvclient does
not depend on this package; StateDigest.from_dict / to_dict convert.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple


def sha256_json(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class DigestComponent:
    name: str
    hash: str
    strict: bool = True
    # entry -> short content hash / value. Optional; makes diffs explainable.
    manifest: Optional[Dict[str, str]] = None

    @staticmethod
    def of(name: str, entries: Dict[str, str], strict: bool = True) -> "DigestComponent":
        return DigestComponent(name, sha256_json(entries), strict, dict(entries))


@dataclass(frozen=True)
class StateDigest:
    components: Dict[str, DigestComponent]
    # Digests are comparable only when the policy and the root image match.
    policy_hash: str = ""
    root_image_id: str = ""
    version: str = "1"
    probe_seconds: float = 0.0

    def strict_hash(self) -> str:
        return sha256_json(
            {n: c.hash for n, c in sorted(self.components.items()) if c.strict}
        )

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "policy_hash": self.policy_hash,
            "root_image_id": self.root_image_id,
            "probe_seconds": self.probe_seconds,
            "components": {
                n: {"hash": c.hash, "strict": c.strict, "manifest": c.manifest}
                for n, c in self.components.items()
            },
        }

    @staticmethod
    def from_dict(data: dict) -> "StateDigest":
        return StateDigest(
            components={
                n: DigestComponent(n, c["hash"], c.get("strict", True), c.get("manifest"))
                for n, c in data["components"].items()
            },
            policy_hash=data.get("policy_hash", ""),
            root_image_id=data.get("root_image_id", ""),
            version=data.get("version", "1"),
            probe_seconds=data.get("probe_seconds", 0.0),
        )


@dataclass(frozen=True)
class DigestDiff:
    comparable: bool
    equal_strict: bool
    # component -> up to top_k (entry, a_value, b_value)
    differing: Dict[str, List[Tuple[str, Optional[str], Optional[str]]]] = field(default_factory=dict)
    advisory_differing: FrozenSet[str] = frozenset()
    reason: str = ""


def diff_digests(a: StateDigest, b: StateDigest, top_k: int = 20) -> DigestDiff:
    if a.policy_hash != b.policy_hash or a.root_image_id != b.root_image_id:
        return DigestDiff(
            comparable=False,
            equal_strict=False,
            reason=(
                f"policy/root mismatch: ({a.policy_hash!r},{a.root_image_id!r}) vs "
                f"({b.policy_hash!r},{b.root_image_id!r})"
            ),
        )

    differing: Dict[str, List[Tuple[str, Optional[str], Optional[str]]]] = {}
    advisory = set()
    for name in sorted(set(a.components) | set(b.components)):
        ca, cb = a.components.get(name), b.components.get(name)
        if ca is not None and cb is not None and ca.hash == cb.hash:
            continue
        strict = (ca or cb).strict
        if not strict:
            advisory.add(name)
            continue
        entries: List[Tuple[str, Optional[str], Optional[str]]] = []
        ma = (ca.manifest if ca else None) or {}
        mb = (cb.manifest if cb else None) or {}
        for entry in sorted(set(ma) | set(mb)):
            if ma.get(entry) != mb.get(entry):
                entries.append((entry, ma.get(entry), mb.get(entry)))
            if len(entries) >= top_k:
                break
        if ca is None or cb is None:
            entries.insert(0, ("<component>", "present" if ca else None, "present" if cb else None))
        differing[name] = entries

    return DigestDiff(
        comparable=True,
        equal_strict=not differing,
        differing=differing,
        advisory_differing=frozenset(advisory),
    )


@dataclass(frozen=True)
class DigestPolicy:
    """What a probe hashes. Digests are comparable only under the same policy."""

    strict_components: FrozenSet[str] = frozenset({"fs", "env", "cwd"})
    advisory_components: FrozenSet[str] = frozenset({"tmp", "procs"})
    watch_roots: Tuple[str, ...] = ()
    exclude_globs: Tuple[str, ...] = ()
    # Paths found to be volatile by the baseline noise-floor study, per task.
    learned_excludes: Tuple[str, ...] = ()
    mode: str = "tree_walk"

    def policy_hash(self) -> str:
        return sha256_json(
            {
                "strict": sorted(self.strict_components),
                "advisory": sorted(self.advisory_components),
                "roots": list(self.watch_roots),
                "exclude": list(self.exclude_globs),
                "learned": list(self.learned_excludes),
                "mode": self.mode,
            }
        )[:16]
