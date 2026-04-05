from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class TornadoFactor:
    name: str
    impact: float


def tornado_rank(factors: Dict[str, float]) -> List[TornadoFactor]:
    """Sort factors by absolute impact descending."""
    ranked: List[Tuple[str, float]] = sorted(
        factors.items(), key=lambda kv: abs(kv[1]), reverse=True
    )
    return [TornadoFactor(name=k, impact=v) for k, v in ranked]
