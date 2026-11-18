"""Value-of-information gate (Section 5.4) - the economic core.

For candidate c against incumbent p:

    P(flip) = alpha * h(dir(c)) + (1 - alpha) * A
    A       = a * j + (1 - a) * (1 - j)
    dV      = w_acc * dAcc^ - |dC|
    VoI     = P(flip) * |dV|
    validate  iff  VoI > theta * tokenCost(validation)

h(dir) is the directive's historical flip rate from plan memory.  The
agreement term A turns the cheap local judge's opinion j = P_judge(c > p)
into a flip estimate, trusting it only as much as the judge has historically
agreed with the LLM validator (rate a).  With no history a = 0.5, so A = 0.5
and the term is uninformative, as it should be.

dAcc^ is the memory's prediction of the accuracy gain (clipped at 0 so a
candidate expected to be worse never *buys* validation through |dV|), and dC
is the lifetime cost difference over ``reuse_runs`` executions.  Both sides
of the inequality are in US dollars.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .cost_model import CostEstimate, HybridCostModel
from .memory import PlanMemory
from .pipeline import Pipeline


@dataclass
class GateDecision:
    validate: bool
    voi: float
    threshold: float
    p_flip: float
    p_raw: float
    h_dir: float
    h_trials: int
    judge_p: float | None
    agreement: float
    d_acc: float
    d_cost: float
    d_value: float
    val_cost_usd: float
    val_cost_tokens: float
    rationale: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class VoIGate:
    def __init__(self, gate_settings, memory: PlanMemory, cost_model: HybridCostModel) -> None:
        self.cfg = gate_settings
        self.memory = memory
        self.cost = cost_model

    def flip_probability(self, directive: str, judge_p: float | None) -> tuple[float, float, float, int, float]:
        h, trials = self.memory.flip_rate(directive)
        a, n_agree = self.memory.agreement_rate(directive)
        if judge_p is None or n_agree < self.cfg.min_agreement_samples:
            term = 0.5
        else:
            term = a * judge_p + (1 - a) * (1 - judge_p)
        p_raw = self.cfg.alpha * h + (1 - self.cfg.alpha) * term
        p = self.memory.calibrated(p_raw) if self.cfg.calibrate else p_raw
        return min(max(p, 0.01), 0.99), p_raw, h, trials, a

    def decide(self, directive: str, family: str, cand: Pipeline, inc: Pipeline, stats: dict,
               cost_c: CostEstimate, cost_p: CostEstimate, judge_p: float | None,
               n_items: int, n_criteria: int, theta: float | None = None) -> GateDecision:
        theta = self.cfg.theta if theta is None else theta
        p_flip, p_raw, h, trials, a = self.flip_probability(directive, judge_p)
        gain, _ = self.memory.predicted_gain(directive, family, self.cfg.prior_gain)
        d_acc = max(gain, 0.0)
        d_cost = self.cfg.reuse_runs * (cost_c.usd - cost_p.usd)
        d_value = self.cfg.w_acc * d_acc - abs(d_cost)
        voi = p_flip * abs(d_value)
        val_usd, val_tok = self.cost.validation_cost(cand, inc, stats, n_items, n_criteria)
        threshold = theta * val_usd
        ok = voi > threshold
        why = (f"P(flip)={p_flip:.2f} [h={h:.2f} over {trials}, judge={'-' if judge_p is None else f'{judge_p:.2f}'},"
               f" agree={a:.2f}] x |dV|=${abs(d_value):.5f} (dAcc^={d_acc:+.3f}, dC=${d_cost:+.5f})"
               f" = VoI ${voi:.5f} {'>' if ok else '<='} theta*tokenCost ${threshold:.5f}"
               f" -> {'VALIDATE' if ok else 'SKIP'}")
        return GateDecision(ok, voi, threshold, p_flip, p_raw, h, trials, judge_p, a, d_acc, d_cost,
                            d_value, val_usd, val_tok, why)
