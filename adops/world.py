"""HIDDEN GROUND TRUTH and stochastic dynamics.

Nothing in tools.py, brief.py, agent.py or policies.py may read this module's state.
The agent only ever sees noisy, partially delayed samples produced by `step()`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from random import Random
from typing import Dict, List, Optional

# -- dynamics constants (tuned for plausibility, not for any particular policy) --
IMPRESSION_DECAY = 0.60      # weekly audience refresh: old exposure fades
SAT_FREE = 3.0               # effective frequency tolerated before saturation bites
MAX_WEEKLY_FREQ = 4.0        # delivery ceiling; beyond this the budget under-delivers
CPM_PUSH_COEF = 0.30         # auction pressure when spending past efficient capacity
CVR_DILUTION_COEF = 0.25     # audience-quality dilution when scaling past capacity
CTR_NOISE_SD = 0.035
CVR_NOISE_SD = 0.05
CPM_NOISE_SD = 0.05
REV_NOISE_SD = 0.12
CREATIVE_SHARE_FLOOR = 0.15  # platform never fully starves a creative
FATIGUE_FLOOR = 0.32
FATIGUE_CVR_COUPLING = 0.30  # a worn-out creative also converts slightly worse


def binomial(rng: Random, n: float, p: float) -> int:
    """Exact for small n, normal approximation for large n. Always seeded."""
    n = int(max(0, round(n)))
    if n == 0 or p <= 0:
        return 0
    p = min(1.0, p)
    if n < 60:
        return sum(1 for _ in range(n) if rng.random() < p)
    mean = n * p
    sd = math.sqrt(max(n * p * (1.0 - p), 1e-9))
    return int(max(0, min(n, round(rng.gauss(mean, sd)))))


def lognoise(rng: Random, sd: float) -> float:
    if sd <= 0:
        return 1.0
    return math.exp(rng.gauss(0.0, sd))


@dataclass
class HiddenCreative:
    base_ctr: float
    fatigue_rate: float          # how fast CTR decays with cumulative exposure
    ad_set_id: str = ""
    cum_impressions: float = 0.0
    decay: float = IMPRESSION_DECAY   # creative wear-out can persist across weeks

    def fatigue(self, audience_size: float) -> float:
        if audience_size <= 0:
            return 1.0
        x = self.cum_impressions / audience_size
        return max(FATIGUE_FLOOR, math.exp(-self.fatigue_rate * x))


@dataclass
class HiddenAdSet:
    base_cvr: float
    base_cpm: float
    audience_size: float
    saturation_strength: float
    aov: float
    efficient_weekly_spend: float = 600.0   # spend beyond which returns diminish
    cum_impressions: float = 0.0
    learning_periods: int = 0
    learning_efficiency: float = 1.0
    decay: float = IMPRESSION_DECAY   # audience refresh rate

    def effective_frequency(self) -> float:
        if self.audience_size <= 0:
            return 0.0
        return self.cum_impressions / self.audience_size

    def pressure(self) -> float:
        return min(8.0, max(0.0, self.effective_frequency() - SAT_FREE))

    def learning_multiplier(self) -> float:
        return self.learning_efficiency if self.learning_periods > 0 else 1.0


@dataclass
class World:
    """Container for all hidden state."""
    rng: Random
    ad_sets: Dict[str, HiddenAdSet] = field(default_factory=dict)
    creatives: Dict[str, HiddenCreative] = field(default_factory=dict)

    # -- mutation hooks used by the environment (never by tools directly) --
    def start_learning(self, ad_set_id: str, periods: int, efficiency: float) -> None:
        h = self.ad_sets.get(ad_set_id)
        if h is None:
            return
        h.learning_periods = max(h.learning_periods, periods)
        h.learning_efficiency = efficiency

    def add_creative(self, creative_id: str, ad_set_id: str,
                     quality_lo: float = 0.80, quality_hi: float = 1.30) -> HiddenCreative:
        """A creative test is a genuine gamble: the new creative may be better or worse."""
        siblings = [c for c in self.creatives.values() if c.ad_set_id == ad_set_id]
        if siblings:
            base = sum(s.base_ctr for s in siblings) / len(siblings)
        else:
            base = 0.011
        quality = self.rng.uniform(quality_lo, quality_hi)
        hc = HiddenCreative(base_ctr=base * quality, fatigue_rate=0.35,
                            ad_set_id=ad_set_id)  # fresh creative: no accrued fatigue
        self.creatives[creative_id] = hc
        return hc

    def decay(self) -> None:
        for h in self.ad_sets.values():
            h.cum_impressions *= h.decay
            if h.learning_periods > 0:
                h.learning_periods -= 1
        for c in self.creatives.values():
            c.cum_impressions *= c.decay


def simulate_ad_set(world: World, ad_set_id: str, daily_budget: float,
                    active_creative_ids: List[str]) -> Dict[str, dict]:
    """Simulate one weekly period for one ad set. Returns TRUE (undelayed) results
    keyed by creative id. All randomness comes from world.rng."""
    h = world.ad_sets[ad_set_id]
    rng = world.rng
    out: Dict[str, dict] = {}
    if not active_creative_ids or daily_budget <= 0:
        return out

    intended_spend = daily_budget * 7.0
    pressure = h.pressure()

    # Two independent diminishing-return mechanisms with different observable
    # signatures:
    #   (a) capacity push  -> spending past efficient capacity inflates CPM and
    #       dilutes audience quality (visible as rising CPM + softening CVR)
    #   (b) saturation     -> repeated exposure of a small audience (visible as
    #       rising frequency, flat reach, rising CPM, falling CVR)
    push = max(0.0, intended_spend / max(1.0, h.efficient_weekly_spend) - 1.0)
    cpm = (h.base_cpm
           * (1.0 + h.saturation_strength * pressure * 0.30)
           * (1.0 + CPM_PUSH_COEF * (push ** 0.9))
           * lognoise(rng, CPM_NOISE_SD))
    cpm = max(0.5, cpm)

    wanted_impr = intended_spend / cpm * 1000.0
    cap_impr = h.audience_size * MAX_WEEKLY_FREQ
    impressions = min(wanted_impr, cap_impr)
    spend = impressions * cpm / 1000.0          # under-delivery when capped

    reach = h.audience_size * (1.0 - math.exp(-impressions / max(h.audience_size, 1.0)))
    reach = max(1.0, reach)

    # Platform allocates budget across creatives towards the better performer,
    # but never fully starves one.
    expected = {}
    for cid in active_creative_ids:
        hc = world.creatives[cid]
        expected[cid] = max(1e-6, hc.base_ctr * hc.fatigue(h.audience_size))
    total_exp = sum(expected.values())
    raw_share = {cid: expected[cid] / total_exp for cid in active_creative_ids}
    n = len(active_creative_ids)
    floor = min(CREATIVE_SHARE_FLOOR, 1.0 / n)
    shares = {cid: floor + (1.0 - floor * n) * raw_share[cid] for cid in active_creative_ids}

    cvr_saturation = 1.0 / (1.0 + h.saturation_strength * pressure * 0.35)
    cvr_dilution = 1.0 / (1.0 + CVR_DILUTION_COEF * push)
    learn = h.learning_multiplier()

    for cid in active_creative_ids:
        hc = world.creatives[cid]
        share = shares[cid]
        impr_c = impressions * share
        spend_c = spend * share
        reach_c = reach * share

        fat = hc.fatigue(h.audience_size)
        ctr = hc.base_ctr * fat * lognoise(rng, CTR_NOISE_SD)
        clicks = binomial(rng, impr_c, ctr)

        cvr = (h.base_cvr * cvr_saturation * cvr_dilution * learn
               * (1.0 - FATIGUE_CVR_COUPLING * (1.0 - fat))
               * lognoise(rng, CVR_NOISE_SD))
        conversions = binomial(rng, clicks, cvr)
        revenue = conversions * h.aov * lognoise(rng, REV_NOISE_SD)

        hc.cum_impressions += impr_c
        out[cid] = {
            "spend": spend_c,
            "impressions": int(round(impr_c)),
            "reach": int(round(reach_c)),
            "clicks": int(clicks),
            "conversions": float(conversions),
            "revenue": float(revenue),
        }

    h.cum_impressions += impressions
    return out
