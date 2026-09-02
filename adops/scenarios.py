"""Scenario definitions: account structure + hidden parameters + business objective.

`description` and `correct_action_family` are for evaluation/documentation ONLY.
They are never exposed to the agent.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from random import Random
from typing import Dict, List

from .schema import Account, AdSet, Campaign, Creative, Objective
from .world import HiddenAdSet, HiddenCreative, World

PLATFORM = "meta"
ACCOUNT_NAME = "Synthetic Premium Furniture Brand"

# ---------------------------------------------------------------- structure
STRUCTURE = [
    ("cmp_pros", "Prospecting - Handcrafted Collections", [
        ("as_broad", "Broad Interest - Home & Living", 95.0,
         [("cr_broad_a", "Video - Artisan Workshop"), ("cr_broad_b", "Carousel - Bestsellers")]),
        ("as_lal", "Lookalike 2% - Purchasers", 95.0,
         [("cr_lal_a", "Static - Dining Collection"), ("cr_lal_b", "Video - Room Styling")]),
        ("as_interior", "Interest - Interior Design", 80.0,
         [("cr_int_a", "Carousel - Living Room"), ("cr_int_b", "Static - Material Story")]),
    ]),
    ("cmp_rt", "Retargeting - Site Visitors", [
        ("as_rt30", "Visitors 30d", 50.0,
         [("cr_rt30_a", "Dynamic - Viewed Products"), ("cr_rt30_b", "Static - Craft Guarantee")]),
        ("as_cart", "Cart Abandoners 14d", 45.0,
         [("cr_cart_a", "Dynamic - Cart Reminder"), ("cr_cart_b", "Static - Delivery Promise")]),
    ]),
    ("cmp_seasonal", "Seasonal - Festive Living", [
        ("as_seas_broad", "Festive Broad", 78.0,
         [("cr_seas_a", "Video - Festive Table"), ("cr_seas_b", "Carousel - Gift Picks")]),
        ("as_seas_gift", "Gifting Intent", 37.0,
         [("cr_gift_a", "Static - Gift Guide"), ("cr_gift_b", "Video - Unboxing")]),
    ]),
]

# hidden defaults: cvr, cpm, audience, saturation_strength, aov
AD_SET_TRUTH = {
    # base_cvr, base_cpm, audience_size, saturation_strength, aov, efficient_weekly_spend
    "as_broad":      dict(base_cvr=0.021, base_cpm=10.5, audience_size=420000, saturation_strength=0.6, aov=330.0, efficient_weekly_spend=750.0),
    "as_lal":        dict(base_cvr=0.028, base_cpm=12.0, audience_size=260000, saturation_strength=0.7, aov=340.0, efficient_weekly_spend=700.0),
    "as_interior":   dict(base_cvr=0.019, base_cpm=11.5, audience_size=300000, saturation_strength=0.7, aov=320.0, efficient_weekly_spend=650.0),
    "as_rt30":       dict(base_cvr=0.062, base_cpm=19.0, audience_size=15000,  saturation_strength=0.9, aov=350.0, efficient_weekly_spend=380.0),
    "as_cart":       dict(base_cvr=0.085, base_cpm=22.0, audience_size=8000,   saturation_strength=1.0, aov=360.0, efficient_weekly_spend=330.0),
    "as_seas_broad": dict(base_cvr=0.022, base_cpm=11.0, audience_size=280000, saturation_strength=0.7, aov=300.0, efficient_weekly_spend=600.0),
    "as_seas_gift":  dict(base_cvr=0.024, base_cpm=12.5, audience_size=150000, saturation_strength=0.7, aov=290.0, efficient_weekly_spend=300.0),
}

# hidden defaults per creative: base_ctr, fatigue_rate
CREATIVE_TRUTH = {
    "cr_broad_a": dict(base_ctr=0.0135, fatigue_rate=0.35),
    "cr_broad_b": dict(base_ctr=0.0118, fatigue_rate=0.30),
    "cr_lal_a":   dict(base_ctr=0.0142, fatigue_rate=0.32),
    "cr_lal_b":   dict(base_ctr=0.0128, fatigue_rate=0.36),
    "cr_int_a":   dict(base_ctr=0.0125, fatigue_rate=0.34),
    "cr_int_b":   dict(base_ctr=0.0116, fatigue_rate=0.30),
    "cr_rt30_a":  dict(base_ctr=0.0210, fatigue_rate=0.40),
    "cr_rt30_b":  dict(base_ctr=0.0185, fatigue_rate=0.34),
    "cr_cart_a":  dict(base_ctr=0.0240, fatigue_rate=0.42),
    "cr_cart_b":  dict(base_ctr=0.0205, fatigue_rate=0.36),
    "cr_seas_a":  dict(base_ctr=0.0130, fatigue_rate=0.33),
    "cr_seas_b":  dict(base_ctr=0.0121, fatigue_rate=0.31),
    "cr_gift_a":  dict(base_ctr=0.0126, fatigue_rate=0.32),
    "cr_gift_b":  dict(base_ctr=0.0114, fatigue_rate=0.30),
}


@dataclass
class Scenario:
    key: str
    title: str
    description: str                       # evaluation-only, never shown to the agent
    correct_action_family: List[str]       # evaluation-only
    objective: Objective
    budget_overrides: Dict[str, float] = field(default_factory=dict)
    ad_set_overrides: Dict[str, dict] = field(default_factory=dict)
    creative_overrides: Dict[str, dict] = field(default_factory=dict)
    burn_in_periods: int = 12
    eval_include: bool = True
    # entities where a correct intervention would land; evaluation-only, never shown
    on_target_entities: List[str] = field(default_factory=list)


def _objective(target_cpa: float, daily_total: float, horizon: int = 6) -> Objective:
    """Budget cap = the planned spend at the scenario's starting allocation, plus 2%
    headroom. The agent can therefore reallocate freely but cannot simply buy more."""
    return Objective(
        horizon_periods=horizon,
        target_cpa=target_cpa,
        total_budget=round(daily_total * 7.0 * horizon * 1.02, 2),
    )


DEFAULT_DAILY_TOTAL = sum(a[2] for c in STRUCTURE for a in c[2])   # 480.0


def _flat_prospecting() -> Dict[str, dict]:
    """S4 only: remove every latent quality difference between prospecting ad sets so
    that no reallocation has positive expected value and restraint is optimal."""
    group = ["as_broad", "as_lal", "as_interior", "as_seas_broad", "as_seas_gift"]
    cvr = sum(AD_SET_TRUTH[a]["base_cvr"] for a in group) / len(group)
    cpm = sum(AD_SET_TRUTH[a]["base_cpm"] for a in group) / len(group)
    aov = sum(AD_SET_TRUTH[a]["aov"] for a in group) / len(group)
    out = {a: dict(base_cvr=round(cvr, 5), base_cpm=round(cpm, 3), aov=round(aov, 1))
           for a in group}
    return out


def _flat_creatives() -> Dict[str, dict]:
    """S4 only: equalise creatives within each ad set, so pausing one never helps."""
    out: Dict[str, dict] = {}
    for _, _, ad_sets in STRUCTURE:
        for _, _, _, creatives in ad_sets:
            ids = [c[0] for c in creatives]
            ctr = sum(CREATIVE_TRUTH[c]["base_ctr"] for c in ids) / len(ids)
            fat = sum(CREATIVE_TRUTH[c]["fatigue_rate"] for c in ids) / len(ids)
            for c in ids:
                out[c] = dict(base_ctr=round(ctr, 5), fatigue_rate=round(fat, 3))
    return out


SCENARIOS: Dict[str, Scenario] = {
    "S0_calibration": Scenario(
        key="S0_calibration",
        title="Healthy account at its calibrated equilibrium allocation",
        description=("Internal calibration reference: default latent parameters, no "
                     "injected defect. Used to derive the balanced starting budgets."),
        correct_action_family=["none"],
        objective=_objective(36.0, 480.0),
        eval_include=False,
    ),
    "S1_creative_fatigue": Scenario(
        key="S1_creative_fatigue",
        title="Creative fatigue in the largest prospecting ad set",
        description=("cr_broad_a is a high-CTR hero creative with a high fatigue rate and "
                     "persistent wear-out. CTR and CPC degrade over time; the ad set's "
                     "targeting and audience are healthy. Cutting budget suppresses volume "
                     "without fixing efficiency."),
        correct_action_family=["pause_creative:cr_broad_a", "creative_test:as_broad"],
        on_target_entities=["cr_broad_a", "as_broad"],
        objective=_objective(37.0, 480.0),
        budget_overrides={"as_broad": 150.0, "as_interior": 60.0, "as_seas_broad": 43.0},
        # as_broad's capacity is raised to match its larger budget, so the only defect
        # in this scenario is the worn-out creative -- not over-scaling.
        ad_set_overrides={"as_broad": dict(efficient_weekly_spend=1100.0)},
        creative_overrides={"cr_broad_a": dict(base_ctr=0.0260, fatigue_rate=2.00, decay=0.97)},
    ),
    "S2_audience_saturation": Scenario(
        key="S2_audience_saturation",
        title="Over-scaled retargeting ad set against a small audience",
        description=("as_cart has been scaled to a budget its 16k audience cannot absorb. "
                     "Exposure persists (low refresh), so CPM inflates and CVR degrades. "
                     "Both creatives are healthy -- pausing one does not help."),
        correct_action_family=["cut_budget:as_cart"],
        on_target_entities=["as_cart"],
        objective=_objective(41.0, 530.0),
        budget_overrides={"as_cart": 95.0},
        ad_set_overrides={"as_cart": dict(decay=0.80)},
    ),
    "S3_misallocation": Scenario(
        key="S3_misallocation",
        title="Budget concentrated on the structurally weaker ad set",
        description=("as_lal converts far better than as_interior but holds less than a "
                     "third of its budget. No fatigue, no saturation, no trend -- the "
                     "signal is purely cross-sectional and statistically detectable."),
        correct_action_family=["shift_budget:as_interior->as_lal"],
        on_target_entities=["as_interior", "as_lal"],
        objective=_objective(39.0, 495.0),
        budget_overrides={"as_lal": 35.0, "as_interior": 155.0},
        ad_set_overrides={"as_interior": dict(base_cvr=0.0140)},
    ),
    "S4_noise_trap": Scenario(
        key="S4_noise_trap",
        title="No underlying problem; low-volume ad sets produce noisy metrics",
        description=("Every latent parameter is healthy. Any observed swing is sampling "
                     "noise and mean-reverts. The correct behaviour is restraint: every "
                     "structural edit costs a learning-phase penalty for nothing."),
        correct_action_family=["none"],
        on_target_entities=[],
        objective=_objective(38.0, 480.0),
        ad_set_overrides=_flat_prospecting(),
        creative_overrides=_flat_creatives(),
    ),
}


def build_account(scenario: Scenario) -> Account:
    acct = Account(id="act_synthetic_001", name=ACCOUNT_NAME, platform=PLATFORM)
    for cid, cname, ad_sets in STRUCTURE:
        camp = Campaign(id=cid, name=cname)
        for aid, aname, budget, creatives in ad_sets:
            budget = scenario.budget_overrides.get(aid, budget)
            ad_set = AdSet(id=aid, name=aname, campaign_id=cid, daily_budget=float(budget))
            for crid, crname in creatives:
                acct.creatives[crid] = Creative(id=crid, name=crname, ad_set_id=aid)
                ad_set.creative_ids.append(crid)
            acct.ad_sets[aid] = ad_set
            camp.ad_set_ids.append(aid)
        acct.campaigns[cid] = camp
    return acct


def build_world(scenario: Scenario, rng: Random) -> World:
    world = World(rng=rng)
    for aid, truth in AD_SET_TRUTH.items():
        params = dict(truth)
        params.update(scenario.ad_set_overrides.get(aid, {}))
        world.ad_sets[aid] = HiddenAdSet(**params)
    for _, _, ad_sets in STRUCTURE:
        for aid, _, _, creatives in ad_sets:
            for crid, _ in creatives:
                params = dict(CREATIVE_TRUTH[crid])
                params.update(scenario.creative_overrides.get(crid, {}))
                world.creatives[crid] = HiddenCreative(ad_set_id=aid, **params)
    return world


def get(key: str) -> Scenario:
    if key not in SCENARIOS:
        raise KeyError("unknown scenario %r; available: %s" % (key, ", ".join(sorted(SCENARIOS))))
    return copy.deepcopy(SCENARIOS[key])


def keys(eval_only: bool = True) -> List[str]:
    return sorted(k for k, v in SCENARIOS.items() if v.eval_include or not eval_only)
