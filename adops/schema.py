"""Observable data model. Nothing here is hidden from the agent."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

ACTIVE = "active"
PAUSED = "paused"


@dataclass
class Creative:
    id: str
    name: str
    ad_set_id: str
    status: str = ACTIVE
    launched_period: int = 0
    is_test: bool = False


@dataclass
class AdSet:
    id: str
    name: str
    campaign_id: str
    daily_budget: float
    status: str = ACTIVE
    creative_ids: List[str] = field(default_factory=list)


@dataclass
class Campaign:
    id: str
    name: str
    status: str = ACTIVE
    ad_set_ids: List[str] = field(default_factory=list)


@dataclass
class Account:
    id: str
    name: str
    platform: str
    campaigns: Dict[str, Campaign] = field(default_factory=dict)
    ad_sets: Dict[str, AdSet] = field(default_factory=dict)
    creatives: Dict[str, Creative] = field(default_factory=dict)

    # -- lookup helpers -------------------------------------------------
    def entity_type(self, entity_id: str) -> Optional[str]:
        if entity_id in self.campaigns:
            return "campaign"
        if entity_id in self.ad_sets:
            return "ad_set"
        if entity_id in self.creatives:
            return "creative"
        return None

    def exists(self, entity_id: str) -> bool:
        return self.entity_type(entity_id) is not None

    def active_ad_sets(self, campaign_id: Optional[str] = None) -> List[AdSet]:
        out = []
        for a in self.ad_sets.values():
            if a.status != ACTIVE:
                continue
            if campaign_id and a.campaign_id != campaign_id:
                continue
            out.append(a)
        return out

    def active_creatives(self, ad_set_id: str) -> List[Creative]:
        return [
            self.creatives[c]
            for c in self.ad_sets[ad_set_id].creative_ids
            if self.creatives[c].status == ACTIVE
        ]

    def account_daily_budget(self) -> float:
        return sum(a.daily_budget for a in self.ad_sets.values() if a.status == ACTIVE)

    def name_of(self, entity_id: str) -> str:
        t = self.entity_type(entity_id)
        if t == "campaign":
            return self.campaigns[entity_id].name
        if t == "ad_set":
            return self.ad_sets[entity_id].name
        if t == "creative":
            return self.creatives[entity_id].name
        return entity_id


@dataclass
class PerfRow:
    """One reported row: period x creative. All raw counts; metrics are derived."""
    period: int
    date: str
    platform: str
    campaign_id: str
    campaign_name: str
    ad_set_id: str
    ad_set_name: str
    creative_id: str
    creative_name: str
    spend: float
    impressions: int
    reach: int
    clicks: int
    conversions: float
    revenue: float

    def as_dict(self) -> dict:
        return asdict(self)


CSV_COLUMNS = [
    "period", "date", "platform",
    "campaign_id", "campaign_name",
    "ad_set_id", "ad_set_name",
    "creative_id", "creative_name",
    "spend", "impressions", "reach", "clicks", "conversions", "revenue",
]


@dataclass
class Objective:
    """The business objective and its hard/soft constraints."""
    horizon_periods: int = 6
    target_cpa: float = 45.0
    total_budget: float = 21000.0
    pacing_tolerance: float = 0.15
    currency: str = "EUR"
    description: str = "Maximise conversions within CPA target and total budget."

    def planned_weekly_spend(self) -> float:
        return self.total_budget / float(self.horizon_periods)


@dataclass
class Limits:
    """Hard guardrails. Enforced in code, never by the model."""
    max_budget_change_pct: float = 0.25
    min_daily_budget: float = 20.0
    max_daily_budget: float = 400.0
    max_account_daily_budget: Optional[float] = None   # set from scenario: 1.15x start
    min_campaign_daily_budget: float = 40.0
    max_changes_per_period: int = 2
    max_total_changes: int = 8
    min_clicks_for_pause: int = 120
    min_clicks_for_budget_cut: int = 80
    max_active_creative_tests: int = 3
    max_tool_calls_per_period: int = 14
    max_total_tool_calls: int = 140
    learning_penalty_periods: int = 1
    learning_efficiency: float = 0.85
    budget_change_learning_threshold: float = 0.20
