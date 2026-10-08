"""Basic tests for RuleThresholds and AlertFilter integration.

These are lightweight to ensure configuration wiring works.
Run with: pytest -q
"""
from datetime import datetime

from smartstablemodel.metamodel_rules_extensive import (
    ContextualMetaModelEvaluator,
    RuleThresholds,
    AlertFilterConfig,
    AlertFilter,
)


def make_event(**overrides):
    base = {
        "timestamp": "2025-08-28 23:59:59",
        "label_distribution": {},
        "loudness_metrics": {"segment": {"value": 0.0}},
        "time_context": {"hour": 23, "is_night": True},
    }
    base.update(overrides)
    return base


def test_kick_mid_threshold_override():
    # Raise mid threshold so a mid-prob kick does NOT trigger
    th = RuleThresholds(kick_single_mid=0.8)
    evalr = ContextualMetaModelEvaluator(thresholds=th)
    ev = make_event(label_distribution={"horse_kick": 0.75})
    res = evalr.evaluate(ev)
    assert res.primary_alert is None


def test_kick_mid_default_triggers():
    th = RuleThresholds()  # default mid is 0.7
    evalr = ContextualMetaModelEvaluator(thresholds=th)
    ev = make_event(label_distribution={"horse_kick": 0.75})
    res = evalr.evaluate(ev)
    assert res.primary_alert == "possible_burst"


def test_alert_filter_min_confidence_blocks():
    th = RuleThresholds(kick_single_mid=0.6)
    evalr = ContextualMetaModelEvaluator(thresholds=th)
    ev = make_event(label_distribution={"horse_kick": 0.65})
    res = evalr.evaluate(ev)
    assert res.primary_alert is not None  # rule fires
    flt = AlertFilter(AlertFilterConfig(min_confidence=0.9))
    rec = flt.process(ev, res)
    assert rec is None  # blocked by high min_confidence


def test_alert_filter_cooldown_blocks_repeated():
    th = RuleThresholds(kick_single_mid=0.6)
    evalr = ContextualMetaModelEvaluator(thresholds=th)
    flt = AlertFilter(AlertFilterConfig(min_confidence=0.5, cooldown_seconds=60))
    ev1 = make_event(timestamp="2025-08-28 23:59:59", label_distribution={"horse_kick": 0.75})
    ev2 = make_event(timestamp="2025-08-28 23:59:10", label_distribution={"horse_kick": 0.8})  # earlier second one (out of order ok)
    r1 = evalr.evaluate(ev1)
    a1 = flt.process(ev1, r1)
    assert a1 is not None
    r2 = evalr.evaluate(ev2)
    a2 = flt.process(ev2, r2)
    assert a2 is None  # cooldown suppresses


def test_kick_cluster_escalation():
    th = RuleThresholds(kick_cluster_prob_threshold=0.5, kick_cluster_hits=3)
    evalr = ContextualMetaModelEvaluator(thresholds=th)
    # Feed several events
    res = None
    for i in range(3):
        ev = make_event(label_distribution={"horse_kick": 0.7})
        res = evalr.evaluate(ev)
    assert res is not None and res.primary_alert == "confirmed_burst"
