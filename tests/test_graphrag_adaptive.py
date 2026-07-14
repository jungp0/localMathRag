from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "third_party" / "ragflow" / "rag" / "graphrag" / "adaptive.py"
SPEC = importlib.util.spec_from_file_location("localmathrag_graphrag_adaptive", MODULE_PATH)
assert SPEC and SPEC.loader
ADAPTIVE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ADAPTIVE
SPEC.loader.exec_module(ADAPTIVE)


def test_execution_plan_scales_with_model_slots() -> None:
    constrained = ADAPTIVE.resolve_execution_plan(
        8,
        environ={
            "LOCALMATHRAG_MODEL_PARALLEL_SLOTS": "1",
            "MAX_CONCURRENT_PROCESS_AND_EXTRACT_CHUNK": "1",
        },
    )
    assert constrained.document_slots == 1
    assert constrained.chunk_slots_per_document == 1

    capable = ADAPTIVE.resolve_execution_plan(
        20,
        environ={
            "LOCALMATHRAG_MODEL_PARALLEL_SLOTS": "8",
            "MAX_CONCURRENT_PROCESS_AND_EXTRACT_CHUNK": "3",
        },
    )
    assert capable.document_slots == 3
    assert capable.chunk_slots_per_document == 3


def test_execution_plan_honors_explicit_override_and_document_boundary() -> None:
    explicit = ADAPTIVE.resolve_execution_plan(
        2,
        configured_document_slots=9,
        environ={"LOCALMATHRAG_MODEL_PARALLEL_SLOTS": "16"},
    )
    assert explicit.document_slots == 2

    invalid = ADAPTIVE.resolve_execution_plan(
        4,
        configured_document_slots=-1,
        environ={"LOCALMATHRAG_MODEL_PARALLEL_SLOTS": "1"},
    )
    assert invalid.document_slots == 1


def test_activity_watchdog_learns_without_shrinking_below_minimum() -> None:
    now = [0.0]
    watchdog = ADAPTIVE.AdaptiveActivityWatchdog(
        10,
        60,
        observed_gap_multiplier=3,
        clock=lambda: now[0],
    )
    assert watchdog.timeout_seconds == 10

    now[0] = 4
    watchdog.touch()
    assert watchdog.timeout_seconds == 12

    now[0] = 9
    assert watchdog.remaining_seconds() == 7


def test_activity_watchdog_caps_slow_observations() -> None:
    now = [0.0]
    watchdog = ADAPTIVE.AdaptiveActivityWatchdog(10, 30, clock=lambda: now[0])
    now[0] = 100
    watchdog.touch()
    assert watchdog.timeout_seconds == 30
