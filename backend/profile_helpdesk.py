from __future__ import annotations

import os
import sys
import time
import tracemalloc

sys.path.insert(0, "backend")

from app.intelligence.providers.helpdesk_provider import get_helpdesk_provider


def report(stage: str, start_time: float) -> None:
    current, peak = tracemalloc.get_traced_memory()

    print(
        f"[PROFILE] {stage:<40} "
        f"time={time.perf_counter() - start_time:8.2f}s "
        f"current={current / 1024 / 1024:8.1f}MB "
        f"peak={peak / 1024 / 1024:8.1f}MB"
    )


def main() -> None:
    print("Starting Help Desk snapshot profiling...")
    print()

    tracemalloc.start()

    provider = get_helpdesk_provider()

    start = time.perf_counter()

    print("[PROFILE] Building snapshot...")
    snapshot = provider.build_snapshot()

    report("build_snapshot COMPLETE", start)

    print()
    print("[RESULT]")
    print(f"Tickets: {len(snapshot.tickets_by_id)}")
    print(f"Workflow events: {sum(len(v) for v in snapshot.workflow_events_by_ticket.values())}")
    print(f"Workflow intervals: {sum(len(v) for v in snapshot.workflow_intervals_by_ticket.values())}")
    print(f"Handoffs: {sum(len(v) for v in snapshot.handoffs_by_ticket.values())}")
    print(f"Agents: {len(snapshot.agents_by_id)}")
    print(f"Ticket analytics: {len(snapshot.ticket_analytics_by_id)}")
    print(f"Bottleneck groups: {len(snapshot.bottlenecks_by_ticket)}")
    print(f"Variant groups: {len(snapshot.variants_by_ticket)}")

    current, peak = tracemalloc.get_traced_memory()

    print()
    print("[MEMORY]")
    print(f"Current traced: {current / 1024 / 1024:.1f} MB")
    print(f"Peak traced:    {peak / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()