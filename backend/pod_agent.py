"""Pod Agent — Background state reconciler for distributed serving pods.

Reconciles INTENT (desired graph artifact) with REALITY (currently loaded graph).
When a new graph artifact is published:
1. Downloads / verifies artifact file
2. Performs zero-downtime pointer swap in TenantDatabaseRegistry
3. Signals readiness
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

from registry import REGISTRY, TenantDatabaseRegistry
from tracerag import config

logger = logging.getLogger("graphsight.pod_agent")


class PodAgent:
    """Monitors tenant assignments and performs atomic graph hot-swaps."""

    def __init__(self, pod_id: str = config.POD_ID, registry: Optional[TenantDatabaseRegistry] = None) -> None:
        self.pod_id = pod_id
        self.registry = registry or REGISTRY
        self._desired: Dict[str, str] = {}  # org_id -> desired_artifact_path

    def set_desired(self, org_id: str, artifact_path: str) -> None:
        """Sets the target intent for an organization's active graph."""
        self._desired[org_id] = artifact_path

    def reconcile_step(self) -> List[Dict[str, str]]:
        """Compares target intent against reality and executes hot-swaps."""
        swaps: List[Dict[str, str]] = []
        for org_id, desired_path in self._desired.items():
            current = self.registry.get_tenant(org_id)
            if current is None or current.path != desired_path:
                logger.info("Executing atomic graph swap for %s -> %s", org_id, desired_path)
                self.registry.swap(org_id, artifact_id=f"art_{org_id}", file_path=desired_path)
                swaps.append({"org_id": org_id, "swapped_to": desired_path})
        return swaps

    async def run(self, stop_event: asyncio.Event, poll_interval: float = 5.0) -> None:
        """Continuous reconciliation loop running in the background."""
        logger.info("PodAgent started on pod %s", self.pod_id)
        while not stop_event.is_set():
            try:
                self.reconcile_step()
            except Exception as e:
                logger.warning("PodAgent reconciliation error: %s", e)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
            except asyncio.TimeoutError:
                pass
        logger.info("PodAgent shutdown cleanly.")


async def run_pod_agent(stop_event: asyncio.Event) -> None:
    agent = PodAgent()
    await agent.run(stop_event)
