"""Unit tests for registry, pod_agent, and mcp_server helpers."""

import pytest
from registry import REGISTRY, default_tenant_loader
from pod_agent import PodAgent
from mcp_server import Tenant, set_tenant_provider, get_current_tenant
from tracerag.db import TraceDB
from tracerag.router import TraceRouter


@pytest.fixture
def test_graph(tmp_path):
    db_file = tmp_path / "agent_test.sqlite3"
    db = TraceDB(db_file)
    router = TraceRouter(db)
    return router, db, str(db_file)


def test_registry_and_pod_agent_reconciliation(test_graph, tmp_path):
    router, db, path = test_graph
    
    # 1. Register tenant loader
    REGISTRY.set_handle_loader(default_tenant_loader)
    REGISTRY.bind("org_acme", db=db, router=router, path=path)

    agent = PodAgent()
    assert REGISTRY.get_tenant("org_acme") is not None

    # 2. Desired points to new path
    new_db_file = tmp_path / "new_store.sqlite3"
    agent.set_desired("org_acme", str(new_db_file))

    # 3. Step reconcile
    swaps = agent.reconcile_step()
    assert len(swaps) == 1
    assert swaps[0]["org_id"] == "org_acme"
    assert swaps[0]["swapped_to"] == str(new_db_file)

    # 4. Repeated reconcile produces 0 swaps (converged)
    assert len(agent.reconcile_step()) == 0


def test_mcp_tenant_provider(test_graph):
    router, db, _ = test_graph
    tenant_obj = Tenant(router=router, db=db, graph_id="test_graph")

    set_tenant_provider(lambda org_id: tenant_obj)
    active = get_current_tenant()
    assert active.graph_id == "test_graph"
