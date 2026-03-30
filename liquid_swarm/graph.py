"""Liquid Swarm Graph — Build and compile the StateGraph.

Architecture:
  START ──[conditional_edge: route_to_workers]──► N × worker_node
                                                       │
                                                       ▼
                                                  reduce_node ──► END

Uses plain StateGraph (NOT AsyncStateGraph, which doesn't exist).
Async support is native — just use `async def` nodes and `await graph.ainvoke()`.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from liquid_swarm.nodes import reduce_node, route_to_workers, worker_node
from liquid_swarm.state import SwarmState


def build_swarm_graph() -> CompiledStateGraph:
    """Build and compile the Liquid Swarm graph.

    Returns:
        A compiled LangGraph StateGraph ready for ainvoke().
    """
    builder = StateGraph(SwarmState)

    # ── Nodes ────────────────────────────────────────────────────────────
    builder.add_node("worker_node", worker_node)
    builder.add_node("reduce_node", reduce_node)

    # ── Edges ────────────────────────────────────────────────────────────
    # Fan-Out: START -> N parallel workers via Send()
    builder.add_conditional_edges(START, route_to_workers)

    # Fan-In: All workers -> Reduce (LangGraph waits for all superstep
    # nodes to complete before proceeding to reduce_node)
    builder.add_edge("worker_node", "reduce_node")

    # Reduce -> END
    builder.add_edge("reduce_node", END)

    return builder.compile()
