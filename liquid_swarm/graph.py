"""Liquid Swarm Graph — Build and compile the StateGraph.

Architecture:
  START ─→ bootstrap_node ─[conditional_edge: route_to_workers]─► N × worker_node
                                                                       │
                                                                       ▼
                                            END ◄─ archivar_node ◄─ reduce_node

Uses plain StateGraph (NOT AsyncStateGraph, which doesn't exist).
Async support is native — just use `async def` nodes and `await graph.ainvoke()`.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver
from liquid_swarm.nodes import reduce_node, route_to_workers, worker_node
from liquid_swarm.memory import bootstrap_node, archivar_node
from liquid_swarm.state import SwarmState

def build_swarm_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """Build and compile the Liquid Swarm graph.

    Returns:
        A compiled LangGraph StateGraph ready for ainvoke().
    """
    builder = StateGraph(SwarmState)

    # ── Nodes ────────────────────────────────────────────────────────────
    builder.add_node("bootstrap_node", bootstrap_node)
    builder.add_node("worker_node", worker_node)
    builder.add_node("reduce_node", reduce_node)
    builder.add_node("archivar_node", archivar_node)

    # ── Edges ────────────────────────────────────────────────────────────
    # START -> Bootstrap
    builder.add_edge(START, "bootstrap_node")
    
    # Fan-Out: Bootstrap -> N parallel workers
    builder.add_conditional_edges("bootstrap_node", route_to_workers)

    # Fan-In: All workers -> Reduce
    builder.add_edge("worker_node", "reduce_node")

    # Reduce -> Archivar
    builder.add_edge("reduce_node", "archivar_node")
    
    # Archivar -> END
    builder.add_edge("archivar_node", END)

    # HITL: Human-in-the-Loop checkpointing.
    if checkpointer is not None:
        return builder.compile(
            checkpointer=checkpointer,
            interrupt_before=["reduce_node"]
        )
        
    return builder.compile()
