"""
graph.py — Constructs and compiles the LangGraph StateGraph.

Graph flow:
    memory_retrieval
         ↓
      planning
         ↓
      execution ←──────────────────┐
         ↓                         │
   [should_replan]                 │
    /     |     \\                   │
execute replan synthesize         │
   │       └──────────────────────→┘
   │
   └──→ synthesize
              ↓
        save_memory
              ↓
             END

FIX (Gap 3): save_memory_node is a dedicated graph node so sessions are
saved regardless of how the graph is invoked (API, direct, tests).
"""

from langgraph.graph import END, StateGraph

from .nodes import (
    execution_node,
    memory_retrieval_node,
    planning_node,
    replan_node,
    save_memory_node,
    should_replan,
    synthesis_node,
)
from .state import AgentState


def build_graph() -> "CompiledGraph":  # type: ignore[name-defined]
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("memory_retrieval", memory_retrieval_node)
    graph.add_node("planning", planning_node)
    graph.add_node("execution", execution_node)
    graph.add_node("replan", replan_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("save_memory", save_memory_node)  # FIX Gap 3

    # Entry point
    graph.set_entry_point("memory_retrieval")

    # Linear edges
    graph.add_edge("memory_retrieval", "planning")
    graph.add_edge("planning", "execution")

    # Conditional routing after each execution step
    graph.add_conditional_edges(
        "execution",
        should_replan,
        {
            "execute": "execution",
            "replan": "replan",
            "synthesize": "synthesis",
        },
    )

    # Replan loops back to execution with revised plan
    graph.add_edge("replan", "execution")

    # After synthesis, always save memory before ending
    graph.add_edge("synthesis", "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile()


# Module-level singleton — import this in api/main.py and mcp_server/server.py
agent_graph = build_graph()
