"""
graph.py
========
LangGraph agent graph definition and compilation.

Assembles the nodes from nodes.py into a directed state graph
and compiles it with a memory checkpointer for conversation persistence.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from rag_agent.agent.nodes import (
    generation_node,
    query_rewrite_node,
    retrieval_node,
    should_retry_retrieval,
)
from rag_agent.agent.state import AgentState


class AgentGraphBuilder:
    """
    Constructs and compiles the LangGraph agent state graph.

    The graph implements a three-node RAG pipeline:

        [START]
           │
           ▼
    query_rewrite_node   ← rewrites query for better retrieval
           │
           ▼
    retrieval_node       ← fetches relevant chunks from ChromaDB
           │
           ▼ (conditional edge via should_retry_retrieval)
           │
     ┌─────┴──────┐
     │            │
  "generate"    "end"
     │            │
     ▼            ▼
generation_node  [END]   ← hallucination guard fires here
     │
     ▼
   [END]

    The checkpointer (MemorySaver) enables multi-turn conversation:
    each thread_id maintains its own message history and state,
    persisted in memory for the lifetime of the application session.

    Interview talking point: the graph structure makes the agent's
    decision logic explicit and auditable. Compare this to a black-box
    agent loop where control flow is implicit.

    Example
    -------
    >>> builder = AgentGraphBuilder()
    >>> graph = builder.build()
    >>> config = {"configurable": {"thread_id": "user-session-001"}}
    >>> result = graph.invoke(
    ...     {"messages": [HumanMessage(content="Explain LSTMs")]},
    ...     config=config
    ... )
    >>> print(result["final_response"].answer)
    """

    def __init__(self) -> None:
        self._checkpointer = MemorySaver()

    def build(self):
        """
        Assemble nodes and edges, then compile the graph.

        Returns
        -------
        CompiledStateGraph
            A compiled LangGraph graph ready to invoke or stream.

        Notes
        -----
        The compiled graph is thread-safe and can be shared across
        multiple Streamlit sessions via st.cache_resource.
        """
        graph = StateGraph(AgentState)

        graph.add_node("query_rewrite", query_rewrite_node)
        graph.add_node("retrieval", retrieval_node)
        graph.add_node("generation", generation_node)

        graph.add_edge(START, "query_rewrite")
        graph.add_edge("query_rewrite", "retrieval")
        graph.add_conditional_edges(
            "retrieval",
            should_retry_retrieval,
            {"generate": "generation", "end": END},
        )
        graph.add_edge("generation", END)

        return graph.compile(checkpointer=self._checkpointer)


@lru_cache(maxsize=1)
def get_compiled_graph():
    """
    Return the singleton compiled graph.

    Uses lru_cache so the graph is built only once per process.
    In Streamlit, wrap with st.cache_resource instead:

        @st.cache_resource
        def get_graph():
            return AgentGraphBuilder().build()

    Returns
    -------
    CompiledStateGraph
    """
    return AgentGraphBuilder().build()