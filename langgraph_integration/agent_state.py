"""
Agent state definition for LangGraph medical workflow
"""

from typing import Annotated
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
import operator

# Define the state schema for your multi-agent system
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    user_query: str
    graphrag_context: str
    graphrag_entities: list[dict]
    acr_recommendations: dict
    enriched_rationales: dict
    acr_analysis: str
    analysis_result: str
    final_answer: str
    next_step: str
    neo4j_password: str 