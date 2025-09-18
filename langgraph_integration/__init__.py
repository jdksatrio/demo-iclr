"""
LangGraph Integration for Medical Graph RAG + ACR Retrieval

This package provides LangGraph nodes and workflows for integrating
the Medical Graph RAG system and ACR procedure recommendations 
into multi-agent LLM workflows.

Enhanced Usage:
    from langgraph_integration.enhanced_medical_workflow import run_enhanced_medical_workflow
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(model="gpt-4.1", temperature=0.1, api_key=os.getenv("OPENAI_API_KEY"))
    
    # Run enhanced supervisor with enriched rationales
    result = await run_enhanced_medical_workflow(
        user_query="Patient with chest pain and elevated troponins",
        llm=llm,
        neo4j_password="your_neo4j_password"
    )
"""

# Import AgentState
from .agent_state import AgentState

# Import ACR retrieval components
from .acr_retrieval_node import (
    ACRRetrievalNode,
    )

# Import enhanced medical workflow
try:
    from .enhanced_medical_workflow import (
        run_enhanced_medical_workflow,
        EnrichedRationaleRetriever,
        EnhancedMedicalSupervisorNode,
        create_enhanced_medical_workflow
    )
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False

__all__ = [
    # Agent state
    "AgentState",
    
    # ACR retrieval components
    "ACRRetrievalNode",
]

# Add enhanced components if available
if ENHANCED_AVAILABLE:
    __all__.extend([
        "run_enhanced_medical_workflow",
        "EnrichedRationaleRetriever", 
        "EnhancedMedicalSupervisorNode",
        "create_enhanced_medical_workflow"
    ])

__version__ = "1.0.0" 