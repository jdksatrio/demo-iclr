"""
ACR retrieval node (langgraph)
"""

import os
import asyncio
from typing import Dict, Any, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph
import sys

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from langgraph_integration.agent_state import AgentState

class ACRRetrievalNode:
    def __init__(
        self,
        retrieval_method: str = "colbert",
        neo4j_uri: str = "neo4j+s://d761b877.databases.neo4j.io",
        neo4j_user: str = "neo4j", 
        neo4j_password: str = None,
        embedding_provider: str = "pubmedbert",
        embedding_model: str = "NeuML/pubmedbert-base-embeddings",
        colbert_index_path: str = None,
        debug: bool = True
    ):
        self.retrieval_method = retrieval_method
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model

        self.colbert_index_path = colbert_index_path
        
        self.retriever = None
        
        if self.debug:
            print(f"ACR Retrieval Node initialized with method: {retrieval_method}")
            print(f"Index path: {colbert_index_path}")

    async def initialize(self, neo4j_password: str = None):
        if neo4j_password:
            self.neo4j_password = neo4j_password
            
        if self.retrieval_method == "colbert":
            await self._initialize_colbert()
        elif self.retrieval_method == "neo4j":
            await self._initialize_neo4j(neo4j_password)

    async def _initialize_colbert(self):

        from retrieve_acr.colbert_acr_retriever import ColBERTACRRetriever
        self.retriever = ColBERTACRRetriever(
            index_path=self.colbert_index_path,
        )

    async def _initialize_neo4j(self, neo4j_password: str = None):
        from retrieve_acr.medical_procedure_recommender_vectorized import MedicalProcedureRecommenderVectorized
            
        final_password = neo4j_password or self.neo4j_password
            
        self.retriever = MedicalProcedureRecommenderVectorized(
            neo4j_uri=self.neo4j_uri,
            neo4j_user=self.neo4j_user,
            neo4j_password=final_password,
            embedding_provider=self.embedding_provider,
            embedding_model=self.embedding_model,
        )


    def close(self):
        if self.retriever and hasattr(self.retriever, 'close'):
            self.retriever.close()

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """Execute ACR retrieval based on user query"""
        # try:
        messages = state.get("messages", [])
        if not messages:
            user_query = state.get("user_query", "")
        else:
            last_message = messages[-1]
            user_query = last_message.content if hasattr(last_message, 'content') else str(last_message)

        
        if not self.retriever:
            await self.initialize()
        
        if self.debug:
            print(f"Processing query: '{user_query}'")
        
        recommendations = self.retriever.recommend_procedures(user_query)
        
        if recommendations and "error" not in recommendations:
            response_content = self._format_recommendations_summary(recommendations)
            response_message = AIMessage(
                content=response_content,
                additional_kwargs={
                    "source": "acr_retrieval_node",
                    "retrieval_method": self.retrieval_method,
                    "acr_recommendations": recommendations
                }
            )
            next_step = "acr_analysis"
        
        if self.debug:
            print(f"ACR retrieval completed")
        
        return {
            "messages": [response_message],
            "acr_recommendations": recommendations,
            "next_step": next_step
        }
            

    def _format_recommendations_summary(self, recommendations: Dict) -> str:
        """Format ACR recommendations into a readable summary"""
        if "error" in recommendations:
            return f"Error: {recommendations['error']}"
        
        if recommendations.get("retrieval_method") == "colbert":
            summary = f"ColBERT ACR Recommendations for: {recommendations['query']}\n\n"
            
            if "best_variant" in recommendations:
                summary += f"**Top Match:**\n"
                summary += f"• {recommendations['best_variant']['content']}\n"
                summary += f"• Confidence: {recommendations['best_variant']['relevance_score']:.3f}\n\n"
                
                procedures = recommendations.get('usually_appropriate_procedures', [])
                if procedures:
                    summary += f"**Top ACR Criteria ({len(procedures)} total):**\n"
                    for i, proc in enumerate(procedures[:5], 1):
                        summary += f"{i}. {proc['title'][:120]}...\n"
                        summary += f"   • Score: {proc['relevance_score']:.4f}\n"
                    
                    if len(procedures) > 5:
                        summary += f"... and {len(procedures) - 5} more criteria\n"
                else:
                    summary += "**No procedures found for this query.**\n"
            
            return summary
        
        summary = f"Neo4j ACR Recommendations for: {recommendations['query']}\n\n"
        
        if "best_variant" in recommendations:
            summary += f"**Top Matching Condition:**\n"
            summary += f"• {recommendations['top_condition']['condition_id']}\n"
            summary += f"• Similarity: {recommendations['top_condition']['condition_similarity']:.3f}\n\n"
            
            summary += f"**Best Matching Variant:**\n"
            summary += f"• {recommendations['best_variant']['variant_id']}\n"
            summary += f"• Similarity: {recommendations['best_variant']['variant_similarity']:.3f}\n\n"
            
            procedures = recommendations['usually_appropriate_procedures']
            if procedures:
                summary += f"**Usually Appropriate Procedures ({len(procedures)}):**\n"
                for i, procedure in enumerate(procedures[:5], 1):
                    summary += f"{i}. {procedure['procedure_id']}\n"
                    if procedure.get('dosage'):
                        summary += f"   • Radiation Dosage: {procedure['dosage']}\n"
                
                if len(procedures) > 5:
                    summary += f"... and {len(procedures) - 5} more procedures\n"
            else:
                summary += "**No usually appropriate procedures found for this variant.**\n"
        
        else:
            summary += f"**Similar Conditions Found:**\n"
            for i, condition in enumerate(recommendations['similar_conditions'][:3], 1):
                summary += f"{i}. {condition['condition_id']} (similarity: {condition['similarity_score']:.3f})\n"
            
            total_appropriate = len(recommendations['aggregated_procedures']['USUALLY_APPROPRIATE'])
            total_maybe = len(recommendations['aggregated_procedures']['MAYBE_APPROPRIATE'])
            
            summary += f"\n**Procedure Summary:**\n"
            summary += f"• Usually Appropriate: {total_appropriate} procedures\n"
            summary += f"• Maybe Appropriate: {total_maybe} procedures\n"
        
        return summary