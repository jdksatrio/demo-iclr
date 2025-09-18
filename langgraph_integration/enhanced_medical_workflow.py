from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from neo4j import GraphDatabase
import os
import asyncio

# Import AgentState from separate file
from .agent_state import AgentState

# Import existing nodes
try:
    from .acr_retrieval_node import ACRRetrievalNode
    from .enhanced_graphrag_agent import EnhancedGraphRAGAgent
except ImportError:
    # Fallback for direct execution
    from acr_retrieval_node import ACRRetrievalNode
    from enhanced_graphrag_agent import EnhancedGraphRAGAgent


class EnrichedRationaleRetriever:
    """
    Retrieves enriched clinical rationales, evidence quality, and references 
    from the Perplexity-enhanced Neo4j knowledge graph.
    """
    
    def __init__(self, neo4j_uri: str = "neo4j+s://d761b877.databases.neo4j.io", 
                 neo4j_user: str = "neo4j", 
                 neo4j_password: str = None):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.driver = None
    
    def initialize(self, neo4j_password: str = None):
        """Initialize Neo4j connection"""
        password = neo4j_password or self.neo4j_password or os.environ.get("NEO4J_PASSWORD")
        if not password:
            raise ValueError("NEO4J_PASSWORD environment variable is required for AuraDB connection")
        
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, password)
        )
        
        # Verify connectivity
        self.driver.verify_connectivity()
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def get_enriched_rationales(self, acr_recommendations: Dict) -> Dict[str, Any]:
        """
        Retrieve enriched rationales by traversing from ACR variant through appropriateness relationships.
        """
        
        enriched_data = {
            "procedures": [],
            "summary": {
                "total_procedures": 0,
                "enriched_procedures": 0,
                "missing_enrichment": 0
            }
        }
        
        # try:
        # Extract variant ID from ACR recommendations
        variant_id = self._extract_variant_id(acr_recommendations)
        
        if not variant_id:
            return {"error": "No variant found in ACR recommendations"}
        
        print(f"Debug: Using variant ID: {variant_id}")
        
        # Query enriched data by traversing from variant through appropriateness relationships
        query = """
        MATCH (v:Variant {id: $variant_id})-[r]->(p:Procedure)
        WHERE type(r) IN ['USUALLY_APPROPRIATE', 'MAYBE_APPROPRIATE']
        AND p.clinical_rationale IS NOT NULL
        RETURN r.rationale as edge_rationale,
               p.id as procedure_node_id,
               type(r) as appropriateness,
               p.clinical_rationale as rationale,
               p.evidence_quality as evidence,
               p.references as references,
               p.enriched_timestamp as enriched_at,
               p.enrichment_source as source,
               p.enrichment_model as model,
               p.gid as gid
        ORDER BY 
            CASE type(r) 
                WHEN 'USUALLY_APPROPRIATE' THEN 1 
                WHEN 'MAYBE_APPROPRIATE' THEN 2 
                ELSE 3 
            END,
            r.rationale
        """
        
        with self.driver.session() as session:
            result = session.run(query, {"variant_id": variant_id})
            
            for record in result:
                edge_rationale = record["edge_rationale"]
                procedure_node_id = record["procedure_node_id"]
                appropriateness = record["appropriateness"]
                rationale = record["rationale"]
                evidence = record["evidence"] 
                references = record["references"]
                enriched_at = record["enriched_at"]
                source = record["source"]
                model = record["model"]
                
                print(f"Found enriched data for: {procedure_node_id} ({appropriateness})")
                
                procedure_data = {
                    "procedure_id": edge_rationale,  # Use edge rationale as primary identifier
                    "procedure_node_id": procedure_node_id,  # Keep original procedure node id
                    "appropriateness": appropriateness,
                    "clinical_rationale": rationale,
                    "evidence_quality": evidence,
                    "references": references,
                    "is_enriched": True,
                    "enrichment_metadata": {
                        "enriched_at": str(enriched_at) if enriched_at else None,
                        "source": source,
                        "model": model
                    }
                }
                
                enriched_data["procedures"].append(procedure_data)
                enriched_data["summary"]["total_procedures"] += 1
                enriched_data["summary"]["enriched_procedures"] += 1
        
        print(f"Enrichment Summary: {enriched_data['summary']['enriched_procedures']}/{enriched_data['summary']['total_procedures']} procedures enriched")
        
        # If no enriched procedures were found, let's check if the variant has any procedures at all
        if enriched_data["summary"]["total_procedures"] == 0:
            # Debug: Check if variant exists and has any procedures at all
            print("Debug: No procedures found for variant. Checking if variant exists and has any procedures...")
            
            debug_query = """
            MATCH (v:Variant {id: $variant_id})-[r]->(p:Procedure)
            RETURN r.rationale as edge_rationale,
                   p.id as procedure_node_id,
                   type(r) as appropriateness,
                   p.clinical_rationale IS NOT NULL as has_rationale
            ORDER BY type(r), r.rationale
            LIMIT 10
            """
            
            result = session.run(debug_query, {"variant_id": variant_id})
            found_procedures = list(result)
            
            if found_procedures:
                print(f"Found {len(found_procedures)} procedures for variant (showing first 5):")
                for proc in found_procedures[:5]:
                    enriched_status = "ENRICHED" if proc['has_rationale'] else "Not enriched"
                    print(f"  - {proc['procedure_node_id']} ({proc['appropriateness']}): {enriched_status}")
            else:
                print("No procedures found for this variant at all")
                
                # Check if variant exists
                variant_check = session.run("MATCH (v:Variant {id: $variant_id}) RETURN v", {"variant_id": variant_id})
                variant_exists = variant_check.single() is not None
                
                if variant_exists:
                    print(f"Variant {variant_id} exists but has no procedures")
                else:
                    print(f"Variant {variant_id} does not exist in database")
        
        return enriched_data
        
        # except Exception as e:
        #     print(f"Enriched rationale error: {str(e)}")
        #     return {"error": f"Failed to retrieve enriched rationales: {str(e)}"}
    
    def _extract_variant_id(self, acr_recommendations: Dict) -> str:
        """Extract variant ID from ACR recommendations"""
        
        try:
            print(f"Debug: ACR recommendations structure: {list(acr_recommendations.keys())}")
            
            # Check for new format first (best_variant)
            if "best_variant" in acr_recommendations:
                variant_id = acr_recommendations["best_variant"]["variant_id"]
                print(f"Debug: Found variant ID from best_variant: {variant_id}")
                return variant_id
            
            # Fallback to old format (similar_conditions)
            elif "similar_conditions" in acr_recommendations:
                conditions = acr_recommendations["similar_conditions"]
                if conditions and len(conditions) > 0:
                    # Use the first (most similar) condition
                    condition_id = conditions[0]["condition_id"] 
                    print(f"Debug: Found condition ID from similar_conditions: {condition_id}")
                    return condition_id
            
            return None
            
        except Exception as e:
            print(f"Error extracting variant ID: {e}")
            return None


class EnhancedMedicalSupervisorNode:
    """
    Enhanced supervisor agent that incorporates enriched clinical rationales,
    evidence quality assessments, and authoritative references into decision-making.
    """
    
    def __init__(self, llm, neo4j_password: str = None):
        self.llm = llm
        self.rationale_retriever = EnrichedRationaleRetriever(neo4j_password=neo4j_password)
    
    async def initialize(self, neo4j_password: str = None):
        """Initialize the rationale retriever"""
        self.rationale_retriever.initialize(neo4j_password)
    
    def close(self):
        """Close connections"""
        self.rationale_retriever.close()
    
    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """Enhanced supervisor analysis with enriched rationales"""
        
        # Initialize if needed
        if not self.rationale_retriever.driver:
            neo4j_password = state.get("neo4j_password") or os.getenv("NEO4J_PASSWORD")
            await self.initialize(neo4j_password)
        
        user_query = state.get("user_query", "")
        graphrag_context = state.get("graphrag_context", "")
        acr_recommendations = state.get("acr_recommendations", {})
        
        # TEMPORARY DEBUG: Show full context from both Neo4j agents
        print("\n" + "="*80)
        print("🔍 DEBUG: FULL CONTEXT FROM NEO4J AGENTS")
        print("="*80)
        
        print("\n📊 1. GRAPHRAG AGENT FULL CONTEXT:")
        print("-" * 50)
        if graphrag_context:
            print(f"Length: {len(graphrag_context)} characters")
            print(f"Content:\n{graphrag_context}")
        else:
            print("No GraphRAG context retrieved")
        
        print("\n🏥 2. ACR AGENT FULL RECOMMENDATIONS:")
        print("-" * 50)
        if acr_recommendations:
            import json
            print(f"Structure: {list(acr_recommendations.keys())}")
            print(f"Full content:\n{json.dumps(acr_recommendations, indent=2, default=str)}")
        else:
            print("No ACR recommendations retrieved")
        
        print("\n" + "="*80)
        print("🔍 END DEBUG: NEO4J AGENTS CONTEXT")
        print("="*80 + "\n")
        
        # Retrieve enriched rationales for recommended procedures
        enriched_rationales = self.rationale_retriever.get_enriched_rationales(acr_recommendations)
        
        # Create enhanced analysis prompt
        analysis_prompt = self._create_enhanced_prompt(
            user_query, graphrag_context, acr_recommendations, enriched_rationales
        )
        
        try:
            # Get enhanced analysis from LLM
            analysis_result = await self.llm.ainvoke([HumanMessage(content=analysis_prompt)])
            
            # Create final answer that directly uses Neo4j procedures instead of LLM interpretation
            final_answer = await self._create_direct_neo4j_response(user_query, acr_recommendations, enriched_rationales, analysis_result.content)
            
            return {
                "messages": [analysis_result],
                "analysis_result": analysis_result.content,
                "final_answer": final_answer,
                "enriched_rationales": enriched_rationales,
                "next_step": "complete"
            }
            
        except Exception as e:
            error_message = AIMessage(content=f"Enhanced supervisor analysis failed: {str(e)}")
            return {
                "messages": [error_message],
                "analysis_result": f"Analysis error: {str(e)}",
                "final_answer": f"Analysis error: {str(e)}",
                "enriched_rationales": enriched_rationales,
                "next_step": "error_handling"
            }
    
    def _create_enhanced_prompt(self, user_query: str, graphrag_context: str, 
                              acr_recommendations: Dict, enriched_rationales: Dict) -> str:
        """Create enhanced analysis prompt with enriched rationales"""
        
        prompt = f"""
        You are an advanced medical AI supervisor providing comprehensive clinical analysis with access to:
        1. Medical knowledge graphs
        2. ACR Appropriateness Criteria  
        3. **ENRICHED CLINICAL RATIONALES** from expert medical sources
        4. **EVIDENCE QUALITY ASSESSMENTS**
        5. **AUTHORITATIVE REFERENCES**
        
        **Patient Query:** {user_query}
        """
        
        # Add GraphRAG context
        if graphrag_context:
            prompt += f"""
        
        **Medical Knowledge Graph Context:**
        {graphrag_context}
        """
        
        # Add ACR recommendations
        if acr_recommendations and "error" not in acr_recommendations:
            prompt += f"""
        
        **ACR Procedure Recommendations:**
        {self._format_acr_summary(acr_recommendations)}
        """
        
        # Add enriched rationales - THE KEY ENHANCEMENT
        if enriched_rationales and "error" not in enriched_rationales:
            prompt += f"""
        
        **ENRICHED CLINICAL RATIONALES & EVIDENCE:**
        {self._format_enriched_rationales(enriched_rationales)}
        """
        
        # Extract actual procedures from Neo4j recommendations
        procedures_list = []
        if acr_recommendations and "usually_appropriate_procedures" in acr_recommendations:
            procedures_list = acr_recommendations["usually_appropriate_procedures"]
        
        prompt += f"""
        
        **Please provide a comprehensive imaging recommendation in this EXACT format:**
        
        Based on the condition [brief clinical summary], [1-2 sentence medical assessment]. The appropriate imaging options for the patient include:
        """
        
        # Generate procedure entries using ACTUAL Neo4j data, not templates
        for i, procedure in enumerate(procedures_list[:5]):  # Max 5 procedures
            procedure_id = procedure.get('procedure_id', procedure.get('title', f'Procedure {i+1}'))
            appropriateness = procedure.get('appropriateness', 'Usually Appropriate')
            
            prompt += f"""
        **Imaging:** {procedure_id} : {appropriateness}
        **Rationale:** [Provide clinical justification for {procedure_id} using enriched rationales and evidence]
        **References:**
        1. [ACR source or enriched reference for {procedure_id}]
        2. [Relevant medical source for {procedure_id}]
        
        ------
        """
        
        prompt += f"""
        **CRITICAL REQUIREMENTS:**
        1. Use the EXACT procedure names provided above from Neo4j ACR database
        2. Do NOT modify or paraphrase the procedure names - use them exactly as shown
        3. Use the exact format with "Imaging:", "Rationale:", "References:" for EACH procedure
        4. Include ------ separators between procedures
        5. Each procedure shows its appropriateness category from ACR database
        6. Use enriched rationales to provide detailed clinical justifications
        7. DO NOT hallucinate procedure names - only use the {len(procedures_list)} procedures listed above
        """
        
        return prompt
    
    def _format_acr_summary(self, recommendations: Dict) -> str:
        """Format ACR recommendations summary with enhanced context"""
        if "error" in recommendations:
            return f"ACR Error: {recommendations['error']}"
        
        summary = f"Query: {recommendations['query']}\n"
        
        # Check for enhanced format with condition context
        if "parent_condition" in recommendations and "best_variant" in recommendations:
            # Enhanced format with condition context
            parent_condition = recommendations["parent_condition"]
            best_variant = recommendations["best_variant"]
            context = recommendations.get("context_for_output", {})
            
            summary += f"\n**Clinical Category:** {parent_condition['condition_id']}\n"
            summary += f"**Specific Scenario:** {best_variant['variant_id']}\n"
            summary += f"**Variant Similarity:** {best_variant['variant_similarity']:.3f}\n"
            summary += f"**Search Method:** {recommendations.get('search_method', 'direct_variant')}\n"
            summary += f"**Confidence:** {context.get('search_confidence', 'medium')}\n"
            
            # Show top variant comparisons if available
            if "all_top_variants" in recommendations:
                top_variants = recommendations["all_top_variants"]
                summary += f"\n**Top {len(top_variants)} Variant Similarities:**\n"
                for variant in top_variants:
                    summary += f"  {variant['rank']}. {variant['variant_id']} (similarity: {variant['similarity']:.3f})\n"
                    summary += f"     Condition: {variant['condition_id']}\n"
            
            procedures = recommendations['usually_appropriate_procedures']
            if procedures:
                summary += f"\n**Usually Appropriate Procedures ({len(procedures)}):**\n"
                for proc in procedures[:5]:  # Show top 5
                    summary += f"• {proc['procedure_id']}\n"
                    
        elif "best_variant" in recommendations:
            # Original format fallback
            summary += f"\nBest Match: {recommendations['best_variant']['variant_id']}\n"
            summary += f"Similarity: {recommendations['best_variant']['variant_similarity']:.3f}\n"
            
            procedures = recommendations['usually_appropriate_procedures']
            if procedures:
                summary += f"\nUsually Appropriate Procedures ({len(procedures)}):\n"
                for proc in procedures[:5]:  # Show top 5
                    summary += f"• {proc['procedure_id']}\n"
        
        return summary
    
    def _format_enriched_rationales(self, enriched_data: Dict) -> str:
        """Format enriched rationales for the prompt - KEY ENHANCEMENT"""
        if "error" in enriched_data:
            return f"Enrichment Error: {enriched_data['error']}"
        
        summary = f"""Summary: {enriched_data['summary']['enriched_procedures']} of {enriched_data['summary']['total_procedures']} procedures have enriched rationales.\n"""
        
        if enriched_data['summary']['enriched_procedures'] == 0:
            return summary + "\nNo enriched rationales available for recommended procedures."
        
        summary += "\n**ENRICHED PROCEDURE ANALYSIS:**\n\n"
        
        for proc_data in enriched_data['procedures']:
            if proc_data['is_enriched']:
                summary += f"### {proc_data['procedure_id']}\n\n"
                
                # Clinical Rationale
                summary += f"**Clinical Rationale:**\n{proc_data['clinical_rationale'][:500]}...\n\n"
                
                # Evidence Quality
                if proc_data['evidence_quality']:
                    summary += f"**Evidence Quality:**\n{proc_data['evidence_quality'][:300]}...\n\n"
                
                # References
                if proc_data['references']:
                    summary += f"**References:**\n{proc_data['references'][:400]}...\n\n"
                
                # Metadata
                if proc_data['enrichment_metadata']['enriched_at']:
                    summary += f"*Enriched: {proc_data['enrichment_metadata']['enriched_at']} via {proc_data['enrichment_metadata']['source']}*\n\n"
                
                summary += "---\n\n"
        
        return summary
    
    async def _create_direct_neo4j_response(self, user_query: str, acr_recommendations: Dict, 
                                     enriched_rationales: Dict, llm_analysis: str) -> str:
        """
        Create final response using DIRECT Neo4j procedures instead of LLM interpretation.
        This ensures evaluation and user-facing systems have identical behavior.
        """
        if "error" in acr_recommendations:
            return f"ACR retrieval failed: {acr_recommendations['error']}"
        
        # Extract procedures directly from Neo4j
        procedures = acr_recommendations.get("usually_appropriate_procedures", [])
        
        if not procedures:
            return "No appropriate imaging procedures found in ACR database."
        
        # Build response using actual Neo4j data
        response = f"Based on the clinical presentation: {user_query}\n\n"
        response += "The following imaging procedures are recommended according to ACR Appropriateness Criteria:\n\n"
        
        for i, procedure in enumerate(procedures, 1):
            # Use procedure_id (from Procedure node) as the procedure name
            procedure_name = procedure.get('procedure_id', procedure.get('title', f'Procedure {i}'))
            appropriateness = procedure.get('appropriateness', 'Usually Appropriate')
            
            response += f"**{i}. {procedure_name}** - {appropriateness}\n"
            
            # Use edge rationale as clinical rationale if available
            edge_rationale = procedure.get('edge_rationale', '')
            if edge_rationale:
                # Rephrase the edge rationale to be more concise
                clinical_rationale = await self._rephrase_edge_rationale(edge_rationale, user_query)
                response += f"   Clinical rationale: {clinical_rationale}\n"
            else:
                # Add enriched rationale if available, otherwise generate simple rationale
                rationale_found = False
                if enriched_rationales and "procedures" in enriched_rationales:
                    for enriched_proc in enriched_rationales["procedures"]:
                        if enriched_proc.get("procedure_id") == procedure.get('procedure_id') and enriched_proc.get("is_enriched"):
                            rationale = enriched_proc.get("clinical_rationale", "")
                            if rationale:
                                response += f"   Clinical rationale: {rationale[:200]}...\n"
                                rationale_found = True
            response += "\n"
        
        return response
    

    async def _rephrase_edge_rationale(self, edge_rationale: str, user_query: str) -> str:
        """
        Use LLM to rephrase the edge rationale to be more concise and user-friendly.
        """
        if not edge_rationale or len(edge_rationale.strip()) < 10:
            return edge_rationale
        
        #try:
        rephrase_prompt = f"""
        Rephrase this medical imaging rationale to be more concise and user-friendly, 
        while maintaining the key clinical information. Make it 3-4 sentences maximum.

        Original rationale: {edge_rationale}
        
        Patient query: {user_query}
        
        Provide only the rephrased rationale, nothing else.
        """
        
        # Use the LLM to rephrase
        from langchain_core.messages import HumanMessage
        response = await self.llm.ainvoke([HumanMessage(content=rephrase_prompt)])
        
        rephrased = response.content.strip()
        

        return rephrased
            
def create_enhanced_medical_workflow(llm, neo4j_password: str = None) -> StateGraph:
    """
    Create an enhanced medical workflow with enriched rationale integration.
    
    Args:
        llm: LangChain LLM instance
        neo4j_password: Neo4j password for both ACR and enriched data retrieval
        
    Returns:
        StateGraph: Enhanced workflow with rationale integration
    """
    
    graphrag_node = EnhancedGraphRAGAgent(llm, neo4j_password=neo4j_password)

    from .enhanced_acr_agent import EnhancedACRAgent
    acr_node = EnhancedACRAgent(
        llm, 
        colbert_index_path=os.path.join(os.path.dirname(__file__), '..', 'retrieve_acr', '.ragatouille/colbert/indexes/acr_variants_index'),
        neo4j_password=neo4j_password
    )
    print("Using Enhanced ACR Agent with ColBERT")

    
    enhanced_supervisor = EnhancedMedicalSupervisorNode(llm, neo4j_password=neo4j_password)
    
    # Create the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("graphrag_retrieval", graphrag_node)
    workflow.add_node("acr_retrieval", acr_node)
    workflow.add_node("enhanced_supervisor", enhanced_supervisor)
    workflow.add_node("final_response", lambda state: {
        "final_answer": state.get("final_answer", "No analysis available"),
        "enriched_rationales": state.get("enriched_rationales", {}),
        "next_step": "complete"
    })
    
    # Define enhanced workflow
    workflow.set_entry_point("graphrag_retrieval")
    workflow.add_edge("graphrag_retrieval", "acr_retrieval")
    workflow.add_edge("acr_retrieval", "enhanced_supervisor")
    workflow.add_edge("enhanced_supervisor", "final_response")
    workflow.add_edge("final_response", END)
    
    return workflow.compile()


async def run_enhanced_medical_workflow(
    user_query: str, 
    llm, 
    neo4j_password: str = None
) -> Dict[str, Any]:
    """
    Run the enhanced medical workflow with enriched rationale integration.
    
    Args:
        user_query: Medical question or case description
        llm: LangChain LLM instance  
        neo4j_password: Neo4j password
        
    Returns:
        Enhanced workflow result with enriched clinical analysis
    """
    
    # Create enhanced workflow
    workflow = create_enhanced_medical_workflow(llm, neo4j_password)
    
    # Initial state
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
        "neo4j_password": neo4j_password
    }
    
    try:
        # Run the enhanced workflow
        result = await workflow.ainvoke(initial_state)
        
        return {
            "success": True,
            "query": user_query,
            "final_answer": result.get("final_answer", ""),
            "graphrag_context": result.get("graphrag_context", ""),
            "acr_recommendations": result.get("acr_recommendations", {}),
            "enriched_rationales": result.get("enriched_rationales", {}),
            "analysis_result": result.get("analysis_result", ""),
            "messages": result.get("messages", [])
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": user_query
        } 