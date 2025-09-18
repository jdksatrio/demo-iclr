from typing import Dict, Any, List, Optional
import asyncio
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import os
import sys
import json

# Add the retrieve_acr path to import the core functionality
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'retrieve_acr'))
from medical_procedure_recommender_vectorized import MedicalProcedureRecommenderVectorized
from colbert_acr_retriever import ColBERTACRRetriever


class EnhancedACRAgent:
    """
    Enhanced ACR Agent with LLM-powered query optimization for better ACR condition matching.
    
    This agent uses Chain of Thought reasoning to:
    1. Analyze the medical query for ACR-relevant components  
    2. Understand ACR condition naming patterns and structure
    3. Formulate optimal search queries for ACR database
    4. Execute strategic retrieval with multiple query approaches
    
    Supports both Neo4j vectorized search and high-performance ColBERT retrieval.
    """
    
    def __init__(
        self,
        llm,
        colbert_index_path: str = None,
        neo4j_password: str = None,
        neo4j_uri: str = "neo4j+s://d761b877.databases.neo4j.io",
        neo4j_user: str = "neo4j"
    ):
        self.llm = llm
        self.colbert_index_path = colbert_index_path
        self.neo4j_password = neo4j_password
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.colbert_retriever = None
        self.neo4j_recommender = None
        self.initialized = False
    
    async def initialize(self, neo4j_password: str = None):
        """Initialize ColBERT for variant search and Neo4j for procedure fetching"""
        if self.initialized:
            return
            
        try:
            # Initialize ColBERT for variant search
            self.colbert_retriever = ColBERTACRRetriever(
                index_path=self.colbert_index_path,
                debug=True
            )
            
            # Initialize Neo4j for procedure fetching
            password = neo4j_password or self.neo4j_password or os.environ.get("NEO4J_PASSWORD", "medgraphrag")
            self.neo4j_recommender = MedicalProcedureRecommenderVectorized(
                neo4j_uri=self.neo4j_uri,
                neo4j_user=self.neo4j_user,
                neo4j_password=password,
                embedding_provider="pubmedbert",
                embedding_model="NeuML/pubmedbert-base-embeddings"
            )
            
            self.initialized = True
            print(f"Enhanced ACR Agent initialized with ColBERT + Neo4j")
            
        except Exception as e:
            print(f"Enhanced ACR Agent initialization failed: {e}")
            raise
    
    def close(self):
        """Close connections"""
        if self.colbert_retriever:
            self.colbert_retriever.close()
        if self.neo4j_recommender:
            self.neo4j_recommender.close()
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main agent execution with LLM-powered query optimization
        """
        if not self.initialized:
            await self.initialize(state.get("neo4j_password"))

        user_query = state.get("user_query", "")

        if not user_query:
            # Extract query from messages if not in state
            for message in reversed(state.get("messages", [])):
                if isinstance(message, HumanMessage):
                    user_query = message.content
                    break
        
        if not user_query:
            return self._create_error_response("No user query found")
        

        # Step 1: LLM analyzes query and formulates optimal ACR search queries
        #query_analysis = await self._analyze_query_for_acr(user_query)
        # Step 2: Execute strategic ACR retrieval with optimized queries
        acr_results = await self._execute_acr_retrieval(user_query)
        
        # Step 3: LLM evaluates and selects best results
        final_recommendations = await self._evaluate_acr_results(user_query, acr_results)
        
        # Create response message
        response_message = AIMessage(
            content=f"Enhanced ACR Agent optimized query and retrieved procedure recommendations",
            additional_kwargs={
                #"query_analysis": query_analysis,
                "acr_results": acr_results,
                "final_recommendations": final_recommendations,
                "source": "enhanced_acr_agent"
            }
        )
        
        return {
            "messages": [response_message],
            "acr_recommendations": final_recommendations,
            #"acr_analysis": query_analysis,
            "next_step": "enhanced_supervisor"
        }
    

    
    async def _execute_acr_retrieval(self, user_query: str) -> Dict[str, Any]:
        """
        Execute strategic ACR retrieval using direct variant search + condition context
        """

        direct_result = await self._direct_variant_search_with_context(user_query)
        return {
            "primary_result": direct_result,
            "alternative_results": [],
            "search_metadata": {
                "queries_tried": [f"Original: {user_query}"],
                "best_similarity": direct_result.get("best_variant", {}).get("variant_similarity", 0.0),
                "total_procedures": len(direct_result.get("usually_appropriate_procedures", [])),
                "search_method": "original_only",
                "best_query": user_query
            }
        }

    
    async def _evaluate_acr_results(self, user_query: str, acr_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to evaluate ACR results and select the best recommendations with smart variant selection
        """
        primary_result = acr_results.get("primary_result", {})
        
        if "error" in primary_result or not primary_result:
            return primary_result
        
        # Get all top variants for evaluation
        all_variants = primary_result.get("all_variants", [])
        
        # Ensure all_variants is a list
        if not isinstance(all_variants, list):
            print(f"Warning: all_variants is {type(all_variants)}, expected list")
            return self._add_evaluation_metadata(primary_result, acr_results, "invalid_variants")
        
        for i, variant in enumerate(all_variants):
            variant["rank"] = i + 1
        
        if len(all_variants) <= 1:
            return self._add_evaluation_metadata(primary_result, acr_results, "single_variant")


        variants_for_prompt = all_variants[:10]  # limit to 10 variants for prompt brevity
        formatted_variants = self._format_variants_for_llm_selection(variants_for_prompt)

        evaluation_prompt = f"""
You are a radiology triage assistant.  
Pick ONE variant that best matches the provider’s query using this hierarchy (stop at first mismatch):

1. CLINICAL SIGNS\tMust match exactly, including negations.  
   • e.g. “no fever” ≠ “fever”; “no trauma” = “unrelated to trauma”.

2. AGE GROUP\tAdult (≥18 y) vs Pediatric (<18 y) vs Neonate (<1 mo).  
   Never cross groups.

3. IMAGING PHASE\tinitial / next study / surveillance-follow-up.  
   “Next study” only if prior imaging mentioned; “initial” only if none.

4. PRIMARY INDICATION\tMain reason for imaging (e.g. “suspected bone tumour”).  
   Prefer general over over-specific unless the query names the subtype.

5. ANATOMIC REGION & SYMPTOM SPECIFICITY.

Reject a variant as soon as it fails a higher rule.

ORIGINAL QUERY:
"{user_query}"

RANKED VARIANTS (from ColBERT semantic search):
{formatted_variants}


Return **JSON only**  
```json
{{
  "selected_variant_rank": "<1-10>",
  "confidence": "low|medium|high",
  "reasoning": "<=25 words"
}}
```
"""

        print("\n📝 LLM Variant-selection prompt:\n" + "-"*60)
        print(evaluation_prompt)

        try:
            evaluation_response = await self.llm.ainvoke([HumanMessage(content=evaluation_prompt)])
            response_content = evaluation_response.content.strip()
            print("\n🤖 LLM response:\n" + "-"*60)
            print(response_content)

            decision = self._parse_json_response(response_content)
            selected_rank = int(decision.get("selected_variant_rank", 1))
            confidence = decision.get("confidence", "medium").lower()
            reasoning = decision.get("reasoning", "LLM decision")

            # Always trust LLM medical reasoning over ColBERT similarity ranking
            # ColBERT gets us the right variants in top 10, LLM picks the clinically correct one
            should_override = (
                1 <= selected_rank <= len(all_variants) and
                selected_rank != 1  # Only override if LLM selected a different variant
            )

            if should_override:
                print(f"🔄 LLM overrides to rank {selected_rank} based on medical reasoning.")
                chosen_variant = all_variants[selected_rank-1]
                result_with_override = await self._rebuild_result_with_variant(primary_result, chosen_variant, user_query)
                return self._add_evaluation_metadata(result_with_override, acr_results, "llm_override", llm_reasoning=reasoning, llm_confidence=confidence, original_rank=selected_rank)
            else:
                print("✅ LLM confirms ColBERT rank-1 variant is clinically appropriate.")

                # Enrich procedures using Neo4j to ensure full list, not just ColBERT stub
                try:
                    rank1_variant = all_variants[0]
                    enriched_result = await self._rebuild_result_with_variant(primary_result, rank1_variant, user_query)
                except Exception as enrich_err:
                    print(f"⚠️  Procedure enrichment failed: {enrich_err}")
                    enriched_result = primary_result

                return self._add_evaluation_metadata(enriched_result, acr_results, "llm_confirmed", llm_reasoning=reasoning, llm_confidence=confidence, original_rank=1)

        except Exception as e:
            print(f"LLM evaluation failed: {e}")
            return self._add_evaluation_metadata(primary_result, acr_results, "llm_error")
    
    
    def _format_variants_for_llm_selection(self, variants: List[Dict[str, Any]]) -> str:
        """Format variants for LLM selection evaluation with inferred clinical phase"""


        formatted = []
        for variant in variants:
            rank = variant.get("rank", "Unknown")
            content = variant.get("content", variant.get("variant_id", "Unknown"))
            similarity = variant.get("similarity", 0.0)
            condition = variant.get("condition_id", "Unknown")

            formatted.append(f"**Rank {rank}:** {content}")

        return "\n".join(formatted)
    
    async def _rebuild_result_with_variant(self, original_result: Dict[str, Any], selected_variant: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Rebuild ACR result with LLM-selected variant"""
        variant_id = selected_variant["variant_id"]
        
        try:
            # Resolve variant ID if placeholder (colbert_variant_X)
            if variant_id.startswith("colbert_variant"):
                resolved_id = self.neo4j_recommender.resolve_variant_id_by_text(selected_variant.get("content", ""))
                if resolved_id:
                    print(f"🔗 Resolved ColBERT placeholder to Neo4j variant id: {resolved_id}")
                    variant_id = resolved_id

            # Get procedures for the (resolved) variant
            procedures = self.neo4j_recommender.get_procedures_for_variant(variant_id, "USUALLY_APPROPRIATE")
            
            # Ensure procedures is a list
            if not isinstance(procedures, list):
                print(f"Warning: get_procedures_for_variant returned {type(procedures)}, expected list")
                procedures = []
            
            # Rebuild result structure
            new_result = original_result.copy()
            new_result["best_variant"] = {
                "variant_id": variant_id,
                "variant_similarity": selected_variant["similarity"],
                "variant_description": variant_id  # Could be enhanced with actual description
            }
            new_result["usually_appropriate_procedures"] = procedures
            new_result["context_for_output"] = {
                "clinical_category": selected_variant.get("condition_id", "Unknown"),
                "specific_scenario": variant_id,
                "search_confidence": "llm_override"
            }
            
            print(f"🔄 Rebuilt result with variant: {variant_id}")
            print(f"Found {len(procedures)} procedures for selected variant")
            
            return new_result
            
        except Exception as e:
            print(f"Error in _rebuild_result_with_variant: {e}")
            print(f"   Variant ID: {variant_id}")
            print(f"   Procedures type: {type(procedures) if 'procedures' in locals() else 'undefined'}")
            
            # Return original result as fallback
            return original_result
    
    def _add_evaluation_metadata(self, result: Dict[str, Any], 
                                acr_results: Dict[str, Any], evaluation_type: str,
                                llm_reasoning: str = "", llm_confidence: str = "",
                                negation_detected: bool = False, original_rank: int = 1) -> Dict[str, Any]:
        """Add evaluation metadata to result"""
        if "evaluation" not in result:
            result["evaluation"] = {}
        
        result["evaluation"].update({
            "queries_tried": acr_results.get("search_metadata", {}).get("queries_tried", []),
            "best_similarity": acr_results.get("search_metadata", {}).get("best_similarity", 0.0),
            "evaluation_type": evaluation_type,
            "llm_reasoning": llm_reasoning,
            "llm_confidence": llm_confidence,
            "negation_issue_detected": negation_detected,
            "selected_rank": original_rank
        })
        
        if evaluation_type == "llm_override":
            print(f"🎉 LLM Override Applied: {llm_reasoning}")
        
        return result
    
    def _format_acr_result_for_evaluation(self, result: Dict[str, Any]) -> str:
        """Format ACR result for LLM evaluation"""
        if "error" in result:
            return f"Error: {result['error']}"
        
        if "best_variant" in result:
            variant_id = result["best_variant"].get("variant_id", "Unknown")
            similarity = result["best_variant"].get("variant_similarity", 0.0)
            procedures = result.get("usually_appropriate_procedures", [])
            
            return f"Variant: {variant_id} (similarity: {similarity:.3f}), Procedures: {len(procedures)}"
        
        return "No valid result"
    
    def _format_alternative_results(self, alternatives: List[Dict[str, Any]]) -> str:
        """Format alternative results for LLM evaluation"""
        if not alternatives:
            return "None"
        
        formatted = []
        for i, alt in enumerate(alternatives):
            query = alt.get("query", "Unknown")
            result = alt.get("result", {})
            if "best_variant" in result:
                similarity = result["best_variant"].get("variant_similarity", 0.0)
                formatted.append(f"{i+1}. Query: '{query}' (similarity: {similarity:.3f})")
        
        return "\n".join(formatted) if formatted else "None valid"
    
    
    async def _direct_variant_search_with_context(self, query_text: str) -> Dict[str, Any]:
        """
        Hybrid search: ColBERT for variant finding + Neo4j for procedure fetching
        """
        try:
            print(f"Hybrid Search: '{query_text}'")
            
            # Step 1: Use ColBERT to find best matching ACR variant
            colbert_results = self.colbert_retriever.search_acr_variants(query_text, k=10)
            
            if not colbert_results:
                return {"error": "No ACR variants found"}
            
            # Format all results with proper structure
            formatted_results = []
            for i, result in enumerate(colbert_results):
                formatted_result = {
                    "rank": i + 1,
                    "variant_id": result["content"],
                    "content": result["content"],
                    "similarity": result["score"],
                    "condition_id": "colbert_matched_condition",
                    "relevance_score": result["score"]
                }
                formatted_results.append(formatted_result)
            
            best_result = formatted_results[0]
            variant_content = best_result["content"]
            
            print(f"ColBERT found variant: {variant_content}")
            print(f"Similarity: {best_result['similarity']:.4f}")
            
            # Step 2: Extract actual variant ID from the content and use Neo4j to get procedures
            # The variant content should contain the actual variant ID
            # Need to find this variant in Neo4j and get its procedures
            procedures = self.neo4j_recommender.get_procedures_for_variant(variant_content, "USUALLY_APPROPRIATE")
            
            # Build proper result structure
            result = {
                "query": query_text,
                "retrieval_method": "hybrid_colbert_neo4j",
                "best_variant": {
                    "variant_id": variant_content,
                    "content": variant_content,
                    "variant_similarity": best_result["similarity"],
                    "relevance_score": best_result["similarity"]
                },
                "usually_appropriate_procedures": procedures if isinstance(procedures, list) else [],
                "all_variants": formatted_results,  # Use formatted results here
                "total_results": len(procedures) if isinstance(procedures, list) else 0
            }
            
            print(f"Neo4j returned {len(result['usually_appropriate_procedures'])} procedures")
            return result
                
        except Exception as e:
            print(f"Hybrid variant search failed: {e}")
            return {"error": f"Hybrid variant search failed: {str(e)}"}
    
    def _direct_variant_search(self, query_text: str) -> Dict[str, Any]:
        """
        Synchronous fallback for direct variant search
        """
        try:
            # Simple fallback using the original recommender
            return self.neo4j_recommender.recommend_procedures(query_text)
        except Exception as e:
            return {"error": f"Fallback search failed: {str(e)}"}
    
    def _parse_json_response(self, response_content: str) -> Dict[str, Any]:
        """
        Robust JSON parsing with multiple cleaning strategies
        """
        import re
        
        # Strategy 1: Direct parsing
        try:
            return json.loads(response_content)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Remove markdown code blocks
        cleaned = response_content
        
        # Remove ```json ... ``` blocks
        if '```json' in cleaned:
            cleaned = re.sub(r'```json\s*\n?', '', cleaned)
            cleaned = re.sub(r'\n?```\s*$', '', cleaned)
        
        # Remove generic ``` blocks
        elif '```' in cleaned:
            cleaned = re.sub(r'```\s*\n?', '', cleaned)
            cleaned = re.sub(r'\n?```\s*$', '', cleaned)
        
        cleaned = cleaned.strip()
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Find JSON within text
        # Look for { ... } pattern
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Strategy 4: Remove common prefixes/suffixes
        lines = cleaned.split('\n')
        json_lines = []
        in_json = False
        
        for line in lines:
            if line.strip().startswith('{') or in_json:
                in_json = True
                json_lines.append(line)
                if line.strip().endswith('}') and line.count('}') >= line.count('{'):
                    break
        
        if json_lines:
            try:
                return json.loads('\n'.join(json_lines))
            except json.JSONDecodeError:
                pass
        
        # Strategy 5: Last resort - extract between first { and last }
        first_brace = cleaned.find('{')
        last_brace = cleaned.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            try:
                json_content = cleaned[first_brace:last_brace + 1]
                return json.loads(json_content)
            except json.JSONDecodeError:
                pass
        
        # If all strategies fail, raise the original error
        print(f"All JSON parsing strategies failed")
        print(f"Raw response (first 200 chars): {response_content[:200]}")
        print(f"Raw response (last 200 chars): {response_content[-200:]}")
        raise json.JSONDecodeError("Could not parse JSON from LLM response", response_content, 0)

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        error_msg = AIMessage(
            content=f"Enhanced ACR Agent error: {error_message}",
            additional_kwargs={"error": True, "source": "enhanced_acr_agent"}
        )
        
        return {
            "messages": [error_msg],
            "acr_recommendations": {"error": error_message},
            "next_step": "error_handling"
        }