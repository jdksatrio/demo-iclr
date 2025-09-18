from typing import Dict, Any, List, Optional
import asyncio
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from neo4j import GraphDatabase
import os


class EnhancedGraphRAGAgent:
    """
    Enhanced GraphRAG Agent with LLM-powered reasoning for intelligent medical knowledge retrieval.
    
    This agent uses Chain of Thought reasoning to:
    1. Analyze the medical query for key concepts
    2. Plan strategic search approaches  
    3. Execute multi-step knowledge retrieval
    4. Synthesize comprehensive medical context
    """
    
    def __init__(
        self,
        llm,
        neo4j_uri: str = "neo4j+s://d761b877.databases.neo4j.io",
        neo4j_user: str = "neo4j", 
        neo4j_password: str = None
    ):
        self.llm = llm
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.neo4j_driver = None
        self.initialized = False
        self._embedding_model = None
        
        # Medical entity types for focused search (based on actual Neo4j schema)
        self.medical_entity_types = [
            "ANATOMY", "BIOMARKER", "CONDITION", "DEVICE", "DIAGNOSIS", 
            "DIAGNOSTIC TEST", "DIAGNOSTIC_TEST", "DISEASE", "MEDICAL CONDITION", 
            "MEDICAL DEVICE", "MEDICATION", "PROCEDURE", "ORGANIZATION", "PERSON"
        ]
    
    async def initialize(self, neo4j_password: str = None):
        """Initialize Neo4j connection and embedding model"""
        if self.initialized:
            return
            
        try:
            password = neo4j_password or self.neo4j_password or os.environ.get("NEO4J_PASSWORD")
            if not password:
                raise ValueError("NEO4J_PASSWORD environment variable is required for AuraDB connection")
            
            self.neo4j_driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, password)
            )
            
            # Verify connectivity
            self.neo4j_driver.verify_connectivity()
            
            # Initialize embedding model
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer("NeuML/pubmedbert-base-embeddings")
            
            self.initialized = True
            print(f"Enhanced GraphRAG Agent initialized at {self.neo4j_uri}")
            
        except Exception as e:
            print(f"Enhanced GraphRAG Agent initialization failed: {e}")
            raise
    
    def close(self):
        """Close Neo4j connection"""
        if self.neo4j_driver:
            self.neo4j_driver.close()
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main agent execution with LLM-powered reasoning
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
        
        # ✅ ADD ENHANCED QUERY VALIDATION
        import pandas as pd
        if (pd.isna(user_query) or 
            not str(user_query).strip() or 
            str(user_query).lower() in ['nan', 'none', 'null'] or
            len(str(user_query).strip()) < 3):
            return self._create_error_response(f"Invalid query received: '{user_query}'")
        
        # Ensure query is a proper string
        user_query = str(user_query).strip()
        
        try:
            # Step 1: LLM analyzes the query and plans retrieval strategy
            analysis_result = await self._analyze_query_with_llm(user_query)
            
            # Step 2: Execute strategic knowledge retrieval based on LLM analysis
            retrieved_context = await self._execute_strategic_retrieval(user_query, analysis_result)
            
            # Step 3: LLM synthesizes comprehensive medical context
            final_context = await self._synthesize_medical_context(user_query, analysis_result, retrieved_context)
            
            # Create response message
            response_message = AIMessage(
                content=f"Enhanced GraphRAG Agent analyzed query and retrieved comprehensive medical context",
                additional_kwargs={
                    "query_analysis": analysis_result,
                    "raw_context": retrieved_context,
                    "synthesized_context": final_context,
                    "source": "enhanced_graphrag_agent"
                }
            )
            
            return {
                "messages": [response_message],
                "graphrag_context": final_context,
                "graphrag_analysis": analysis_result,
                "next_step": "acr_retrieval"
            }
            
        except Exception as e:
            return self._create_error_response(f"Enhanced GraphRAG Agent failed: {str(e)}")
    
    async def _analyze_query_with_llm(self, user_query: str) -> Dict[str, Any]:
        """
        Use LLM to analyze the medical query and plan retrieval strategy
        """
        analysis_prompt = f"""
        You are an expert medical knowledge retrieval strategist. Analyze this clinical query and plan the most effective knowledge retrieval approach.

        **Clinical Query:** {user_query}

        Please provide a strategic analysis in this EXACT JSON format:

        {{
            "primary_medical_concepts": ["concept1", "concept2", "concept3"],
            "clinical_context": "brief description of clinical scenario",
            "search_strategy": {{
                "primary_entities": ["key entities to search for"],
                "related_concepts": ["broader related medical concepts"],
                "diagnostic_focus": ["specific diagnostic considerations"],
                "anatomical_systems": ["relevant anatomical systems/organs"]
            }},
            "retrieval_priorities": ["priority1", "priority2", "priority3"],
            "evidence_requirements": ["type of evidence needed"],
            "risk_factors": ["potential clinical risks to consider"]
        }}

        **Instructions:**
        1. Extract 3-5 primary medical concepts (diseases, conditions, procedures, anatomy, diagnoses)
        2. Identify the clinical context (emergency, routine, specialized, etc.)
        3. Plan multi-layered search strategy (primary + related + diagnostic focus)
        4. Prioritize what information is most critical for clinical decision-making
        5. Consider what evidence quality is needed
        6. Identify any high-risk factors that need special attention
        
        **Available Entity Types in Database:** anatomy, biomarker, condition, device, diagnosis, diagnostic test, disease, medical condition, medical device, medication, procedure

        **Focus on:** Diagnostic imaging context, clinical decision support, and evidence-based recommendations.
        """
        
        try:
            analysis_response = await self.llm.ainvoke([HumanMessage(content=analysis_prompt)])
            
            # Parse JSON response
            import json
            analysis_data = json.loads(analysis_response.content)
            
            print(f"GraphRAG Agent Query Analysis:")
            print(f"  Primary concepts: {analysis_data.get('primary_medical_concepts', [])}")
            print(f"  Clinical context: {analysis_data.get('clinical_context', 'Unknown')}")
            print(f"  Search priorities: {analysis_data.get('retrieval_priorities', [])}")
            
            return analysis_data
            
        except json.JSONDecodeError:
            # Fallback: Extract key concepts manually if JSON parsing fails
            print("JSON parsing failed, using fallback analysis")
            return {
                "primary_medical_concepts": self._extract_medical_concepts_fallback(user_query),
                "clinical_context": "general clinical inquiry",
                "search_strategy": {
                    "primary_entities": [user_query],
                    "related_concepts": [],
                    "diagnostic_focus": [],
                    "anatomical_systems": []
                },
                "retrieval_priorities": ["semantic_search"],
                "evidence_requirements": ["clinical_guidelines"],
                "risk_factors": []
            }
        
        except Exception as e:
            print(f"Query analysis failed: {e}")
            return {"error": str(e)}
    
    async def _execute_strategic_retrieval(self, user_query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute multi-step knowledge retrieval based on LLM analysis
        """
        if "error" in analysis:
            return {"error": "Analysis failed, using basic retrieval"}
        
        retrieved_data = {
            "primary_entities": [],
            "related_concepts": [],
            "diagnostic_evidence": [],
            "clinical_guidelines": [],
            "anatomical_context": []
        }
        
        try:
            # Step 1: Primary concept search
            primary_concepts = analysis.get("primary_medical_concepts", [])
            for concept in primary_concepts[:3]:  # Limit to top 3
                entities = await self._semantic_search_concept(concept)
                retrieved_data["primary_entities"].extend(entities)
            
            # Step 2: Related concept expansion
            search_strategy = analysis.get("search_strategy", {})
            related_concepts = search_strategy.get("related_concepts", [])
            for concept in related_concepts[:2]:  # Limit to top 2
                entities = await self._semantic_search_concept(concept)
                retrieved_data["related_concepts"].extend(entities)
            
            # Step 3: Diagnostic-focused search
            diagnostic_focus = search_strategy.get("diagnostic_focus", [])
            for focus in diagnostic_focus[:2]:
                entities = await self._search_diagnostic_entities(focus)
                retrieved_data["diagnostic_evidence"].extend(entities)
            
            # Step 4: Anatomical system search
            anatomical_systems = search_strategy.get("anatomical_systems", [])
            for system in anatomical_systems[:2]:
                entities = await self._search_anatomical_context(system)
                retrieved_data["anatomical_context"].extend(entities)
            
            print(f"Retrieved entities: {len(retrieved_data['primary_entities'])} primary, "
                  f"{len(retrieved_data['related_concepts'])} related")
            
            return retrieved_data
            
        except Exception as e:
            print(f"Strategic retrieval failed: {e}")
            # Fallback to basic semantic search
            fallback_entities = await self._semantic_search_concept(user_query)
            return {"primary_entities": fallback_entities, "error": str(e)}
    
    async def _semantic_search_concept(self, concept: str) -> List[Dict[str, Any]]:
        """
        Perform semantic search for a specific medical concept
        """
        try:
            query_embedding = self._get_query_embedding(concept)
            if query_embedding is None:
                return []
            
            with self.neo4j_driver.session() as session:
                similarity_query = """
                CALL db.index.vector.queryNodes('graphrag_embeddings_index', $k, $query_embedding)
                YIELD node, score
                WHERE score > 0.4
                
                OPTIONAL MATCH (node)-[r]-(related)
                WHERE related.entity_type IN $entity_types
                  AND related.id IS NOT NULL
                
                RETURN node.id as entity_id,
                       node.entity_type as entity_type,
                       node.description as description,
                       score as similarity_score,
                       collect(DISTINCT {
                           entity: related.id,
                           type: related.entity_type,
                           description: related.description
                       })[0..2] as related_entities
                ORDER BY similarity_score DESC
                LIMIT 5
                """
                
                result = session.run(similarity_query, {
                    "query_embedding": query_embedding,
                    "k": 8,
                    "entity_types": self.medical_entity_types
                })
                
                entities = []
                for record in result:
                    entities.append({
                        "entity_id": record["entity_id"],
                        "entity_type": record["entity_type"],
                        "description": record["description"],
                        "similarity_score": record["similarity_score"],
                        "related_entities": record["related_entities"],
                        "search_concept": concept
                    })
                
                return entities
                
        except Exception as e:
            print(f"Semantic search for '{concept}' failed: {e}")
            return []
    
    async def _search_diagnostic_entities(self, diagnostic_focus: str) -> List[Dict[str, Any]]:
        """
        Search for diagnostic-specific entities (tests, procedures, criteria)
        """
        try:
            with self.neo4j_driver.session() as session:
                diagnostic_query = """
                MATCH (n)
                WHERE (n.entity_type IN ['DIAGNOSTIC_TEST', 'DIAGNOSTIC TEST', 'PROCEDURE', 'DIAGNOSIS']
                       OR n.id CONTAINS $focus_term
                       OR n.description CONTAINS $focus_term)
                  AND n.id IS NOT NULL
                
                OPTIONAL MATCH (n)-[r]-(related)
                WHERE related.entity_type IN ['DISEASE', 'CONDITION', 'MEDICAL CONDITION', 'ANATOMY']
                
                RETURN n.id as entity_id,
                       n.entity_type as entity_type,
                       n.description as description,
                       collect(DISTINCT related.id)[0..2] as related_entities
                ORDER BY 
                    CASE WHEN n.id CONTAINS $focus_term THEN 1 ELSE 2 END,
                    n.id
                LIMIT 3
                """
                
                result = session.run(diagnostic_query, {
                    "focus_term": diagnostic_focus.lower()
                })
                
                entities = []
                for record in result:
                    entities.append({
                        "entity_id": record["entity_id"],
                        "entity_type": record["entity_type"], 
                        "description": record["description"],
                        "related_entities": record["related_entities"],
                        "search_type": "diagnostic_focus"
                    })
                
                return entities
                
        except Exception as e:
            print(f"Diagnostic search for '{diagnostic_focus}' failed: {e}")
            return []
    
    async def _search_anatomical_context(self, anatomical_system: str) -> List[Dict[str, Any]]:
        """
        Search for anatomical context and related pathology
        """
        try:
            with self.neo4j_driver.session() as session:
                anatomical_query = """
                MATCH (n)
                WHERE (n.entity_type = 'ANATOMY'
                       OR n.id CONTAINS $system_term
                       OR n.description CONTAINS $system_term)
                  AND n.id IS NOT NULL
                
                OPTIONAL MATCH (n)-[r]-(pathology)
                WHERE pathology.entity_type IN ['DISEASE', 'CONDITION', 'MEDICAL CONDITION', 'DIAGNOSIS']
                
                RETURN n.id as entity_id,
                       n.entity_type as entity_type,
                       n.description as description,
                       collect(DISTINCT pathology.id)[0..3] as related_pathology
                ORDER BY 
                    CASE WHEN n.entity_type = 'ANATOMY' THEN 1 ELSE 2 END,
                    n.id
                LIMIT 3
                """
                
                result = session.run(anatomical_query, {
                    "system_term": anatomical_system.lower()
                })
                
                entities = []
                for record in result:
                    entities.append({
                        "entity_id": record["entity_id"],
                        "entity_type": record["entity_type"],
                        "description": record["description"], 
                        "related_pathology": record["related_pathology"],
                        "search_type": "anatomical_context"
                    })
                
                return entities
                
        except Exception as e:
            print(f"Anatomical search for '{anatomical_system}' failed: {e}")
            return []
    
    async def _synthesize_medical_context(self, user_query: str, analysis: Dict[str, Any], retrieved_data: Dict[str, Any]) -> str:
        """
        Use LLM to synthesize comprehensive medical context from retrieved data
        """
        synthesis_prompt = f"""
        You are an expert medical knowledge synthesizer. Create a comprehensive clinical context summary from the retrieved medical knowledge.

        **Original Query:** {user_query}

        **Query Analysis:** {analysis.get('clinical_context', 'General clinical inquiry')}

        **Retrieved Medical Knowledge:**
        
        **Primary Entities ({len(retrieved_data.get('primary_entities', []))}):**
        {self._format_entities_for_synthesis(retrieved_data.get('primary_entities', []))}
        
        **Related Concepts ({len(retrieved_data.get('related_concepts', []))}):**
        {self._format_entities_for_synthesis(retrieved_data.get('related_concepts', []))}
        
        **Diagnostic Evidence ({len(retrieved_data.get('diagnostic_evidence', []))}):**
        {self._format_entities_for_synthesis(retrieved_data.get('diagnostic_evidence', []))}
        
        **Anatomical Context ({len(retrieved_data.get('anatomical_context', []))}):**
        {self._format_entities_for_synthesis(retrieved_data.get('anatomical_context', []))}

        **Instructions:**
        1. Synthesize the most relevant medical knowledge into a coherent clinical context
        2. Focus on information directly relevant to the query
        3. Prioritize diagnostic and therapeutic insights
        4. Organize information by clinical relevance
        5. Keep the synthesis concise but comprehensive (300-500 words)
        6. Include specific medical entities and their relationships
        7. Highlight any critical clinical considerations

        **Output Format:**
        **Clinical Context:** [Brief overview of the medical scenario]

        **Key Medical Entities:** [List most relevant entities found]

        **Diagnostic Considerations:** [Relevant diagnostic information]

        **Clinical Relationships:** [Important medical relationships discovered]

        **Additional Context:** [Any other relevant medical knowledge]
        """
        
        try:
            synthesis_response = await self.llm.ainvoke([HumanMessage(content=synthesis_prompt)])
            
            synthesized_context = synthesis_response.content
            print(f"GraphRAG Agent synthesized context: {len(synthesized_context)} characters")
            
            return synthesized_context
            
        except Exception as e:
            print(f"Context synthesis failed: {e}")
            # Fallback to basic context formatting
            return self._create_fallback_context(user_query, retrieved_data)
    
    def _format_entities_for_synthesis(self, entities: List[Dict[str, Any]]) -> str:
        """Format entities for LLM synthesis"""
        if not entities:
            return "None found"
        
        formatted = []
        for i, entity in enumerate(entities[:5]):  # Limit to top 5
            entity_id = entity.get('entity_id', 'Unknown')
            entity_type = entity.get('entity_type', 'Unknown')
            description = entity.get('description', '')
            similarity = entity.get('similarity_score', 0)
            
            # Clean and truncate description
            if description:
                clean_desc = description.split('<SEP>')[0] if '<SEP>' in description else description
                clean_desc = clean_desc.strip('"').strip()[:200]
            else:
                clean_desc = "No description"
            
            if similarity > 0:
                formatted.append(f"{i+1}. {entity_id} ({entity_type}, {similarity:.2f}): {clean_desc}")
            else:
                formatted.append(f"{i+1}. {entity_id} ({entity_type}): {clean_desc}")
        
        return "\n".join(formatted)
    
    def _create_fallback_context(self, user_query: str, retrieved_data: Dict[str, Any]) -> str:
        """Create basic context when synthesis fails"""
        primary_entities = retrieved_data.get('primary_entities', [])
        
        if not primary_entities:
            return f"**Clinical Context:** {user_query} (limited context available)"
        
        # Extract top entities
        top_entities = []
        for entity in primary_entities[:3]:
            entity_id = entity.get('entity_id', 'Unknown')
            if entity_id:
                top_entities.append(entity_id.replace('_', ' ').title())
        
        context = f"**Clinical Context:** Query related to {', '.join(top_entities)} with {len(primary_entities)} relevant medical entities identified."
        
        return context
    
    def _extract_medical_concepts_fallback(self, query: str) -> List[str]:
        """Fallback method to extract medical concepts when LLM analysis fails"""
        # Simple keyword extraction
        medical_keywords = []
        query_lower = query.lower()
        
        # Common medical terms (aligned with Neo4j entity types)
        common_terms = [
            'pain', 'bleeding', 'fever', 'headache', 'chest', 'abdominal', 'pelvic',
            'cardiac', 'pulmonary', 'neurological', 'gastrointestinal', 'urological',
            'appendicitis', 'pneumonia', 'fracture', 'tumor', 'infection', 'inflammation',
            'condition', 'disease', 'diagnosis', 'procedure', 'anatomy', 'medication'
        ]
        
        for term in common_terms:
            if term in query_lower:
                medical_keywords.append(term)
        
        return medical_keywords[:5] if medical_keywords else [query]
    
    def _get_query_embedding(self, text: str):
        """Get embedding for query using PubMedBERT model"""
        try:
            if not self._embedding_model:
                return None
            
            embedding = self._embedding_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            print(f"Failed to get embedding for '{text}': {e}")
            return None
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        error_msg = AIMessage(
            content=f"Enhanced GraphRAG Agent error: {error_message}",
            additional_kwargs={"error": True, "source": "enhanced_graphrag_agent"}
        )
        
        return {
            "messages": [error_msg],
            "graphrag_context": f"Error: {error_message}",
            "next_step": "error_handling"
        }