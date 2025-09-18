import os
import json
import requests
import numpy as np
from typing import List, Dict, Tuple, Optional
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI


class MedicalProcedureRecommenderVectorized:
    """
    A system for recommending medical procedures based on ACR appropriateness criteria
    using Neo4j knowledge graph and VECTOR INDEXES for fast similarity search.
    
    This version uses Neo4j vector indexes instead of manual similarity calculations.
    """
    
    def __init__(self, neo4j_uri: str = "neo4j+s://d761b877.databases.neo4j.io", 
                 neo4j_user: str = "neo4j", 
                 neo4j_password: str = None,
                 embedding_provider: str = "pubmedbert",  # "openai", "ollama", or "pubmedbert"
                 embedding_model: str = "NeuML/pubmedbert-base-embeddings",
                 openai_api_key: str = None,
                 ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize the recommender system with vector index support.
        
        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            embedding_provider: "openai", "ollama", or "pubmedbert"
            embedding_model: Embedding model name
            openai_api_key: OpenAI API key (if using OpenAI)
            ollama_base_url: Ollama API base URL (if using Ollama)
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = os.environ.get("NEO4J_PASSWORD")
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.ollama_base_url = ollama_base_url
        self.stored_embedding_dim = None  # Will be detected from database
        
        # Initialize embedding provider
        if embedding_provider == "pubmedbert":
            self.openai_client = None
            print(f"🚀 Using PubMedBERT with model {self.embedding_model}")
        elif embedding_provider == "openai":
            self.openai_client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
            print(f"🚀 Using OpenAI API with model {self.embedding_model}")
        else:
            self.openai_client = None
            print(f"🚀 Using Ollama at {self.ollama_base_url} with model {self.embedding_model}")
        
        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(
            self.neo4j_uri, 
            auth=(self.neo4j_user, self.neo4j_password)
        )
        
        # Verify connectivity
        self.driver.verify_connectivity()
        
        print(f"🔌 Connected to Neo4j at {self.neo4j_uri}")
        
        # Check vector indexes
        self._check_vector_indexes()
        
        # Detect embedding dimensions from stored data
        self._detect_embedding_dimensions()
    
    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
    
    def _check_vector_indexes(self):
        """Check if required vector indexes exist."""
        print("🔍 Checking vector indexes...")
        
        with self.driver.session() as session:
            # Check for vector indexes
            index_query = """
            SHOW INDEXES
            YIELD name, labelsOrTypes, properties, state, type
            WHERE type = "VECTOR"
            RETURN name, labelsOrTypes, properties, state
            """
            
            try:
                result = session.run(index_query)
                indexes = list(result)
                
                condition_index = None
                variant_index = None
                
                for record in indexes:
                    name = record['name']
                    labels = record['labelsOrTypes']
                    state = record['state']
                    
                    if 'Condition' in labels:
                        condition_index = name
                        print(f"✅ Found Condition vector index: {name} ({state})")
                    elif 'Variant' in labels:
                        variant_index = name
                        print(f"✅ Found Variant vector index: {name} ({state})")
                
                self.condition_index_name = condition_index
                self.variant_index_name = variant_index
                
                if not condition_index:
                    print("⚠️ WARNING: No Condition vector index found!")
                    print("   Falling back to manual similarity calculation")
                
                if not variant_index:
                    print("⚠️ WARNING: No Variant vector index found!")
                    print("   Falling back to manual similarity calculation")
                
            except Exception as e:
                print(f"❌ Error checking vector indexes: {e}")
                self.condition_index_name = None
                self.variant_index_name = None
    
    def _detect_embedding_dimensions(self):
        """
        Detect the dimensions of embeddings stored in the database.
        """
        try:
            # Check a few conditions to get embedding dimensions
            query = """
            MATCH (c:Condition)
            WHERE c.embedding IS NOT NULL
            RETURN c.embedding as embedding
            LIMIT 1
            """
            
            with self.driver.session() as session:
                result = session.run(query)
                record = result.single()
                
                if record and record["embedding"]:
                    self.stored_embedding_dim = len(record["embedding"])
                    print(f"📊 Detected stored embedding dimension: {self.stored_embedding_dim}")
                    
                    # Test current model dimension
                    try:
                        test_embedding = self.get_embedding("test")
                        current_model_dim = len(test_embedding)
                        print(f"📊 Current model embedding dimension: {current_model_dim}")
                        
                        if self.stored_embedding_dim != current_model_dim:
                            print(f"⚠️ WARNING: Dimension mismatch!")
                            print(f"  Stored embeddings: {self.stored_embedding_dim} dimensions")
                            print(f"  Query embeddings: {current_model_dim} dimensions")
                        else:
                            print("✅ Embedding dimensions match!")
                            
                    except Exception as e:
                        print(f"Could not test current model: {e}")
                else:
                    print("No embeddings found in database to detect dimensions")
                    
        except Exception as e:
            print(f"Could not detect embedding dimensions: {e}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text using the configured provider (OpenAI, Ollama, or PubMedBERT).
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding as numpy array
        """
        try:
            if self.embedding_provider == "pubmedbert":
                # Use PubMedBERT via sentence-transformers
                if not hasattr(self, 'pubmedbert_model'):
                    from sentence_transformers import SentenceTransformer
                    self.pubmedbert_model = SentenceTransformer("NeuML/pubmedbert-base-embeddings")
                embedding = self.pubmedbert_model.encode(text, convert_to_numpy=True)
                return embedding
            elif self.embedding_provider == "openai":
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                embedding = response.data[0].embedding
                return np.array(embedding)
            else:
                # Ollama
                response = requests.post(
                    f"{self.ollama_base_url}/api/embeddings",
                    json={
                        "model": self.embedding_model,
                        "prompt": text
                    }
                )
                response.raise_for_status()
                embedding = response.json()["embedding"]
                return np.array(embedding)
        except Exception as e:
            print(f"Error getting embedding from {self.embedding_provider}: {e}")
            raise
    
    def find_similar_conditions_vectorized(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find conditions similar to the query text using VECTOR INDEX.
        
        Args:
            query_text: Input medical query
            top_k: Number of similar conditions to return
            
        Returns:
            List of tuples (condition_id, similarity_score)
        """
        if not self.condition_index_name:
            print("⚠️ No condition vector index available, falling back to manual calculation")
            return self.find_similar_conditions_manual(query_text, top_k)
        
        # Get embedding for the query
        query_embedding = self.get_embedding(query_text)
        
        # Use vector index for fast similarity search
        # WORKAROUND: Neo4j vector index bug with top_k=1, always request at least 2
        actual_top_k = max(2, top_k)
        
        query = f"""
        CALL db.index.vector.queryNodes('{self.condition_index_name}', $actual_top_k, $query_embedding)
        YIELD node, score
        WHERE node.gid = "ACR_BATCH_1"
        RETURN node.id as condition_id, score as similarity_score
        ORDER BY score DESC
        """
        
        with self.driver.session() as session:
            result = session.run(query, actual_top_k=actual_top_k, query_embedding=query_embedding.tolist())
            
            similarities = []
            for record in result:
                similarities.append((record["condition_id"], record["similarity_score"]))
            
            # Return only the requested number of results
            similarities = similarities[:top_k]
            
            print(f"🚀 Found {len(similarities)} similar conditions using vector index")
            return similarities
    
    def find_similar_variants_vectorized(self, query_text: str, condition_id: str = None, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find variants similar to the query text using VECTOR INDEX.
        
        Args:
            query_text: Input medical query
            condition_id: Optional condition ID to filter variants
            top_k: Number of similar variants to return
            
        Returns:
            List of tuples (variant_id, similarity_score)
        """
        if not self.variant_index_name:
            print("⚠️ No variant vector index available, falling back to manual calculation")
            return self.find_most_similar_variant_manual(query_text, condition_id, top_k)
        
        # Get embedding for the query
        query_embedding = self.get_embedding(query_text)
        
        # Use vector index for fast similarity search
        if condition_id:
            # Filter by condition if specified
            query = f"""
            CALL db.index.vector.queryNodes('{self.variant_index_name}', $top_k_expanded, $query_embedding)
            YIELD node, score
            WHERE node.gid = "ACR_BATCH_1"
            AND EXISTS {{
                MATCH (c:Condition {{id: $condition_id}})-[:HAS_VARIANT]->(node)
            }}
            RETURN node.id as variant_id, score as similarity_score
            ORDER BY score DESC
            LIMIT $top_k
            """
            params = {"top_k": top_k, "top_k_expanded": max(15, top_k * 5), "query_embedding": query_embedding.tolist(), "condition_id": condition_id}
        else:
            # Search all variants
            query = f"""
            CALL db.index.vector.queryNodes('{self.variant_index_name}', $top_k, $query_embedding)
            YIELD node, score
            WHERE node.gid = "ACR_BATCH_1"
            RETURN node.id as variant_id, score as similarity_score
            ORDER BY score DESC
            """
            params = {"top_k": top_k, "query_embedding": query_embedding.tolist()}
        
        with self.driver.session() as session:
            result = session.run(query, **params)
            
            similarities = []
            for record in result:
                similarities.append((record["variant_id"], record["similarity_score"]))
            
            print(f"🚀 Found {len(similarities)} similar variants using vector index")
            return similarities
    
    def find_similar_conditions_manual(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        FALLBACK: Find conditions similar to the query text using manual calculation.
        
        Args:
            query_text: Input medical query
            top_k: Number of similar conditions to return
            
        Returns:
            List of tuples (condition_id, similarity_score)
        """
        print("🐌 Using manual similarity calculation for conditions...")
        
        # Get embedding for the query
        query_embedding = self.get_embedding(query_text)
        
        # Get all conditions with embeddings
        query = """
        MATCH (c:Condition)
        WHERE c.gid = "ACR_BATCH_1" AND c.embedding IS NOT NULL
        RETURN c.id as condition_id, c.embedding as embedding
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            
            similarities = []
            for record in result:
                condition_embedding = np.array(record["embedding"])
                
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    condition_embedding.reshape(1, -1)
                )[0][0]
                
                similarities.append((record["condition_id"], similarity))
            
            # Sort by similarity score (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
    
    def find_most_similar_variant_manual(self, query_text: str, condition_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        FALLBACK: Find the most similar variants using manual calculation.
        
        Args:
            query_text: Input medical query
            condition_id: ID of the condition to search variants within
            top_k: Number of variants to return
            
        Returns:
            List of tuples (variant_id, similarity_score)
        """
        print("🐌 Using manual similarity calculation for variants...")
        
        # Get embedding for the query
        query_embedding = self.get_embedding(query_text)
        
        # Get all variants for this condition
        query = """
        MATCH (c:Condition {id: $condition_id})-[:HAS_VARIANT]->(v:Variant)
        WHERE v.gid = "ACR_BATCH_1" AND v.embedding IS NOT NULL
        RETURN v.id as variant_id, v.embedding as embedding
        """
        
        with self.driver.session() as session:
            result = session.run(query, condition_id=condition_id)
            
            similarities = []
            for record in result:
                variant_embedding = np.array(record["embedding"])
                
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    variant_embedding.reshape(1, -1)
                )[0][0]
                
                similarities.append((record["variant_id"], similarity))
            
            # Sort by similarity score (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
    
    def get_procedures_for_variant(self, variant_id: str, appropriateness_filter: str = None) -> List[Dict]:
        """
        Get procedures for a specific variant, optionally filtered by appropriateness.
        
        Args:
            variant_id: ID of the variant
            appropriateness_filter: Optional filter ('USUALLY_APPROPRIATE', 'MAYBE_APPROPRIATE', 'USUALLY_NOT_APPROPRIATE')
            
        Returns:
            List of procedures
        """
        if appropriateness_filter:
            query = """
            MATCH (v:Variant {id: $variant_id})-[r]->(p:Procedure)
            WHERE type(r) = $appropriateness
            RETURN type(r) as appropriateness, r.rationale as edge_rationale, p.id as procedure_id, 
                   p.peds_rrl_dosage as dosage, p as procedure_properties
            """
            params = {"variant_id": variant_id, "appropriateness": appropriateness_filter}
        else:
            query = """
            MATCH (v:Variant {id: $variant_id})-[r]->(p:Procedure)
            WHERE type(r) IN ['USUALLY_APPROPRIATE', 'MAYBE_APPROPRIATE', 'USUALLY_NOT_APPROPRIATE']
            RETURN type(r) as appropriateness, r.rationale as edge_rationale, p.id as procedure_id, 
                   p.peds_rrl_dosage as dosage, p as procedure_properties
            """
            params = {"variant_id": variant_id}
        
        with self.driver.session() as session:
            try:
                result = session.run(query, **params)

                procedures = []
                for record in result:
                    # Safely extract properties
                    procedure_properties = record.get('procedure_properties')
                    if procedure_properties is not None:
                        properties_dict = dict(procedure_properties)
                    else:
                        properties_dict = {}

                    procedure_info = {
                        'procedure_id': record.get('procedure_id', 'Unknown'),
                        'title': record.get('procedure_id', 'Unknown'),
                        'edge_rationale': record.get('edge_rationale', ''),
                        'appropriateness': record.get('appropriateness', 'Unknown'),
                        'dosage': record.get('dosage', ''),
                        'properties': properties_dict
                    }
                    procedures.append(procedure_info)

                return procedures
            except Exception as e:
                print(f"Error in get_procedures_for_variant: {e}")
                print(f"Query: {query}")
                print(f"Params: {params}")
                return []
    
    def recommend_procedures(self, query_text: str, top_conditions: int = 1) -> Dict:
        """
        Recommend procedures based on query text using VECTOR INDEXES.
        
        Args:
            query_text: Medical query text
            top_conditions: Number of top conditions to consider
            
        Returns:
            Dictionary with recommendations
        """
        print(f"\n🎯 === VECTORIZED ACR RECOMMENDATION ===")
        print(f"🎯 Query: '{query_text}'")
        print(f"🎯 Top conditions: {top_conditions}")
        
        recommendations = {
            'query': query_text
        }
        
        # Step 1: Find similar conditions using vector index
        print(f"\n🔍 Step 1: Finding similar conditions...")
        similar_conditions = self.find_similar_conditions_vectorized(query_text, top_conditions)
        

        
        if not similar_conditions:
            print("❌ No similar conditions found")
            return recommendations
        
        # Use the new approach: find top condition, then best variant
        if not similar_conditions:
            return {"error": "No similar conditions found"}
        
        # Get the top condition
        top_condition_id, condition_similarity = similar_conditions[0]
        print(f"\n📋 Processing condition: {top_condition_id} (similarity: {condition_similarity:.4f})")
        
        # Find the most similar variant within that condition
        try:
            similar_variants = self.find_similar_variants_vectorized(query_text, top_condition_id, top_k=1)
            
            if not similar_variants:
                return {"error": f"No variants found for condition: {top_condition_id}"}
            
            best_variant_id, variant_similarity = similar_variants[0]
            print(f"   📋 Best variant: {best_variant_id} (similarity: {variant_similarity:.4f})")
            
            # Get USUALLY_APPROPRIATE procedures for this variant
            procedures = self.get_procedures_for_variant(best_variant_id, "USUALLY_APPROPRIATE")
            print(f"      📋 Found {len(procedures)} usually appropriate procedures")
            
            # Return in the expected format
            recommendations.update({
                "top_condition": {
                    "condition_id": top_condition_id,
                    "condition_similarity": condition_similarity
                },
                "best_variant": {
                    "variant_id": best_variant_id,
                    "variant_similarity": variant_similarity
                },
                "usually_appropriate_procedures": procedures
            })
            
            print(f"\n✅ Generated recommendations for top condition")
            return recommendations
            
        except Exception as e:
            print(f"Error finding variant: {e}")
            return {"error": f"Error processing variants for condition {top_condition_id}: {str(e)}"}
    
    def print_recommendations(self, recommendations: Dict):
        """
        Print recommendations in a readable format.
        
        Args:
            recommendations: Dictionary with recommendations
        """
        print(f"\n🎯 === ACR PROCEDURE RECOMMENDATIONS ===")
        
        for condition_id, condition_data in recommendations.items():
            print(f"\n🏥 CONDITION: {condition_id}")
            print(f"   Similarity: {condition_data['similarity']:.4f}")
            
            for variant_id, variant_data in condition_data['variants'].items():
                print(f"\n   📋 VARIANT: {variant_id}")
                print(f"      Similarity: {variant_data['similarity']:.4f}")
                
                procedures = variant_data['procedures']
                if procedures:
                    print(f"      PROCEDURES ({len(procedures)}):")
                    for i, proc in enumerate(procedures[:5], 1):  # Show top 5
                        print(f"         {i}. {proc['title']}")
                        print(f"            Appropriateness: {proc['appropriateness']}")
                        if proc['dosage']:
                            print(f"            Dosage: {proc['dosage']}")
                else:
                    print(f"      No procedures found")
        
        print(f"\n🎯 === END RECOMMENDATIONS ===")

    # ---------------------- NEW HELPER ----------------------
    def resolve_variant_id_by_text(self, variant_text: str) -> str:
        """Given the free-text of a variant, return the true Variant.id stored in Neo4j.

        We match on either v.description or v.id exactly (case-sensitive) and return the v.id
        so that downstream calls to get_procedures_for_variant() work.
        Returns None if not found.
        """
        query = (
            "MATCH (v:Variant)\n"
            "WHERE v.description = $txt OR v.id = $txt\n"
            "RETURN v.id AS vid LIMIT 1"
        )

        try:
            with self.driver.session() as session:
                res = session.run(query, txt=variant_text)
                record = res.single()
                if record:
                    return record["vid"]
        except Exception as e:
            print(f"Error resolving variant text '{variant_text}': {e}")
        return None


def main():
    """Test the vectorized recommender"""
    
    # Initialize the vectorized recommender
    recommender = MedicalProcedureRecommenderVectorized(
        embedding_provider="ollama",
        embedding_model="nomic-embed-text:latest"
    )
    
    # Test query
    test_query = "26 year old woman with chest pain"
    
    print(f"\n🧪 Testing vectorized ACR recommendations...")
    print(f"🧪 Query: '{test_query}'")
    
    # Get recommendations
    recommendations = recommender.recommend_procedures(test_query, top_conditions=2)
    
    # Print results
    recommender.print_recommendations(recommendations)
    
    # Close connection
    recommender.close()


if __name__ == "__main__":
    main() 