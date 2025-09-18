"""
ColBERT-based ACR Retrieval System
"""

import os
import sys
from typing import List, Dict, Tuple, Optional
from ragatouille import RAGPretrainedModel


class ColBERTACRRetriever:
    """ACR retrieval system using fine-tuned ColBERT model."""
    
    def __init__(self, index_path: str = None, debug: bool = True):
        self.debug = debug
        self.model = None
        self.initialized = False
        
        if index_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            index_path = os.path.join(current_dir, ".ragatouille/colbert/indexes/acr_variants_index")
        
        self.index_path = index_path
        
        # Initialize model path - use the index_path directly since it points to the model directory
        self.model_path = index_path
        if self.debug:
            print(f"Model path: {self.model_path}")
            print(f"Index path: {self.index_path}")
    
    def _ensure_model_loaded(self):
        if self.model is None:
            if self.debug:
                print("Loading ColBERT model...")
            
            if self.debug:
                print(f"Using index path: {self.index_path}")
            
            # Load the model using the correct method
            self.model = RAGPretrainedModel.from_index(
                self.index_path,
                n_gpu=-1,
                verbose=1 if self.debug else 0
            )
            self.initialized = True
            
            if self.debug:
                print("ColBERT model loaded successfully")
    
    def search_acr_variants(self, query: str, k: int = 10) -> List[Dict[str, any]]:
        self._ensure_model_loaded()

        results = self.model.search(query, k=k)
        
        formatted_results = []
        for i, result in enumerate(results):
            formatted_result = {
                'rank': i + 1,
                'score': result['score'],
                'content': result['content'],
                'variant_id': f"colbert_variant_{i+1}",
                'relevance_score': result['score']
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def recommend_procedures(self, query_text: str, top_conditions: int = 1) -> Dict:
        colbert_results = self.search_acr_variants(query_text, k=10)
        
        best_result = colbert_results[0]
        
        recommendations = {
            "query": query_text,
            "retrieval_method": "colbert",
            "best_variant": {
                "variant_id": best_result['variant_id'],
                "content": best_result['content'],
                "variant_similarity": best_result['score'],
                "relevance_score": best_result['score']
            },
            "top_condition": {
                "condition_id": "colbert_matched_condition",
                "condition_similarity": best_result['score'],
                "description": "Condition matched via ColBERT model"
            },
            "usually_appropriate_procedures": self._extract_procedures_from_results(colbert_results),
            "all_variants": colbert_results,
            "total_results": len(colbert_results)
        }
        
        return recommendations
    
    def _extract_procedures_from_results(self, results: List[Dict]) -> List[Dict]:
        procedures = []
        
        for result in results:
            procedure = {
                'procedure_id': result['variant_id'],
                'title': result['content'],
                'appropriateness': 'USUALLY_APPROPRIATE',
                'relevance_score': result['score'],
                'rank': result['rank'],
                'dosage': None,
                'source': 'colbert'
            }
            procedures.append(procedure)
        
        return procedures
    
    def close(self):
        self.model = None
        self.initialized = False