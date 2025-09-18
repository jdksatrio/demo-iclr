"""
Enhanced Medical Supervisor - Simplified

Integrates GraphRAG knowledge, ACR procedure recommendations, 
and enriched clinical rationales from Perplexity API.
"""

import asyncio
import os
import warnings
import json
from pathlib import Path
from getpass import getpass
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Suppress verbose output
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.*")
warnings.filterwarnings("ignore", message=".*torch.load.*")
warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
warnings.filterwarnings("ignore", message=".*User provided device_type.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.GradScaler.*")
warnings.filterwarnings("ignore", message=".*GradScaler is enabled, but CUDA is not available.*")

# Suppress ColBERT verbose output
os.environ["COLBERT_LOAD_TORCH_EXTENSION_VERBOSE"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Suppress logging
import logging
logging.getLogger().setLevel(logging.ERROR)

# Load environment variables from .env file
load_dotenv()

def fix_colbert_paths():
    """Fix ColBERT metadata paths for current machine"""
    try:
        # Get current project directory
        current_dir = Path(__file__).parent.parent.absolute()
        
        # Define the checkpoint path relative to current directory
        checkpoint_path = current_dir / "retrieve_acr" / ".ragatouille" / "colbert" / "indexes" / "acr_variants_index" / "checkpoints" / "colbert"
        
        # Files to update
        metadata_file = current_dir / "retrieve_acr" / ".ragatouille" / "colbert" / "indexes" / "acr_variants_index" / "metadata.json"
        plan_file = current_dir / "retrieve_acr" / ".ragatouille" / "colbert" / "indexes" / "acr_variants_index" / "plan.json"
        
        if not checkpoint_path.exists():
            print(f"ColBERT model not found, skipping path fix")
            return
        
        # Update metadata.json
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            old_path = metadata['config']['checkpoint']
            new_path = str(checkpoint_path)
            
            if old_path != new_path:
                metadata['config']['checkpoint'] = new_path
                
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"Updated ColBERT checkpoint path")
        
        # Update plan.json
        if plan_file.exists():
            with open(plan_file, 'r') as f:
                plan = json.load(f)
            
            old_path = plan['config']['checkpoint']
            new_path = str(checkpoint_path).replace('\\', '\\/')
            
            if old_path != new_path:
                plan['config']['checkpoint'] = new_path
                
                with open(plan_file, 'w') as f:
                    json.dump(plan, f, indent=4)
                
                print(f"Updated ColBERT plan path")
        
    except Exception as e:
        print(f"Warning: Could not fix ColBERT paths: {e}")

# Fix ColBERT paths on startup
fix_colbert_paths()

# Import enhanced supervisor workflow
try:
    from .enhanced_medical_workflow import run_enhanced_medical_workflow
    print("Enhanced supervisor loaded")
except ImportError:
    try:
        from enhanced_medical_workflow import run_enhanced_medical_workflow
        print("Enhanced supervisor loaded")
    except ImportError as e:
        print(f"Enhanced supervisor not available: {e}")
        exit(1)

def main():
    """Main interface for the enhanced medical supervisor"""
    
    print("Enhanced Medical Supervisor")
    print("==================================================")
    print("Integrates: GraphRAG + ACR + Enriched Clinical Rationales")
    
    # Get Neo4j credentials
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not neo4j_password or neo4j_password == "your_neo4j_password_here":
        print("Warning: NEO4J_PASSWORD not found in environment variables or still has placeholder value.")
        neo4j_password = getpass("Enter Neo4j password: ")
    else:
        print("✓ Neo4j password loaded from environment")
    
    # Check OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key or openai_api_key == "your_openai_api_key_here":
        print("Warning: OPENAI_API_KEY not found in environment variables or still has placeholder value.")
        openai_api_key = getpass("Enter OpenAI API key: ")
    else:
        print("✓ OpenAI API key loaded from environment")
    
    # Initialize OpenAI LLM
    llm = ChatOpenAI(
        model="gpt-4.1",
        temperature=0.0,
        api_key=openai_api_key
    )
    print("Using OpenAI GPT-4.1")
    
    print("System ready!")
    
    print("\n--------------------------------------------------")
    
    while True:
        try:
            query = input("\nEnter medical query (or 'quit'): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            print(f"\nProcessing: '{query}'")
            print("Integrating GraphRAG + ACR + Enriched Rationales...")
            
            # Run enhanced workflow
            result = asyncio.run(run_enhanced_medical_workflow(
                user_query=query,
                llm=llm,
                neo4j_password=neo4j_password
            ))
            
            if result["success"]:
                print("\nEnhanced Analysis Complete!")
                
                # Get results
                graphrag_context = result.get("graphrag_context", "")
                acr_recommendations = result.get("acr_recommendations", {})
                enriched_rationales = result.get("enriched_rationales", {})
                
                print(f"\nData Sources:")
                print(f"   GraphRAG: {'Available' if graphrag_context else 'Unavailable'} (length: {len(graphrag_context) if graphrag_context else 0})")
                print(f"   ACR: {'Available' if acr_recommendations and 'error' not in acr_recommendations else 'Unavailable'}")
                
                # Enhanced rationale status
                if enriched_rationales and "error" not in enriched_rationales:
                    summary = enriched_rationales.get("summary", {})
                    enriched_count = summary.get("enriched_procedures", 0)
                    total_count = summary.get("total_procedures", 0)
                    print(f"   Enriched: Available ({enriched_count}/{total_count} procedures)")
                else:
                    print(f"   Enriched: Unavailable")
                
                # Show GraphRAG context preview
                if graphrag_context and graphrag_context.strip():
                    print(f"\nGraphRAG Preview: {graphrag_context[:200]}...")
                else:
                    print(f"\nGraphRAG returned empty context")
                
                # Show main analysis
                final_answer = result.get("final_answer", "")
                if final_answer:
                    print(f"\nENHANCED ANALYSIS:")
                    print("==================================================")
                    print(final_answer)
                
            else:
                print(f"Analysis failed: {result.get('error', 'Unknown error')}")
            
        except KeyboardInterrupt:
            print(f"\nError: {str(e)}")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\nEnhanced medical analysis session ended!")


if __name__ == "__main__":
    main() 