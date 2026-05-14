import sys
sys.path.insert(0, '.')

import json
from pathlib import Path
from src.config import DATA_DIR, DATABASE_URL
from sentence_transformers import SentenceTransformer
import psycopg2
import numpy as np

def embed_policies():
    """Generate and store policy embeddings"""
    
    # Load policies
    policy_file = DATA_DIR / "policies.json"
    with open(policy_file) as f:
        policies = json.load(f)
    
    print(f"Loading {len(policies)} policies...")
    
    # Load embedding model
    print("Loading embedding model (this takes ~30s on first run)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Connect to DB
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    # Embed and insert each policy
    for policy_id, policy_data in policies.items():
        text = f"{policy_data['title']}. {policy_data['description']}"
        
        # Generate embedding
        embedding = model.encode(text)
        embedding_list = embedding.tolist()
        
        # Insert into database
        try:
            cursor.execute("""
                INSERT INTO policies (policy_id, title, description, embedding, category, severity, examples)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (policy_id) DO UPDATE SET embedding = EXCLUDED.embedding;
            """, (
                policy_id,
                policy_data["title"],
                policy_data["description"],
                embedding_list,
                policy_data.get("category", "general"),
                policy_data.get("severity", 2),
                policy_data.get("examples", [])
            ))
            
            print(f"✓ Embedded {policy_id}: {policy_data['title']}")
        
        except Exception as e:
            print(f"✗ Failed to embed {policy_id}: {e}")
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print("\n✅ All policies embedded successfully")

if __name__ == "__main__":
    embed_policies()
