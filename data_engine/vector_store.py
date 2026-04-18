import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict

class VectorStoreManager:
    def __init__(self, persist_directory: str = "data_engine/storage/chroma_db"):
        self.persist_directory = persist_directory
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize Persistent Client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Initialize Embedding Model (Running locally)
        print("Loading local embedding model: all-MiniLM-L6-v2")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="cuad_contracts",
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks_in_batches(self, chunks: List[Dict], batch_size: int = 10):
        """
        Adds chunks to ChromaDB in specified batch sizes.
        """
        total = len(chunks)
        print(f"Starting ingestion of {total} chunks in batches of {batch_size}")
        
        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            
            ids = [c["id"] for c in batch]
            texts = [c["text"] for c in batch]
            metadatas = [c["metadata"] for c in batch]
            
            # Generate embeddings locally
            embeddings = self.model.encode(texts).tolist()
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            print(f"Ingested chunks {i} to {min(i + batch_size, total)}")

    def query(self, query_text: str, top_k: int = 10) -> Dict:
        """
        Queries the collection for the top K most similar chunks.
        """
        query_embedding = self.model.encode([query_text]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        return results

if __name__ == "__main__":
    # Test run
    vstore = VectorStoreManager()
    # Dummy data for testing
    sample_chunks = [
        {"id": "test1", "text": "This is a test contract clause about liability.", "metadata": {"source": "test.txt"}},
        {"id": "test2", "text": "Confidentiality shall be maintained at all times.", "metadata": {"source": "test.txt"}}
    ]
    vstore.add_chunks_in_batches(sample_chunks)
    res = vstore.query("liability")
    print(f"Query results: {res['documents']}")
