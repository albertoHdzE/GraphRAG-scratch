from processor import DocumentProcessor
from vector_store import VectorStoreManager
import os

def run_ingestion():
    print("Initializing Level 0 Data Engine Ingestion")
    
    # Paths
    contracts_dir = "contracts"
    storage_dir = "storage/chroma_db"
    
    # Initialize components
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=100)
    vstore = VectorStoreManager(persist_directory=storage_dir)
    
    # 1. Process files into chunks
    chunks = processor.process_directory(contracts_dir)
    
    if not chunks:
        print("No chunks to ingest. Check data_engine/contracts/ directory.")
        return
    
    # 2. Ingest into ChromaDB
    vstore.add_chunks_in_batches(chunks, batch_size=10)
    
    print("Ingestion complete. Level 0 Foundation Ready.")

if __name__ == "__main__":
    # Ensure we run from the data_engine directory context for simplified paths
    run_ingestion()
