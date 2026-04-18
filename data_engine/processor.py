import os
import uuid
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def process_directory(self, directory_path: str) -> List[Dict]:
        """
        Processes all txt files in a directory into chunks with metadata.
        """
        all_chunks = []
        if not os.path.exists(directory_path):
            print(f"Error: Directory {directory_path} not found")
            return []

        files = [f for f in os.listdir(directory_path) if f.endswith(".txt")]
        print(f"Processing {len(files)} files found in {directory_path}")

        for filename in files:
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                
                # Split text into chunks
                chunks = self.splitter.split_text(text)
                
                # Create metadata-enriched chunks
                for i, chunk_text in enumerate(chunks):
                    chunk_id = str(uuid.uuid4())
                    all_chunks.append({
                        "id": chunk_id,
                        "text": chunk_text,
                        "metadata": {
                            "source": filename,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "prev_id": all_chunks[-1]["id"] if i > 0 else "",
                            "next_id": "" 
                        }
                    })
                    
                    # Update previous chunk's next_id
                    if i > 0:
                        all_chunks[-2]["metadata"]["next_id"] = chunk_id
                        
            except Exception as e:
                print(f"Error processing {filename}: {e}")

        print(f"Total chunks created: {len(all_chunks)}")
        return all_chunks

if __name__ == "__main__":
    # Test run
    processor = DocumentProcessor()
    processed_data = processor.process_directory("data_engine/contracts")
    if processed_data:
        print(f"Sample chunk: {processed_data[0]['text'][:100]}...")
        print(f"Metadata: {processed_data[0]['metadata']}")
