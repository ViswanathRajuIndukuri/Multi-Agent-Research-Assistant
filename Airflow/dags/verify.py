import os
import json
from dotenv import load_dotenv
import pinecone
from typing import Dict, List, Optional

class ChunkVerifier:
    def __init__(self):
        """Initialize verifier with Pinecone connection."""
        load_dotenv()
        self.pc = pinecone.Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

    def print_chunk_info(self, chunk_id: str, metadata: Dict):
        """Print formatted chunk information."""
        print(f"\n{'='*80}")
        print(f"Chunk ID: {chunk_id}")
        print(f"{'='*80}")
        
        # Print content preview (first 200 characters)
        content_preview = metadata['content'][:200] + '...' if len(metadata['content']) > 200 else metadata['content']
        print("\nContent Preview:")
        print(f"{content_preview}")
        
        # Print element counts
        print(f"\nElement Counts:")
        print(f"Images: {metadata['num_images']}")
        print(f"Tables: {metadata['num_tables']}")
        
        # Print image details if present
        if 'image_paths' in metadata:
            print("\nImages:")
            for i, (path, img_id, caption) in enumerate(zip(
                metadata['image_paths'],
                metadata['image_ids'],
                metadata.get('image_captions', [''] * len(metadata['image_paths']))
            )):
                print(f"\nImage {i+1}:")
                print(f"  ID: {img_id}")
                print(f"  Path: {path}")
                if caption:
                    print(f"  Caption: {caption[:100]}..." if len(caption) > 100 else f"  Caption: {caption}")
        
        # Print table details if present
        if 'table_paths' in metadata:
            print("\nTables:")
            for i, (path, table_id, markdown) in enumerate(zip(
                metadata['table_paths'],
                metadata['table_ids'],
                metadata.get('table_markdown', [''] * len(metadata['table_paths']))
            )):
                print(f"\nTable {i+1}:")
                print(f"  ID: {table_id}")
                print(f"  Path: {path}")
                if markdown:
                    print(f"  Content Preview: {markdown[:100]}...")

    def verify_index(self, index_name: str):
        """Verify and display chunks with elements in a given index."""
        try:
            index = self.pc.Index(index_name)
            
            # Query all vectors
            results = index.query(
                vector=[0] * 1536,  # Dummy vector for metadata query
                top_k=10000,
                include_metadata=True
            )
            
            # Filter chunks with elements
            chunks_with_elements = []
            for match in results.matches:
                metadata = match.metadata
                if metadata['has_images'] == 'True' or metadata['has_tables'] == 'True':
                    chunks_with_elements.append({
                        'id': match.id,
                        'metadata': metadata
                    })
            
            # Print summary
            print(f"\nFound {len(chunks_with_elements)} chunks with elements")
            print(f"{'='*80}")
            
            # Print detailed information for each chunk
            for chunk in chunks_with_elements:
                self.print_chunk_info(chunk['id'], chunk['metadata'])
            
            # Save results to file
            output_file = f"chunks_with_elements_{index_name}.json"
            with open(output_file, 'w') as f:
                json.dump(chunks_with_elements, f, indent=2)
            print(f"\nDetailed results saved to {output_file}")
            
        except Exception as e:
            print(f"Error verifying index: {str(e)}")

def main():
    verifier = ChunkVerifier()
    
    # Get list of indexes
    indexes = verifier.pc.list_indexes()
    
    if not indexes:
        print("No indexes found!")
        return
        
    print("\nAvailable indexes:")
    for i, index in enumerate(indexes, 1):
        print(f"{i}. {index}")
        
    # Let user choose index
    while True:
        try:
            choice = int(input("\nEnter the number of the index to verify (0 to exit): "))
            if choice == 0:
                break
            if 1 <= choice <= len(indexes):
                index_name = indexes[choice - 1]
                print(f"\nVerifying index: {index_name}")
                verifier.verify_index(index_name)
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
            continue

if __name__ == "__main__":
    main()