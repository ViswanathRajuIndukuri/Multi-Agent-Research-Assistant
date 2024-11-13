import os
from pathlib import Path
from typing import Dict, Any, List
import pinecone
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from sentence_transformers import SentenceTransformer

class PDFProcessor:
    def __init__(self):
        """Initialize the processor with Pinecone and embedding model."""
        load_dotenv()  # Load environment variables
        
        # Initialize Pinecone
        api_key = os.getenv('PINECONE_API_KEY')
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        self.pc = pinecone.Pinecone(api_key=api_key)
        
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Setup document converter
        self.setup_document_converter()
        
        # Create base directory for storing images
        self.base_image_dir = Path("document_images")
        self.base_image_dir.mkdir(exist_ok=True)
        
        print("Initialized PDFProcessor:")
        print(f"- Image storage location: {self.base_image_dir}")
        print("- Using embedding model: all-MiniLM-L6-v2")
        print("- Connected to Pinecone")

    def generate_embedding(self, text: str):
        """Generate embedding for text using sentence-transformers."""
        return self.embedding_model.encode(text).tolist()

    def setup_document_converter(self):
        """Set up Docling document converter."""
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_page_images = True
        pipeline_options.generate_table_images = True
        pipeline_options.generate_picture_images = True

        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def process_single_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF document and save images locally."""
        print(f"Processing {pdf_path}...")
        
        # Create unique directory for this PDF's images
        pdf_name = Path(pdf_path).stem
        pdf_image_dir = self.base_image_dir / pdf_name
        pdf_image_dir.mkdir(exist_ok=True)
        
        # Convert document using Docling
        conv_result = self.doc_converter.convert(pdf_path)
        doc = conv_result.document
        
        # Extract content
        content = {
            'text': doc.export_to_markdown(),
            'tables': [],
            'images': [],
            'metadata': {
                'filename': pdf_name,
                'doc_id': pdf_name.lower().replace(' ', '_')
            }
        }
        
        # Process all content items
        image_counter = 0
        for element, level in doc.iterate_items():
            if hasattr(element, 'image') and element.image:
                page_num = element.metadata.get('page_num') if hasattr(element, 'metadata') else None
                if page_num is None:
                    continue
                
                # Determine image type and create filename
                if hasattr(element, 'table'):
                    image_type = 'table'
                elif hasattr(element, 'figure'):
                    image_type = 'figure'
                else:
                    image_type = 'image'
                
                # Create unique filename
                image_filename = f"{image_type}_{page_num}_{image_counter}.png"
                image_path = pdf_image_dir / image_filename
                
                # Save image
                element.image.pil_image.save(image_path)
                print(f"  Saved {image_type} {image_counter} from page {page_num}")
                
                # Store image information
                image_info = {
                    'page_no': page_num,
                    'image_path': str(image_path),
                    'type': image_type,
                    'image_id': image_counter
                }
                content['images'].append(image_info)
                image_counter += 1
        
        # Process tables (metadata)
        for idx, table in enumerate(doc.tables):
            page_num = table.metadata.get('page_num') if hasattr(table, 'metadata') else None
            if page_num is None:
                continue
                
            table_content = {
                'table_id': idx,
                'markdown': table.export_to_markdown(),
                'html': table.export_to_html(),
                'page_no': page_num
            }
            
            # If table has image and wasn't processed above
            if hasattr(table, 'image') and not any(img['type'] == 'table' and img['page_no'] == page_num 
                                                 for img in content['images']):
                image_filename = f"table_{page_num}_{idx}.png"
                image_path = pdf_image_dir / image_filename
                table.image.pil_image.save(image_path)
                print(f"  Saved table {idx} from page {page_num}")
                
                table_content['image_path'] = str(image_path)
            
            content['tables'].append(table_content)
        
        return content
    
    def create_index_for_pdf(self, pdf_path: str):
        """Create a unique index for each PDF."""
        # Clean and format the PDF name for index
        pdf_name = Path(pdf_path).stem.lower()
        cleaned_name = ''.join(c if c.isalnum() else '-' for c in pdf_name)
        cleaned_name = cleaned_name.strip('-')[:40]  # Limit length
        index_name = f"pdf-{cleaned_name}"
        
        # Create index if it doesn't exist
        if index_name not in self.pc.list_indexes():
            self.pc.create_index(
                name=index_name,
                dimension=384,  # matches all-MiniLM-L6-v2's output dimension
                metric="cosine",
                spec=pinecone.ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        return index_name

    def process_and_store(self, pdf_dir: str):
        """Process all PDFs in directory and store in Pinecone."""
        # Get list of PDF files
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
        
        def chunk_text(text, max_chunk_size=30000):
            words = text.split()
            chunks = []
            current_chunk = []
            current_size = 0
            
            for word in words:
                word_size = len(word.encode('utf-8'))
                if current_size + word_size > max_chunk_size:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_size = word_size
                else:
                    current_chunk.append(word)
                    current_size += word_size
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            return chunks

        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            print(f"\nProcessing {pdf_file}...")
            
            try:
                # Create unique index for this PDF
                index_name = self.create_index_for_pdf(pdf_path)
                print(f"Created/Retrieved index: {index_name}")
                
                # Process PDF content
                content = self.process_single_pdf(pdf_path)
                print(f"Found {len(content['images'])} images and {len(content['tables'])} tables")
                
                # Get index reference
                index = self.pc.Index(index_name)
                
                # Split text content into chunks
                text_chunks = chunk_text(content['text'])
                print(f"Split text into {len(text_chunks)} chunks")
                
                # Find images associated with pages
                def find_images_for_text(chunk_text, content):
                    """Find images that might be relevant to this chunk of text."""
                    relevant_images = []
                    for image in content['images']:
                        if f"Page {image['page_no']}" in chunk_text or f"page {image['page_no']}" in chunk_text:
                            relevant_images.append(image)
                    return relevant_images

                # Store text content chunks
                for chunk_id, chunk in enumerate(text_chunks):
                    chunk_images = find_images_for_text(chunk, content)
                    embedding = self.generate_embedding(chunk)
                    
                    text_vector = {
                        'id': f"{content['metadata']['doc_id']}_chunk_{chunk_id}",
                        'values': embedding,
                        'metadata': {
                            'type': 'text',
                            'content': chunk,
                            'chunk_id': chunk_id,
                            'total_chunks': len(text_chunks),
                            'doc_id': content['metadata']['doc_id'],
                            'filename': content['metadata']['filename'],
                            'images': [
                                {
                                    'path': img['image_path'],
                                    'type': img['type'],
                                    'page': img['page_no']
                                } for img in chunk_images
                            ],
                            'has_images': bool(chunk_images)
                        }
                    }
                    index.upsert([text_vector])
                    print(f"  Processed chunk {chunk_id + 1}/{len(text_chunks)}", end='\r')
                print()  # New line after progress
                
                # Store tables
                for table in content['tables']:
                    embedding = self.generate_embedding(table['markdown'])
                    
                    table_vector = {
                        'id': f"{content['metadata']['doc_id']}_table_{table['table_id']}",
                        'values': embedding,
                        'metadata': {
                            'type': 'table',
                            'markdown': table['markdown'],
                            'html': table['html'],
                            'doc_id': content['metadata']['doc_id'],
                            'filename': content['metadata']['filename'],
                            'page_no': table.get('page_no'),
                            'image_path': table.get('image_path')
                        }
                    }
                    index.upsert([table_vector])
                
                print(f"Successfully processed {pdf_file}")
                
            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")
                continue

    def get_metadata_structure(self):
        """Returns example metadata structure for documentation."""
        return {
            "Text Chunk Metadata": {
                "type": "text",
                "content": "Actual text content of the chunk",
                "chunk_id": 0,
                "total_chunks": "Total number of chunks in document",
                "doc_id": "unique_document_identifier",
                "filename": "original_pdf_name",
                "images": [
                    {
                        "path": "document_images/pdf_name/image_1_0.png",
                        "type": "image/figure/table",
                        "page": "page number"
                    }
                ],
                "has_images": True
            },
            "Table Chunk Metadata": {
                "type": "table",
                "markdown": "Table content in markdown",
                "html": "Table content in HTML",
                "chunk_id": 0,
                "total_chunks": "Total chunks if table was split",
                "doc_id": "unique_document_identifier",
                "filename": "original_pdf_name",
                "page_no": "page number",
                "image_path": "document_images/pdf_name/table_1_0.png"
            }
        }

def main():
    try:
        # Initialize processor
        print("\n=== Initializing PDF Processor ===")
        processor = PDFProcessor()
        
        # Set the PDF directory
        pdf_dir = "downloaded_pdfs"  # Adjust path as needed
        if not os.path.exists(pdf_dir):
            raise ValueError(f"PDF directory not found: {pdf_dir}")
        
        # List available PDFs
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
        if not pdf_files:
            raise ValueError(f"No PDF files found in {pdf_dir}")
        
        print(f"\nFound {len(pdf_files)} PDF files:")
        for pdf in pdf_files:
            print(f"  - {pdf}")
        
        # Process each PDF
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n=== Processing PDF {i}/{len(pdf_files)}: {pdf_file} ===")
            try:
                pdf_path = os.path.join(pdf_dir, pdf_file)
                processor.process_and_store(pdf_dir)
                print(f"\n✓ Successfully processed {pdf_file}")
                print("  Images saved in:", processor.base_image_dir / Path(pdf_file).stem)
            except Exception as e:
                print(f"\n✗ Error processing {pdf_file}:")
                print(f"  {str(e)}")
                continue
        
        print("\n=== Processing Complete ===")
        print(f"- Processed {len(pdf_files)} PDFs")
        print(f"- Images saved in: {processor.base_image_dir}")
        print("- Each chunk's metadata includes:")
        print("  * Document ID and filename")
        print("  * Chunk position and total chunks")
        print("  * Paths to relevant images")
        print("  * Table HTML/Markdown (for table chunks)")
        
    except Exception as e:
        print(f"\n✗ Fatal error:")
        print(f"  {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())