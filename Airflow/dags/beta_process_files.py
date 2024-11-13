#storing all images' paths in chunks
import os
from pathlib import Path
import logging
import time
import re
from typing import Dict, Any, List, Optional
import pinecone
from dotenv import load_dotenv
from openai import OpenAI
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem

class CombinedPDFProcessor:
    def __init__(self):
        """Initialize the processor with Pinecone and OpenAI."""
        load_dotenv()
        logging.basicConfig(level=logging.INFO)
        self._log = logging.getLogger(__name__)
        
        # Initialize Pinecone
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.pc = pinecone.Pinecone(api_key=pinecone_api_key)
        
        # Initialize OpenAI
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Setup image directories
        self.base_image_dir = Path("document_images")
        self.base_image_dir.mkdir(exist_ok=True)
        
        # Setup document converter
        self.setup_document_converter()

    def setup_document_converter(self):
        """Set up Docling document converter."""
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_page_images = False
        pipeline_options.generate_table_images = True
        pipeline_options.generate_picture_images = True

        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    extract_text=True,
                    extract_tables=True,
                    extract_images=True
                )
            }
        )

    def chunk_text(self, text: str, max_chunk_size: int = 4000) -> List[str]:
        """Split text into chunks of maximum size."""
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

    def extract_page_numbers_from_text(self, text: str) -> List[int]:
        """Extract page numbers from text using various patterns."""
        patterns = [
            r'page\s+(\d+)',  # "page 123"
            r'pg\.\s*(\d+)',  # "pg. 123"
            r'p\.\s*(\d+)',   # "p. 123"
            r'\[(\d+)\]'      # "[123]"
        ]
        
        page_numbers = []
        for pattern in patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                try:
                    page_numbers.append(int(match.group(1)))
                except (ValueError, IndexError):
                    continue
        
        return list(set(page_numbers))  # Remove duplicates

    def find_relevant_elements(self, chunk_text: str, content: Dict) -> Dict[str, List]:
        """Find images and tables that might be relevant to this chunk of text."""
        page_numbers = self.extract_page_numbers_from_text(chunk_text)
        
        relevant_elements = {
            'images': [],
            'tables': []
        }

        # Add all elements if no specific page numbers found
        if not page_numbers:
            relevant_elements['images'] = content['images']
            relevant_elements['tables'] = content['tables']
            return relevant_elements

        # Find relevant images
        for img in content['images']:
            # Add image if its counter appears in the text
            if f"picture-{img['image_id']:03d}" in chunk_text or str(img['image_id']) in chunk_text:
                relevant_elements['images'].append(img)

        # Find relevant tables
        for table in content['tables']:
            # Add table if its counter appears in the text
            if f"table-{table['table_id']:03d}" in chunk_text or str(table['table_id']) in chunk_text:
                relevant_elements['tables'].append(table)

        return relevant_elements

    def generate_embedding(self, text: str):
        """Generate embedding using OpenAI's API."""
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def process_document(self, doc, pdf_name: str, output_dir: Path) -> Dict[str, Any]:
        """Process document for images, tables, and text."""
        content = {
            'text': doc.export_to_markdown(),
            'tables': [],
            'images': [],
            'metadata': {
                'filename': pdf_name,
                'doc_id': pdf_name.lower().replace(' ', '_')
            }
        }
        
        image_counter = 0
        table_counter = 0
        
        # Process tables and images
        for element, level in doc.iterate_items():
            try:
                if hasattr(element, 'image') and element.image:
                    counter = None
                    if isinstance(element, TableItem):
                        table_counter += 1
                        counter = table_counter
                        image_type = 'table'
                        
                        # Create unique filename
                        image_filename = f"table-{counter:03d}-{pdf_name}.png"
                        image_path = output_dir / image_filename
                        element.image.pil_image.save(image_path)
                        
                        # Store table information
                        table_info = {
                            'table_id': counter,
                            'image_path': str(image_path),
                            'markdown': element.export_to_markdown(),
                            'html': element.export_to_html()
                        }
                        content['tables'].append(table_info)
                        self._log.info(f"Saved table {counter}")
                        
                    elif isinstance(element, PictureItem):
                        image_counter += 1
                        counter = image_counter
                        image_type = 'picture'
                        
                        # Create unique filename
                        image_filename = f"picture-{counter:03d}-{pdf_name}.png"
                        image_path = output_dir / image_filename
                        element.image.pil_image.save(image_path)
                        
                        # Store image information
                        image_info = {
                            'image_id': counter,
                            'image_path': str(image_path),
                            'type': 'picture',
                            'caption': getattr(element, 'caption', '')
                        }
                        content['images'].append(image_info)
                        self._log.info(f"Saved picture {counter}")

            except Exception as e:
                self._log.error(f"Error processing element: {e}")
                continue

        self._log.info(f"Successfully processed {len(content['tables'])} tables and {len(content['images'])} images")
        return content

    def process_and_store(self, pdf_dir: str):
        """Process all PDFs in directory and store in Pinecone."""
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            self._log.info(f"\nProcessing {pdf_file}...")
            
            try:
                # Create unique index for this PDF
                index_name = self.create_index_for_pdf(pdf_path)
                self._log.info(f"Created/Retrieved index: {index_name}")
                
                # Process PDF content
                pdf_name = Path(pdf_path).stem
                pdf_image_dir = self.base_image_dir / pdf_name
                pdf_image_dir.mkdir(exist_ok=True)
                
                # Convert document
                conv_result = self.doc_converter.convert(pdf_path)
                doc = conv_result.document
                
                # Process document content
                content = self.process_document(doc, pdf_name, pdf_image_dir)
                
                # Get index reference
                index = self.pc.Index(index_name)
                
                # Process text chunks
                text_chunks = self.chunk_text(content['text'])
                self._log.info(f"Number of text chunks: {len(text_chunks)}")
                
                # Process and store chunks
                for chunk_id, chunk in enumerate(text_chunks):
                    # Find relevant elements for this chunk
                    relevant = self.find_relevant_elements(chunk, content)
                    
                    # Create embedding
                    embedding = self.generate_embedding(chunk)
                    
                    # Create metadata with flattened structure
                    metadata = {
                        'type': 'text',
                        'text': chunk,
                        'chunk_id': str(chunk_id),
                        'total_chunks': str(len(text_chunks)),
                        'doc_id': content['metadata']['doc_id'],
                        'filename': content['metadata']['filename'],
                        'has_tables': str(bool(relevant['tables'])),
                        'has_images': str(bool(relevant['images'])),
                        'num_tables': str(len(relevant['tables'])),
                        'num_images': str(len(relevant['images'])),
                    }

                    # Add table paths and IDs as lists of strings
                    if relevant['tables']:
                        metadata.update({
                            'table_paths': [str(t['image_path']) for t in relevant['tables']],
                            'table_ids': [str(t['table_id']) for t in relevant['tables']],
                            'table_markdown': [t['markdown'][:1000] for t in relevant['tables']]
                        })

                    # Add image paths and IDs as lists of strings
                    if relevant['images']:
                        metadata.update({
                            'image_paths': [str(img['image_path']) for img in relevant['images']],
                            'image_ids': [str(img['image_id']) for img in relevant['images']],
                            'image_captions': [img.get('caption', '') for img in relevant['images']]
                        })

                    # Create vector for this chunk
                    vector = {
                        'id': f"{content['metadata']['doc_id']}_chunk_{chunk_id}",
                        'values': embedding,
                        'metadata': metadata
                    }

                    # Upsert to Pinecone
                    try:
                        index.upsert([vector])
                        self._log.info(f"Indexed chunk {chunk_id}")
                    except Exception as e:
                        self._log.error(f"Failed to index chunk {chunk_id}: {str(e)}")
                        self._log.error(f"Metadata keys: {metadata.keys()}")

            except Exception as e:
                self._log.error(f"Failed to process {pdf_file}. Error: {e}")
                continue

    def create_index_for_pdf(self, pdf_path: str):
        """Create or get Pinecone index for PDF."""
        pdf_name = Path(pdf_path).stem.lower()
        cleaned_name = ''.join(c if c.isalnum() else '-' for c in pdf_name)
        cleaned_name = cleaned_name[:40].strip('-')
        index_name = f"pdf-{cleaned_name}"
        
        if index_name not in self.pc.list_indexes():
            self.pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=pinecone.ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        return index_name

def main():
    processor = CombinedPDFProcessor()
    pdf_dir = "../downloaded_pdfs"
    processor.process_and_store(pdf_dir)

if __name__ == "__main__":
    main()