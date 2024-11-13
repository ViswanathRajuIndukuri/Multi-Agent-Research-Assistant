# Part 1: Core Setup and Document Processing

import os
from pathlib import Path
import logging
import time
import re
from typing import Dict, Any, List, Optional, Set, Tuple
import pinecone
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem

class CombinedPDFProcessor:
    def __init__(self):
        """Initialize the processor with all required components."""
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
        
        # Setup document converter with appropriate options
        self.setup_document_converter()

    def setup_document_converter(self):
        """Set up document converter with proper pipeline options."""
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = 2.0  # Higher resolution for better quality
        pipeline_options.generate_page_images = True
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

    def process_document(self, doc, pdf_name: str, output_dir: Path) -> Dict[str, Any]:
        """Process document and track image/table positions."""
        content = {
            'text': '',  # Will build this from individual elements
            'images': [],
            'tables': [],
            'metadata': {
                'filename': pdf_name,
                'doc_id': pdf_name.lower().replace(' ', '_')
            }
        }
        
        current_position = 0
        image_counter = 0
        table_counter = 0
        text_chunks = []
        
        # Process all elements sequentially
        for element, level in doc.iterate_items():
            try:
                element_position = current_position
                
                # Handle different types of elements
                if isinstance(element, TableItem):
                    table_counter += 1
                    image_filename = f"table-{table_counter:03d}-{pdf_name}.png"
                    image_path = output_dir / image_filename
                    
                    # Save table image
                    element.image.pil_image.save(image_path)
                    
                    # Get table text content
                    table_text = ""
                    if hasattr(element, 'text'):
                        table_text = element.text
                    elif hasattr(element, 'get_text'):
                        table_text = element.get_text()
                    
                    table_info = {
                        'table_id': table_counter,
                        'image_path': str(image_path),
                        'position': str(element_position),
                        'text': table_text,
                        'page_number': str(getattr(element, 'page_number', ''))
                    }
                    
                    # Export table data if available
                    if hasattr(element, 'export_to_dataframe'):
                        table_df = element.export_to_dataframe()
                        table_info['data'] = table_df.to_dict('records')
                    
                    content['tables'].append(table_info)
                    text_chunks.append(table_text)
                    current_position += len(table_text)
                    
                elif isinstance(element, PictureItem):
                    image_counter += 1
                    image_filename = f"picture-{image_counter:03d}-{pdf_name}.png"
                    image_path = output_dir / image_filename
                    
                    # Save image
                    element.image.pil_image.save(image_path)
                    
                    # Get image caption/text
                    image_text = ""
                    if hasattr(element, 'caption'):
                        image_text = element.caption
                    elif hasattr(element, 'text'):
                        image_text = element.text
                    
                    image_info = {
                        'image_id': image_counter,
                        'image_path': str(image_path),
                        'position': str(element_position),
                        'caption': image_text,
                        'page_number': str(getattr(element, 'page_number', ''))
                    }
                    content['images'].append(image_info)
                    text_chunks.append(image_text)
                    current_position += len(image_text)
                
                else:
                    # Handle text elements
                    text = ""
                    if hasattr(element, 'text'):
                        text = element.text
                    elif hasattr(element, 'get_text'):
                        text = element.get_text()
                        
                    if text:
                        text_chunks.append(text)
                        current_position += len(text)

            except Exception as e:
                self._log.error(f"Error processing element of type {type(element).__name__}: {str(e)}")
                continue
                    
        # Combine all text
        content['text'] = '\n'.join(text_chunks)
        return content

    def chunk_text(self, content: Dict, max_chunk_size: int = 4000) -> List[Dict]:
        """Split text into chunks while preserving word boundaries."""
        chunks = []
        words = content['text'].split()
        current_chunk = []
        current_size = 0
        chunk_start_pos = 0
        current_pos = 0
        
        for word in words:
            word_size = len(word.encode('utf-8'))
            word_length = len(word) + 1  # +1 for space
            
            if current_size + word_size > max_chunk_size and current_chunk:
                # Complete current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'start_pos': str(chunk_start_pos),
                    'end_pos': str(current_pos)
                })
                
                # Start new chunk
                current_chunk = [word]
                current_size = word_size
                chunk_start_pos = current_pos
                
            else:
                current_chunk.append(word)
                current_size += word_size
                
            current_pos += word_length
        
        # Add the last chunk
        if current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk),
                'start_pos': str(chunk_start_pos),
                'end_pos': str(current_pos)
            })
        
        return chunks

    def generate_embedding(self, text: str):
        """Generate embedding using OpenAI's API."""
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

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
    
    # Part 2: Chunking and Processing Logic

    # def chunk_text(self, content: Dict, max_chunk_size: int = 4000) -> List[Dict]:
    #     """Split document into chunks while preserving context."""
    #     chunks = []
    #     text = content['text']
    #     words = text.split()
    #     current_chunk = []
    #     current_size = 0
    #     chunk_start_pos = 0
    #     current_pos = 0
        
    #     for word in words:
    #         word_size = len(word.encode('utf-8'))
    #         word_length = len(word) + 1  # Add 1 for space
            
    #         if current_size + word_size > max_chunk_size and current_chunk:
    #             # Complete current chunk
    #             chunk_text = ' '.join(current_chunk)
    #             chunks.append({
    #                 'text': chunk_text,
    #                 'start': chunk_start_pos,
    #                 'end': current_pos
    #             })
                
    #             # Start new chunk
    #             current_chunk = [word]
    #             current_size = word_size
    #             chunk_start_pos = current_pos
    #         else:
    #             current_chunk.append(word)
    #             current_size += word_size
            
    #         current_pos += word_length
        
    #     # Add the last chunk
    #     if current_chunk:
    #         chunk_text = ' '.join(current_chunk)
    #         chunks.append({
    #             'text': chunk_text,
    #             'start': chunk_start_pos,
    #             'end': current_pos
    #         })
        
    #     return chunks

    def find_relevant_elements(self, chunk: Dict, content: Dict) -> Dict[str, List]:
        """Find images and tables relevant to a specific chunk."""
        relevant_elements = {
            'images': [],
            'tables': []
        }
        
        chunk_text = chunk['text'].lower()
        # Convert string positions back to integers for comparison
        chunk_start = int(chunk['start_pos'])
        chunk_end = int(chunk['end_pos'])
        context_window = 1000  # Context window in characters
        
        # Find image references
        image_refs = set()
        for pattern in [
            r'figure\s+(\d+)',
            r'fig\.\s*(\d+)',
            r'image\s+(\d+)',
            r'picture\s+(\d+)'
        ]:
            matches = re.finditer(pattern, chunk_text)
            for match in matches:
                try:
                    image_refs.add(int(match.group(1)))
                except ValueError:
                    continue

        # Find table references
        table_refs = set()
        for pattern in [
            r'table\s+(\d+)',
            r'tbl\.\s*(\d+)',
            r'tab\.\s*(\d+)'
        ]:
            matches = re.finditer(pattern, chunk_text)
            for match in matches:
                try:
                    table_refs.add(int(match.group(1)))
                except ValueError:
                    continue
        
        # Process images
        for img in content['images']:
            # Check for direct references
            if int(img['image_id']) in image_refs:
                relevant_elements['images'].append(img)
                continue
            
            # Check proximity - convert position to int for comparison
            img_pos = int(img['position'])
            if chunk_start - context_window <= img_pos <= chunk_end + context_window:
                relevant_elements['images'].append(img)
        
        # Process tables
        for table in content['tables']:
            # Check for direct references
            if int(table['table_id']) in table_refs:
                relevant_elements['tables'].append(table)
                continue
            
            # Check proximity - convert position to int for comparison
            table_pos = int(table['position'])
            if chunk_start - context_window <= table_pos <= chunk_end + context_window:
                relevant_elements['tables'].append(table)
        
        return relevant_elements

    def process_and_store(self, pdf_dir: str):
        """Main processing pipeline."""
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            try:
                # Initialize processing
                pdf_path = os.path.join(pdf_dir, pdf_file)
                self._log.info(f"\nProcessing {pdf_file}...")
                
                index_name = self.create_index_for_pdf(pdf_path)
                index = self.pc.Index(index_name)
                
                # Process document
                pdf_name = Path(pdf_path).stem
                pdf_image_dir = self.base_image_dir / pdf_name
                pdf_image_dir.mkdir(exist_ok=True)
                
                # Convert and process document
                doc = self.doc_converter.convert(pdf_path).document
                content = self.process_document(doc, pdf_name, pdf_image_dir)
                
                # Create chunks with proper handling of markdown and references
                chunks = self.chunk_text(content)
                self._log.info(f"Created {len(chunks)} chunks from document")
                
                # Process each chunk
                for chunk_id, chunk in enumerate(chunks):
                    # Find relevant elements for this specific chunk
                    relevant_elements = self.find_relevant_elements(chunk, content)
                    
                    # Create metadata
                    metadata = {
                        'type': 'text',
                        'text': chunk['text'],
                        'chunk_id': str(chunk_id),
                        'total_chunks': str(len(chunks)),
                        'doc_id': content['metadata']['doc_id'],
                        'filename': content['metadata']['filename'],
                        'start_pos': chunk['start_pos'],
                        'end_pos': chunk['end_pos']
                    }
                    
                    # Add relevant elements to metadata
                    if relevant_elements['images']:
                        metadata.update({
                            'image_paths': [img['image_path'] for img in relevant_elements['images']],
                            'image_ids': [str(img['image_id']) for img in relevant_elements['images']],
                            'image_captions': [img.get('caption', '') for img in relevant_elements['images']],
                            'image_pages': [str(img.get('page_number', '')) for img in relevant_elements['images']]
                        })
                    
                    if relevant_elements['tables']:
                        metadata.update({
                            'table_paths': [tbl['image_path'] for tbl in relevant_elements['tables']],
                            'table_ids': [str(tbl['table_id']) for tbl in relevant_elements['tables']],
                            'table_pages': [str(tbl.get('page_number', '')) for tbl in relevant_elements['tables']]
                        })
                        
                        # Add table data if available
                        table_texts = [tbl.get('text', '') for tbl in relevant_elements['tables']]
                        if any(table_texts):
                            metadata['table_texts'] = table_texts
                    
                    # Create embedding and store
                    try:
                        embedding = self.generate_embedding(chunk['text'])
                        vector = {
                            'id': f"{content['metadata']['doc_id']}_chunk_{chunk_id}",
                            'values': embedding,
                            'metadata': metadata
                        }
                        
                        index.upsert([vector])
                        self._log.info(
                            f"Indexed chunk {chunk_id} with "
                            f"{len(relevant_elements['images'])} images and "
                            f"{len(relevant_elements['tables'])} tables"
                        )
                        
                    except Exception as e:
                        self._log.error(f"Failed to process chunk {chunk_id}: {str(e)}")
                        continue
                
                self._log.info(f"Successfully processed {pdf_file}")
                
            except Exception as e:
                self._log.error(f"Failed to process {pdf_file}. Error: {e}")
                continue


def main():
    """Main execution function."""
    # Initialize processor
    processor = CombinedPDFProcessor()
    
    # Set the directory containing PDF files
    pdf_dir = "../downloaded_pdfs"
    
    # Process all PDFs in the directory
    processor.process_and_store(pdf_dir)


if __name__ == "__main__":
    main()
