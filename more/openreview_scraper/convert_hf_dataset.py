import os
from datasets import Dataset
import json
import fitz  # PyMuPDF for PDF processing
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm

import os
from datasets import Dataset
import json
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import logging
import numpy as np
from PIL import Image
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetConverter:
    def __init__(self, pdf_dir: str, json_dir: str):
        """
        Initialize the converter with directories containing PDFs and JSON files.
        
        Args:
            pdf_dir (str): Directory containing PDF files
            json_dir (str): Directory containing JSON annotation files
        """
        self.pdf_dir = pdf_dir
        self.json_dir = json_dir
        
        # Verify that required directories exist
        if not os.path.exists(pdf_dir):
            raise ValueError(f"PDF directory does not exist: {pdf_dir}")
        if not os.path.exists(json_dir):
            raise ValueError(f"JSON directory does not exist: {json_dir}")
            
        # Import fitz here to handle import errors gracefully
        try:
            import fitz
            self.fitz = fitz
        except ImportError as e:
            raise ImportError(
                "Failed to import PyMuPDF (fitz). Please install it with: pip install pymupdf"
            ) from e

    def convert_pdf_to_images(self, pdf_path: str, dpi: int = 300) -> List[Image.Image]:
        """
        Convert PDF pages to PIL Images.
        
        Args:
            pdf_path (str): Path to the PDF file
            dpi (int): DPI for rendering (default: 300)
            
        Returns:
            List[Image.Image]: List of PIL Images, one per page
        """
        try:
            doc = self.fitz.open(pdf_path)
            images = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get the pixel matrix with the specified DPI
                pix = page.get_pixmap(matrix=self.fitz.Matrix(dpi/72, dpi/72))
                
                # Convert to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
                
            doc.close()
            return images
            
        except Exception as e:
            logger.error(f"Error converting PDF to images {pdf_path}: {str(e)}")
            return []

    def load_json_annotation(self, json_path: str) -> Dict:
        """
        Load annotations from a JSON file.
        
        Args:
            json_path (str): Path to the JSON file
            
        Returns:
            dict: Loaded JSON content
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON {json_path}: {str(e)}")
            return {}

    def get_matching_files(self) -> List[Tuple[str, str]]:
        """
        Find matching PDF and JSON files based on their names.
        
        Returns:
            List[Tuple[str, str]]: List of tuples containing matching (pdf_path, json_path)
        """
        pdf_files = {os.path.splitext(f)[0]: f for f in os.listdir(self.pdf_dir) 
                    if f.lower().endswith('.pdf')}
        json_files = {os.path.splitext(f)[0]: f for f in os.listdir(self.json_dir) 
                     if f.lower().endswith('.json')}
        
        matching_files = []
        for name in pdf_files.keys():
            if name in json_files:
                pdf_path = os.path.join(self.pdf_dir, pdf_files[name])
                json_path = os.path.join(self.json_dir, json_files[name])
                matching_files.append((pdf_path, json_path))
        
        logger.info(f"Found {len(matching_files)} matching PDF-JSON pairs")
        return matching_files

    def image_to_bytes(self, img: Image.Image, format: str = 'PNG') -> bytes:
        """
        Convert PIL Image to bytes.
        
        Args:
            img (Image.Image): PIL Image
            format (str): Image format (default: 'PNG')
            
        Returns:
            bytes: Image encoded as bytes
        """
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=format)
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr

    def convert_to_dataset(self, dpi: int = 300) -> Dataset:
        """
        Convert PDF and JSON files to a HuggingFace dataset.
        
        Args:
            dpi (int): DPI for rendering PDF pages (default: 300)
            
        Returns:
            Dataset: HuggingFace dataset containing the converted data
        """
        matching_files = self.get_matching_files()
        if not matching_files:
            raise ValueError("No matching PDF-JSON pairs found")
            
        data = {
            'id': [],
            'page_number': [],
            'image': [],
            'annotations': []
        }
        
        logger.info("Processing files...")
        for pdf_path, json_path in tqdm(matching_files):
            file_id = os.path.splitext(os.path.basename(pdf_path))[0]
            
            # Convert PDF to images
            images = self.convert_pdf_to_images(pdf_path, dpi=dpi)
            if not images:
                logger.warning(f"No images extracted from {pdf_path}")
                continue
            
            # Load annotations from JSON
            annotations = self.load_json_annotation(json_path)
            if not annotations:
                logger.warning(f"No annotations loaded from {json_path}")
            
            # Add each page as a separate example
            for page_num, img in enumerate(images):
                data['id'].append(file_id)
                data['page_number'].append(page_num)
                data['image'].append(self.image_to_bytes(img))
                # You might want to modify this part depending on how your annotations
                # are structured (per page or per document)
                data['annotations'].append(annotations)
        
        # Convert to DataFrame first, then to HuggingFace Dataset
        df = pd.DataFrame(data)
        dataset = Dataset.from_pandas(df)
        
        logger.info(f"Created dataset with {len(dataset)} examples")
        return dataset

    def save_dataset(self, dataset: Dataset, output_dir: str):
        """
        Save the dataset to disk.
        
        Args:
            dataset (Dataset): HuggingFace dataset to save
            output_dir (str): Directory to save the dataset
        """
        os.makedirs(output_dir, exist_ok=True)
        dataset.save_to_disk(output_dir)
        logger.info(f"Dataset saved to {output_dir}")

def main():
    # Example usage
    pdf_dir = "./output_pdfs"
    json_dir = "./output_reviews"
    output_dir = "./huggingface_dataset"
    
    try:
        # Create converter instance
        converter = DatasetConverter(pdf_dir, json_dir)
        
        # Convert files to dataset
        dataset = converter.convert_to_dataset(dpi=300)  # Adjust DPI as needed
        
        # Print dataset info
        print("\nDataset Info:")
        print(dataset)
        
        # Save dataset
        converter.save_dataset(dataset, output_dir)
        
    except Exception as e:
        logger.error(f"Error during conversion: {str(e)}")
        raise

if __name__ == "__main__":
    main()