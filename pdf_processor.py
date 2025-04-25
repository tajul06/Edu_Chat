import fitz  # PyMuPDF
import re
import os
from typing import Optional, List, Tuple

def extract_text_from_pdf(pdf_path: str, skip_first: int = 0, skip_last: int = 0) -> str:
    """
    Extract text from PDF while handling multi-column layouts.
    
    Args:
        pdf_path: Path to the PDF file
        skip_first: Number of first pages to skip (optional)
        skip_last: Number of last pages to skip (optional)
        
    Returns:
        Extracted text from the PDF
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    import time
    start_time = time.time()
    print(f"Starting PDF extraction: {os.path.basename(pdf_path)}")
        
    try:
        doc = fitz.open(pdf_path)
        extracted_text = []
        total_pages = len(doc)
        processed_pages = 0
        
        print(f"PDF has {total_pages} pages (skipping first {skip_first} and last {skip_last})")

        for page_num in range(total_pages):
            # Skip pages if configured
            if skip_first > 0 and page_num < skip_first:
                continue
            if skip_last > 0 and page_num >= total_pages - skip_last:
                continue
            
            processed_pages += 1
            
            # Print progress every few pages
            if page_num % 5 == 0 or page_num == total_pages - 1:
                print(f"Processing page {page_num+1}/{total_pages}")
                
            page = doc[page_num]
            blocks = page.get_text("blocks")  # Extract text in block structure
            
            # Sort blocks by Y then X for better reading order (handles columns)
            blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
            
            page_text = "".join([b[4] for b in blocks])
            extracted_text.append(page_text)

        doc.close()
        result = "\n\n".join(extracted_text)
        print(f"Extracted {len(result)} characters from {processed_pages} pages in {time.time() - start_time:.2f} seconds")
        return result
        
    except Exception as e:
        print(f"Error extracting PDF: {str(e)}")
        raise Exception(f"Error extracting text from PDF: {str(e)}")


def clean_text(text: str) -> str:
    """
    Clean text by removing headers, footers, watermarks and other common artifacts.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    cleaned_lines = []
    for line in text.split("\n"):
        if not re.search(r'(Page \d+|Confidential|Company Name|Watermark)', line, re.IGNORECASE):
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def structure_text(text: str) -> str:
    """
    Add structure to text by identifying section headers.
    
    Args:
        text: Cleaned text to structure
        
    Returns:
        Structured text with marked section headers
    """
    structured_text = ""
    for line in text.split("\n"):
        # Detect potential section headers
        if re.match(r'^[A-Z][A-Za-z0-9 ]+:$', line) or re.match(r'^[0-9]+\.[0-9]* [A-Z].*$', line):
            structured_text += f"\n\n## {line} ##\n\n"
        else:
            structured_text += line + "\n"
    return structured_text


def preserve_special_elements(text: str) -> str:
    """
    Preserve special elements like equations, code blocks, and quotes.
    
    Args:
        text: Text to process
        
    Returns:
        Text with preserved special elements
    """
    formatted_text = ""
    for line in text.split("\n"):
        if re.match(r'\$.*\$', line):  # Equations (LaTeX style)
            formatted_text += f"\n(EQUATION) {line} \n"
        elif re.match(r'\s{4,}.*', line):  # Code blocks (indented)
            formatted_text += f"\n(CODE) {line} \n"
        elif re.match(r'".*"', line):  # Quotes
            formatted_text += f"\n(QUOTE) {line} \n"
        else:
            formatted_text += line + "\n"
    return formatted_text


def process_pdf(pdf_path: str, skip_first: int = 6, skip_last: int = 2) -> str:
    """
    Process a PDF file through the complete pipeline:
    1. Extract text
    2. Clean text
    3. Structure text
    4. Preserve special elements
    
    Args:
        pdf_path: Path to the PDF file
        skip_first: Number of first pages to skip (default: 6)
        skip_last: Number of last pages to skip (default: 2)
        
    Returns:
        Processed text ready for semantic chunking
    """
    import time
    start_time = time.time()
    
    print(f"\n{'='*50}")
    print(f"Processing PDF: {os.path.basename(pdf_path)}")
    
    # Get PDF metadata first to show total pages
    metadata = get_pdf_metadata(pdf_path)
    total_pages = metadata["page_count"]
    print(f"PDF has {total_pages} pages. Processing pages {skip_first+1} to {total_pages-skip_last}")
    print(f"{'='*50}")
    
    print("Step 1/4: Extracting text...")
    raw_text = extract_text_from_pdf(pdf_path, skip_first=skip_first, skip_last=skip_last)
    
    print(f"Step 2/4: Cleaning text... ({len(raw_text)} characters)")
    cleaned_text = clean_text(raw_text)
    
    print(f"Step 3/4: Structuring text...")
    structured_text = structure_text(cleaned_text)
    
    print(f"Step 4/4: Preserving special elements...")
    final_text = preserve_special_elements(structured_text)
    
    elapsed = time.time() - start_time
    print(f"✓ PDF processing complete! ({elapsed:.2f} seconds)")
    print(f"   Title: {metadata['title']}")
    print(f"   Pages processed: {total_pages - skip_first - skip_last} of {total_pages}")
    print(f"   Size: {metadata['file_size_kb']} KB")
    print(f"{'='*50}\n")
    
    return final_text


def get_pdf_metadata(pdf_path: str) -> dict:
    """
    Extract metadata from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing metadata
    """
    try:
        doc = fitz.open(pdf_path)
        metadata = {
            "title": doc.metadata.get("title", "Unknown"),
            "author": doc.metadata.get("author", "Unknown"),
            "subject": doc.metadata.get("subject", ""),
            "keywords": doc.metadata.get("keywords", ""),
            "page_count": len(doc),
            "file_size_kb": round(os.path.getsize(pdf_path) / 1024, 2)
        }
        doc.close()
        return metadata
    except Exception as e:
        return {"error": str(e)}
