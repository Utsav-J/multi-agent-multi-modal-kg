import os
import re
from typing import Optional
from markitdown import MarkItDown

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser

from pyprojroot import here

USE_MARKER=True

def convert_to_markdown(file_path:str="attention_is_all_you_need.pdf", output_folder="markdown_outputs"):
    """
    assumes your data is in the "data" folder (as it should be)
    converts it into markdown using markitdown
    """
    md = MarkItDown(enable_plugins=True, enable_builtins=True)
    input_file_path = os.path.join("data",file_path)
    result = md.convert(input_file_path)
    output_path = f"{output_folder}/{file_path.split('.')[0]}.md"
    with open(output_path, "w", encoding="utf-8") as markdown_file:
        markdown_file.write(result.text_content)
    return output_path

def clean_markdown(input_file_path: str, output_file_path: Optional[str] = None):
    """
    Cleans up a markdown file converted from PDF by:
    - Removing content before the table of contents (copyright, ISBN, etc.)
    - Removing "This page intentionally left blank" lines
    - Removing page numbers and footer artifacts
    - Removing publisher information at the end
    - Cleaning up excessive whitespace
    """
    if output_file_path is None:
        output_file_path = input_file_path.replace(".md", "_cleaned.md")
    
    with open(input_file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    lines = content.split('\n')
    cleaned_lines = []
    
    # Find where the actual content starts (after "Contents" or "Foreword")
    start_index = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == "contents" or line.strip().lower() == "foreword":
            start_index = i
            break
    
    # Process lines from the table of contents onward
    in_content = False
    for i in range(start_index, len(lines)):
        line = lines[i]
        
        # Skip "This page intentionally left blank"
        if "this page intentionally left blank" in line.lower():
            continue
        
        # Skip lines that are just page numbers (standalone numbers)
        if re.match(r'^\s*\d+\s*$', line):
            continue
        
        # Skip lines with Roman numerals (i, ii, iii, iv, v, vi, vii, viii, ix, x, etc.)
        if re.match(r'^\s*[ivxlcdm]+\s*$', line.lower()):
            continue
        
        # Skip publisher location lines (New York Chicago San Francisco...)
        if re.search(r'(New York|Chicago|San Francisco|Lisbon|London|Madrid|Mexico City)', line):
            continue
        
        # Skip lines that look like garbled text/artifacts (mostly special characters)
        if re.match(r'^[\W_]{10,}$', line):  # 10+ non-word characters
            continue
        
        # Skip very short lines with only special characters
        if len(line.strip()) > 0 and len(line.strip()) < 5 and not re.match(r'^[A-Za-z0-9\s]+$', line.strip()):
            continue
        
        cleaned_lines.append(line)
    
    # Join lines and clean up excessive blank lines
    cleaned_content = '\n'.join(cleaned_lines)
    
    # Replace 3+ consecutive newlines with just 2
    cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)
    
    # Remove trailing whitespace from each line
    cleaned_content = '\n'.join(line.rstrip() for line in cleaned_content.split('\n'))
    
    # Write cleaned content
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(cleaned_content.strip())
    
    print(f"Cleaned markdown saved to: {output_file_path}")
    return output_file_path

def use_markitdown():
    output_path = convert_to_markdown()
    clean_markdown(output_path)

def use_marker(converter:PdfConverter,file_path:str="attention_is_all_you_need.pdf"):
    final_path = here(f"data/{file_path}")
    rendered = converter(str(final_path))
    full_text, images, out_meta = text_from_rendered(rendered)
    with open("output.md", "w") as f:
        f.write(full_text)

def main():
    if USE_MARKER:
        os.environ['TORCH_DEVICE'] = 'cuda'  # Forces CUDA; fallback to 'cpu' if issues
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0 (your laptop's primary)
        artifact_dict = create_model_dict()
        config = {
            "output_dir": here("markdown_outputs"),
        }
        config_parser = ConfigParser(config)
        converter = PdfConverter(
            artifact_dict=artifact_dict,
            config=config_parser.generate_config_dict()
        )
        use_marker(converter)
    else:
        use_markitdown()
    
if __name__ == "__main__":
    main()