import os
import base64
import pathlib
import re
from typing import Optional
import pymupdf4llm
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pyprojroot import here
import sys
sys.path.append(str(here()))

try:
    from utils.prompts import captioning_prompt
except ImportError:
    # Fallback if utils is not found directly
    print("Warning: Could not import prompts from utils.")
    captioning_prompt = "" 

load_dotenv()

IMAGE_PATTERN = re.compile(r"!\[(.*?)\]\((.*?)\)")

def encode_image_to_base64(image_path):
    """
    Encodes an image file into a Base64 string.
    """
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            encoded_image = base64.b64encode(image_data)
            return encoded_image.decode("utf-8")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"An error occurred while encoding image: {e}")
        return None

def caption_image(image_location: str) -> str:
    """
    Generate a scientific caption for a single image using Google GenAI.
    """
    image_string = encode_image_to_base64(image_location)
    if not image_string:
        raise ValueError(f"Could not encode image: {image_location}")

    ext = pathlib.Path(image_location).suffix.lower()
    mime_type = "image/png"
    if ext in {".jpg", ".jpeg"}:
        mime_type = "image/jpeg"
    elif ext == ".gif":
        mime_type = "image/gif"

    client = genai.Client()
    model = "gemma-3-4b-it" # Using the model specified in the reference
    formatted_prompt = captioning_prompt.format(image_path=f"images/{image_location}")
    
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    mime_type=mime_type, data=base64.b64decode(image_string)
                ),
                types.Part.from_text(text=formatted_prompt),
            ],
        ),
    ]

    try:
        response_obj = client.models.generate_content(
            model=model,
            contents=contents,
        )
        response_text = getattr(response_obj, "text", None)
        if not response_text:
            print(f"Warning: Empty response for {image_location}")
            return f"![Image]({image_location})" # Fallback
            
        print(f"Generated caption for {image_location}")
        return response_text
    except Exception as e:
        print(f"Failed to generate caption for {image_location}: {e}")
        return f"![Image]({image_location})\n\n**Error generating caption:** {e}"

def convert_pdf_to_markdown(pdf_path: pathlib.Path, output_md_path: pathlib.Path, image_output_dir: pathlib.Path):
    """
    Converts a PDF to Markdown, extracting images to the specified directory.
    """
    print(f"Converting {pdf_path} to Markdown...")
    
    # Ensure image directory exists
    image_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        md_text = pymupdf4llm.to_markdown(
            str(pdf_path),
            write_images=True,
            image_path=str(image_output_dir),
        )
        
        # Write the initial markdown
        output_md_path.parent.mkdir(parents=True, exist_ok=True)
        output_md_path.write_text(md_text, encoding="utf-8")
        print(f"Initial markdown saved to {output_md_path}")
        return True
    except Exception as e:
        print(f"Error converting PDF: {e}")
        return False

def annotate_markdown_images(md_path: pathlib.Path, output_path: pathlib.Path) -> str:
    """
    Reads markdown, finds images, generates captions, and writes the annotated markdown.
    """
    print(f"Annotating images in {md_path}...")
    original_text = md_path.read_text(encoding="utf-8")

    def _replacer(match: re.Match) -> str:
        img_path = match.group(2)
        # The image path in markdown might be relative. 
        # pymupdf4llm usually outputs paths relative to the markdown file or absolute.
        # We need to resolve it to an absolute path or a path relative to CWD for reading.
        
        # If path starts with /, it might be absolute. If not, it's relative to md_path.
        # However, pymupdf4llm usually writes images to the folder we gave it.
        # Let's try to resolve the file path.
        
        potential_path = pathlib.Path(img_path)
        if not potential_path.is_absolute():
            # Try resolving relative to the markdown file's directory
            potential_path = md_path.parent / img_path
            
        if not potential_path.exists():
            print(f"Warning: Image file not found at {potential_path}, skipping captioning.")
            return match.group(0) # Return original text

        try:
            caption_block = caption_image(str(potential_path))
            return caption_block
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            return f"![Caption generation failed]({img_path})\n\n**Error:** {e}"

    updated_text = IMAGE_PATTERN.sub(_replacer, original_text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(updated_text, encoding="utf-8")
    print(f"Annotated markdown saved to {output_path}")
    return updated_text

def process_document(filename: str):
    """
    Main workflow:
    1. Takes filename (expected in data/ folder).
    2. Converts PDF to Markdown (images extracted to markdown_outputs/images).
    3. Annotates Markdown with AI captions.
    4. Saves final output.
    """
    # Setup paths
    project_root = here()
    data_dir = project_root / "data"
    output_dir = project_root / "markdown_outputs"
    images_dir = output_dir / "images"
    
    input_pdf = data_dir / filename
    if not input_pdf.exists():
        print(f"Error: Input file not found: {input_pdf}")
        return

    base_name = input_pdf.stem
    intermediate_md = output_dir / f"{base_name}_raw.md"
    final_md = output_dir / f"{base_name}_annotated.md"

    print(f"Processing {filename}...")
    print(f"Input: {input_pdf}")
    print(f"Output: {final_md}")

    # Step 1: Convert PDF to Markdown
    success = convert_pdf_to_markdown(input_pdf, intermediate_md, images_dir)
    if not success:
        print("Conversion failed. Aborting.")
        return

    # Step 2: Annotate images
    annotate_markdown_images(intermediate_md, final_md)
    
    print("Processing complete!")

if __name__ == "__main__":
    # Simple CLI to get filename
    import sys
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        # Default for testing
        filename = "attention_is_all_you_need.pdf"
        print(f"No filename provided. Using default: {filename}")
    
    process_document(filename)

