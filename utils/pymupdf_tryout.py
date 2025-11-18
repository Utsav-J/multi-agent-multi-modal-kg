import base64
import pathlib
import re
from typing import List, Tuple, Optional
from prompts import captioning_prompt
import pymupdf4llm
from google import genai
from pyprojroot import here
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
book_path = here("data/attention_is_all_you_need.pdf")
image_extraction_path = here("markdown_outputs/images")
IMAGE_PATTERN = re.compile(r"!\[(.*?)\]\((.*?)\)")


def encode_image_to_base64(image_path):
    """
    Encodes an image file into a Base64 string.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The Base64 encoded string of the image, or None if an error occurs.
    """
    try:
        with open(image_path, "rb") as image_file:
            # Read the binary data of the image
            image_data = image_file.read()
            # Encode the binary data to Base64
            encoded_image = base64.b64encode(image_data)
            # Decode the bytes-like object to a UTF-8 string
            return encoded_image.decode("utf-8")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def caption_image(
    image_location: str,
) -> str:
    """Generate a scientific caption for a single image.

    Args:
        image_location: Absolute or relative path to the image file.

    Returns:
        Caption text returned by the model.
    """
    image_string = encode_image_to_base64(image_location)
    if not image_string:
        raise ValueError(
            f"Could not encode image. Check if the path is correct: {image_location}"
        )

    ext = pathlib.Path(image_location).suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        mime_type = "image/jpeg"
    elif ext == ".png":
        mime_type = "image/png"
    elif ext == ".gif":
        mime_type = "image/gif"
    else:
        mime_type = "image/png"

    client = genai.Client()
    model = "gemma-3-4b-it"
    formatted_prompt = captioning_prompt.format(image_path=image_location)
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

    response_obj = client.models.generate_content(
        model=model,
        contents=contents,  # type: ignore
    )
    print(response_obj)
    response_text = getattr(response_obj, "text", None)
    if not response_text:
        raise ValueError("Caption generation returned empty response.")

    print(f"Caption for {image_location}: {response_text}")
    return response_text


def convert_pdf_to_markdown():
    md_text = pymupdf4llm.to_markdown(
        book_path,
        write_images=True,
        image_path=str(image_extraction_path),
    )
    pathlib.Path("pymu_output.md").write_bytes(md_text.encode())


def annotate_markdown_images(md_path: str, output_path: Optional[str] = None) -> str:
    """Replace image markdown expressions with AI-generated formatted captions.

    Process:
      1. Read markdown file.
      2. Find image references of form ![alt](path) anywhere in the text.
      3. For each image path, call caption_image to get formatted caption block.
      4. Replace the entire image expression with the caption block.

    Args:
        md_path: Path to source markdown file.
        output_path: Optional path to write annotated markdown. If None, file is not written.

    Returns:
        Updated markdown content as a string.
    """
    original_text = pathlib.Path(md_path).read_text(encoding="utf-8")

    def _replacer(match: re.Match) -> str:
        img_path = match.group(2)
        try:
            caption_block = caption_image(img_path)
            print(caption_block)
            return caption_block
        except Exception as e:
            return f"![Image caption generation failed]({img_path})\n\n**Error:** {e}"

    updated_text = IMAGE_PATTERN.sub(_replacer, original_text)

    if output_path:
        pathlib.Path(output_path).write_text(updated_text, encoding="utf-8")

    return updated_text


if __name__ == "__main__":
    # Example usage demo (will caption a single default image and then batch annotate markdown if present)
    # try:
    #     caption = caption_image()
    # except Exception as e:
    #     print(f"Single image caption failed: {e}")
    md_file = pathlib.Path("pymu_output.md")
    if md_file.exists():
        print("Annotating images in pymu_output.md ...")
        annotated = annotate_markdown_images(
            str(md_file),
            output_path="pymu_output_annotated.md",
        )
        print("Annotation complete. Written to pymu_output_annotated.md")
    else:
        print(
            "Markdown file pymu_output.md not found. Run convert_pdf_to_markdown() first."
        )
