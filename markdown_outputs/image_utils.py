"""
Utility functions for extracting image information from markdown files.

For each image in a markdown file, collects:
- Image path (from markdown)
- Position in document (page number)
- Surrounding section header (if any)
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ImageInfo:
    """Information about an image in a markdown document."""

    image_path: str
    page_number: Optional[int]
    section_header: Optional[str]
    line_number: int


def extract_page_number_from_filename(image_path: str) -> Optional[int]:
    """
    Extract page number from image filename.

    Example: 'attention_is_all_you_need.pdf-2-0.png' -> 2
    """
    # Pattern: filename.pdf-PAGE-INDEX.png
    match = re.search(r"-(\d+)-\d+\.(png|jpg|jpeg|gif|svg)", image_path, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def find_page_number_nearby(
    lines: List[str], image_line_idx: int, lookback: int = 10
) -> Optional[int]:
    """
    Find a standalone page number near the image.

    Looks for lines that are just a number (possibly with whitespace).
    """
    start_idx = max(0, image_line_idx - lookback)

    for i in range(image_line_idx - 1, start_idx - 1, -1):
        line = lines[i].strip()
        # Check if line is just a number
        if line.isdigit():
            return int(line)

    return None


def is_section_header(line: str) -> bool:
    """
    Check if a line is a section header.

    Patterns:
    - ## **Title**
    - **N** **Title**
    - **N.M** **Title**
    - **Title** (for special sections like Abstract, References)
    """
    line = line.strip()

    # Pattern 1: ## **Title** or ## Title
    if re.match(r"^##+\s*\*?\*?[^*]", line):
        return True

    # Pattern 2: **N** **Title** or **N.M** **Title**
    if re.match(r"^\*\*\d+(\.\d+)?\*\*\s+\*\*[^*]", line):
        return True

    # Pattern 3: **Title** (standalone, like Abstract, References)
    # But exclude if it's part of a list or other formatting
    if re.match(r"^\*\*[A-Z][^*]*\*\*$", line) and len(line) < 100:
        # Common section names
        common_sections = [
            "Abstract",
            "Introduction",
            "Background",
            "Conclusion",
            "References",
            "Acknowledgements",
            "Acknowledgments",
        ]
        for section in common_sections:
            if section in line:
                return True

    return False


def extract_section_header_text(line: str) -> str:
    """
    Extract clean section header text from a markdown header line.

    Examples:
    - "## **Attention Is All You Need**" -> "Attention Is All You Need"
    - "**1** **Introduction**" -> "1 Introduction"
    - "**3.1** **Encoder and Decoder Stacks**" -> "3.1 Encoder and Decoder Stacks"
    """
    line = line.strip()

    # Remove markdown formatting
    # Remove ## prefix
    line = re.sub(r"^##+\s*", "", line)
    # Remove ** markers but keep the text
    line = re.sub(r"\*\*", "", line)
    # Clean up extra whitespace
    line = " ".join(line.split())

    return line


def find_nearest_section_header(lines: List[str], image_line_idx: int) -> Optional[str]:
    """
    Find the most recent section header before the image.
    """
    for i in range(image_line_idx - 1, -1, -1):
        if is_section_header(lines[i]):
            return extract_section_header_text(lines[i])

    return None


def extract_images_from_markdown(markdown_path: str) -> List[ImageInfo]:
    """
    Extract all image information from a markdown file.

    Args:
        markdown_path: Path to the markdown file

    Returns:
        List of ImageInfo objects containing image path, page number, and section header
    """
    markdown_path = Path(markdown_path)

    if not markdown_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {markdown_path}")

    with open(markdown_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Pattern to match markdown image syntax: ![](path) or ![alt](path)
    image_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

    images = []

    for line_idx, line in enumerate(lines):
        matches = image_pattern.finditer(line)
        for match in matches:
            image_path = match.group(2)

            # Try to find page number
            page_number = None

            # First, try to extract from filename
            page_number = extract_page_number_from_filename(image_path)

            # If not found, look for nearby standalone number
            if page_number is None:
                page_number = find_page_number_nearby(lines, line_idx)

            # Find nearest section header
            section_header = find_nearest_section_header(lines, line_idx)

            images.append(
                ImageInfo(
                    image_path=image_path,
                    page_number=page_number,
                    section_header=section_header,
                    line_number=line_idx + 1,  # 1-indexed
                )
            )

    return images


def print_image_info(images: List[ImageInfo]):
    """
    Print image information in a readable format.
    """
    print(f"Found {len(images)} image(s):\n")

    for i, img in enumerate(images, 1):
        print(f"Image {i}:")
        print(f"  Path: {img.image_path}")
        print(f"  Page: {img.page_number if img.page_number else 'Unknown'}")
        print(f"  Section: {img.section_header if img.section_header else 'None'}")
        print(f"  Line: {img.line_number}")
        print()


def get_image_info_dict(images: List[ImageInfo]) -> List[Dict]:
    """
    Convert ImageInfo objects to dictionaries for easier serialization.

    Returns:
        List of dictionaries with image information
    """
    return [
        {
            "image_path": img.image_path,
            "page_number": img.page_number,
            "section_header": img.section_header,
            "line_number": img.line_number,
        }
        for img in images
    ]


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        markdown_file = sys.argv[1]
    else:
        # Default to the example file
        markdown_file = "attention_functional_roles_raw.md"

    markdown_path = Path(__file__).parent / markdown_file

    try:
        images = extract_images_from_markdown(markdown_path)
        print_image_info(images)

        # Also return as dict for programmatic use
        image_dicts = get_image_info_dict(images)
        print("\nAs dictionary:")
        import json

        print(json.dumps(image_dicts, indent=2))

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
