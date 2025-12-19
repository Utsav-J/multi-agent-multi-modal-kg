"""
Utility functions for extracting image information from markdown files and
assigning stable image identifiers.

For each image in a markdown file, we can:
- Collect:
  - Image path (from markdown)
  - Position in document (page number)
  - Surrounding section header (if any)
- Optionally assign a **deterministic image ID** of the form:
  - `img_<paper_id>_<page>_<index>`
  - And embed a JSON snippet with metadata in the **generated markdown** instead
    of writing a separate manifest file, for example:
    - "img_attention_is_all_you_need_2_0": {
        "path": "...png",
        "page": 2,
        "section": "Model Architecture"
      }
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json


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


def _parse_image_filename_for_id(image_path: str) -> Optional[Tuple[str, int, int]]:
    """
    Parse an image filename into (paper_id, page, index).

    Expected pattern (filename portion only):
        <paper_id>.pdf-<page>-<index>.<ext>

    Example:
        attention_is_all_you_need.pdf-2-0.png
            -> ("attention_is_all_you_need", 2, 0)

    Returns:
        (paper_id, page, index) if parsing succeeds, otherwise None.
    """
    filename = Path(image_path).name

    match = re.match(
        r"(?P<paper_id>.+?)\.pdf-(?P<page>\d+)-(?P<index>\d+)\.(?P<ext>png|jpg|jpeg|gif|svg)$",
        filename,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    paper_id = match.group("paper_id")
    page = int(match.group("page"))
    index = int(match.group("index"))
    return paper_id, page, index


def assign_stable_image_ids(
    markdown_path: str,
    output_markdown_path: Optional[str] = None,
    paper_id: Optional[str] = None,
) -> Path:
    """
    Create a copy of the markdown file where image references are replaced by
    **stable image IDs**, and embed image metadata as JSON snippets directly
    in the new markdown file.

    The markdown image syntax:
        ![](E:/.../attention_is_all_you_need.pdf-2-0.png)

    Becomes:
        ![](img_attention_is_all_you_need_2_0)

    And, immediately after the image, we embed a JSON entry in the markdown:
        "img_attention_is_all_you_need_2_0": {
          "path": "E:/.../attention_is_all_you_need.pdf-2-0.png",
          "page": 2,
          "section": "Model Architecture"
        }

    Notes:
    - The **original markdown file is not modified**; a new copy is written.
    - Image IDs are deterministic given the image filename pattern:
        img_<paper_id>_<page>_<index>

    Args:
        markdown_path: Path to the input markdown file.
        output_markdown_path: Optional explicit output markdown path. If not
            provided, a sibling file with suffix "_with_image_ids.md" is used.
        paper_id: Optional paper_id override. Normally inferred from the
            image filename (<paper_id>.pdf-<page>-<index>.<ext>). If supplied,
            this value is used whenever parsing fails.

    Returns:
        output_markdown_path as a Path object.
    """
    markdown_path = Path(markdown_path)
    if not markdown_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {markdown_path}")

    if output_markdown_path is None:
        output_markdown_path = markdown_path.with_name(
            markdown_path.stem + "_with_image_ids.md"
        )
    else:
        output_markdown_path = Path(output_markdown_path)

    with open(markdown_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Pattern to match markdown image syntax: ![](path) or ![alt](path)
    image_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

    # Keep track of "current" section header as we scan top‑down, so we can
    # associate each image occurrence with the nearest preceding section.
    current_section: Optional[str] = None

    def _make_image_id(image_path: str, section: Optional[str]) -> str:
        parsed = _parse_image_filename_for_id(image_path)

        nonlocal paper_id
        # Prefer filename-based parsing; fall back to an explicit / inferred
        # paper_id when needed.
        if parsed:
            parsed_paper_id, page, idx = parsed
            if paper_id is None:
                paper_id_local = parsed_paper_id
            else:
                paper_id_local = paper_id
        else:
            # Fallback when filename does not match the expected pattern:
            # - page from helper (may be None)
            # - index is 0 for now (deterministic per path)
            page = extract_page_number_from_filename(image_path) or 0
            if paper_id is None:
                # Derive a reasonably stable paper_id from the markdown filename
                paper_id_local = markdown_path.stem
            else:
                paper_id_local = paper_id
            idx = 0

        image_id = f"img_{paper_id_local}_{page}_{idx}"

        return image_id, page, section

    new_lines: List[str] = []

    for line in lines:
        image_metadata_snippets: List[str] = []
        # Track section headers as we go
        if is_section_header(line):
            current_section = extract_section_header_text(line)

        def _replace(match: re.Match) -> str:
            alt_text = match.group(1)
            image_path = match.group(2)

            image_id, page, section = _make_image_id(image_path, current_section)

            # Collect JSON-style metadata snippet to append after this line.
            # Example:
            # "img_attention_functional_roles_5_0": {
            #   "path": "E:/.../attention_functional_roles.pdf-5-0.png",
            #   "page": 5,
            #   "section": "..."
            # }
            snippet_lines = [
                "```json",
                f'"{image_id}": {{',
                f'  "path": "{image_path}",',
                f'  "page": {page},',
                f'  "section": "{section}",',
                "}",
                "```",
            ]
            image_metadata_snippets.extend(snippet_lines)

            # We do NOT write the image markdown itself in the output file,
            # only the JSON metadata snippet that follows this line.
            # Returning an empty string here removes the image from the line.
            return ""

        new_line = image_pattern.sub(_replace, line)
        new_lines.append(new_line)

        # If we collected any snippets for this line, append them just below
        # the (possibly multiple) images we rewrote.
        if image_metadata_snippets:
            for snippet in image_metadata_snippets:
                new_lines.append(snippet + "\n")

    # Write the new markdown file (copy with IDs and embedded metadata)
    with open(output_markdown_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    return output_markdown_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Image utilities for markdown files.")
    parser.add_argument(
        "markdown_file",
        nargs="?",
        default="attention_functional_roles_raw.md",
        help="Path to the markdown file (default: attention_functional_roles_raw.md)",
    )
    parser.add_argument(
        "--assign-ids",
        action="store_true",
        help=(
            "Assign stable image IDs in a *copy* of the markdown file and "
            "embed image metadata as JSON snippets in the new markdown."
        ),
    )
    parser.add_argument(
        "--output-markdown",
        type=str,
        default=None,
        help=(
            "Optional path for the markdown copy with image IDs. "
            "Defaults to '<stem>_with_image_ids.md' next to input."
        ),
    )
    parser.add_argument(
        "--paper-id",
        type=str,
        default=None,
        help=(
            "Optional explicit paper_id to use in image IDs. "
            "By default inferred from image filenames."
        ),
    )

    args = parser.parse_args()
    markdown_path = Path(__file__).parent / args.markdown_file

    if args.assign_ids:
        out_md = assign_stable_image_ids(
            markdown_path,
            output_markdown_path=args.output_markdown,
            paper_id=args.paper_id,
        )
        print(f"Wrote markdown with image IDs to: {out_md}")
    else:
        # Original behaviour: just extract and print image info
        try:
            images = extract_images_from_markdown(markdown_path)
            print_image_info(images)

            # Also return as dict for programmatic use
            image_dicts = get_image_info_dict(images)
            print("\nAs dictionary:")
            print(json.dumps(image_dicts, indent=2))

        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise SystemExit(1)
