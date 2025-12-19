"""
Utilities for processing markdown files that contain embedded image metadata
JSON blocks and enriching them with image captions produced by
`image_captioning_utils.generate`.

Usage (from project root):

    uv run markdown_outputs/markdown_image_processing.py attention_functional_roles_raw_with_image_ids.md

This will:
  - Read the markdown file
  - Find fenced ```json blocks that define image metadata of the form:
        "img_attention_functional_roles_1_2": {
          "path": "E:/.../attention_functional_roles.pdf-1-2.png",
          "page": 1,
          "section": "..."
        }
  - For each such image:
      - Run the Gemini captioning pipeline
      - Merge the returned caption JSON into the existing metadata block
  - Write a new markdown file alongside the original with suffix
        `_with_captions.md`
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from image_captioning_utils import generate

# Set up logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = (
    LOG_DIR
    / f"markdown_image_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)

IMAGE_ID_PATTERN = re.compile(r'\s*"(?P<img_id>img_[^"]+)"\s*:\s*\{')


def _parse_metadata_snippet(lines: List[str]) -> Tuple[str, Dict]:
    """
    Parse a small metadata snippet of the form:

        "img_attention_functional_roles_1_2": {
          "path": "...",
          "page": 1,
          "section": "...",
        }

    Returns:
        (image_id, metadata_dict)
    """
    if not lines:
        logger.error("Empty metadata snippet provided")
        raise ValueError("Empty metadata snippet")

    # First line should contain the image id
    m = IMAGE_ID_PATTERN.match(lines[0].strip())
    if not m:
        logger.error(f"Could not parse image id from line: {lines[0]!r}")
        raise ValueError(f"Could not parse image id from line: {lines[0]!r}")
    img_id = m.group("img_id")
    logger.debug(f"Parsing metadata snippet for image ID: {img_id}")

    metadata: Dict[str, object] = {}

    # Remaining lines until the closing brace
    for line in lines[1:]:
        stripped = line.strip()
        if stripped.startswith("}"):
            break

        # Expect lines like:  "path": "....",
        # or                "page": 1,
        field_match = re.match(r'"(?P<key>[^"]+)":\s*(?P<value>.+?)(,)?$', stripped)
        if not field_match:
            continue

        key = field_match.group("key")
        raw_val = field_match.group("value").strip()

        # Heuristic parsing based on key
        if raw_val.endswith(","):
            raw_val = raw_val[:-1].rstrip()

        if key in {"path", "section"}:
            # String value with quotes
            try:
                metadata[key] = json.loads(raw_val)
            except json.JSONDecodeError:
                # Fallback: strip quotes manually
                metadata[key] = raw_val.strip('"')
        elif key == "page":
            try:
                metadata[key] = int(re.sub(r"[^\d]", "", raw_val))
            except ValueError:
                metadata[key] = raw_val
        else:
            # Fallback: try JSON, else raw string
            try:
                metadata[key] = json.loads(raw_val)
            except json.JSONDecodeError:
                metadata[key] = raw_val

    logger.debug(f"Parsed metadata for {img_id}: {list(metadata.keys())}")
    return img_id, metadata


def _format_combined_snippet(img_id: str, combined: Dict) -> List[str]:
    """
    Format the combined metadata + caption dict back into a snippet like:

        "img_x": {
          "path": "...",
          "page": 1,
          "section": "...",
          "image_relevance": "...",
          "image_type": "...",
          "semantic_role": "...",
          "caption": "...",
          "depicted_concepts": [...],
          "confidence": "..."
        }
    """
    # Order keys for readability
    ordered_keys = [
        "path",
        "page",
        "section",
        "image_relevance",
        "image_type",
        "semantic_role",
        "caption",
        "depicted_concepts",
        "confidence",
    ]

    # Reorder the combined dict according to ordered_keys
    ordered_dict = {k: combined[k] for k in ordered_keys if k in combined}
    # Add any remaining keys that weren't in ordered_keys
    for k, v in combined.items():
        if k not in ordered_dict:
            ordered_dict[k] = v

    # Format the inner object (without the image_id key) with proper indentation
    inner_json = json.dumps(ordered_dict, indent=2, ensure_ascii=False)

    # Split into lines
    inner_lines = inner_json.split("\n")

    # Build the final lines with the image_id key
    lines: List[str] = [f'"{img_id}": {{']
    # Skip the first line (opening brace) and last line (closing brace) of inner_json
    # Add 2-space indentation to all inner content lines
    for inner_line in inner_lines[1:-1]:
        lines.append("  " + inner_line)
    lines.append("}")

    # Ensure each line ends with a newline
    formatted_lines = [line + "\n" for line in lines]
    logger.debug(f"Formatted snippet for {img_id} into {len(formatted_lines)} lines")
    return formatted_lines


def caption_markdown_images(markdown_path: str) -> Path:
    """
    Process a markdown file, enriching each image metadata JSON snippet
    with captioning output.

    The original markdown is NOT modified; a new file with suffix
    `_with_captions.md` is created.
    """
    logger.info(f"Starting markdown image captioning process for: {markdown_path}")

    md_path = Path(markdown_path)
    if not md_path.is_absolute():
        md_path = Path(__file__).parent / md_path
    if not md_path.exists():
        logger.error(f"Markdown file not found: {md_path}")
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    logger.info(f"Reading markdown file: {md_path}")
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    logger.debug(f"Read {len(lines)} lines from markdown file")

    out_lines: List[str] = []
    in_json_block = False
    current_block_lines: List[str] = []
    json_block_count = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not in_json_block and stripped.startswith("```json"):
            # Start of a JSON fenced block
            json_block_count += 1
            logger.debug(f"Found JSON block #{json_block_count}")
            in_json_block = True
            current_block_lines = [line]
            i += 1
            # Accumulate until closing fence
            while i < len(lines):
                block_line = lines[i]
                current_block_lines.append(block_line)
                if (
                    block_line.strip().startswith("```")
                    and block_line.strip() != "```json"
                ):
                    break
                i += 1

            # Process this JSON block
            processed_block = _process_json_block(current_block_lines)
            out_lines.extend(processed_block)

            in_json_block = False
            i += 1
            continue

        # Outside JSON block: copy line as‑is
        out_lines.append(line)
        i += 1

    logger.info(f"Processed {json_block_count} JSON blocks")
    logger.debug(f"Generated {len(out_lines)} output lines")

    out_path = md_path.with_name(md_path.stem + "_with_captions.md")
    logger.info(f"Writing output to: {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

    logger.info(f"Successfully wrote caption-enriched markdown to: {out_path}")
    return out_path


def _process_json_block(block_lines: List[str]) -> List[str]:
    """
    Given an entire fenced ```json block as a list of lines (including fences),
    find any image metadata snippets inside and enrich them with caption data.

    If the block does not contain any `img_...` snippets, it is returned
    unchanged.
    """
    if len(block_lines) < 3:
        logger.debug("JSON block too short, skipping processing")
        return block_lines

    # First and last are fences
    header = block_lines[0]
    footer = block_lines[-1]
    inner = block_lines[1:-1]

    i = 0
    new_inner: List[str] = []
    images_found = 0
    images_processed = 0
    images_failed = 0

    while i < len(inner):
        line = inner[i]
        if IMAGE_ID_PATTERN.match(line.strip()):
            images_found += 1
            # Collect this snippet until the closing brace
            snippet_lines = [line]
            i += 1
            while i < len(inner):
                snippet_lines.append(inner[i])
                if inner[i].strip().startswith("}"):
                    i += 1
                    break
                i += 1

            try:
                img_id, metadata = _parse_metadata_snippet(snippet_lines)
                image_path = metadata.get("path")
                if not image_path:
                    # If there's no path, we can't caption; keep original snippet
                    logger.warning(
                        f"No path found for image {img_id}, skipping captioning"
                    )
                    new_inner.extend(snippet_lines)
                    continue

                logger.info(f"Processing image {img_id} from path: {image_path}")
                # Run captioning pipeline
                try:
                    caption_json_str = generate(image_path)
                    caption_data = json.loads(caption_json_str)
                    logger.debug(
                        f"Generated caption for {img_id}: relevance={caption_data.get('image_relevance')}, type={caption_data.get('image_type')}"
                    )

                    # Merge metadata and captioning output
                    combined = {**metadata, **caption_data}

                    # Format back to snippet lines
                    combined_lines = _format_combined_snippet(img_id, combined)
                    new_inner.extend(combined_lines)
                    images_processed += 1
                    logger.info(f"Successfully processed image {img_id}")
                except Exception as e:
                    logger.error(
                        f"Error generating caption for {img_id} ({image_path}): {e}",
                        exc_info=True,
                    )
                    images_failed += 1
                    # On captioning error, fall back to original snippet
                    new_inner.extend(snippet_lines)
            except Exception as e:
                # On any parsing error, fall back to original snippet
                logger.error(f"Error parsing metadata snippet: {e}", exc_info=True)
                images_failed += 1
                new_inner.extend(snippet_lines)
        else:
            new_inner.append(line)
            i += 1

    if images_found > 0:
        logger.info(
            f"JSON block summary: {images_found} images found, {images_processed} processed successfully, {images_failed} failed"
        )

    return [header] + new_inner + [footer]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: uv run markdown_outputs/markdown_image_processing.py <markdown_file>"
        )
        print(
            "Example: uv run markdown_outputs/markdown_image_processing.py "
            "attention_functional_roles_raw_with_image_ids.md"
        )
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Markdown Image Processing - Starting")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info("=" * 60)

    md_file = sys.argv[1]
    try:
        out_path = caption_markdown_images(md_file)
        logger.info("=" * 60)
        logger.info("Markdown Image Processing - Completed Successfully")
        logger.info(f"Output file: {out_path}")
        logger.info("=" * 60)
        print(f"Wrote caption-enriched markdown to: {out_path}")
        print(f"Log file saved to: {LOG_FILE}")
    except Exception as e:
        logger.error(f"Fatal error during processing: {e}", exc_info=True)
        logger.error("Markdown Image Processing - Failed")
        print(f"Error: {e}")
        print(f"See log file for details: {LOG_FILE}")
        sys.exit(1)
