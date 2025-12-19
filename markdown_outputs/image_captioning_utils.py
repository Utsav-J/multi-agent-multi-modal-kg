# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
import sys
import json
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


def generate(image_path: str):
    """
    Generate a structured caption for a single image using Gemini.
    """
    client = genai.Client(
        api_key=os.environ.get("GOOGLE_API_KEY"),
    )

    model = "gemini-2.5-flash"  # Recommended version for structured multimodal

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                types.Part.from_text(
                    text="Analyze this scientific image according to the system instructions."
                ),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            required=[
                "image_relevance",
                "image_type",
                "semantic_role",
                "caption",
                "depicted_concepts",
                "confidence",
            ],
            properties={
                "image_relevance": genai.types.Schema(
                    type=genai.types.Type.STRING,
                    enum=["high", "low"],
                ),
                "image_type": genai.types.Schema(
                    type=genai.types.Type.STRING,
                    enum=[
                        "architecture",
                        "diagram",
                        "plot",
                        "attention_map",
                        "table_image",
                        "equation_render",
                        "logo",
                        "other",
                    ],
                ),
                "semantic_role": genai.types.Schema(
                    type=genai.types.Type.STRING,
                    enum=[
                        "defines",
                        "explains",
                        "illustrates",
                        "supports_result",
                        "decorative",
                    ],
                ),
                "caption": genai.types.Schema(
                    type=genai.types.Type.STRING,
                ),
                "depicted_concepts": genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=genai.types.Schema(
                        type=genai.types.Type.STRING,
                    ),
                ),
                "confidence": genai.types.Schema(
                    type=genai.types.Type.STRING,
                    enum=["high", "medium", "low"],
                ),
            },
        ),
        system_instruction=[
            types.Part.from_text(
                text="""You are an expert scientific document analyst and multimodal knowledge extraction system.

Your task is to analyze an image extracted from a scientific research paper and produce a **precise, information-dense caption** suitable for **knowledge graph construction**.

## Core Objective

Generate a caption that:

* Faithfully describes what is visually present in the image
* Identifies scientifically relevant structures, components, or results
* Uses terminology consistent with research papers
* Can be directly used to extract entities and relationships

This is **not** a descriptive alt-text task.
This is **semantic scientific grounding**.

---

## Image Relevance Assessment (Mandatory First Step)

Before writing a caption, classify the image into one of the following categories:

1. **Scientifically relevant**

   * Architecture diagrams
   * Model pipelines
   * Experimental plots or charts
   * Attention visualizations
   * Algorithm flow diagrams
   * Equation renderings or tables as images

2. **Low scientific relevance**

   * Publisher logos
   * Conference branding
   * Decorative icons
   * Author photos
   * Page ornaments

---

## Captioning Rules by Image Type

### If the image is **low scientific relevance**:

* Produce a **minimal caption**
* Do **not** infer or hallucinate scientific meaning
* Do **not** link it to concepts or methods

**Allowed example:**

> “Publisher or conference logo with no direct scientific content.”

---

### If the image is **scientifically relevant**:

Produce a **structured, precise caption** that includes:

1. **What the image depicts**

   * Architecture, process, plot, visualization, etc.

2. **Key components or entities shown**

   * Models, layers, modules, variables, axes, or flows

3. **Scientific role**

   * What the image explains, defines, or supports

4. **Relationships**

   * How components interact or are connected

5. **Scope limitation**

   * Only describe what is visually present
   * Do not infer results or claims not shown in the image

---

## Output Constraints (Very Important)

* Do **not** mention:

  * “the paper”
  * “the authors”
  * “this figure shows” (avoid narrative phrasing)
* Do **not** reference surrounding text unless it is visible in the image
* Do **not** invent labels, values, or equations
* Do **not** explain *why* something works — only *what is shown*

---

## Output Format (Strict)

Return a **single JSON object** with the following structure and nothing else:

```json
{
  \"image_relevance\": \"high | low\",
  \"image_type\": \"architecture | diagram | plot | attention_map | table_image | equation_render | logo | other\",
  \"semantic_role\": \"defines | explains | illustrates | supports_result | decorative\",
  \"caption\": \"Concise, information-dense scientific caption.\",
  \"depicted_concepts\": [
    \"Concept 1\",
    \"Concept 2\"
  ],
  \"confidence\": \"high | medium | low\"
}
```

### Field Rules

* `image_relevance`

  * `\"low\"` only for logos or decorative images
* `depicted_concepts`

  * Empty list allowed only if relevance is `\"low\"`
* `caption`

  * 1–3 sentences maximum
  * No markdown
  * No bullet points

---

## Quality Bar

Assume that:

* The caption will be used to automatically extract entities and relations
* Errors will propagate into a knowledge graph
* Precision is more important than completeness

If uncertain, **describe less rather than more** and lower the confidence score.

---

## Final Reminder

You are performing **scientific visual grounding**, not summarization.

Only describe what the image explicitly contains and what can be reliably identified.

"""
            ),
        ],
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    return response.text


def process_images_by_prefix(prefix: str):
    """
    Scan markdown_outputs/images/ for images starting with prefix and process them.
    """
    images_dir = Path(__file__).parent / "images"
    if not images_dir.exists():
        print(f"Error: Directory {images_dir} does not exist.")
        return

    # Find all image files matching the prefix
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".svg"}
    matching_images = sorted(
        [
            f
            for f in images_dir.iterdir()
            if f.is_file()
            and f.name.startswith(prefix)
            and f.suffix.lower() in image_extensions
        ]
    )

    if not matching_images:
        print(f"No images found with prefix: {prefix}")
        return

    print(f"Found {len(matching_images)} images matching prefix '{prefix}'")

    for img_path in matching_images:
        print(f"\n--- Processing: {img_path.name} ---")
        try:
            caption_json_str = generate(str(img_path))
            # The API returns a string, which should be valid JSON
            caption_data = json.loads(caption_json_str)
            print(json.dumps(caption_data, indent=2))
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")


def process_single_image(image_path: str):
    """
    Process a single image given by an absolute (or relative) path.
    """
    img_path = Path(image_path)
    if not img_path.is_file():
        print(f"Error: image file not found: {image_path}")
        return

    print(f"\n--- Processing single image: {img_path} ---")
    try:
        caption_json_str = generate(str(img_path))
        caption_data = json.loads(caption_json_str)
        print(json.dumps(caption_data, indent=2))
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run image_captioning_utils.py <prefix>")
        print("   or: uv run image_captioning_utils.py --image <absolute_image_path>")
        print(
            "Example (prefix): uv run image_captioning_utils.py attention_is_all_you_need.pdf"
        )
        print(
            "Example (image):  uv run image_captioning_utils.py --image E:/path/to/image.png"
        )
        sys.exit(1)

    # Mode 1: explicit single image path
    if sys.argv[1] == "--image":
        if len(sys.argv) < 3:
            print("Error: --image requires an absolute image path argument.")
            sys.exit(1)
        image_path = sys.argv[2]
        process_single_image(image_path)
    else:
        # Mode 2: prefix-based bulk processing
        prefix = sys.argv[1]
        process_images_by_prefix(prefix)

"""
Example uses: (hey ai coder pls dont delete this section while making any changes pls)
$ uv run markdown_outputs/image_captioning_utils.py abc
Found 2 images matching prefix 'abc'

--- Processing: abc-1.png ---
{
  "image_relevance": "high",
  "image_type": "architecture",
  "semantic_role": "defines",
  "caption": "The image illustrates the architecture of a Transformer model, consisting of an Encoder and a Decoder. The Encoder takes Input Embeddings combined with Positional Encodings through N identical layers, each with a Multi-Head Attention and a Feed Forward sub-layer, both followed by Add & Norm. The Decoder processes Output Embeddings with Positional Encodings through N identical layers, featuring a Masked Multi-Head Attention, a Multi-Head Attention (receiving encoder output), and a Feed Forward sub-layer, all followed by Add & Norm, concluding with a Linear and Softmax layer for Output Probabilities.",
  "depicted_concepts": [
    "Transformer architecture",
    "Encoder",
    "Decoder",
    "Input Embedding",
    "Output Embedding",
    "Positional Encoding",
    "Multi-Head Attention",
    "Masked Multi-Head Attention",
    "Feed Forward Network",
    "Add & Norm",
    "Linear layer",
    "Softmax layer",
    "Output Probabilities"
  ],
  "confidence": "high"
}


uv run markdown_outputs/image_captioning_utils.py --image "E:/Python Stuff/MAS-for-multimodal-knowledge-graph/m
arkdown_outputs/images/attention_functional_roles.pdf-1-1.png"

--- Processing single image: E:\Python Stuff\MAS-for-multimodal-knowledge-graph\markdown_outputs\images\attention_functional_roles.pdf-1-1.png ---
{
  "image_relevance": "high",
  "image_type": "diagram",
  "semantic_role": "illustrates",
  "caption": "The image illustrates two round-bottom flasks, each containing a solution with dispersed purple particles. The right flask is labeled 'Volume: 20mL Solution B' and contains fewer particles than the partially visible flask on the left, which shows a higher concentration of similar purple particles.",
  "depicted_concepts": [
    "Round-bottom flask",
    "Solution",
    "Particles",
    "Volume",
    "Concentration"
  ],
  "confidence": "high"
}

"""
