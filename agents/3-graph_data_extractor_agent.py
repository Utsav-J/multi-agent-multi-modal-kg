import argparse
import sys
import os
import logging
import json
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Load environment variables
load_dotenv()

# Import from knowledge_graph module
from knowledge_graph.models import GraphDocument
from knowledge_graph.prompts import render_graph_construction_instructions

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import tool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Define paths
CHUNKS_DIR = project_root / "chunking_outputs"
OUTPUT_DIR = project_root / "knowledge_graph_outputs"
REGISTRY_PATH = OUTPUT_DIR / "global_entity_registry.json"


def load_global_registry() -> set:
    if REGISTRY_PATH.exists():
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            logger.warning("Failed to load registry, starting fresh.")
    return set()


def save_global_registry(entities: set):
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(sorted(list(entities)), f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save registry: {e}")


@tool
def extract_graph_from_chunks_tool(
    chunks_filename: str,
    include_metadata: bool = False,
    batch_size: int = 1,
    token_limit: int = 5500,
) -> str:
    """
    Extracts knowledge graph nodes and relationships from a JSONL file containing text chunks.
    Uses a global entity registry to maintain consistency across documents.

    Args:
        chunks_filename (str): The name of the JSONL file in 'chunking_outputs' (e.g., 'doc_chunks_5k.jsonl').
        include_metadata (bool): Whether to include original chunk metadata. Defaults to True.
        batch_size (int): Number of chunks to process in a single LLM call. Defaults to 1.
        token_limit (int): Approximate maximum number of tokens per batch.

    Returns:
        str: A message indicating success and the path to the output JSONL file.
    """
    logger.info(
        f"Tool invoked: extract_graph_from_chunks_tool for file '{chunks_filename}' "
        f"with include_metadata={include_metadata}, batch_size={batch_size}, token_limit={token_limit}"
    )

    try:
        input_path = CHUNKS_DIR / chunks_filename
        if not input_path.exists():
            error_msg = f"Error: Input file not found at {input_path}"
            logger.error(error_msg)
            return error_msg

        # Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_filename = input_path.stem + "_graph.jsonl"
        output_path = OUTPUT_DIR / output_filename

        # Initialize Google GenAI client
        # Using the same model as in construction.py
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

        results = []
        seen_entity_ids = load_global_registry()
        logger.info(f"Loaded {len(seen_entity_ids)} existing entities from registry.")

        # Read chunks
        chunks = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))

        logger.info(f"Processing {len(chunks)} chunks from {chunks_filename}")

        # Process each chunk
        with open(output_path, "w", encoding="utf-8") as out_f:
            current_batch = []
            current_batch_tokens = 0
            processed_batches = 0

            def process_batch(batch_chunks):
                if not batch_chunks:
                    return

                # Combine content from batch
                combined_content = ""
                for chunk in batch_chunks:
                    chunk_content = chunk.get("content", "")
                    if chunk_content:
                        combined_content += chunk_content + "\n\n"

                if not combined_content.strip():
                    return

                # Get batch IDs for logging
                batch_ids = [c.get("id") for c in batch_chunks]
                logger.info(
                    f"Processing batch of {len(batch_chunks)} chunks (IDs: {batch_ids})"
                )

                try:
                    # Generate content using the logic from construction.py
                    # Pass existing entities to context
                    existing_entities_list = sorted(list(seen_entity_ids))

                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=render_graph_construction_instructions(
                            chunk=combined_content,
                            existing_entities=existing_entities_list,
                        ),
                        config=genai.types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=GraphDocument,
                        ),
                    )

                    if response.parsed:
                        graph_doc: GraphDocument = response.parsed

                        # Track extracted entities for context in future chunks
                        for node in graph_doc.nodes:
                            seen_entity_ids.add(node.id)

                        # Save registry immediately after processing the batch
                        save_global_registry(seen_entity_ids)

                        # Convert to dict for serialization
                        graph_dict = graph_doc.model_dump()

                        # Add metadata from the original chunk to keep traceability
                        if include_metadata:
                            # For batched processing, we store lists of original metadata
                            graph_dict["original_chunk_ids"] = [
                                c.get("id") for c in batch_chunks
                            ]
                            graph_dict["original_chunk_indices"] = [
                                c.get("chunk_index") for c in batch_chunks
                            ]
                            graph_dict["original_metadata"] = [
                                c.get("metadata") for c in batch_chunks
                            ]

                        out_f.write(json.dumps(graph_dict) + "\n")
                        out_f.flush()
                    else:
                        logger.warning(f"No parsed response for batch {batch_ids}")

                except Exception as e:
                    logger.error(f"Error processing batch {batch_ids}: {str(e)}")

            # Iterate through chunks and form batches
            for i, chunk_data in enumerate(chunks):
                content = chunk_data.get("content", "")
                if not content:
                    continue

                # Estimate tokens (approx 4 chars per token)
                chunk_tokens = len(content) // 4

                # Determine effective token limit
                # If token_limit is not provided (0) and batch_size is default (1),
                # default to 3000 tokens as per requirement.
                effective_token_limit = token_limit
                if effective_token_limit == 0 and batch_size == 1:
                    effective_token_limit = 3000

                # Determine if we need to flush the current batch
                should_flush = False

                if effective_token_limit > 0:
                    # Adaptive batching based on tokens
                    # Flush if adding this chunk would exceed limit (and batch is not empty)
                    if current_batch and (
                        current_batch_tokens + chunk_tokens > effective_token_limit
                    ):
                        should_flush = True
                else:
                    # Fixed batch size (only if batch_size > 1 explicitly provided)
                    if len(current_batch) >= batch_size:
                        should_flush = True

                if should_flush:
                    process_batch(current_batch)
                    current_batch = []
                    current_batch_tokens = 0
                    processed_batches += 1

                current_batch.append(chunk_data)
                current_batch_tokens += chunk_tokens

            # Process any remaining chunks
            if current_batch:
                process_batch(current_batch)

        # Save updated registry
        # save_global_registry(seen_entity_ids)
        logger.info(f"Updated registry with {len(seen_entity_ids)} entities.")

        return f"Successfully processed chunks. Graph data saved to: {output_filename}"

    except Exception as e:
        error_msg = f"Error during graph extraction: {str(e)}"
        logger.exception(error_msg)
        return error_msg


def main():
    parser = argparse.ArgumentParser(description="Graph Data Extractor Agent")
    parser.add_argument(
        "filename",
        nargs="?",
        default="rag_paper_annotated_chunks_5k.jsonl",
        help="The input JSONL filename (default: rag_paper_annotated_chunks_5k.jsonl)",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Do not include original chunk metadata in the output",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of chunks to process in a single LLM call (default: 1)",
    )
    parser.add_argument(
        "--token-limit",
        type=int,
        default=0,
        help="Adaptive batching token limit (approx). Defaults to 3000 if batch-size is 1.",
    )

    args = parser.parse_args()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0, convert_system_message_to_human=True
    )

    tools = [extract_graph_from_chunks_tool]

    sys_prompt = (
        "You are a helpful AI assistant specializing in knowledge graph construction. "
        "Your goal is to process text chunks and extract structured graph data (nodes and relationships). "
        "Use the 'extract_graph_from_chunks_tool' to process the JSONL file provided by the user. "
        "You can choose whether to include metadata based on user instructions. "
        "You can also specify the batch size or token limit for processing chunks. "
        "Always report the path of the generated output file."
    )

    agent = create_agent(model=llm, tools=tools, system_prompt=sys_prompt)

    # Construct user input based on arguments
    user_input = f"Extract graph from {args.filename}"
    default_user_input = f"Extract graph from neuronal_attention_circuits_raw_chunks_5k.jsonl without metadata using a token limit of 5500"
    if args.no_metadata:
        user_input += " without including metadata"
    else:
        user_input += " and include metadata"

    if args.token_limit > 0:
        user_input += f" using a token limit of {args.token_limit}"
    elif args.batch_size > 1:
        user_input += f" using a batch size of {args.batch_size}"

    logger.info(f"Starting graph construction agent with input: {default_user_input}")

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": default_user_input}]}
        )
        logger.info(f"Agent Result: {result['messages'][-1].content}")
    except Exception as e:
        logger.exception("Agent execution failed.")


if __name__ == "__main__":
    main()
