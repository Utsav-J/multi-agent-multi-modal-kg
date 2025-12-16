import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MARKDOWN_DIR = PROJECT_ROOT / "markdown_outputs"
CHUNKS_DIR = PROJECT_ROOT / "chunking_outputs"
GRAPH_DIR = PROJECT_ROOT / "knowledge_graph_outputs"


def run_command(command, description):
    logger.info(f"--- Starting: {description} ---")
    logger.info(f"Command: {command}")
    try:
        # Using shell=True for Windows compatibility with 'uv' if needed,
        # but list format is safer. 'uv' should be in path.
        subprocess.run(command, check=True, shell=True)
        logger.info(f"--- Completed: {description} ---\n")
    except subprocess.CalledProcessError as e:
        logger.error(f"--- Failed: {description} ---")
        logger.error(f"Error: {e}")
        raise e


def main():
    # 1. Scan for PDFs
    if not DATA_DIR.exists():
        logger.error(f"Data directory not found: {DATA_DIR}")
        return

    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in data directory.")
        return

    # Limit to 5 documents as per requirement
    if len(pdf_files) > 5:
        logger.warning(f"Found {len(pdf_files)} PDFs. Limiting to first 5.")
        pdf_files = pdf_files[:5]

    logger.info(f"Processing {len(pdf_files)} documents: {[f.name for f in pdf_files]}")

    for pdf_file in pdf_files:
        base_name = pdf_file.stem

        # Step 1: PDF Processor (Agent 1)
        # Outputs: markdown_outputs/{base_name}_annotated.md (always created now)
        run_command(
            f'uv run agents/1-pdf_processor_agent.py "{pdf_file.name}"',
            f"Agent 1: Process PDF {pdf_file.name}",
        )

        annotated_md = f"{base_name}_annotated.md"

        # Step 2: Chunker (Agent 2)
        # Outputs: chunking_outputs/{base_name}_annotated_chunks_5k.jsonl AND _2k.jsonl
        run_command(
            f'uv run agents/2-chunker_agent.py "{annotated_md}"',
            f"Agent 2: Chunk Markdown {annotated_md}",
        )

        chunks_5k = f"{base_name}_annotated_chunks_5k.jsonl"

        # Step 3: Graph Extractor (Agent 3)
        # Outputs: knowledge_graph_outputs/{base_name}_annotated_chunks_5k_graph.jsonl
        # Uses 5k chunks for graph construction
        run_command(
            f'uv run agents/3-graph_data_extractor_agent.py "{chunks_5k}"',
            f"Agent 3: Extract Graph from {chunks_5k}",
        )

    # Step 4: Graph Ingestion (Agent 5)
    # Ingests all generated graph files
    run_command(
        "uv run agents/5-jsonl_graph_ingestion_agent.py",
        "Agent 5: Ingest All Graph Data",
    )

    logger.info("Pipeline processing complete!")
    logger.info("You can now use Agent 6 to query the graph:")
    logger.info('uv run agents/6-graph_query_agent.py "Your question here"')


if __name__ == "__main__":
    main()
