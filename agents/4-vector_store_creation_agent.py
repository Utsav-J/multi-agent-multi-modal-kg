import sys
import os
import argparse
import logging
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.agents import create_agent
from langchain_core.tools import tool

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Load environment variables
load_dotenv()
EMBEDDING_MODEL = "gemini-embedding-001"
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Define paths
CHUNKS_DIR = project_root / "chunking_outputs"
VECTOR_STORE_DIR = project_root / "vector_store_outputs"


@tool
def scan_and_ingest_chunks(dummy_arg: str = "") -> str:
    """
    Scans the chunking_outputs directory for files ending in '_2k.jsonl',
    generates embeddings for the chunks, and ingests them into a FAISS vector store.

    Args:
        dummy_arg (str): Not used, just for tool signature.

    Returns:
        str: A message indicating the result of the vector store creation.
    """
    logger.info("Tool invoked: scan_and_ingest_chunks")

    try:
        # Ensure output directory exists
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

        # Check if chunks directory exists
        if not CHUNKS_DIR.exists():
            return f"Error: Input directory not found at {CHUNKS_DIR}"

        # Find all matching files
        chunk_files = list(CHUNKS_DIR.glob("*_2k.jsonl"))

        if not chunk_files:
            return (
                "No files ending with '_2k.jsonl' found in chunking_outputs directory."
            )

        logger.info(
            f"Found {len(chunk_files)} files to process: {[f.name for f in chunk_files]}"
        )

        documents = []
        total_chunks = 0

        # Process each file
        for file_path in chunk_files:
            logger.info(f"Processing file: {file_path.name}")
            try:
                file_chunk_count = 0
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            chunk_data = json.loads(line)
                            content = chunk_data.get("content", "")

                            # Skip empty chunks
                            if not content:
                                continue

                            # Extract metadata (assuming 'metadata' field exists, or construct it)
                            # We also keep 'id' and source filename
                            metadata = chunk_data.get("metadata", {})
                            if isinstance(metadata, dict):
                                metadata["chunk_id"] = chunk_data.get(
                                    "id", f"{file_path.stem}_{line_num}"
                                )
                                metadata["source_file"] = file_path.name
                            else:
                                metadata = {
                                    "chunk_id": chunk_data.get(
                                        "id", f"{file_path.stem}_{line_num}"
                                    ),
                                    "source_file": file_path.name,
                                }

                            doc = Document(page_content=content, metadata=metadata)
                            documents.append(doc)
                            file_chunk_count += 1

                        except json.JSONDecodeError:
                            logger.warning(
                                f"Failed to parse JSON at line {line_num} in {file_path.name}"
                            )

                logger.info(
                    f"Extracted {file_chunk_count} chunks from {file_path.name}"
                )
                total_chunks += file_chunk_count

            except Exception as e:
                logger.error(f"Error reading file {file_path.name}: {e}")

        if not documents:
            return "No valid chunks found to ingest."

        logger.info(f"Total chunks to ingest: {total_chunks}")
        logger.info(f"Initializing embeddings model: ({EMBEDDING_MODEL})")

        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

        logger.info("Creating FAISS vector store...")
        try:
            vector_store = FAISS.from_documents(documents, embeddings)

            # Save to disk
            index_path = VECTOR_STORE_DIR / "index"
            vector_store.save_local(str(index_path))

            msg = f"Successfully created vector store with {total_chunks} chunks. Saved to {index_path}"
            logger.info(msg)
            return msg

        except Exception as e:
            error_msg = f"Failed to create/save vector store: {e}"
            logger.error(error_msg)
            return error_msg

    except Exception as e:
        error_msg = f"Unexpected error in scan_and_ingest_chunks: {e}"
        logger.exception(error_msg)
        return error_msg


def main():
    parser = argparse.ArgumentParser(description="Vector Store Creation Agent")
    parser.add_argument(
        "--input",
        type=str,
        default="start chunking",
        help="Input trigger for the agent (default: 'start chunking')",
    )
    args = parser.parse_args()

    # Initialize LLM for the agent
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        convert_system_message_to_human=True,
    )

    tools = [scan_and_ingest_chunks]

    sys_prompt = (
        "You are an AI assistant responsible for initializing and populating a vector store. "
        "Your task is to use the 'scan_and_ingest_chunks' tool to process text chunks from the 'chunking_outputs' directory "
        "and create a FAISS vector index. "
        "Always report the outcome of the tool execution clearly."
    )

    agent = create_agent(model=llm, tools=tools, system_prompt=sys_prompt)

    logger.info(f"Starting Vector Store Creation Agent with input: '{args.input}'")

    try:
        result = agent.invoke({"messages": [{"role": "user", "content": args.input}]})
        logger.info(f"Agent Result: {result['messages'][-1].content}")

    except Exception as e:
        logger.exception("Agent execution failed.")


if __name__ == "__main__":
    main()
