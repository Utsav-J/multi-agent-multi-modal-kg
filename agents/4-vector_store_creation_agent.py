import sys
import argparse
import logging
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

load_dotenv()

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


class EmbeddingGemmaWrapper(Embeddings):
    """Wrapper for Google's EmbeddingGemma model via SentenceTransformers."""

    def __init__(self, model_name: str = "google/embeddinggemma-300m"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            prompt_name="document",
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(
            text,
            prompt_name="query",
            normalize_embeddings=True,
        )
        return embedding.tolist()

@tool
def scan_and_ingest_chunks(dummy_arg: str = "") -> str:
    """
    Scans the chunking_outputs directory for files ending in '_2k.jsonl',
    generates embeddings for the chunks using EmbeddingGemma, and ingests them into a FAISS vector store.

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
        logger.info("Initializing EmbeddingGemma model...")

        embeddings = EmbeddingGemmaWrapper(model_name="google/embeddinggemma-300m")

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
        "and create a FAISS vector index using EmbeddingGemma embeddings. "
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

'''
(mas-for-multimodal-knowledge-graph) 
UTSAV@Utsav MINGW64 /e/Python Stuff/MAS-for-multimodal-knowledge-graph (main)
$ uv run agents/4-vector_store_creation_agent.py 
2025-12-17 18:51:03,189 - __main__ - INFO - Starting Vector Store Creation Agent with input: 'start chunking'
2025-12-17 18:51:06,040 - __main__ - INFO - Tool invoked: scan_and_ingest_chunks
2025-12-17 18:51:06,041 - __main__ - INFO - Found 4 files to process: ['attention_functional_roles_raw_chunks_2k.jsonl', 'attention_is_all_you_need_raw_chunks_2k.jsonl', 'neuronal_attention_circuits_raw_chunks_2k.jsonl', 'sliding_window_attention_annotated_chunks_2k.jsonl']
2025-12-17 18:51:06,041 - __main__ - INFO - Processing file: attention_functional_roles_raw_chunks_2k.jsonl
2025-12-17 18:51:06,042 - __main__ - INFO - Extracted 13 chunks from attention_functional_roles_raw_chunks_2k.jsonl
2025-12-17 18:51:06,042 - __main__ - INFO - Processing file: attention_is_all_you_need_raw_chunks_2k.jsonl
2025-12-17 18:51:06,043 - __main__ - INFO - Extracted 7 chunks from attention_is_all_you_need_raw_chunks_2k.jsonl
2025-12-17 18:51:06,043 - __main__ - INFO - Processing file: neuronal_attention_circuits_raw_chunks_2k.jsonl
2025-12-17 18:51:06,044 - __main__ - INFO - Extracted 17 chunks from neuronal_attention_circuits_raw_chunks_2k.jsonl
2025-12-17 18:51:06,045 - __main__ - INFO - Processing file: sliding_window_attention_annotated_chunks_2k.jsonl
2025-12-17 18:51:06,046 - __main__ - INFO - Extracted 11 chunks from sliding_window_attention_annotated_chunks_2k.jsonl
2025-12-17 18:51:06,046 - __main__ - INFO - Total chunks to ingest: 48
2025-12-17 18:51:06,047 - __main__ - INFO - Initializing EmbeddingGemma model...
2025-12-17 18:51:06,049 - sentence_transformers.SentenceTransformer - INFO - Use pytorch device_name: cpu
2025-12-17 18:51:06,049 - sentence_transformers.SentenceTransformer - INFO - Load pretrained SentenceTransformer: google/embeddinggemma-300m
2025-12-17 18:51:12,955 - sentence_transformers.SentenceTransformer - INFO - 14 prompts are loaded, with the keys: ['query', 'document', 'BitextMining', 'Clustering', 'Classification', 'InstructionRetrieval', 'MultilabelClassification', 'PairClassification', 'Reranking', 'Retrieval', 'Retrieval-query', 'Retrieval-document', 'STS', 'Summarization']
2025-12-17 18:51:12,956 - __main__ - INFO - Creating FAISS vector store...
Batches: 100%|███████████████████████████████████████████████| 2/2 [03:09<00:00, 94.77s/it]
2025-12-17 18:54:22,989 - faiss.loader - INFO - Loading faiss with AVX2 support.
2025-12-17 18:54:26,138 - faiss.loader - INFO - Successfully loaded faiss with AVX2 support.
2025-12-17 18:54:27,119 - __main__ - INFO - Successfully created vector store with 48 chunks. Saved to E:\Python Stuff\MAS-for-multimodal-knowledge-graph\vector_store_outputs\index    
2025-12-17 18:54:31,016 - __main__ - INFO - Agent Result: The vector store has been successfully initialized and populated with 48 chunks. The FAISS index is saved at `E:\Python Stuff\MAS-for-multimodal-knowledge-graph\vector_store_outputs\index`.
(mas-for-multimodal-knowledge-graph) 
UTSAV@Utsav MINGW64 /e/Python Stuff/MAS-for-multimodal-knowledge-graph (main)
$ uv run agents/4-vector_store_creation_agent.py 
2025-12-17 19:02:00,009 - __main__ - INFO - Starting Vector Store Creation Agent with input: 'start chunking'
2025-12-17 19:02:02,183 - __main__ - INFO - Tool invoked: scan_and_ingest_chunks
2025-12-17 19:02:02,185 - __main__ - INFO - Found 4 files to process: ['attention_functional_roles_raw_chunks_2k.jsonl', 'attention_is_all_you_need_raw_chunks_2k.jsonl', 'neuronal_attention_circuits_raw_chunks_2k.jsonl', 'sliding_window_attention_annotated_chunks_2k.jsonl']  
2025-12-17 19:02:02,185 - __main__ - INFO - Processing file: attention_functional_roles_raw_chunks_2k.jsonl
2025-12-17 19:02:02,186 - __main__ - INFO - Extracted 13 chunks from attention_functional_roles_raw_chunks_2k.jsonl
2025-12-17 19:02:02,186 - __main__ - INFO - Processing file: attention_is_all_you_need_raw_chunks_2k.jsonl
2025-12-17 19:02:02,187 - __main__ - INFO - Extracted 7 chunks from attention_is_all_you_need_raw_chunks_2k.jsonl
2025-12-17 19:02:02,187 - __main__ - INFO - Processing file: neuronal_attention_circuits_raw_chunks_2k.jsonl
2025-12-17 19:02:02,188 - __main__ - INFO - Extracted 17 chunks from neuronal_attention_circuits_raw_chunks_2k.jsonl
2025-12-17 19:02:02,189 - __main__ - INFO - Processing file: sliding_window_attention_annotated_chunks_2k.jsonl
2025-12-17 19:02:02,190 - __main__ - INFO - Extracted 11 chunks from sliding_window_attention_annotated_chunks_2k.jsonl
2025-12-17 19:02:02,190 - __main__ - INFO - Total chunks to ingest: 48
2025-12-17 19:02:02,190 - __main__ - INFO - Initializing EmbeddingGemma model...
2025-12-17 19:02:02,193 - sentence_transformers.SentenceTransformer - INFO - Use pytorch device_name: cpu
2025-12-17 19:02:02,193 - sentence_transformers.SentenceTransformer - INFO - Load pretrained SentenceTransformer: google/embeddinggemma-300m
2025-12-17 19:02:11,161 - sentence_transformers.SentenceTransformer - INFO - 14 prompts are loaded, with the keys: ['query', 'document', 'BitextMining', 'Clustering', 'Classification', 'InstructionRetrieval', 'MultilabelClassification', 'PairClassification', 'Reranking', 'Retrieval', 'Retrieval-query', 'Retrieval-document', 'STS', 'Summarization']
2025-12-17 19:02:11,162 - __main__ - INFO - Creating FAISS vector store...
Batches: 100%|███████████████████████████████████████████████| 2/2 [02:53<00:00, 86.60s/it]
2025-12-17 19:05:04,651 - faiss.loader - INFO - Loading faiss with AVX2 support.
2025-12-17 19:05:06,738 - faiss.loader - INFO - Successfully loaded faiss with AVX2 support.
2025-12-17 19:05:07,471 - __main__ - INFO - Successfully created vector store with 48 chunks. Saved to E:\Python Stuff\MAS-for-multimodal-knowledge-graph\vector_store_outputs\index    
2025-12-17 19:05:11,180 - __main__ - INFO - Agent Result: The vector store has been successfully created with 48 chunks and saved to `E:\Python Stuff\MAS-for-multimodal-knowledge-graph\vector_store_outputs\index`.
(mas-for-multimodal-knowledge-graph) 
UTSAV@Utsav MINGW64 /e/Python Stuff/MAS-for-multimodal-knowledge-graph (main)
$
'''