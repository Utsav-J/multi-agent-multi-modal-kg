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


@tool
def extract_graph_from_chunks_tool(chunks_filename: str) -> str:
    """
    Extracts knowledge graph nodes and relationships from a JSONL file containing text chunks.

    Args:
        chunks_filename (str): The name of the JSONL file in 'chunking_outputs' (e.g., 'doc_chunks.jsonl').

    Returns:
        str: A message indicating success and the path to the output JSONL file with graph data.
    """
    logger.info(
        f"Tool invoked: extract_graph_from_chunks_tool for file '{chunks_filename}'"
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

        # Read chunks
        chunks = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))

        logger.info(f"Processing {len(chunks)} chunks from {chunks_filename}")

        # Process each chunk
        with open(output_path, "w", encoding="utf-8") as out_f:
            for i, chunk_data in enumerate(chunks):
                content = chunk_data.get("content", "")
                if not content:
                    continue

                logger.info(
                    f"Processing chunk {i+1}/{len(chunks)} (ID: {chunk_data.get('id')})"
                )

                try:
                    # Generate content using the logic from construction.py
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=render_graph_construction_instructions(chunk=content),
                        config=genai.types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=GraphDocument,
                        ),
                    )

                    if response.parsed:
                        graph_doc: GraphDocument = response.parsed

                        # Convert to dict for serialization
                        # We might want to preserve original chunk metadata or ID
                        graph_dict = graph_doc.model_dump()

                        # Add metadata from the original chunk to keep traceability
                        graph_dict["original_chunk_id"] = chunk_data.get("id")
                        graph_dict["original_chunk_index"] = chunk_data.get(
                            "chunk_index"
                        )
                        graph_dict["original_metadata"] = chunk_data.get("metadata")

                        out_f.write(json.dumps(graph_dict) + "\n")
                        out_f.flush()
                    else:
                        logger.warning(
                            f"No parsed response for chunk {chunk_data.get('id')}"
                        )

                except Exception as e:
                    logger.error(
                        f"Error processing chunk {chunk_data.get('id')}: {str(e)}"
                    )
                    # Continue to next chunk even if one fails
                    continue

        return f"Successfully processed chunks. Graph data saved to: {output_filename}"

    except Exception as e:
        error_msg = f"Error during graph extraction: {str(e)}"
        logger.exception(error_msg)
        return error_msg


def main():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0, convert_system_message_to_human=True
    )

    tools = [extract_graph_from_chunks_tool]

    sys_prompt = (
        "You are a helpful AI assistant specializing in knowledge graph construction. "
        "Your goal is to process text chunks and extract structured graph data (nodes and relationships). "
        "Use the 'extract_graph_from_chunks_tool' to process the JSONL file provided by the user. "
        "Always report the path of the generated output file."
    )

    agent = create_agent(model=llm, tools=tools, system_prompt=sys_prompt)

    if len(sys.argv) > 1:
        # If user provides a full path or just a name, handle it
        arg_path = Path(sys.argv[1])
        filename = arg_path.name
        user_input = f"Extract graph from {filename}"
    else:
        # Default for testing
        user_input = "Extract graph from rag_paper_annotated_chunks.jsonl"

    logger.info(f"Starting graph construction agent with input: {user_input}")

    try:
        result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        logger.info(f"Agent Result: {result['messages'][-1].content}")
    except Exception as e:
        logger.exception("Agent execution failed.")


if __name__ == "__main__":
    main()
