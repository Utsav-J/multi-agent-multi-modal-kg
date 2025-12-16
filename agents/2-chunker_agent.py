import sys
import os
import logging
import json
from pathlib import Path
from dotenv import load_dotenv

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Load environment variables
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Define paths
MARKDOWN_DIR = project_root / "markdown_outputs"
OUTPUT_DIR = project_root / "chunking_outputs"


@tool
def chunk_markdown_tool(markdown_filename: str) -> str:
    """
    Splits a Markdown file into text chunks using token-based chunking.
    Generates two outputs:
      1. 5000-token chunks (for Graph Construction)
      2. 2000-token chunks (for RAG/Search)

    Args:
        markdown_filename (str): The name of the markdown file in 'markdown_outputs' to chunk.

    Returns:
        str: A message indicating success and the paths to the output JSONL files.
    """
    logger.info(f"Tool invoked: chunk_markdown_tool for file '{markdown_filename}'")
    try:
        input_path = MARKDOWN_DIR / markdown_filename
        if not input_path.exists():
            error_msg = f"Error: Input file not found at {input_path}"
            logger.error(error_msg)
            return error_msg

        # Read content
        content = input_path.read_text(encoding="utf-8")

        # Prepare output path
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        configs = [
            {"size": 5000, "overlap": 500, "suffix": "_chunks_5k"},
            {"size": 2000, "overlap": 200, "suffix": "_chunks_2k"},
        ]

        generated_files = []

        for config in configs:
            # Initialize splitter with tiktoken encoder
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=config["size"],
                chunk_overlap=config["overlap"],
            )

            # Split text
            docs = text_splitter.create_documents(
                [content], metadatas=[{"source": markdown_filename}]
            )

            output_filename = f"{input_path.stem}{config['suffix']}.jsonl"
            output_path = OUTPUT_DIR / output_filename

            # Write to JSONL
            logger.info(
                f"Writing {len(docs)} chunks (size={config['size']}) to {output_path}"
            )
            with open(output_path, "w", encoding="utf-8") as f:
                for i, doc in enumerate(docs):
                    record = {
                        "id": f"{input_path.stem}{config['suffix']}_{i}",
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "chunk_index": i,
                        "token_size_config": config["size"],
                    }
                    f.write(json.dumps(record) + "\n")

            generated_files.append(output_filename)

        return (
            f"Successfully created chunks. Output files: {', '.join(generated_files)}"
        )

    except Exception as e:
        error_msg = f"Error during chunking: {str(e)}"
        logger.exception(error_msg)
        return error_msg


def main():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0, convert_system_message_to_human=True
    )

    tools = [chunk_markdown_tool]

    sys_prompt = (
        "You are a helpful AI assistant specializing in text processing. "
        "Your goal is to take a markdown file and split it into chunks using the 'chunk_markdown_tool'. "
        "The tool automatically generates two sets of chunks: one with 5000 tokens (for graph construction) and one with 2000 tokens (for RAG). "
        "Always report the paths of the generated JSONL files."
    )

    agent = create_agent(model=llm, tools=tools, system_prompt=sys_prompt)

    if len(sys.argv) > 1:
        # If user provides a full path or just a name, handle it
        arg_path = Path(sys.argv[1])
        filename = arg_path.name
        user_input = f"Chunk the file {filename}"
    else:
        # Default for testing - scan directory if no file provided?
        # For now, let's just pick one if exists or warn.
        # But consistent with Agent 1, let's try to be smart or just keep default.
        # Let's keep it simple for now as Main Pipeline will drive this.
        user_input = "Chunk the file sliding_window_attention_annotated.md"

    logger.info(f"Starting chunker agent with input: {user_input}")

    try:
        result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        # Log the final response from the agent
        logger.info(f"Agent Result: {result['messages'][-1].content}")
    except Exception as e:
        logger.exception("Agent execution failed.")


if __name__ == "__main__":
    main()

# ~  sample chunk looks like this 👇
# {
#     "id": "attention_is_all_you_need_annotated_2",
#     "content": "bla bla bla",
#     "metadata": {"source": "attention_is_all_you_need_annotated.md"},
#     "chunk_index": 2,
# }
