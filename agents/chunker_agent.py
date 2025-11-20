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
def chunk_markdown_tool(
    markdown_filename: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> str:
    """
    Splits a Markdown file into smaller text chunks using Recursive Character Chunking and saves them as a JSONL file.

    Args:
        markdown_filename (str): The name of the markdown file in 'markdown_outputs' to chunk (e.g., 'doc_annotated.md').
        chunk_size (int): The maximum size of each chunk in characters. Default is 1000.
        chunk_overlap (int): The number of characters to overlap between chunks. Default is 200.

    Returns:
        str: A message indicating success and the path to the output JSONL file.
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

        # Initialize splitter
        # Using RecursiveCharacterTextSplitter as it's robust for general text and code
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        # Split text
        # create_documents allows us to easily attach metadata if we had it per chunk,
        # but here we just have one big text. split_text returns list of strings.
        # split_documents expects list of Documents.
        # Let's use create_documents to get Document objects which we can then serialize.
        docs = text_splitter.create_documents(
            [content], metadatas=[{"source": markdown_filename}]
        )

        # Prepare output path
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_filename = f"{input_path.stem}_chunks.jsonl"
        output_path = OUTPUT_DIR / output_filename

        # Write to JSONL
        logger.info(f"Writing {len(docs)} chunks to {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            for i, doc in enumerate(docs):
                # Create a structured record
                record = {
                    "id": f"{input_path.stem}_{i}",
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "chunk_index": i,
                }
                f.write(json.dumps(record) + "\n")

        return f"Successfully created {len(docs)} chunks. Output saved to: {output_filename}"

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
        "Your goal is to take a markdown file and split it into smaller, manageable chunks for downstream usage (like RAG). "
        "Use the available chunking tool to process the file provided by the user. "
        "Always report the path of the generated JSONL file."
    )

    agent = create_agent(model=llm, tools=tools, system_prompt=sys_prompt)

    if len(sys.argv) > 1:
        # If user provides a full path or just a name, handle it
        # The tool expects just the filename in markdown_outputs
        arg_path = Path(sys.argv[1])
        filename = arg_path.name
        user_input = f"Chunk the file {filename}"
    else:
        # Default for testing
        user_input = "Chunk the file attention_is_all_you_need_annotated.md"

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
