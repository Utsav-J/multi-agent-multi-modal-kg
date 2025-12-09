import sys
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add project root to sys.path to enable imports from utils
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Load environment variables
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from utils.process_pdf import convert_pdf_to_markdown, annotate_markdown_images

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DATA_DIR = project_root / "data"
OUTPUT_DIR = project_root / "markdown_outputs"
IMAGES_DIR = OUTPUT_DIR / "images"


@tool
def convert_pdf_to_markdown_tool(pdf_filename: str) -> str:
    """
    Converts a PDF file from the data directory to Markdown format, extracting images.

    Args:
        pdf_filename (str): The name of the PDF file located in the 'data' directory (e.g., 'paper.pdf').

    Returns:
        str: A message indicating the result of the conversion and the path to the generated raw markdown file.
    """
    logger.info(f"Tool invoked: convert_pdf_to_markdown_tool for file '{pdf_filename}'")
    try:
        input_pdf = DATA_DIR / pdf_filename
        if not input_pdf.exists():
            error_msg = f"Error: Input file not found at {input_pdf}"
            logger.error(error_msg)
            return error_msg

        base_name = input_pdf.stem
        intermediate_md = OUTPUT_DIR / f"{base_name}_raw.md"

        # Ensure directories exist
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting PDF conversion for {input_pdf}")
        success = convert_pdf_to_markdown(input_pdf, intermediate_md, IMAGES_DIR)

        if success:
            msg = f"Successfully converted PDF to Markdown. Raw output saved to: {intermediate_md.name}"
            logger.info(msg)
            return msg
        else:
            error_msg = "Failed to convert PDF to Markdown."
            logger.error(error_msg)
            return error_msg
    except Exception as e:
        error_msg = f"Error during PDF conversion: {str(e)}"
        logger.exception(error_msg)
        return error_msg


@tool
def annotate_markdown_tool(markdown_filename: str) -> str:
    """
    Annotates images in a Markdown file with AI-generated captions.

    Args:
        markdown_filename (str): The name of the markdown file in the 'markdown_outputs' directory to annotate.

    Returns:
        str: A message indicating the result and the path to the final annotated markdown file.
    """
    logger.info(f"Tool invoked: annotate_markdown_tool for file '{markdown_filename}'")
    try:
        input_md = OUTPUT_DIR / markdown_filename
        if not input_md.exists():
            error_msg = f"Error: Markdown file not found at {input_md}"
            logger.error(error_msg)
            return error_msg

        base_name = input_md.stem.replace("_raw", "")
        final_md = OUTPUT_DIR / f"{base_name}_annotated.md"

        logger.info(f"Starting annotation for {input_md}")
        # The annotate_markdown_images function returns the text, but also writes to file
        # We don't need the returned text here, just the side effect
        annotate_markdown_images(input_md, final_md)

        msg = f"Successfully annotated markdown images. Final output saved to: {final_md.name}"
        logger.info(msg)
        return msg
    except Exception as e:
        error_msg = f"Error during annotation: {str(e)}"
        logger.exception(error_msg)
        return error_msg


def main():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0, convert_system_message_to_human=True
    )

    tools = [convert_pdf_to_markdown_tool, annotate_markdown_tool]

    sys_prompt = (
        "You are a helpful AI assistant specializing in processing PDF documents. ",
        "Your goal is to convert PDFs to Markdown and then annotate any extracted images with descriptive captions. ",
        "Always start by converting the PDF. If that succeeds, proceed to annotate the resulting markdown file. ",
        "Report the final status to the user.",
    )

    agent = create_agent(model=llm, tools=tools, system_prompt=sys_prompt)

    if len(sys.argv) > 1:
        user_input = f"Process the file {sys.argv[1]}"
    else:
        user_input = "Process the file rag_paper.pdf"

    logger.info(f"Starting agent with input: {user_input}")
    try:
        agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        logger.info("Agent execution completed successfully.")
    except Exception as e:
        logger.exception("Agent execution failed with an error.")


if __name__ == "__main__":
    main()

# # Process the default file
# uv run agents/1-pdf_processor_agent.py
