import sys
import os
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def connect_to_neo4j() -> Neo4jGraph:
    """Establishes a connection to the Neo4j database."""
    url = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    return Neo4jGraph(url=url, username=username, password=password)


def main():
    parser = argparse.ArgumentParser(description="Graph Query Agent")
    parser.add_argument(
        "query",
        nargs="?",
        default="Tell me about attention mechanism",
        help="The natural language query to ask the knowledge graph",
    )
    args = parser.parse_args()

    logger.info(f"Starting Graph Query Agent with query: {args.query}")

    try:
        graph = connect_to_neo4j()
        
        # Refresh schema to ensure we have the latest data
        graph.refresh_schema()
        logger.info("Connected to Neo4j and refreshed schema.")

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", temperature=0
        )

        chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=True,
            allow_dangerous_requests=True
        )

        response = chain.invoke({"query": args.query})
        
        print("\n" + "="*50)
        print(f"Query: {args.query}")
        print("-" * 50)
        print(f"Answer: {response['result']}")
        print("="*50 + "\n")

    except Exception as e:
        logger.exception("Agent execution failed.")


if __name__ == "__main__":
    main()



