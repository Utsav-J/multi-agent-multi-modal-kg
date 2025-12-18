import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI

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

CYPHER_PROMPT = PromptTemplate(
    input_variables=["schema", "query", "entities"],
    template="""
You are an expert Neo4j Cypher generator.

Schema:
{schema}

Known entities in the graph (use ONLY these names):
{entities}

Rules:
- Do NOT invent entity names.
- Use only labels and properties present in the schema.
- If no entity matches, return an empty query.

User question:
{query}

Generate Cypher only.
""",
)


def resolve_entities(graph: Neo4jGraph, query: str, limit: int = 5):
    cypher = """
    CALL db.index.fulltext.queryNodes(
        "entityIndex",
        $query
    )
    YIELD node, score
    RETURN
        labels(node) AS labels,
        node.text AS name,
        score
    ORDER BY score DESC
    LIMIT $limit
    """

    return graph.query(
        cypher,
        params={"query": query, "limit": limit},
    )


def query_graph(query: str):
    try:
        url = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        logger.info(f"Connecting to Neo4j at {url}...")
        graph = Neo4jGraph(url=url, username=username, password=password)

        schema = graph.schema

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            convert_system_message_to_human=True,
        )

        # 1. Resolve entities FIRST
        logger.info("Resolving entities from graph...")
        entities = resolve_entities(graph, query)

        if not entities:
            logger.warning("No matching entities found.")
            return "No relevant entities found in the knowledge graph."

        entity_str = "\n".join(
            f"- {e['name']} ({', '.join(e['labels'])})" for e in entities
        )

        # 2. Build Cypher chain with constrained prompt
        logger.info("Initializing grounded GraphCypherQAChain...")
        chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            cypher_prompt=CYPHER_PROMPT,
            verbose=True,
            return_intermediate_steps=True,
            allow_dangerous_requests=True,
        )

        # 3. Invoke with grounded context
        result = chain.invoke(
            {
                "query": query,  # REQUIRED by GraphCypherQAChain
                "entities": entity_str,
                "schema": schema,
            }
        )
        logger.info("Generated Cypher:")
        steps = result.get("intermediate_steps", [])
        if steps:
            logger.info("Generated Cypher:")
            logger.info(steps[0].get("cypher"))

        return result.get("result", "No result returned.")

    except Exception as e:
        logger.error(f"Error in Graph retrieval: {e}", exc_info=True)
        return f"Error occurred during graph query: {str(e)}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query Neo4j Graph")
    parser.add_argument(
        "query",
        nargs="?",
        default="what is attention mechanism?",
        help="The query to answer.",
    )
    args = parser.parse_args()

    response = query_graph(args.query)
    print("\nFinal Answer:")
    print(response)

"""
basic working
(mas-for-multimodal-knowledge-graph) 
UTSAV@Utsav MINGW64 /e/Python Stuff/MAS-for-multimodal-knowledge-graph (main)
$ uv run utils/neo4j_query.py 
2025-12-18 12:30:48,118 - __main__ - INFO - Connecting to Neo4j at neo4j+s:<>...
2025-12-18 12:30:50,138 - __main__ - INFO - Resolving entities from graph...
2025-12-18 12:30:50,328 - __main__ - INFO - Initializing grounded GraphCypherQAChain...


> Entering new GraphCypherQAChain chain...
Generated Cypher:
cypher
MATCH (am:`Attention Mechanism`)
OPTIONAL MATCH (am)-[:PERFORMS]->(f:Function)
OPTIONAL MATCH (am)-[:USES]->(c:Component)
WITH COLLECT(DISTINCT f.id) AS functionsPerformed, COLLECT(DISTINCT c.id) AS componentsUsed
RETURN functionsPerformed AS FunctionsPerformedByAttentionMechanism,
       componentsUsed AS ComponentsUsedByAttentionMechanism

Full Context:
[{'FunctionsPerformedByAttentionMechanism': ['attention function'], 'ComponentsUsedByAttentionMechanism': ['Key', 'Value', 'Query']}]

> Finished chain.
2025-12-18 12:31:04,314 - __main__ - INFO - Generated Cypher:
2025-12-18 12:31:04,315 - __main__ - INFO - Generated Cypher:
2025-12-18 12:31:04,315 - __main__ - INFO - None

Final Answer:
The attention mechanism performs an attention function and uses Key, Value, and Query components.
(mas-for-multimodal-knowledge-graph)
UTSAV@Utsav MINGW64 /e/Python Stuff/MAS-for-multimodal-knowledge-graph (main)
$
"""
