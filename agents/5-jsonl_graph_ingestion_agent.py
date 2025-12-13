import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()
from langchain_neo4j.graphs.neo4j_graph import Neo4jGraph
from langchain_neo4j.graphs.graph_document import (
    GraphDocument as LCGraphDocument,
    Node as LCNode,
    Relationship as LCRelationship,
)
from langchain_core.documents import Document as LCDocument

# Add the project root to sys.path to allow imports from knowledge_graph
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from knowledge_graph.models import (
    GraphDocument,
    Node,
    Relationship,
    Document,
    MetadataItem,
)


def connect_to_neo4j() -> Neo4jGraph:
    """Establishes a connection to the Neo4j database."""
    url = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    return Neo4jGraph(url=url, username=username, password=password)


def convert_to_langchain_format(custom_graph_doc: GraphDocument) -> LCGraphDocument:
    """
    Converts your custom Pydantic GraphDocument to LangChain's format.
    """
    # Convert Source Document
    # Ensure metadata is in a format acceptable by LangChain (dict)
    metadata = {}
    if custom_graph_doc.source.metadata:
        for m in custom_graph_doc.source.metadata:
            metadata[m.key] = m.value

    lc_source = LCDocument(
        page_content=custom_graph_doc.source.page_content, metadata=metadata
    )

    # Convert Nodes
    lc_nodes = [
        LCNode(
            id=node.id,
            type=node.type,
            properties={},
        )
        for node in custom_graph_doc.nodes
    ]

    # Convert Relationships
    lc_relationships = [
        LCRelationship(
            source=LCNode(id=rel.source.id, type=rel.source.type),
            target=LCNode(id=rel.target.id, type=rel.target.type),
            type=rel.type,
            properties={},
        )
        for rel in custom_graph_doc.relationships
    ]

    return LCGraphDocument(
        nodes=lc_nodes, relationships=lc_relationships, source=lc_source
    )


def load_jsonl_and_ingest(file_path: str, graph: Neo4jGraph):
    """
    Reads a JSONL file, converts lines to GraphDocument, then to LangChain format,
    and ingests into Neo4j.
    """
    print(f"Processing file: {file_path}")

    lc_documents = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    # Extract only the fields required for GraphDocument
                    # The JSONL might have extra fields like 'original_chunk_id' which we ignore for the graph structure itself
                    # unless we want to add them as metadata to the source document.

                    # Reconstruct MetadataItems if present
                    source_data = data.get("source", {})
                    metadata_list = []
                    if source_data.get("metadata"):
                        if isinstance(source_data["metadata"], list):
                            # It's already a list of dicts or MetadataItems
                            for m in source_data["metadata"]:
                                if isinstance(m, dict):
                                    metadata_list.append(MetadataItem(**m))
                                else:
                                    # Assuming it might be already in correct object form if using some other loader,
                                    # but here we load from JSON so it's dicts.
                                    pass

                    # Reconstruct Document
                    source_doc = Document(
                        page_content=source_data.get("page_content", ""),
                        metadata=metadata_list if metadata_list else None,
                    )

                    # Reconstruct Nodes
                    nodes = [Node(**n) for n in data.get("nodes", [])]

                    # Reconstruct Relationships
                    relationships = []
                    for r in data.get("relationships", []):
                        # r['source'] and r['target'] are dicts in the JSON
                        relationships.append(
                            Relationship(
                                source=Node(**r["source"]),
                                target=Node(**r["target"]),
                                type=r["type"],
                            )
                        )

                    # Create Pydantic GraphDocument
                    custom_doc = GraphDocument(
                        nodes=nodes, relationships=relationships, source=source_doc
                    )

                    # Convert to LangChain format
                    lc_doc = convert_to_langchain_format(custom_doc)
                    lc_documents.append(lc_doc)

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line {line_number}: {e}")
                except Exception as e:
                    print(f"Error processing line {line_number}: {e}")

        if lc_documents:
            print(f"Ingesting {len(lc_documents)} documents into Neo4j...")
            graph.add_graph_documents(lc_documents, include_source=True)
            print("Ingestion complete.")
        else:
            print("No valid documents found to ingest.")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest Graph JSONL files into Neo4j")
    parser.add_argument(
        "filename",
        nargs="?",
        help="Specific JSONL file to ingest (relative to knowledge_graph_outputs)",
    )
    args = parser.parse_args()

    # Define the input file path
    output_dir = project_root / "knowledge_graph_outputs"
    files_to_process = []

    if args.filename:
        fpath = output_dir / args.filename
        if fpath.exists():
            files_to_process.append(fpath)
        else:
            print(f"File not found: {fpath}")
    else:
        # Scan for all graph jsonl files
        if output_dir.exists():
            files_to_process = list(output_dir.glob("*_graph.jsonl"))

    # Connect to Neo4j
    try:
        graph = connect_to_neo4j()
        graph.query("RETURN 1")
        print("Connected to Neo4j.")

        if not files_to_process:
            print("No graph files found to ingest.")
        else:
            print(f"Found {len(files_to_process)} files to ingest: {[f.name for f in files_to_process]}")
            for fpath in files_to_process:
                load_jsonl_and_ingest(str(fpath), graph)

    except Exception as e:
        print(f"Failed to connect to Neo4j or execute graph operations: {e}")
