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

    def _metadata_items_to_properties(items: list[MetadataItem] | None) -> dict:
        """
        Convert MetadataItem list to a Neo4j-friendly properties dict.

        - Duplicate keys become lists (Neo4j supports list properties).
        - All values are stored as strings (as defined in MetadataItem).
        """
        props: dict = {}
        if not items:
            return props

        for m in items:
            if not m or not getattr(m, "key", None):
                continue
            k = m.key
            v = m.value
            if k in props:
                if isinstance(props[k], list):
                    props[k].append(v)
                else:
                    props[k] = [props[k], v]
            else:
                props[k] = v
        return props

    # Convert Source Document
    # Ensure metadata is in a format acceptable by LangChain (dict)
    metadata = {}
    if custom_graph_doc.source and custom_graph_doc.source.metadata:
        for m in custom_graph_doc.source.metadata:
            metadata[m.key] = m.value

    # Always include source_id/source_type for provenance even if metadata list is empty.
    if custom_graph_doc.source:
        metadata.setdefault("source_id", custom_graph_doc.source.source_id)
        metadata.setdefault("source_type", custom_graph_doc.source.source_type)

    lc_source = LCDocument(
        # Source is a pointer, not payload.
        page_content=(
            custom_graph_doc.source.source_id if custom_graph_doc.source else ""
        ),
        metadata=metadata,
    )

    # Convert Nodes
    lc_nodes = [
        LCNode(
            id=node.id,
            type=node.type,
            properties=_metadata_items_to_properties(getattr(node, "metadata", None)),
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
                        source_id=source_data.get("source_id", "unknown"),
                        source_type=source_data.get("source_type", "chunk"),
                        metadata=metadata_list,
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
            # IMPORTANT:
            # langchain_neo4j currently uses apoc.merge.node([label], {id}, onCreateProps, onMatchProps)
            # with onMatchProps={}, meaning node properties won't be updated if the node already exists.
            # We run an explicit "upsert properties" pass so metadata becomes visible on existing nodes.
            try:
                upsert_rows = []
                for doc in lc_documents:
                    for n in doc.nodes:
                        upsert_rows.append(
                            {
                                "id": n.id,
                                "type": n.type,
                                "properties": n.properties or {},
                            }
                        )

                if upsert_rows:
                    graph.query(
                        """
                        UNWIND $data AS row
                        CALL apoc.merge.node([row.type], {id: row.id}, row.properties, row.properties)
                        YIELD node
                        RETURN count(node) AS upserted
                        """,
                        params={"data": upsert_rows},
                    )
            except Exception as e:
                print(f"Warning: failed to upsert node properties: {e}")
            print("Ingestion complete.")
        else:
            print("No valid documents found to ingest.")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Define the input file path
    input_file = (
        project_root
        / "knowledge_graph_outputs"
        / "attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k_graph.jsonl"
    )

    # Connect to Neo4j
    try:
        graph = connect_to_neo4j()
        print(1)
        graph.query("RETURN 1")
        print(1)
        print("Connected to Neo4j.")

        print(1)
        load_jsonl_and_ingest(str(input_file), graph)

    except Exception as e:
        print(f"Failed to connect to Neo4j or execute graph operations: {e}")
