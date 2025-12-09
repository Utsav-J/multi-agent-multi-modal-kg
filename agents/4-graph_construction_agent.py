import os
from langchain_neo4j.graphs.neo4j_graph import Neo4jGraph
from langchain_neo4j.graphs.graph_document import (
    GraphDocument as LCGraphDocument,
    Node as LCNode,
    Relationship as LCRelationship,
)
from langchain_core.documents import Document as LCDocument
from knowledge_graph.models import GraphDocument, extracted_sample_data

url = os.getenv("NEO4J_URI", "bolt://localhost:7687")
username = os.getenv("NEO4J_USERNAME", "neo4j")
password = os.getenv("NEO4J_PASSWORD", "password")

graph = Neo4jGraph(url=url, username=username, password=password)


# 2. Define a helper to convert your Custom Pydantic objects to LangChain objects
def convert_to_langchain_format(custom_graph_doc: GraphDocument):
    """
    Converts your custom Pydantic GraphDocument to LangChain's format.
    """
    # Convert Source Document
    lc_source = LCDocument(
        page_content=custom_graph_doc.source.page_content,
        metadata=(
            {m.key: m.value for m in custom_graph_doc.source.metadata}
            if custom_graph_doc.source.metadata
            else {}
        ),
    )

    # Convert Nodes
    lc_nodes = [
        LCNode(
            id=node.id,
            type=node.type,
            properties={},  # Add extra properties here if your node had them
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


# 3. Assuming 'your_extracted_data' is your custom GraphDocument instance
# (The variable containing the data from your prompt)

lc_graph_doc = convert_to_langchain_format(extracted_sample_data)

print(lc_graph_doc)

# 4. Ingest into Neo4j
# include_source=True will automatically create a 'Document' node and link entities to it!
graph.add_graph_documents(
    [lc_graph_doc],
    include_source=True,
)

print("Graph construction complete.")
