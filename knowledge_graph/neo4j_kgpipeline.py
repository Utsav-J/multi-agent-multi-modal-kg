import os
from neo4j import GraphDatabase
from typing import List
from models import GraphDocument  # Import your Pydantic models
from dotenv import load_dotenv
from models import extracted_sample_data

load_dotenv()


class Neo4jKGConstructor:
    """Constructs and manages a Neo4j knowledge graph from GraphDocument objects."""

    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize Neo4j connection.

        Args:
            uri: Neo4j connection URI (e.g., 'bolt://localhost:7687')
            user: Neo4j username
            password: Neo4j password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()

    def create_constraints_and_indexes(self):
        """Create uniqueness constraints and indexes for better performance."""
        with self.driver.session() as session:
            # Create uniqueness constraint on node id (this also creates an index)
            # Note: Adjust constraint name based on your Neo4j version
            constraints = [
                "CREATE CONSTRAINT node_id_unique IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
            ]

            # Create indexes on node type for faster queries
            indexes = [
                "CREATE INDEX node_type_idx IF NOT EXISTS FOR (n:Entity) ON (n.type)",
            ]

            for constraint in constraints:
                try:
                    session.run(constraint)
                    print(f"Created constraint: {constraint}")
                except Exception as e:
                    print(f"Constraint may already exist: {e}")

            for index in indexes:
                try:
                    session.run(index)
                    print(f"Created index: {index}")
                except Exception as e:
                    print(f"Index may already exist: {e}")

    def ingest_graph_document(self, graph_doc: GraphDocument):
        """
        Ingest a single GraphDocument into Neo4j.

        Args:
            graph_doc: GraphDocument containing nodes, relationships, and source
        """
        with self.driver.session() as session:
            # Ingest nodes
            self._ingest_nodes(session, graph_doc.nodes)

            # Ingest relationships
            self._ingest_relationships(session, graph_doc.relationships)

            # Optionally: Store source document information
            self._link_source_document(session, graph_doc)

    def ingest_graph_documents_batch(
        self, graph_docs: List[GraphDocument], batch_size: int = 100
    ):
        """
        Ingest multiple GraphDocuments in batches for better performance.

        Args:
            graph_docs: List of GraphDocument objects
            batch_size: Number of operations per batch
        """
        with self.driver.session() as session:
            # Collect all unique nodes
            all_nodes = []
            for doc in graph_docs:
                all_nodes.extend(doc.nodes)

            # Batch create nodes
            self._batch_create_nodes(session, all_nodes, batch_size)

            # Collect all relationships
            all_relationships = []
            for doc in graph_docs:
                all_relationships.extend(doc.relationships)

            # Batch create relationships
            self._batch_create_relationships(session, all_relationships, batch_size)

            print(f"Ingested {len(graph_docs)} graph documents")

    def _ingest_nodes(self, session, nodes: List):
        """Create or merge nodes into Neo4j."""
        for node in nodes:
            query = """
            MERGE (n:Entity {id: $id})
            SET n.type = $type
            """
            session.run(query, id=node.id, type=node.type)

    def _batch_create_nodes(self, session, nodes: List, batch_size: int):
        """Batch create nodes for better performance."""
        # Deduplicate nodes by id
        unique_nodes = {node.id: node for node in nodes}
        node_list = list(unique_nodes.values())

        for i in range(0, len(node_list), batch_size):
            batch = node_list[i : i + batch_size]
            node_data = [{"id": n.id, "type": n.type} for n in batch]

            query = """
            UNWIND $nodes AS node
            MERGE (n:Entity {id: node.id})
            SET n.type = node.type
            """
            session.run(query, nodes=node_data)

        print(f"Created/merged {len(node_list)} unique nodes")

    def _ingest_relationships(self, session, relationships: List):
        """Create relationships between nodes."""
        for rel in relationships:
            query = (
                """
            MATCH (source:Entity {id: $source_id})
            MATCH (target:Entity {id: $target_id})
            MERGE (source)-[r:`"""
                + rel.type
                + """`]->(target)
            """
            )
            session.run(query, source_id=rel.source.id, target_id=rel.target.id)

    def _batch_create_relationships(
        self, session, relationships: List, batch_size: int
    ):
        """Batch create relationships for better performance."""
        for i in range(0, len(relationships), batch_size):
            batch = relationships[i : i + batch_size]
            rel_data = [
                {"source_id": r.source.id, "target_id": r.target.id, "type": r.type}
                for r in batch
            ]

            query = """
            UNWIND $relationships AS rel
            MATCH (source:Entity {id: rel.source_id})
            MATCH (target:Entity {id: rel.target_id})
            CALL apoc.merge.relationship(source, rel.type, {}, {}, target, {})
            YIELD rel as relationship
            RETURN count(relationship)
            """

            # Fallback if APOC is not available
            try:
                session.run(query, relationships=rel_data)
            except Exception as e:
                print(f"APOC not available, using standard MERGE: {e}")
                self._batch_create_relationships_without_apoc(session, batch)

        print(f"Created {len(relationships)} relationships")

    def _batch_create_relationships_without_apoc(self, session, relationships: List):
        """Fallback method without APOC for dynamic relationship types."""
        for rel in relationships:
            # Use string formatting for relationship type (be cautious with input validation)
            query = f"""
            MATCH (source:Entity {{id: $source_id}})
            MATCH (target:Entity {{id: $target_id}})
            MERGE (source)-[r:{rel.type}]->(target)
            """
            session.run(query, source_id=rel.source.id, target_id=rel.target.id)

    def _link_source_document(self, session, graph_doc: GraphDocument):
        """
        Optionally create a Document node and link it to extracted entities.
        This helps track provenance.
        """
        if graph_doc.source and graph_doc.source.page_content:
            # Create a hash or ID for the document
            import hashlib

            doc_id = hashlib.md5(graph_doc.source.page_content.encode()).hexdigest()

            # Create Document node
            query = """
            MERGE (d:Document {id: $doc_id})
            SET d.content = $content
            """
            session.run(
                query,
                doc_id=doc_id,
                content=graph_doc.source.page_content[:1000],  # Truncate if too long
            )

            # Link document to all nodes extracted from it
            for node in graph_doc.nodes:
                link_query = """
                MATCH (d:Document {id: $doc_id})
                MATCH (n:Entity {id: $node_id})
                MERGE (d)-[:EXTRACTED]->(n)
                """
                session.run(link_query, doc_id=doc_id, node_id=node.id)


def main():
    url = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    kg = Neo4jKGConstructor(uri=url, user=username, password=password)

    try:
        # Create constraints and indexes first
        kg.create_constraints_and_indexes()

        graph_documents = [extracted_sample_data]  # Your list of GraphDocument objects
        kg.ingest_graph_documents_batch(graph_documents, batch_size=100)

    finally:
        kg.close()

if __name__ == "__main__":
    main()
