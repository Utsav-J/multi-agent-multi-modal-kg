"""
Validator Agent for Knowledge Graph Validation

This agent validates knowledge graphs across four dimensions:
1. Document Coverage - How much of the source content is represented
2. Extraction Faithfulness - Whether nodes/edges are grounded in text
3. Graph Structural Quality - Whether the KG is well-formed
4. Semantic Plausibility - Whether relations make sense

Focus is on deterministic structural quality tests first, with framework
for LLM-based tests provided for future implementation.
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dotenv import load_dotenv

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

load_dotenv()

# Neo4j imports
from langchain_neo4j.graphs.neo4j_graph import Neo4jGraph

# Knowledge graph models
from knowledge_graph.models import GraphDocument, Node, Relationship

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
MARKDOWN_DIR = project_root / "markdown_outputs"
VALIDATION_OUTPUT_DIR = project_root / "validation_outputs"


class StructuralEvaluator:
    """Evaluates graph structural quality using deterministic metrics (no LLM)."""

    def __init__(self, graph: Neo4jGraph):
        self.graph = graph

    def evaluate(self) -> Dict[str, float]:
        """
        Compute structural quality metrics:
        - Orphan ratio (nodes with degree = 0)
        - Component fragmentation (connected components / nodes)
        - Type consistency violations
        """
        logger.info("Computing structural quality metrics...")

        metrics = {}

        # Get total nodes and relationships
        node_count_result = self.graph.query("MATCH (n) RETURN count(n) as count")
        total_nodes = node_count_result[0]["count"] if node_count_result else 0

        rel_count_result = self.graph.query("MATCH ()-[r]->() RETURN count(r) as count")
        total_rels = rel_count_result[0]["count"] if rel_count_result else 0

        logger.info(f"Graph has {total_nodes} nodes and {total_rels} relationships")

        # Test C1: Orphan Ratio
        orphan_result = self.graph.query(
            """
            MATCH (n)
            WHERE NOT (n)--()
            RETURN count(n) as orphan_count
            """
        )
        orphan_count = orphan_result[0]["orphan_count"] if orphan_result else 0
        orphan_ratio = orphan_count / total_nodes if total_nodes > 0 else 0.0
        metrics["orphan_ratio"] = orphan_ratio

        logger.info(f"Orphan nodes: {orphan_count} ({orphan_ratio:.2%})")

        # Test C2: Component Fragmentation
        # Use a simpler approach that works without GDS library
        # Count distinct connected components by traversing from each node
        try:
            # Try GDS first if available
            component_result = self.graph.query(
                """
                CALL gds.graph.project.cypher(
                    'temp_graph',
                    'MATCH (n) RETURN id(n) as id',
                    'MATCH (n)-[r]->(m) RETURN id(n) as source, id(m) as target'
                ) YIELD nodeCount
                CALL gds.wcc.stream('temp_graph')
                YIELD nodeId, componentId
                WITH collect(DISTINCT componentId) as components
                RETURN size(components) as component_count
                """
            )
            if (
                component_result
                and component_result[0].get("component_count") is not None
            ):
                component_count = component_result[0]["component_count"]
            else:
                raise Exception("GDS result was None")
        except Exception as e:
            logger.info(
                f"GDS library not available or failed ({e}), using alternative method"
            )
            # Alternative: Use a more efficient connected components algorithm
            # Start from each node and collect all reachable nodes
            try:
                component_result = self.graph.query(
                    """
                    MATCH (n)
                    WITH collect(id(n)) as all_node_ids
                    UNWIND all_node_ids as start_id
                    MATCH (start)
                    WHERE id(start) = start_id
                    OPTIONAL MATCH path = (start)-[*0..]-(connected)
                    WITH start_id, collect(DISTINCT id(connected)) as component_nodes
                    WITH collect(component_nodes) as all_components
                    UNWIND all_components as comp
                    WITH collect(DISTINCT comp) as unique_components
                    RETURN size(unique_components) as component_count
                    """
                )
                if (
                    component_result
                    and component_result[0].get("component_count") is not None
                ):
                    component_count = component_result[0]["component_count"]
                else:
                    raise Exception("Alternative query returned None")
            except Exception as e2:
                logger.warning(
                    f"Alternative component counting failed ({e2}), using simple estimation"
                )
                # Simple fallback: estimate based on connected nodes
                connected_nodes_result = self.graph.query(
                    """
                    MATCH (n)
                    WHERE (n)--()
                    RETURN count(DISTINCT n) as connected_count
                    """
                )
                connected_count = (
                    connected_nodes_result[0]["connected_count"]
                    if connected_nodes_result
                    else 0
                )
                # Rough estimate: connected nodes form one component, isolated nodes are separate
                isolated_count = total_nodes - connected_count
                component_count = max(1, isolated_count) + (
                    1 if connected_count > 0 else 0
                )

        component_ratio = component_count / total_nodes if total_nodes > 0 else 0.0
        metrics["component_ratio"] = component_ratio

        logger.info(
            f"Connected components: {component_count} (ratio: {component_ratio:.4f})"
        )

        # Test C3: Type Consistency
        # Check for nodes without type/id and relationships with missing endpoints
        # In langchain_neo4j, nodes typically have labels (from node.type) and id property

        # Check for nodes without id property (required identifier)
        try:
            unlabeled_result = self.graph.query(
                """
                MATCH (n)
                WHERE n.id IS NULL OR n.id = ''
                RETURN count(n) as unlabeled_count
                """
            )
            unlabeled_count = (
                unlabeled_result[0]["unlabeled_count"] if unlabeled_result else 0
            )
        except Exception as e:
            logger.warning(f"Could not check for unlabeled nodes: {e}")
            unlabeled_count = 0

        # Check for relationships with missing or invalid endpoints
        # All relationships should have valid source and target nodes
        try:
            invalid_rel_result = self.graph.query(
                """
                MATCH (source)-[r]->(target)
                WHERE source.id IS NULL OR target.id IS NULL
                   OR source.id = '' OR target.id = ''
                RETURN count(r) as invalid_count
                """
            )
            invalid_rel_count = (
                invalid_rel_result[0]["invalid_count"] if invalid_rel_result else 0
            )
        except Exception as e:
            logger.warning(f"Could not check for invalid relationships: {e}")
            invalid_rel_count = 0

        total_violations = unlabeled_count + invalid_rel_count
        type_violation_rate = total_violations / max(total_rels, 1)
        metrics["type_violation_rate"] = type_violation_rate

        logger.info(
            f"Type violations: {total_violations} (unlabeled: {unlabeled_count}, "
            f"invalid rels: {invalid_rel_count}, rate: {type_violation_rate:.4f})"
        )

        return metrics


class CoverageEvaluator:
    """Evaluates document coverage (requires LLM - framework provided)."""

    def __init__(self, graph: Neo4jGraph, chunks_dir: Path, markdown_dir: Path):
        self.graph = graph
        self.chunks_dir = chunks_dir
        self.markdown_dir = markdown_dir
        # LLM will be added later
        self.llm = None

    def evaluate(
        self,
        document_id: str,
        chunks_filename: Optional[str] = None,
        sample_size: int = 10,
    ) -> Dict[str, float]:
        """
        Compute coverage metrics:
        - Entity Coverage Score
        - Relationship Coverage Score

        Framework provided - LLM integration needed.
        """
        logger.info(f"Computing coverage metrics for document: {document_id}")

        # Placeholder metrics - to be implemented with LLM
        metrics = {
            "entity_coverage": 0.0,
            "relation_coverage": 0.0,
        }

        logger.warning(
            "Coverage evaluation requires LLM integration. "
            "Returning placeholder metrics. "
            "To implement: sample chunks, use LLM to extract entities/relations, "
            "compare with graph extractions."
        )

        return metrics


class FaithfulnessEvaluator:
    """Evaluates extraction faithfulness (requires LLM - framework provided)."""

    def __init__(self, graph: Neo4jGraph, chunks_dir: Path, markdown_dir: Path):
        self.graph = graph
        self.chunks_dir = chunks_dir
        self.markdown_dir = markdown_dir
        # LLM will be added later
        self.llm = None

    def evaluate(self, document_id: str, sample_size: int = 50) -> Dict[str, float]:
        """
        Compute faithfulness metrics:
        - Node Grounding Check
        - Relationship Grounding Check

        Framework provided - LLM integration needed.
        """
        logger.info(f"Computing faithfulness metrics for document: {document_id}")

        # Placeholder metrics - to be implemented with LLM
        metrics = {
            "node_faithfulness": 0.0,
            "relation_faithfulness": 0.0,
        }

        logger.warning(
            "Faithfulness evaluation requires LLM integration. "
            "Returning placeholder metrics. "
            "To implement: sample nodes/relations, use LLM to check grounding in source text."
        )

        return metrics


class SemanticEvaluator:
    """Evaluates semantic plausibility (requires LLM - framework provided)."""

    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
        # LLM will be added later
        self.llm = None

    def evaluate(self, sample_size: int = 100) -> Dict[str, float]:
        """
        Compute semantic plausibility:
        - Relation-Type Plausibility

        Framework provided - LLM integration needed.
        """
        logger.info("Computing semantic plausibility metrics...")

        # Placeholder metrics - to be implemented with LLM
        metrics = {
            "semantic_plausibility": 0.0,
        }

        logger.warning(
            "Semantic evaluation requires LLM integration. "
            "Returning placeholder metrics. "
            "To implement: sample relations, use LLM to check semantic plausibility."
        )

        return metrics


class ReportAggregator:
    """Aggregates evaluation results into structured report."""

    @staticmethod
    def aggregate(
        document_id: str,
        coverage_metrics: Dict[str, float],
        faithfulness_metrics: Dict[str, float],
        structural_metrics: Dict[str, float],
        semantic_metrics: Dict[str, float],
        notes: Optional[List[str]] = None,
    ) -> Dict:
        """
        Combine all metrics into a single structured report.

        Returns a dictionary matching the validator plan output format.
        """
        all_metrics = {}
        all_metrics.update(coverage_metrics)
        all_metrics.update(faithfulness_metrics)
        all_metrics.update(structural_metrics)
        all_metrics.update(semantic_metrics)

        report = {
            "document_id": document_id,
            "metrics": all_metrics,
            "notes": notes or [],
        }

        return report


class ValidatorAgent:
    """Main validator agent coordinating all evaluation modules."""

    def __init__(
        self,
        graph: Neo4jGraph,
        chunks_dir: Path = CHUNKS_DIR,
        markdown_dir: Path = MARKDOWN_DIR,
        enable_llm_tests: bool = False,
    ):
        self.graph = graph
        self.chunks_dir = chunks_dir
        self.markdown_dir = markdown_dir
        self.enable_llm_tests = enable_llm_tests

        # Initialize evaluators
        self.structural_evaluator = StructuralEvaluator(graph)
        self.coverage_evaluator = CoverageEvaluator(graph, chunks_dir, markdown_dir)
        self.faithfulness_evaluator = FaithfulnessEvaluator(
            graph, chunks_dir, markdown_dir
        )
        self.semantic_evaluator = SemanticEvaluator(graph)
        self.report_aggregator = ReportAggregator()

    def validate_document(
        self, document_id: str, chunks_filename: Optional[str] = None
    ) -> Dict:
        """
        Validate a single document's knowledge graph.

        Args:
            document_id: Identifier for the document. This should be the chunks filename
                stem (without .jsonl extension) or the markdown filename stem (without .md).
                Examples:
                - For chunks file "attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k.jsonl":
                  use "attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k"
                - For markdown file "attention_is_all_you_need_raw_with_image_ids_with_captions.md":
                  use "attention_is_all_you_need_raw_with_image_ids_with_captions"
                Note: The chunks filename stem typically includes "_chunks_5k" suffix, while
                the markdown filename stem does not. The document_id should match what's stored
                in the graph's source_id metadata (typically the chunks filename without extension).
            chunks_filename: Optional chunks filename (with .jsonl extension) for coverage
                evaluation. This should be the full filename from chunking_outputs/, e.g.:
                "attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k.jsonl"
                If not provided and LLM tests are enabled, coverage evaluation will attempt
                to infer the chunks filename from document_id.

        Returns:
            Structured validation report dictionary
        """
        logger.info(f"Starting validation for document: {document_id}")

        notes = []

        # Always run structural evaluation (deterministic)
        logger.info("Running structural quality evaluation...")
        structural_metrics = self.structural_evaluator.evaluate()

        # Run LLM-based evaluations if enabled
        if self.enable_llm_tests:
            logger.info("Running coverage evaluation...")
            coverage_metrics = self.coverage_evaluator.evaluate(
                document_id, chunks_filename
            )

            logger.info("Running faithfulness evaluation...")
            faithfulness_metrics = self.faithfulness_evaluator.evaluate(document_id)

            logger.info("Running semantic evaluation...")
            semantic_metrics = self.semantic_evaluator.evaluate()
        else:
            logger.info("Skipping LLM-based evaluations (not enabled)")
            coverage_metrics = {}
            faithfulness_metrics = {}
            semantic_metrics = {}

        # Generate notes based on metrics
        if structural_metrics.get("orphan_ratio", 0) > 0.1:
            notes.append("High orphan ratio detected - many nodes are disconnected")
        if structural_metrics.get("component_ratio", 0) > 0.1:
            notes.append("High component fragmentation - graph is highly fragmented")
        if structural_metrics.get("type_violation_rate", 0) > 0.05:
            notes.append(
                "Type violations detected - some nodes/relationships have invalid types"
            )

        # Aggregate results
        report = self.report_aggregator.aggregate(
            document_id=document_id,
            coverage_metrics=coverage_metrics,
            faithfulness_metrics=faithfulness_metrics,
            structural_metrics=structural_metrics,
            semantic_metrics=semantic_metrics,
            notes=notes,
        )

        logger.info(f"Validation complete for document: {document_id}")
        return report

    def validate_all_documents(self) -> List[Dict]:
        """
        Validate all documents in the knowledge graph.

        Extracts document IDs from graph source metadata or JSONL files and validates each.
        Document IDs are extracted from source_id metadata, which follows the format:
        "<chunks_filename>::<chunk_id>", where chunks_filename is the chunks file stem
        (without .jsonl extension).

        Returns:
            List of validation reports, one per document
        """
        logger.info("Validating all documents in knowledge graph...")

        # Extract document IDs from graph
        # Documents are stored with source metadata. source_id format: "<chunks_filename>::<chunk_id>"
        # We extract the chunks filename part (before "::") as the document identifier
        # Example: "attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k.jsonl::chunk_0"
        #          -> document_id = "attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k.jsonl"
        doc_result = self.graph.query(
            """
            MATCH (n)
            WHERE n.source_id IS NOT NULL
            WITH DISTINCT n.source_id as source_id
            RETURN source_id
            LIMIT 100
            """
        )

        document_ids = []
        if doc_result:
            # Extract unique document identifiers from source_id
            # Format: "chunks_filename::chunk_id" -> extract "chunks_filename" part
            # This gives us the chunks filename (with .jsonl extension typically)
            seen = set()
            for row in doc_result:
                source_id = row.get("source_id", "")
                # Extract document identifier (chunks filename) from "filename::chunk_id" format
                if "::" in source_id:
                    doc_id = source_id.split("::")[0]
                else:
                    # If no "::" separator, use the whole source_id as document_id
                    doc_id = source_id
                if doc_id and doc_id not in seen:
                    seen.add(doc_id)
                    document_ids.append(doc_id)

        # If no documents found in graph, try to infer from JSONL files in knowledge_graph_outputs/
        # JSONL files are named like: "<chunks_filename_stem>_graph.jsonl"
        # We extract the base name (chunks filename stem) as document_id
        # Example: "attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k_graph.jsonl"
        #          -> doc_id = "attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k"
        if not document_ids:
            logger.info("No documents found in graph metadata, checking JSONL files...")
            jsonl_files = list(OUTPUT_DIR.glob("*_graph.jsonl"))
            for jsonl_file in jsonl_files:
                # Extract document ID from filename by removing "_graph" suffix
                # "attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k_graph.jsonl"
                # -> "attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k_graph"
                # -> "attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k" (document_id)
                doc_id = jsonl_file.stem.replace("_graph", "")
                document_ids.append(doc_id)

        if not document_ids:
            logger.warning(
                "No documents found to validate. Using 'unknown' as document_id."
            )
            document_ids = ["unknown"]

        logger.info(
            f"Found {len(document_ids)} document(s) to validate: {document_ids}"
        )

        reports = []
        for doc_id in document_ids:
            try:
                report = self.validate_document(doc_id)
                reports.append(report)
            except Exception as e:
                logger.error(f"Error validating document {doc_id}: {e}", exc_info=True)
                # Create error report
                error_report = {
                    "document_id": doc_id,
                    "metrics": {},
                    "notes": [f"Validation failed with error: {str(e)}"],
                }
                reports.append(error_report)

        return reports


def connect_to_neo4j() -> Neo4jGraph:
    """Establish connection to Neo4j database."""
    url = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    logger.info(f"Connecting to Neo4j at {url}...")
    return Neo4jGraph(url=url, username=username, password=password)


def save_report(report: Dict, output_dir: Path):
    """Save validation report to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_id = report.get("document_id", "unknown")
    output_file = output_dir / f"{doc_id}_validation_report.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Validation report saved to: {output_file}")


def main():
    """
    Main entry point for validator agent.

    Command-line arguments:
    --document-id: Optional. Specific document ID to validate. If not provided, validates all
        documents found in the graph. Format: chunks filename stem (without .jsonl extension).
        Example: "attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k"
        Note: This should match the chunks filename stem that was used during graph extraction.

    --chunks-filename: Optional. Full chunks filename (with .jsonl extension) for coverage
        evaluation. Only used when LLM tests are enabled. Should be a filename from
        chunking_outputs/ directory.
        Example: "attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k.jsonl"
        If not provided, coverage evaluation will attempt to infer from document_id by
        appending "_chunks_5k.jsonl" suffix.

    --enable-llm-tests: Flag. When set, enables LLM-based evaluation tests:
        - Coverage evaluation (entity and relationship coverage)
        - Faithfulness evaluation (node and relationship grounding)
        - Semantic evaluation (semantic plausibility)
        Note: Currently these return placeholder metrics. Full implementation requires
        LLM integration in the respective evaluator classes.

    --output-dir: Optional. Directory path for saving validation reports. Default:
        "validation_outputs/" (relative to project root). Reports are saved as:
        <document_id>_validation_report.json

    Environment variables required (via .env file):
        NEO4J_URI: Neo4j connection URI (default: "bolt://localhost:7687")
        NEO4J_USERNAME: Neo4j username (default: "neo4j")
        NEO4J_PASSWORD: Neo4j password (default: "password")

    Examples:
        # Validate all documents (structural tests only)
        python agents/7-validator_agent.py

        # Validate specific document
        python agents/7-validator_agent.py --document-id "attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k"

        # Validate with chunks file for coverage evaluation (requires --enable-llm-tests)
        python agents/7-validator_agent.py --document-id "attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k" \\
            --chunks-filename "attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k.jsonl" \\
            --enable-llm-tests

        # Custom output directory
        python agents/7-validator_agent.py --output-dir "custom_reports"
    """
    parser = argparse.ArgumentParser(
        description="Knowledge Graph Validator Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate all documents (structural tests only)
  python agents/7-validator_agent.py

  # Validate specific document
  python agents/7-validator_agent.py --document-id "attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k"

  # Validate with LLM tests enabled (currently placeholders)
  python agents/7-validator_agent.py --document-id "attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k" --enable-llm-tests
        """,
    )
    parser.add_argument(
        "--document-id",
        type=str,
        default=None,
        help=(
            "Specific document ID to validate (if not provided, validates all). "
            "Format: chunks filename stem without .jsonl extension. "
            "Example: 'attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k'"
        ),
    )
    parser.add_argument(
        "--chunks-filename",
        type=str,
        default=None,
        help=(
            "Full chunks filename (with .jsonl extension) for coverage evaluation. "
            "Only used when --enable-llm-tests is set. Should be a filename from chunking_outputs/. "
            "Example: 'attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k.jsonl'"
        ),
    )
    parser.add_argument(
        "--enable-llm-tests",
        action="store_true",
        help=(
            "Enable LLM-based evaluation tests (coverage, faithfulness, semantic). "
            "Note: Currently returns placeholder metrics. Full implementation requires LLM integration."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(VALIDATION_OUTPUT_DIR),
        help=(
            "Output directory for validation reports (default: validation_outputs/). "
            "Reports are saved as: <document_id>_validation_report.json"
        ),
    )

    args = parser.parse_args()

    try:
        # Connect to Neo4j
        graph = connect_to_neo4j()
        logger.info("Connected to Neo4j successfully")

        # Create validator agent
        validator = ValidatorAgent(
            graph=graph,
            enable_llm_tests=args.enable_llm_tests,
        )

        output_dir = Path(args.output_dir)

        # Run validation
        if args.document_id:
            # Validate specific document
            report = validator.validate_document(args.document_id, args.chunks_filename)
            save_report(report, output_dir)
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            # Validate all documents
            reports = validator.validate_all_documents()
            for report in reports:
                save_report(report, output_dir)
            print(f"Validation complete. Generated {len(reports)} report(s).")

    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
