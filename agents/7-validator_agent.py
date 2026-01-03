"""
Validator Agent for Knowledge Graph Validation

This agent validates knowledge graphs across four dimensions:
1. Document Coverage - How much of the source content is represented (LLM-based)
2. Extraction Faithfulness - Whether nodes/edges are grounded in text (LLM-based)
3. Graph Structural Quality - Whether the KG is well-formed (deterministic)
4. Semantic Plausibility - Whether relations make sense (LLM-based)

Uses Google Gemini API for LLM-based evaluations and Neo4j queries for structural analysis.
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

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
# Model that supports structured outputs (json_schema)
MODEL = "gemini-2.5-flash"
CHUNKS_DIR = project_root / "chunking_outputs"
OUTPUT_DIR = project_root / "knowledge_graph_outputs"
MARKDOWN_DIR = project_root / "markdown_outputs"
VALIDATION_OUTPUT_DIR = project_root / "validation_outputs"


# Pydantic models for structured LLM responses
class EntityList(BaseModel):
    """List of entities extracted from text."""

    entities: List[str] = Field(
        description="List of key scientific entities explicitly mentioned"
    )


class RelationshipList(BaseModel):
    """List of relationships extracted from text."""

    relationships: List[Dict[str, str]] = Field(
        description="List of relationships, each with 'source', 'target', and 'type' fields"
    )


class GroundingCheck(BaseModel):
    """Result of grounding check."""

    grounded: bool = Field(
        description="Whether the entity/relation is grounded in the text"
    )
    evidence: str = Field(description="Quoted evidence from text or 'not supported'")


class PlausibilityCheck(BaseModel):
    """Result of semantic plausibility check."""

    plausible: str = Field(description="One of: 'yes', 'no', or 'unclear'")
    reasoning: Optional[str] = Field(
        default=None, description="Brief reasoning for the judgment"
    )


class EntityAndRelationList(BaseModel):
    """Combined list of entities and relationships extracted from text."""

    entities: List[str] = Field(
        default=[],
        description="List of key scientific entities explicitly mentioned in the text. Each entity is a string representing the entity name as it appears in the text.",
    )
    relationships: List[Dict[str, str]] = Field(
        default=[],
        description="List of relationships between entities. Each relationship must be a dictionary with exactly three keys: 'source' (string, source entity name), 'target' (string, target entity name), and 'type' (string, relationship type). Example: {'source': 'Entity1', 'target': 'Entity2', 'type': 'RELATED_TO'}",
    )


# Rate limiting configuration
LLM_REQUEST_DELAY = 0.2  # Seconds to wait between LLM requests to prevent throttling


# Helper functions to convert Pydantic models to Gemini Schema format
def get_entity_and_relation_list_schema():
    """Convert EntityAndRelationList to Gemini Schema format."""
    return genai.types.Schema(
        type=genai.types.Type.OBJECT,
        required=["entities", "relationships"],
        properties={
            "entities": genai.types.Schema(
                type=genai.types.Type.ARRAY,
                items=genai.types.Schema(type=genai.types.Type.STRING),
            ),
            "relationships": genai.types.Schema(
                type=genai.types.Type.ARRAY,
                items=genai.types.Schema(
                    type=genai.types.Type.OBJECT,
                    required=["source", "target", "type"],
                    properties={
                        "source": genai.types.Schema(type=genai.types.Type.STRING),
                        "target": genai.types.Schema(type=genai.types.Type.STRING),
                        "type": genai.types.Schema(type=genai.types.Type.STRING),
                    },
                ),
            ),
        },
    )


def get_grounding_check_schema():
    """Convert GroundingCheck to Gemini Schema format."""
    return genai.types.Schema(
        type=genai.types.Type.OBJECT,
        required=["grounded", "evidence"],
        properties={
            "grounded": genai.types.Schema(type=genai.types.Type.BOOLEAN),
            "evidence": genai.types.Schema(type=genai.types.Type.STRING),
        },
    )


def get_plausibility_check_schema():
    """Convert PlausibilityCheck to Gemini Schema format."""
    return genai.types.Schema(
        type=genai.types.Type.OBJECT,
        required=["plausible"],
        properties={
            "plausible": genai.types.Schema(
                type=genai.types.Type.STRING,
                enum=["yes", "no", "unclear"],
            ),
            "reasoning": genai.types.Schema(type=genai.types.Type.STRING),
        },
    )


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
        # Use a simple estimation based on connected nodes (GDS library not required)
        # Simple estimation: count isolated nodes + 1 for all connected nodes
        # This is accurate if the graph has one main connected component
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
        isolated_count = total_nodes - connected_count
        # If all nodes are connected, we have 1 component; otherwise isolated nodes are separate
        if connected_count == total_nodes:
            component_count = 1
        else:
            # Connected nodes form at least 1 component, isolated nodes are separate components
            component_count = max(1, isolated_count) + (1 if connected_count > 0 else 0)

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
    """Evaluates document coverage using LLM to extract entities/relations from chunks."""

    def __init__(self, graph: Neo4jGraph, chunks_dir: Path, markdown_dir: Path):
        self.graph = graph
        self.chunks_dir = chunks_dir
        self.markdown_dir = markdown_dir
        self.genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def _extract_entities_and_relations_from_chunk(
        self, chunk_content: str
    ) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
        """
        Use LLM to extract both entities and relationships from a chunk in a single call.
        This reduces LLM requests by 50% compared to separate calls.
        """
        system_instruction = (
            "You are analyzing a scientific document chunk. Extract entities and relationships.\n\n"
            "RULES:\n"
            "1. Entities: List of key scientific entities explicitly mentioned (models, methods, concepts, tasks, algorithms, publications, persons, organizations).\n"
            "2. Relationships: List of dictionaries, each with exactly three keys: 'source', 'target', and 'type'.\n"
            "3. Use entity names exactly as they appear in the text.\n"
            "4. Only include relationships that are explicitly stated in the text.\n"
            "5. If no entities or relationships are found, return empty arrays.\n"
            "6. All relationship dictionaries MUST have 'source', 'target', and 'type' keys."
        )

        user_content = f"Extract entities and relationships from this text:\n\n{chunk_content[:4000]}"

        # Combine system instruction and user content into a single prompt
        full_prompt = f"{system_instruction}\n\n{user_content}"

        try:
            response = self.genai_client.models.generate_content(
                model=MODEL,
                contents=full_prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=get_entity_and_relation_list_schema(),
                ),
            )

            # Parse JSON response from Gemini
            try:
                content = response.text
                parsed_json = json.loads(content)
                data = EntityAndRelationList.model_validate(parsed_json)
            except (json.JSONDecodeError, AttributeError, ValueError) as e:
                logger.warning(f"Invalid JSON response from LLM: {e}")
                logger.debug(
                    f"Response content: {getattr(response, 'text', 'No text')[:500]}"
                )
                time.sleep(LLM_REQUEST_DELAY)
                return set(), set()

            # Extract entities
            entities = set(e.strip() for e in data.entities if e.strip())

            # Extract relations - validate and filter
            relations = set()
            for rel in data.relationships:
                # Validate that rel is a dict and has required keys
                if isinstance(rel, dict) and all(
                    k in rel for k in ["source", "target", "type"]
                ):
                    source = str(rel.get("source", "")).strip()
                    target = str(rel.get("target", "")).strip()
                    rel_type = str(rel.get("type", "")).strip()
                    if source and target and rel_type:  # Only add if all are non-empty
                        relations.add((source, rel_type, target))

            # Rate limiting: add delay after LLM request
            time.sleep(LLM_REQUEST_DELAY)

            return entities, relations
        except Exception as e:
            logger.warning(f"Failed to extract entities and relations from chunk: {e}")
            time.sleep(LLM_REQUEST_DELAY)  # Still add delay even on error
            return set(), set()

    def _extract_entities_from_chunk(self, chunk_content: str) -> Set[str]:
        """Legacy method for backward compatibility. Use _extract_entities_and_relations_from_chunk instead."""
        entities, _ = self._extract_entities_and_relations_from_chunk(chunk_content)
        return entities

    def _extract_relations_from_chunk(
        self, chunk_content: str
    ) -> Set[Tuple[str, str, str]]:
        """Legacy method for backward compatibility. Use _extract_entities_and_relations_from_chunk instead."""
        _, relations = self._extract_entities_and_relations_from_chunk(chunk_content)
        return relations

    def _get_extracted_entities_for_chunk(self, chunk_id: str) -> Set[str]:
        """Get entities extracted from graph for a specific chunk."""
        try:
            # Try multiple approaches to find nodes linked to this chunk
            # Approach 1: Check if nodes have source_id property
            result = self.graph.query(
                """
                MATCH (n)
                WHERE n.source_id CONTAINS $chunk_id
                RETURN DISTINCT n.id as entity_id
                """,
                params={"chunk_id": chunk_id},
            )
            entities = set(
                row.get("entity_id", "") for row in result if row.get("entity_id")
            )

            # Note: LangChain Neo4j doesn't create :EXTRACTED relationships.
            # We rely on node.source_id properties instead (Approach 1 above).

            return entities
        except Exception as e:
            logger.warning(f"Failed to query entities for chunk {chunk_id}: {e}")
            return set()

    def _get_extracted_relations_for_chunk(
        self, chunk_id: str
    ) -> Set[Tuple[str, str, str]]:
        """Get relationships extracted from graph for a specific chunk."""
        try:
            # Find relations where both source and target nodes are linked to this chunk
            # Approach 1: Via source_id on nodes
            result = self.graph.query(
                """
                MATCH (source)-[r]->(target)
                WHERE (source.source_id CONTAINS $chunk_id OR target.source_id CONTAINS $chunk_id)
                RETURN DISTINCT source.id as source_id, type(r) as rel_type, target.id as target_id
                """,
                params={"chunk_id": chunk_id},
            )
            relations = set()
            for row in result:
                s = row.get("source_id", "")
                t = row.get("target_id", "")
                r_type = row.get("rel_type", "")
                if s and t and r_type:
                    relations.add((s.strip(), r_type.strip(), t.strip()))

            # Note: LangChain Neo4j doesn't create :EXTRACTED relationships.
            # We rely on node.source_id properties instead (Approach 1 above).

            return relations
        except Exception as e:
            logger.warning(f"Failed to query relations for chunk {chunk_id}: {e}")
            return set()

    def evaluate(
        self,
        document_id: str,
        chunks_filename: Optional[str] = None,
        sample_size: int = 5,  # Reduced from 10 to minimize LLM requests
    ) -> Dict[str, float]:
        """
        Compute coverage metrics:
        - Entity Coverage Score: |Extracted ∩ Mentioned| / |Mentioned|
        - Relationship Coverage Score: |Extracted ∩ Mentioned| / |Mentioned|
        """
        logger.info(f"Computing coverage metrics for document: {document_id}")

        # Determine chunks file
        if not chunks_filename:
            # Try to infer from document_id
            # document_id might be "attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k"
            # chunks file would be "attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k.jsonl"
            if document_id.endswith(".jsonl"):
                chunks_filename = document_id
            else:
                chunks_filename = f"{document_id}.jsonl"

        chunks_path = self.chunks_dir / chunks_filename
        if not chunks_path.exists():
            logger.warning(f"Chunks file not found: {chunks_path}")
            return {"entity_coverage": 0.0, "relation_coverage": 0.0}

        # Load and sample chunks
        chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))

        if not chunks:
            logger.warning(f"No chunks found in {chunks_filename}")
            return {"entity_coverage": 0.0, "relation_coverage": 0.0}

        # Sample chunks
        sample_size = min(sample_size, len(chunks))
        sampled_chunks = random.sample(chunks, sample_size)
        logger.info(f"Sampling {sample_size} chunks from {len(chunks)} total chunks")

        entity_coverage_scores = []
        relation_coverage_scores = []

        for i, chunk in enumerate(sampled_chunks, 1):
            chunk_id = chunk.get("id", "")
            chunk_content = chunk.get("content", "")

            if not chunk_content:
                continue

            logger.info(f"Processing chunk {i}/{sample_size}: {chunk_id}")

            # Extract entities/relations using LLM (single combined call)
            mentioned_entities, mentioned_relations = (
                self._extract_entities_and_relations_from_chunk(chunk_content)
            )

            # Get extracted entities/relations from graph
            extracted_entities = self._get_extracted_entities_for_chunk(chunk_id)
            extracted_relations = self._get_extracted_relations_for_chunk(chunk_id)

            # Calculate coverage for this chunk
            # Normalize entity names for comparison (case-insensitive, strip whitespace)
            def normalize_entity(e: str) -> str:
                return e.strip().lower()

            normalized_mentioned_entities = {
                normalize_entity(e) for e in mentioned_entities
            }
            normalized_extracted_entities = {
                normalize_entity(e) for e in extracted_entities
            }

            if normalized_mentioned_entities:
                intersection = (
                    normalized_mentioned_entities & normalized_extracted_entities
                )
                coverage = len(intersection) / len(normalized_mentioned_entities)
                entity_coverage_scores.append(coverage)
                logger.debug(
                    f"Chunk {chunk_id}: {len(intersection)}/{len(normalized_mentioned_entities)} entities covered"
                )

            if mentioned_relations:
                # Normalize relations for comparison
                def normalize_relation(
                    rel: Tuple[str, str, str],
                ) -> Tuple[str, str, str]:
                    return (
                        normalize_entity(rel[0]),
                        rel[1].strip().upper(),
                        normalize_entity(rel[2]),
                    )

                normalized_mentioned_rels = {
                    normalize_relation(r) for r in mentioned_relations
                }
                normalized_extracted_rels = {
                    normalize_relation(r) for r in extracted_relations
                }

                intersection = normalized_mentioned_rels & normalized_extracted_rels
                coverage = len(intersection) / len(normalized_mentioned_rels)
                relation_coverage_scores.append(coverage)
                logger.debug(
                    f"Chunk {chunk_id}: {len(intersection)}/{len(normalized_mentioned_rels)} relations covered"
                )

        # Aggregate metrics
        entity_coverage = (
            sum(entity_coverage_scores) / len(entity_coverage_scores)
            if entity_coverage_scores
            else 0.0
        )
        relation_coverage = (
            sum(relation_coverage_scores) / len(relation_coverage_scores)
            if relation_coverage_scores
            else 0.0
        )

        logger.info(
            f"Coverage metrics: entity={entity_coverage:.3f}, relation={relation_coverage:.3f}"
        )

        return {
            "entity_coverage": entity_coverage,
            "relation_coverage": relation_coverage,
        }


class FaithfulnessEvaluator:
    """Evaluates extraction faithfulness using LLM to check grounding in source text."""

    def __init__(self, graph: Neo4jGraph, chunks_dir: Path, markdown_dir: Path):
        self.graph = graph
        self.chunks_dir = chunks_dir
        self.markdown_dir = markdown_dir
        self.genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def _get_source_chunk_for_node(self, node_id: str) -> Optional[str]:
        """Get the source chunk text for a node."""
        try:
            # Try to find source_id from node property
            result = self.graph.query(
                """
                MATCH (n {id: $node_id})
                WHERE n.source_id IS NOT NULL
                RETURN n.source_id as source_id
                LIMIT 1
                """,
                params={"node_id": node_id},
            )
            source_id = None
            if result and result[0].get("source_id"):
                source_id = result[0]["source_id"]

            if not source_id:
                return None

            # Extract chunk_id from source_id format: "chunks_file::chunk_id" or "chunks_file_chunk_id"
            chunk_id = None
            if "::" in source_id:
                # Format: "chunks_file::chunk_id"
                chunk_id = source_id.split("::")[-1]
            else:
                # Format might be: "chunks_file_chunk_id" or just the chunk_id
                # Try to find matching chunks file and extract chunk_id
                for chunks_file in self.chunks_dir.glob("*.jsonl"):
                    file_stem = chunks_file.stem
                    if source_id.startswith(file_stem):
                        # Extract chunk_id by removing file stem prefix
                        remaining = source_id[len(file_stem) :].lstrip("_")
                        if remaining:
                            chunk_id = remaining
                            break

                # If no match found, assume source_id is the chunk_id itself
                if not chunk_id:
                    chunk_id = source_id

            # Search for chunk in all chunks files
            if chunk_id:
                for chunks_file in self.chunks_dir.glob("*.jsonl"):
                    with open(chunks_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                chunk = json.loads(line)
                                chunk_id_in_file = chunk.get("id", "")
                                # Match exact or if chunk_id is a suffix of the chunk id in file
                                if (
                                    chunk_id_in_file == chunk_id
                                    or chunk_id_in_file.endswith(f"_{chunk_id}")
                                    or chunk_id_in_file.endswith(f"::{chunk_id}")
                                ):
                                    return chunk.get("content", "")
        except Exception as e:
            logger.warning(f"Failed to get source chunk for node {node_id}: {e}")
        return None

    def _get_source_chunk_for_relation(
        self, source_id: str, target_id: str, rel_type: str
    ) -> Optional[str]:
        """Get the source chunk text for a relationship."""
        try:
            # Try to get source_id from source or target node
            result = self.graph.query(
                """
                MATCH (source {id: $source_id})-[r]->(target {id: $target_id})
                WHERE type(r) = $rel_type
                RETURN source.source_id as source_id, target.source_id as target_source_id
                LIMIT 1
                """,
                params={
                    "source_id": source_id,
                    "target_id": target_id,
                    "rel_type": rel_type,
                },
            )
            source_id_str = None
            if result:
                source_id_str = result[0].get("source_id") or result[0].get(
                    "target_source_id"
                )

            if not source_id_str:
                return None

            # Extract chunk_id from source_id format: "chunks_file::chunk_id" or "chunks_file_chunk_id"
            chunk_id = None
            if "::" in source_id_str:
                # Format: "chunks_file::chunk_id"
                chunk_id = source_id_str.split("::")[-1]
            else:
                # Format might be: "chunks_file_chunk_id" or just the chunk_id
                # Try to find matching chunks file and extract chunk_id
                for chunks_file in self.chunks_dir.glob("*.jsonl"):
                    file_stem = chunks_file.stem
                    if source_id_str.startswith(file_stem):
                        # Extract chunk_id by removing file stem prefix
                        remaining = source_id_str[len(file_stem) :].lstrip("_")
                        if remaining:
                            chunk_id = remaining
                            break

                # If no match found, assume source_id_str is the chunk_id itself
                if not chunk_id:
                    chunk_id = source_id_str

            # Search for chunk in all chunks files
            if chunk_id:
                for chunks_file in self.chunks_dir.glob("*.jsonl"):
                    with open(chunks_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                chunk = json.loads(line)
                                chunk_id_in_file = chunk.get("id", "")
                                # Match exact or if chunk_id is a suffix of the chunk id in file
                                if (
                                    chunk_id_in_file == chunk_id
                                    or chunk_id_in_file.endswith(f"_{chunk_id}")
                                    or chunk_id_in_file.endswith(f"::{chunk_id}")
                                ):
                                    return chunk.get("content", "")
        except Exception as e:
            logger.warning(
                f"Failed to get source chunk for relation {source_id}-[{rel_type}]->{target_id}: {e}"
            )
        return None

    def _check_node_grounding(self, node_id: str, node_type: str, text: str) -> bool:
        """Use LLM to check if a node is grounded in the text."""
        system_instruction = (
            "You are validating knowledge graph extractions. "
            "Is the entity explicitly stated or clearly implied in the provided text? "
            "Answer with 'yes' if the entity is mentioned or clearly implied, 'no' otherwise. "
            "Quote evidence if yes."
        )

        user_content = f"Entity: {node_id} (type: {node_type})\n\nText:\n{text[:4000]}"

        # Combine system instruction and user content into a single prompt
        full_prompt = f"{system_instruction}\n\n{user_content}"

        try:
            response = self.genai_client.models.generate_content(
                model=MODEL,
                contents=full_prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=get_grounding_check_schema(),
                ),
            )

            # Parse JSON response from Gemini
            try:
                content = response.text
                parsed_json = json.loads(content)
                data = GroundingCheck.model_validate(parsed_json)
            except (json.JSONDecodeError, AttributeError, ValueError) as e:
                logger.warning(f"Failed to parse grounding check response: {e}")
                time.sleep(LLM_REQUEST_DELAY)
                return False

            # Rate limiting: add delay after LLM request
            time.sleep(LLM_REQUEST_DELAY)
            return data.grounded
        except Exception as e:
            logger.warning(f"Failed to check grounding for node {node_id}: {e}")
            time.sleep(LLM_REQUEST_DELAY)  # Still add delay even on error
            return False

    def _check_relation_grounding(
        self, source_id: str, target_id: str, rel_type: str, text: str
    ) -> bool:
        """Use LLM to check if a relationship is grounded in the text."""
        system_instruction = (
            "You are validating knowledge graph extractions. "
            "Does the text explicitly support the relation (A —[R]→ B)? "
            "Answer with 'yes' if the relation is explicitly stated, 'no' otherwise. "
            "Quote the supporting sentence if yes, or say 'not supported' if no."
        )

        user_content = (
            f"Relation: {source_id} —[{rel_type}]→ {target_id}\n\nText:\n{text[:4000]}"
        )

        # Combine system instruction and user content into a single prompt
        full_prompt = f"{system_instruction}\n\n{user_content}"

        try:
            response = self.genai_client.models.generate_content(
                model=MODEL,
                contents=full_prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=get_grounding_check_schema(),
                ),
            )

            # Parse JSON response from Gemini
            try:
                content = response.text
                parsed_json = json.loads(content)
                data = GroundingCheck.model_validate(parsed_json)
            except (json.JSONDecodeError, AttributeError, ValueError) as e:
                logger.warning(f"Failed to parse grounding check response: {e}")
                time.sleep(LLM_REQUEST_DELAY)
                return False

            # Rate limiting: add delay after LLM request
            time.sleep(LLM_REQUEST_DELAY)
            return data.grounded
        except Exception as e:
            logger.warning(
                f"Failed to check grounding for relation {source_id}-[{rel_type}]->{target_id}: {e}"
            )
            time.sleep(LLM_REQUEST_DELAY)  # Still add delay even on error
            return False

    def evaluate(self, document_id: str, sample_size: int = 25) -> Dict[str, float]:
        # Reduced from 50 to 25 to minimize LLM requests (50% reduction)
        """
        Compute faithfulness metrics:
        - Node Faithfulness: % of nodes with positive grounding
        - Relation Faithfulness: % of relations grounded
        """
        logger.info(f"Computing faithfulness metrics for document: {document_id}")

        # Sample nodes from graph
        try:
            nodes_result = self.graph.query(
                """
                MATCH (n)
                WHERE n.id IS NOT NULL
                RETURN n.id as node_id, 
                       COALESCE(n.type, labels(n)[0], "Unknown") as node_type
                LIMIT $limit
                """,
                params={
                    "limit": sample_size * 2
                },  # Get more to account for missing chunks
            )
            nodes = [
                (row.get("node_id"), row.get("node_type", "Unknown"))
                for row in nodes_result
                if row.get("node_id")
            ]
            random.shuffle(nodes)
            nodes = nodes[:sample_size]
        except Exception as e:
            logger.warning(f"Failed to sample nodes: {e}")
            nodes = []

        # Sample relationships from graph
        try:
            rels_result = self.graph.query(
                """
                MATCH (source)-[r]->(target)
                WHERE source.id IS NOT NULL AND target.id IS NOT NULL
                RETURN source.id as source_id, type(r) as rel_type, target.id as target_id
                LIMIT $limit
                """,
                params={"limit": sample_size * 2},
            )
            relations = [
                (row.get("source_id"), row.get("rel_type"), row.get("target_id"))
                for row in rels_result
                if all(row.get(k) for k in ["source_id", "rel_type", "target_id"])
            ]
            random.shuffle(relations)
            relations = relations[:sample_size]
        except Exception as e:
            logger.warning(f"Failed to sample relations: {e}")
            relations = []

        logger.info(f"Sampling {len(nodes)} nodes and {len(relations)} relations")

        # Check node grounding
        grounded_nodes = 0
        checked_nodes = 0
        nodes_without_chunks = 0
        for node_id, node_type in nodes:
            text = self._get_source_chunk_for_node(node_id)
            if text:
                if self._check_node_grounding(node_id, node_type, text):
                    grounded_nodes += 1
                checked_nodes += 1
                # Rate limiting handled in _check_node_grounding
                if checked_nodes % 10 == 0:
                    logger.info(f"Checked {checked_nodes}/{len(nodes)} nodes...")
            else:
                nodes_without_chunks += 1
                logger.debug(f"Node {node_id} has no source chunk available")

        if nodes_without_chunks > 0:
            logger.warning(
                f"{nodes_without_chunks} nodes had no source chunks available for grounding check"
            )

        node_faithfulness = grounded_nodes / checked_nodes if checked_nodes > 0 else 0.0

        # Check relation grounding
        grounded_rels = 0
        checked_rels = 0
        rels_without_chunks = 0
        for source_id, rel_type, target_id in relations:
            text = self._get_source_chunk_for_relation(source_id, target_id, rel_type)
            if text:
                if self._check_relation_grounding(source_id, target_id, rel_type, text):
                    grounded_rels += 1
                checked_rels += 1
                # Rate limiting handled in _check_relation_grounding
                if checked_rels % 10 == 0:
                    logger.info(f"Checked {checked_rels}/{len(relations)} relations...")
            else:
                rels_without_chunks += 1
                logger.debug(
                    f"Relation {source_id}-[{rel_type}]->{target_id} has no source chunk available"
                )

        if rels_without_chunks > 0:
            logger.warning(
                f"{rels_without_chunks} relations had no source chunks available for grounding check"
            )

        relation_faithfulness = (
            grounded_rels / checked_rels if checked_rels > 0 else 0.0
        )

        logger.info(
            f"Faithfulness metrics: node={node_faithfulness:.3f} ({grounded_nodes}/{checked_nodes}), "
            f"relation={relation_faithfulness:.3f} ({grounded_rels}/{checked_rels})"
        )

        return {
            "node_faithfulness": node_faithfulness,
            "relation_faithfulness": relation_faithfulness,
        }


class SemanticEvaluator:
    """Evaluates semantic plausibility using LLM to check relation-type compatibility."""

    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
        self.genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def _check_relation_plausibility(
        self,
        source_id: str,
        source_type: str,
        target_id: str,
        target_type: str,
        rel_type: str,
    ) -> str:
        """Use LLM to check if a relation is semantically plausible."""
        system_instruction = (
            "You are validating knowledge graph relations. "
            "Given the entity types, is this relation semantically plausible in scientific literature? "
            "Answer with 'yes' if plausible, 'no' if implausible, or 'unclear' if uncertain. "
            "You are NOT checking if it's true, only if it's reasonable given the entity types."
        )

        user_content = (
            f"Source: {source_id} (type: {source_type})\n"
            f"Relation: {rel_type}\n"
            f"Target: {target_id} (type: {target_type})\n\n"
            "Is this relation semantically plausible?"
        )

        # Combine system instruction and user content into a single prompt
        full_prompt = f"{system_instruction}\n\n{user_content}"

        try:
            response = self.genai_client.models.generate_content(
                model=MODEL,
                contents=full_prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=get_plausibility_check_schema(),
                ),
            )

            # Parse JSON response from Gemini
            try:
                content = response.text
                parsed_json = json.loads(content)
                data = PlausibilityCheck.model_validate(parsed_json)
            except (json.JSONDecodeError, AttributeError, ValueError) as e:
                logger.warning(f"Failed to parse plausibility check response: {e}")
                time.sleep(LLM_REQUEST_DELAY)
                return "unclear"

            # Rate limiting: add delay after LLM request
            time.sleep(LLM_REQUEST_DELAY)
            return data.plausible.lower()
        except Exception as e:
            logger.warning(
                f"Failed to check plausibility for relation {source_id}-[{rel_type}]->{target_id}: {e}"
            )
            time.sleep(LLM_REQUEST_DELAY)  # Still add delay even on error
            return "unclear"

    def evaluate(self, sample_size: int = 50) -> Dict[str, float]:
        # Reduced from 100 to 50 to minimize LLM requests (50% reduction)
        """
        Compute semantic plausibility:
        - Plausibility Score = yes / (yes + no)
        """
        logger.info("Computing semantic plausibility metrics...")

        # Sample relationships from graph
        try:
            rels_result = self.graph.query(
                """
                MATCH (source)-[r]->(target)
                WHERE source.id IS NOT NULL AND target.id IS NOT NULL
                RETURN source.id as source_id, 
                       COALESCE(source.type, labels(source)[0], "Unknown") as source_type,
                       type(r) as rel_type,
                       target.id as target_id, 
                       COALESCE(target.type, labels(target)[0], "Unknown") as target_type
                LIMIT $limit
                """,
                params={"limit": sample_size * 2},
            )
            relations = [
                (
                    row.get("source_id"),
                    row.get("source_type", "Unknown"),
                    row.get("rel_type"),
                    row.get("target_id"),
                    row.get("target_type", "Unknown"),
                )
                for row in rels_result
                if all(row.get(k) for k in ["source_id", "rel_type", "target_id"])
            ]
            random.shuffle(relations)
            relations = relations[:sample_size]
        except Exception as e:
            logger.warning(f"Failed to sample relations: {e}")
            relations = []

        logger.info(f"Sampling {len(relations)} relations for plausibility check")

        yes_count = 0
        no_count = 0
        unclear_count = 0

        for i, (source_id, source_type, rel_type, target_id, target_type) in enumerate(
            relations, 1
        ):
            result = self._check_relation_plausibility(
                source_id, source_type, target_id, target_type, rel_type
            )
            if result == "yes":
                yes_count += 1
            elif result == "no":
                no_count += 1
            else:
                unclear_count += 1

            if i % 20 == 0:
                logger.info(f"Checked {i}/{len(relations)} relations...")

        # Calculate plausibility score: yes / (yes + no)
        # Unclear responses are excluded from the denominator
        total_decided = yes_count + no_count
        plausibility_score = yes_count / total_decided if total_decided > 0 else 0.0

        logger.info(
            f"Plausibility metrics: yes={yes_count}, no={no_count}, unclear={unclear_count}, "
            f"score={plausibility_score:.3f}"
        )

        return {
            "semantic_plausibility": plausibility_score,
        }


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

    def _validate_document(
        self, document_id: str, chunks_filename: Optional[str] = None
    ) -> Dict:
        """
        Internal method to validate a single document's knowledge graph.

        Args:
            document_id: Identifier for the document (chunks filename stem without .jsonl extension).
            chunks_filename: Optional chunks filename (with .jsonl extension) for coverage evaluation.

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
        if structural_metrics.get("component_ratio", 0) > 0.5:
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

    def validate_all_documents(self) -> Dict:
        """
        Validate the entire knowledge graph in Neo4j.

        Since all graph files are ingested into the same Neo4j database,
        we validate the entire graph once rather than per file.

        Returns:
            Single validation report dictionary for the entire knowledge graph
        """
        logger.info("Validating entire knowledge graph...")

        # Use a generic document_id for the entire graph
        document_id = "knowledge_graph"

        # For coverage evaluation, we'll use the first available chunks file
        # or sample from multiple files if needed
        chunks_filename = None

        # Try to find a chunks file for coverage evaluation (if LLM tests enabled)
        if self.enable_llm_tests:
            chunks_files = list(self.chunks_dir.glob("*.jsonl"))
            if chunks_files:
                # Use the first chunks file found (or could aggregate across all)
                chunks_filename = chunks_files[0].name
                logger.info(
                    f"Using chunks file for coverage evaluation: {chunks_filename}"
                )

        try:
            report = self._validate_document(document_id, chunks_filename)
            return report
        except Exception as e:
            logger.error(f"Error validating knowledge graph: {e}", exc_info=True)
            # Create error report
            return {
                "document_id": document_id,
                "metrics": {},
                "notes": [f"Validation failed with error: {str(e)}"],
            }


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
    # Use a consistent filename for the knowledge graph validation report
    output_file = output_dir / "knowledge_graph_validation_report.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Validation report saved to: {output_file}")


def main():
    """
    Main entry point for validator agent.

    Validates the entire knowledge graph in Neo4j database (single report).

    Command-line arguments:
    --enable-llm-tests: Flag. When set, enables LLM-based evaluation tests:
        - Coverage evaluation (entity and relationship coverage)
        - Faithfulness evaluation (node and relationship grounding)
        - Semantic evaluation (semantic plausibility)

    --output-dir: Optional. Directory path for saving validation reports. Default:
        "validation_outputs/" (relative to project root). Reports are saved as:
        <document_id>_validation_report.json

    Environment variables required (via .env file):
        NEO4J_URI: Neo4j connection URI (default: "bolt://localhost:7687")
        NEO4J_USERNAME: Neo4j username (default: "neo4j")
        NEO4J_PASSWORD: Neo4j password (default: "password")
        GOOGLE_API_KEY: Google API key for Gemini (required if --enable-llm-tests is set)

    Examples:
        # Validate all documents (structural tests only)
        python agents/7-validator_agent.py

        # Validate all documents with LLM tests enabled
        python agents/7-validator_agent.py --enable-llm-tests

        # Custom output directory
        python agents/7-validator_agent.py --output-dir "custom_reports"
    """
    parser = argparse.ArgumentParser(
        description="Knowledge Graph Validator Agent - Validates all graphs in knowledge_graph_outputs/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate all documents (structural tests only)
  python agents/7-validator_agent.py

  # Validate all documents with LLM tests enabled
  python agents/7-validator_agent.py --enable-llm-tests
        """,
    )
    parser.add_argument(
        "--enable-llm-tests",
        action="store_true",
        help=(
            "Enable LLM-based evaluation tests (coverage, faithfulness, semantic). "
            "Requires GOOGLE_API_KEY environment variable."
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

        # Validate the entire knowledge graph (single report)
        report = validator.validate_all_documents()

        # Save the single report
        save_report(report, output_dir)

        print(f"\nValidation complete. Generated 1 report.")
        print(f"Report saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
