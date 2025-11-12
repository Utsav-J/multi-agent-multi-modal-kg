from pydantic import BaseModel, Field
from typing import List, Optional

class Node(BaseModel):
    """Represents an entity or concept node in the knowledge graph."""
    id: str = Field(..., description="Unique identifier or label of the node (e.g. 'Elon Reeve Musk').")
    type: str = Field(..., description="Type or category of the node (e.g. Person, Location, Company).")

class Relationship(BaseModel):
    """Represents a directed relationship between two nodes."""
    source: Node = Field(..., description="Source node of the relationship.")
    target: Node = Field(..., description="Target node of the relationship.")
    type: str = Field(..., description="Type of relationship (e.g. FOUNDED, ATTENDED, BORN_IN).")

class MetadataItem(BaseModel):
    """Closed schema key-value item to avoid additionalProperties in JSON Schema."""
    key: str = Field(..., description="Metadata key name.")
    value: str = Field(..., description="Metadata value as string.")

class Document(BaseModel):
    """Represents the original text or chunk from which nodes and relationships were extracted."""
    page_content: str = Field(..., description="Raw text content of the source document or chunk.")
    metadata: Optional[List[MetadataItem]] = Field(default_factory=list, description="Optional metadata as a list of key/value pairs.")

class GraphDocument(BaseModel):
    """Represents a graph extracted from one document or chunk."""
    nodes: List[Node] = Field(..., description="List of nodes (entities or concepts) identified in the document.")
    relationships: List[Relationship] = Field(..., description="List of relationships connecting the nodes.")
    source: Document = Field(..., description="The document or chunk this graph was extracted from.")
