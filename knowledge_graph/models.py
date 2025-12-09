from pydantic import BaseModel, Field
from typing import List, Optional


class Node(BaseModel):
    """Represents an entity or concept node in the knowledge graph."""

    id: str = Field(
        ...,
        description="Unique identifier or label of the node (e.g. 'Elon Reeve Musk').",
    )
    type: str = Field(
        ...,
        description="Type or category of the node (e.g. Person, Location, Company).",
    )


class Relationship(BaseModel):
    """Represents a directed relationship between two nodes."""

    source: Node = Field(..., description="Source node of the relationship.")
    target: Node = Field(..., description="Target node of the relationship.")
    type: str = Field(
        ..., description="Type of relationship (e.g. FOUNDED, ATTENDED, BORN_IN)."
    )


class MetadataItem(BaseModel):
    """Closed schema key-value item to avoid additionalProperties in JSON Schema."""

    key: str = Field(..., description="Metadata key name.")
    value: str = Field(..., description="Metadata value as string.")


class Document(BaseModel):
    """Represents the original text or chunk from which nodes and relationships were extracted."""

    page_content: str = Field(
        ..., description="Raw text content of the source document or chunk."
    )
    metadata: Optional[List[MetadataItem]] = Field(
        default_factory=list,
        description="Optional metadata as a list of key/value pairs.",
    )


class GraphDocument(BaseModel):
    """Represents a graph extracted from one document or chunk."""

    nodes: List[Node] = Field(
        ...,
        description="List of nodes (entities or concepts) identified in the document.",
    )
    relationships: List[Relationship] = Field(
        ..., description="List of relationships connecting the nodes."
    )
    source: Document = Field(
        ..., description="The document or chunk this graph was extracted from."
    )


extracted_sample_data = GraphDocument(
    nodes=[
        Node(id="System", type="Concept"),
        Node(id="Input Data", type="Data"),
        Node(id="Text", type="DataType"),
        Node(id="Code", type="DataType"),
        Node(id="Images", type="DataType"),
        Node(id="Audio", type="DataType"),
        Node(id="Video", type="DataType"),
        Node(id="Retriever Component", type="Component"),
        Node(id="Dense Data", type="DataType"),
        Node(id="Sparse Data", type="DataType"),
        Node(id="Generator Component", type="Component"),
        Node(id="Retrieved Information", type="Data"),
        Node(id="Result", type="Concept"),
        Node(id="Multimodal Format", type="Concept"),
        Node(id="Output", type="Data"),
        Node(id="Figure", type="Document Element"),
        Node(id="Large Language Model", type="Technology"),
        Node(id="LLM Query", type="Concept"),
        Node(id="Retrieval-Augmented Generation", type="Technology"),
        Node(id="External Data", type="Data"),
        Node(id="LLM Response", type="Concept"),
    ],
    relationships=[
        Relationship(
            source=Node(id="System", type="Concept"),
            target=Node(id="Input Data", type="Data"),
            type="RECEIVES",
        ),
        Relationship(
            source=Node(id="Input Data", type="Data"),
            target=Node(id="Text", type="DataType"),
            type="CONTAINS",
        ),
        Relationship(
            source=Node(id="Input Data", type="Data"),
            target=Node(id="Code", type="DataType"),
            type="CONTAINS",
        ),
        Relationship(
            source=Node(id="Input Data", type="Data"),
            target=Node(id="Images", type="DataType"),
            type="CONTAINS",
        ),
        Relationship(
            source=Node(id="Input Data", type="Data"),
            target=Node(id="Audio", type="DataType"),
            type="CONTAINS",
        ),
        Relationship(
            source=Node(id="Input Data", type="Data"),
            target=Node(id="Video", type="DataType"),
            type="CONTAINS",
        ),
        Relationship(
            source=Node(id="Retriever Component", type="Component"),
            target=Node(id="Dense Data", type="DataType"),
            type="OPERATES_ON",
        ),
        Relationship(
            source=Node(id="Retriever Component", type="Component"),
            target=Node(id="Sparse Data", type="DataType"),
            type="OPERATES_ON",
        ),
        Relationship(
            source=Node(id="Retriever Component", type="Component"),
            target=Node(id="Retrieved Information", type="Data"),
            type="SELECTS",
        ),
        Relationship(
            source=Node(id="Retriever Component", type="Component"),
            target=Node(id="Input Data", type="Data"),
            type="SELECTS_FROM",
        ),
        Relationship(
            source=Node(id="Generator Component", type="Component"),
            target=Node(id="Retrieved Information", type="Data"),
            type="UTILIZES",
        ),
        Relationship(
            source=Node(id="Generator Component", type="Component"),
            target=Node(id="Result", type="Concept"),
            type="PRODUCES",
        ),
        Relationship(
            source=Node(id="Result", type="Concept"),
            target=Node(id="Multimodal Format", type="Concept"),
            type="HAS_FORMAT",
        ),
        Relationship(
            source=Node(id="Output", type="Data"),
            target=Node(id="Text", type="DataType"),
            type="CONSISTS_OF",
        ),
        Relationship(
            source=Node(id="Output", type="Data"),
            target=Node(id="Code", type="DataType"),
            type="CONSISTS_OF",
        ),
        Relationship(
            source=Node(id="Output", type="Data"),
            target=Node(id="Images", type="DataType"),
            type="CONSISTS_OF",
        ),
        Relationship(
            source=Node(id="Output", type="Data"),
            target=Node(id="Audio", type="DataType"),
            type="CONSISTS_OF",
        ),
        Relationship(
            source=Node(id="Output", type="Data"),
            target=Node(id="Video", type="DataType"),
            type="CONSISTS_OF",
        ),
        Relationship(
            source=Node(id="Output", type="Data"),
            target=Node(id="Input Data", type="Data"),
            type="MIRRORS",
        ),
        Relationship(
            source=Node(id="Figure", type="Document Element"),
            target=Node(id="Retrieval-Augmented Generation", type="Technology"),
            type="ILLUSTRATES_ARCHITECTURE_OF",
        ),
        Relationship(
            source=Node(id="Retrieval-Augmented Generation", type="Technology"),
            target=Node(id="External Data", type="Data"),
            type="UTILIZES",
        ),
        Relationship(
            source=Node(id="Retrieval-Augmented Generation", type="Technology"),
            target=Node(id="LLM Response", type="Concept"),
            type="ENHANCES",
        ),
    ],
    source=Document(
        page_content="The system receives input data consisting of text, code, images, audio, and video. A retriever component, which can operate on dense or sparse data, selects relevant information from the input data. A generator component then utilizes the retrieved information to produce a result in the same multimodal format. The output consists of text, code, images, audio, and video, mirroring the input. ### Figure - RAG Architecture with and without Retrieval-Augmented Generation ![RAG architecture diagram](rag_paper.pdf-1-1.png) Caption: This figure illustrates the difference between a Large Language Model (LLM) query without and with Retrieval-Augmented Generation (RAG). The RAG approach utilizes external data to enhance the LLM's response.",
        metadata=None,
    ),
)
