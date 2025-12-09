import os
from google import genai
from dotenv import load_dotenv
from neo4j import GraphDatabase
from models import GraphDocument
from prompts import render_graph_construction_instructions

load_dotenv()

sample_chunk = "**Description:**\n- The system receives input data consisting of text, code, images, audio, and video.\n- A retriever component, which can operate on dense or sparse data, selects relevant information from the input data.\n- A generator component then utilizes the retrieved information to produce a result in the same multimodal format.\n- The output consists of text, code, images, audio, and video, mirroring the input.\n```\n\n```\n### Figure - RAG Architecture with and without Retrieval-Augmented Generation\n![RAG architecture diagram](rag_paper.pdf-1-1.png)\n\n**Caption:** This figure illustrates the difference between a Large Language Model (LLM) query without and with Retrieval-Augmented Generation (RAG). The RAG approach utilizes external data to enhance the LLM's response."
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

missing = [
    name
    for name, val in (
        ("NEO4J_URI", NEO4J_URI),
        ("NEO4J_USERNAME", NEO4J_USERNAME),
        ("NEO4J_PASSWORD", NEO4J_PASSWORD),
    )
    if not val
]
if missing:
    raise EnvironmentError(
        f"Missing required environment variable(s): {', '.join(missing)}"
    )
else:
    driver = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )  # type:ignore

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=render_graph_construction_instructions(chunk=sample_chunk),
    config=genai.types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=GraphDocument,
    ),
)

graph_doc: GraphDocument = response.parsed  # type: ignore

print(graph_doc)
print(type(graph_doc))
'''
nodes=[Node(id='System', type='Concept'), Node(id='Input Data', type='Data'), Node(id='Text', type='DataType'), Node(id='Code', type='DataType'), Node(id='Images', type='DataType'), Node(id='Audio', type='DataType'), Node(id='Video', type='DataType'), Node(id='Retriever Component', type='Component'), Node(id='Dense Data', type='DataType'), Node(id='Sparse Data', type='DataType'), Node(id='Generator Component', type='Component'), Node(id='Retrieved Information', type='Data'), Node(id='Result', type='Concept'), Node(id='Multimodal Format', type='Concept'), Node(id='Output', type='Data'), Node(id='Figure', type='Document Element'), Node(id='Large Language Model', type='Technology'), Node(id='LLM Query', type='Concept'), Node(id='Retrieval-Augmented Generation', type='Technology'), Node(id='External Data', type='Data'), Node(id='LLM Response', type='Concept')] relationships=[Relationship(source=Node(id='System', type='Concept'), target=Node(id='Input Data', type='Data'), type='RECEIVES'), Relationship(source=Node(id='Input Data', type='Data'), target=Node(id='Text', type='DataType'), type='CONTAINS'), Relationship(source=Node(id='Input Data', type='Data'), target=Node(id='Code', type='DataType'), type='CONTAINS'), Relationship(source=Node(id='Input Data', type='Data'), target=Node(id='Images', type='DataType'), type='CONTAINS'), Relationship(source=Node(id='Input Data', type='Data'), target=Node(id='Audio', type='DataType'), type='CONTAINS'), Relationship(source=Node(id='Input Data', type='Data'), target=Node(id='Video', type='DataType'), type='CONTAINS'), Relationship(source=Node(id='Retriever Component', type='Component'), target=Node(id='Dense Data', type='DataType'), type='OPERATES_ON'), Relationship(source=Node(id='Retriever Component', type='Component'), target=Node(id='Sparse Data', type='DataType'), type='OPERATES_ON'), Relationship(source=Node(id='Retriever Component', type='Component'), target=Node(id='Retrieved Information', type='Data'), type='SELECTS'), Relationship(source=Node(id='Retriever Component', type='Component'), target=Node(id='Input Data', type='Data'), type='SELECTS_FROM'), Relationship(source=Node(id='Generator Component', type='Component'), target=Node(id='Retrieved Information', type='Data'), type='UTILIZES'), Relationship(source=Node(id='Generator Component', type='Component'), target=Node(id='Result', type='Concept'), type='PRODUCES'), Relationship(source=Node(id='Result', type='Concept'), target=Node(id='Multimodal Format', type='Concept'), type='HAS_FORMAT'), Relationship(source=Node(id='Output', type='Data'), target=Node(id='Text', type='DataType'), type='CONSISTS_OF'), Relationship(source=Node(id='Output', type='Data'), target=Node(id='Code', type='DataType'), type='CONSISTS_OF'), Relationship(source=Node(id='Output', type='Data'), target=Node(id='Images', type='DataType'), type='CONSISTS_OF'), Relationship(source=Node(id='Output', type='Data'), target=Node(id='Audio', type='DataType'), type='CONSISTS_OF'), Relationship(source=Node(id='Output', type='Data'), target=Node(id='Video', type='DataType'), type='CONSISTS_OF'), Relationship(source=Node(id='Output', type='Data'), target=Node(id='Input Data', type='Data'), type='MIRRORS'), Relationship(source=Node(id='Figure', type='Document Element'), target=Node(id='Retrieval-Augmented Generation', type='Technology'), type='ILLUSTRATES_ARCHITECTURE_OF'), Relationship(source=Node(id='Retrieval-Augmented Generation', type='Technology'), target=Node(id='External Data', type='Data'), type='UTILIZES'), Relationship(source=Node(id='Retrieval-Augmented Generation', type='Technology'), target=Node(id='LLM Response', type='Concept'), type='ENHANCES')] source=Document(page_content="The system receives input data consisting of text, code, images, audio, and video. A retriever component, which can operate on dense or sparse data, selects relevant information from the input data. A generator component then utilizes the retrieved information to produce a result in the same multimodal format. The output consists of text, code, images, audio, and video, mirroring the input. ### Figure - RAG Architecture with and without Retrieval-Augmented Generation ![RAG architecture diagram](rag_paper.pdf-1-1.png) Caption: This figure illustrates the difference between a Large Language Model (LLM) query without and with Retrieval-Augmented Generation (RAG). The RAG approach utilizes external data to enhance the LLM's response.", metadata=None)
'''