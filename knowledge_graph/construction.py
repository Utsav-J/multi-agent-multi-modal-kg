import os
from google     import genai
from dotenv     import load_dotenv
from neo4j      import GraphDatabase
from models     import GraphDocument
from prompts    import render_graph_construction_instructions

load_dotenv()

sample_chunk = "This is vital because so many inves-\ntors seem to get into the right stocks but don’t know how to take a proﬁt or\nwhen to sell as a position goes against them. He discusses the psychological\nfactors that prevent most investors from cutting a loss. Can you believe that\nan investor of Mark’s stature is right only 50 percent of the time and has still\nmade a fortune! That is due to his use of risk management. I have always believed that to be a successful investor you have a large\ntuition bill to pay in hard losses at the University of Wall Street before you\ncan graduate and start making money. Mark supplies the best textbook on\ngrowth stock investing, and so you can avoid paying such a high tuition. With his help, a determined effort, and discipline, you can get an Ivy League\neducation for the cost of a hardcover book. If you are a seasoned inves-\ntor this book is a graduate-level class that will certainly add to your invest-\nment knowledge, as it has mine. Mark has also saved me a great deal of time\nbecause he wrote the book that I always wanted to write and did it better\nthan I ever could! Enjoy, and may you all have investment success!"

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

missing = [name for name, val in (
    ('NEO4J_URI', NEO4J_URI),
    ('NEO4J_USERNAME', NEO4J_USERNAME),
    ('NEO4J_PASSWORD', NEO4J_PASSWORD),
) if not val]
if missing:
    raise EnvironmentError(f"Missing required environment variable(s): {', '.join(missing)}")
else:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) #type:ignore

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=render_graph_construction_instructions(chunk=sample_chunk),
    config=genai.types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=GraphDocument,
    ),
)

graph_doc: GraphDocument = response.parsed # type: ignore

print(graph_doc)
print(type(graph_doc))
 