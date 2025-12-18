import json
from typing import List
from groq import Groq
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


# 1. Define the schema using Pydantic
class SearchQueries(BaseModel):
    queries: List[str] = Field(
        description="A list of 3-5 diverse, concise search queries."
    )


def generate_rag_subqueries(user_query: str, client: Groq = Groq()) -> List[str]:
    """
    Use Groq to rewrite the user query into multiple retrieval-optimized
    subqueries using the json_schema response format.
    """
    # groq_client = Groq()
    prompt = (
        "You are a retrieval query optimization module sitting in front of a vector database.\n"
        "\n"
        "Your ONLY job is to turn a single user question into a SMALL SET of search queries that will\n"
        "retrieve the most relevant chunks from a *scientific and technical* corpus.\n"
        "\n"
        "STRICT RULES:\n"
        "- DO NOT answer the question.\n"
        "- DO NOT explain your reasoning.\n"
        "- DO NOT invent new concepts, entities, models, tasks, or paper titles that are not "
        "  implied by the user question.\n"
        "- Preserve key technical terms (e.g. model names, method names, dataset names, acronyms) exactly\n"
        "  as they appear in the user question whenever possible.\n"
        "- If the question is ambiguous, generate variants that cover the most plausible interpretations,\n"
        "  but still stay conservative and close to the original wording.\n"
        "- Favor *recall-friendly*, retrieval-oriented queries over chatty or conversational rephrasings.\n"
        "- Each query must be standalone and make sense without prior context.\n"
        "- Maximum 5 queries. If the question is very narrow, 1-3 high-quality queries is enough.\n"
        "\n"
        "OUTPUT FORMAT:\n"
        "- You MUST return your output in the JSON schema provided (SearchQueries), and nothing else.\n"
        "- The 'queries' field should contain only the final list of optimized search queries.\n"
    )

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",  # Or your preferred Groq model
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_query},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "search_queries",
                    "schema": SearchQueries.model_json_schema(),
                },
            },
        )

        # 4. Parse and validate the response
        content = response.choices[0].message.content
        structured_data = SearchQueries.model_validate(json.loads(content))

        seen = set()
        unique_queries = []
        for q in structured_data.queries:
            if q and q not in seen:
                seen.add(q)
                unique_queries.append(q)

        return unique_queries or [user_query]

    except Exception as e:
        print(f"Error during structured output generation: {e}")
        return [user_query]


if __name__ == "__main__":

    # query = "What are the latest advancements in perovskite solar cell stability?"
    query = "what is attention mechanism"
    sub_queries = generate_rag_subqueries(query)
    print(type(sub_queries))
    print("User Query:", query)
    print("\nGenerated Sub-queries:")
    for i, q in enumerate(sub_queries, 1):
        print(f"{i}. {q}")
