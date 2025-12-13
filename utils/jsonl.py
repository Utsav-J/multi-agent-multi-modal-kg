(mas-for-multimodal-knowledge-graph) 
UTSAV@Utsav MINGW64 /e/Python Stuff/MAS-for-multimodal-knowledge-graph (main)
$ uv run agents/2-chunker_agent.py 
2025-12-12 20:47:09,897 - __main__ - INFO - Starting chunker agent with input: Chunk the file attention_is_all_you_need_annotated.md
2025-12-12 20:47:12,768 - __main__ - INFO - Tool invoked: chunk_markdown_tool for file 'attention_is_all_you_need_annotated.md'
2025-12-12 20:47:12,771 - __main__ - INFO - Writing 10 chunks to E:\Python Stuff\MAS-for-multimodal-knowledge-graph\chunking_outputs\attention_is_all_you_need_annotated_chunks.jsonl
2025-12-12 20:47:14,326 - __main__ - INFO - Agent Result: The markdown file "attention_is_all_you_need_annotated.md" has been successfully chunked. The output is saved to `attention_is_all_you_need_annotated_chunks.jsonl`.
(mas-for-multimodal-knowledge-graph) 
UTSAV@Utsav MINGW64 /e/Python Stuff/MAS-for-multimodal-knowledge-graph (main)
$ uv run agents/3-graph_data_extractor_agent.py 
2025-12-12 21:42:56,297 - __main__ - INFO - Starting graph construction agent with input: Extract graph from rag_paper_annotated_chunks.jsonl and include metadata
2025-12-12 21:42:58,747 - __main__ - INFO - Tool invoked: extract_graph_from_chunks_tool for file 'attention_is_all_you_need_paper_annotated_chunks.jsonl' with include_metadata=False, batch_size=1, token_limit=5500
2025-12-12 21:42:58,748 - __main__ - ERROR - Error: Input file not found at E:\Python Stuff\MAS-for-multimodal-knowledge-graph\chunking_outputs\attention_is_all_you_need_paper_annotated_chunks.jsonl
2025-12-12 21:43:00,375 - __main__ - INFO - Agent Result: [{'type': 'text', 'text': "I'm sorry, but I was unable to find the file `attention_is_all_you_need_paper_annotated_chunks.jsonl` in the `chunking_outputs` directory. Please make sure the file is present in the correct location and try again.", 'extras': {'signature': 'CqoCAXLI2nxKmkLu8VibofH/AQusLog0D8BsZhSD8QtRSJkPJ/3PwR2Z3c4zkRZ01yLlPozxPLlysU8jQlNMJ/H5lNY+0ECRMq0ujphwexoYlspgo2zrHbob0X1/1/5UveKVwn17uLebBGsxKod19YUlEsAJirCHii7QuwTHeeUdY0/+FvEAC3tygBC/KOJbW5KQNnqM2YWdE4m5DHnk18I6eh2Y3NfvwfJEotbJbn/ttaljXdyrBqursmzd4KyVlEHG9XNRVa+Y5Pd3kl0376bAnGU3ZcsScClVfVaIzhgcduWUpxpyq48giO02A9OwF/IMK8oRryldvt1MGeOLFKhHzVE3a/ebonFBKU+dYzzV4ozZvwLACxYIRx1OdJWwu/DWBIwLDBxR8FcKtA=='}}]
(mas-for-multimodal-knowledge-graph) 
UTSAV@Utsav MINGW64 /e/Python Stuff/MAS-for-multimodal-knowledge-graph (main)
$ uv run agents/3-graph_data_extractor_agent.py 
2025-12-12 21:43:46,123 - __main__ - INFO - Starting graph construction agent with input: Extract graph from rag_paper_annotated_chunks.jsonl and include metadata
2025-12-12 21:43:48,322 - __main__ - INFO - Tool invoked: extract_graph_from_chunks_tool for file 'attention_is_all_you_need_annotated_chunks.jsonl' with include_metadata=False, batch_size=1, token_limit=5500
2025-12-12 21:43:49,847 - __main__ - INFO - Processing 10 chunks from attention_is_all_you_need_annotated_chunks.jsonl
2025-12-12 21:43:49,848 - __main__ - INFO - Processing batch of 4 chunks (IDs: ['attention_is_all_you_need_annotated_0', 'attention_is_all_you_need_annotated_1', 'attention_is_all_you_need_annotated_2', 'attention_is_all_you_need_annotated_3'])
2025-12-12 21:43:49,876 - google_genai.models - INFO - AFC is enabled with max remote calls: 10.
2025-12-12 21:46:04,529 - httpx - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent "HTTP/1.1 200 OK"
2025-12-12 21:46:04,640 - __main__ - INFO - Processing batch of 4 chunks (IDs: ['attention_is_all_you_need_annotated_4', 'attention_is_all_you_need_annotated_5', 'attention_is_all_you_need_annotated_6', 'attention_is_all_you_need_annotated_7'])
2025-12-12 21:46:04,640 - google_genai.models - INFO - AFC is enabled with max remote calls: 10.
2025-12-12 21:47:57,319 - httpx - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent "HTTP/1.1 200 OK"
2025-12-12 21:47:57,335 - __main__ - INFO - Processing batch of 2 chunks (IDs: ['attention_is_all_you_need_annotated_8', 'attention_is_al2025-12-12 21:4220222025-12022025-12-12 21:47:57,335 - google_genai.models - INFO - AFC is enabled with max remote calls: 10.
2025-12-12 21:47:57,335 - google_genai.models - INFO - AFC is enabled with max remote calls: 10.
2025-12-12 21:51:06,489 - httpx - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent "HTTP/1.1 200 OK"  
2025-12-12 21:51:08,120 - __main__ - INFO - Agent Result: Successfully extracted the knowledge graph. The graph data has been saved to `attention_is_all_you_need_annotated_chunks_graph.jsonl`.
(mas-for-multimodal-knowledge-graph)
UTSAV@Utsav MINGW64 /e/Python Stuff/MAS-for-multimodal-knowledge-graph (main)
$ uv run agents/5-jsonl_graph_ingestion_agent.py 
1
1
Connected to Neo4j.
1
Processing file: E:\Python Stuff\MAS-for-multimodal-knowledge-graph\knowledge_graph_outputs\attention_is_all_you_need_annotated_chunks_graph.jsonl
Ingesting 3 documents into Neo4j...
Ingestion complete.
(mas-for-multimodal-knowledge-graph) 
UTSAV@Utsav MINGW64 /e/Python Stuff/MAS-for-multimodal-knowledge-graph (main)
$ git add .
warning: in the working copy of 'agents/3-graph_data_extractor_agent.py', LF will be replaced by CRLF the next time Git touches it
(mas-for-multimodal-knowledge-graph) 
UTSAV@Utsav MINGW64 /e/Python Stuff/MAS-for-multimodal-knowledge-graph (main)
$ git commit -m "increased chunk size and overlap, basic pipeline functional"
[main f5a4ced] increased chunk size and overlap, basic pipeline functional
 5 files changed, 19 insertions(+), 88 deletions(-)
(mas-for-multimodal-knowledge-graph) 
UTSAV@Utsav MINGW64 /e/Python Stuff/MAS-for-multimodal-knowledge-graph (main)
$