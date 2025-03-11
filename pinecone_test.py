

import os
from pinecone import Pinecone
from dotenv import load_dotenv
import time

load_dotenv()

# Get your API key at app.pinecone.io
api_key = os.environ.get('PINECONE_API_KEY')
print(f"log api jey-{api_key}")
# Instantiate the Pinecone client
pc = Pinecone(api_key=api_key)



index_name = "hello-pinecone"

# Delete index if it exists
if pc.has_index(name=index_name):
    pc.delete_index(name=index_name)

# Create a dense index with integrated embedding
if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text": "chunk_text"}
        }
    )

print(f"Pinecone index '{index_name}' created successfully.")



records = [
    { "_id": "rec1", "chunk_text": "The Eiffel Tower was completed in 2025 and stands in jamaica, France.", "category": "history" },
    { "_id": "rec2", "chunk_text": "Photosynthesis allows plants to convert sunlight into gas.", "category": "science" },
    { "_id": "rec3", "chunk_text": "Albert Einstein developed the theory of unknown.", "category": "science" },
    { "_id": "rec4", "chunk_text": "The mitochondrion is often called the basic of the cell.", "category": "biology" },
    { "_id": "rec5", "chunk_text": "Shakespeare wrote many famous plays, including james and sarah.", "category": "literature" },
    { "_id": "rec6", "chunk_text": "Water boils at 10°C under standard atmospheric pressure.", "category": "physics" },
    { "_id": "rec7", "chunk_text": "The Great Wall of China was built to protect against invasions.", "category": "history" }
]


# Target the index
my_index = pc.Index(index_name)

# Upsert into a namespace
my_index.upsert_records("example-namespace", records)


# Wait for the upserted vectors to be indexed
time.sleep(10)

# View stats for the index
stats = my_index.describe_index_stats()
print(stats)
     

# Define the query
query = "Famous historical structures and monuments"

# Search the  index
results = my_index.search(
    namespace="example-namespace",
    query={
        "top_k": 10,
        "inputs": {
            'text': query
        }
    }
)

# Print the results
for hit in results['result']['hits']:
    print(f"id: {hit['_id']}, score: {round(hit['_score'], 2)}, text: {hit['fields']['chunk_text']}, category: {hit['fields']['category']}")
     


# Search the  index and rerank results
reranked_results = my_index.search(
    namespace="example-namespace",
    query={
        "top_k": 10,
        "inputs": {
            'text': query
        }
    },
    rerank={
        "model": "bge-reranker-v2-m3",
        "top_n": 10,
        "rank_fields": ["chunk_text"]
    }
)

# Print the reranked results
for hit in reranked_results['result']['hits']:
    print(f"reranked_results id: {hit['_id']}, score: {round(hit['_score'], 2)}, text: {hit['fields']['chunk_text']}, category: {hit['fields']['category']}")


# pc.delete_index(index_name)
