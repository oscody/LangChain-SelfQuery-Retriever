import os
import json
import re
import yaml
import pinecone
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import NotFoundException

# Load excluded files
with open("exclude_files.json", "r") as f:
    exclude_files = json.load(f)

allowed_dirs = {"1. Projects", "2. Areas", "3. Resources", "4. Archives"}
base_dir = "/Users/bogle/Dev/obsidian/Bogle"

# Define metadata field info
metadata_field_info = [
    {"name": "name", "description": "The name of the document", "type": "string"},
    {"name": "tags", "description": "Tags associated with the document", "type": "list"},
    {"name": "date-created", "description": "The date the document was created", "type": "date"},
]

# Function to extract metadata from Markdown front matter
def extract_metadata(content):
    """Extracts metadata from Markdown front matter (YAML block) at the beginning of a file."""
    metadata = {}

    # match = re.search(r"^---\s*\n(.*?)(?:\n---|\Z)", content, re.DOTALL)
    # match = re.search(r'^---\s+(.*?)\s+---', content, re.DOTALL)
    # Case 1: Content starts with YAML front matter delimiters
    if content.startswith('---'):
        pattern = r'^---\s*\n(.*?)\n---\s*'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            try:
                front_matter_text = match.group(1)
                front_matter = yaml.safe_load(front_matter_text)
                metadata["name"] = front_matter.get("name", "").strip()
                metadata["tags"] = front_matter.get("tags", [])
                metadata["date-created"] = front_matter.get("date-created", "").strip()
            except yaml.YAMLError as e:
                print(f"Error parsing YAML front matter: {e}")
        else:
            print("YAML front matter block not found.")
    
    # Case 2: No YAML front matter delimiters – assume first block has key-value pairs
    else:
        # Assume the metadata block is at the very beginning until a double newline
        header_block = content.split("\n\n", 1)[0]
        
        # Regex to capture keys and their values.
        # This pattern captures keys as sequences of word characters or hyphens,
        # then a colon and optional whitespace,
        # then captures any text (non-greedily) until it sees another key pattern or the end.
        pattern = r'([\w-]+):\s*([^:]+?)(?=\s+[\w-]+:|$)'
        pairs = re.findall(pattern, header_block)
        
        for key, value in pairs:
            metadata[key] = value.strip()
    
    return metadata

# Load documents
documents = []
for sub_dir in allowed_dirs:
    full_path = os.path.join(base_dir, sub_dir)
    loader = DirectoryLoader(full_path, glob="**/*.md", recursive=True)

    try:
        docs = loader.load()

        # Filter out excluded files
        filtered_docs = [
            doc for doc in docs
            if os.path.relpath(doc.metadata.get("source", ""), base_dir) not in exclude_files
        ]

        # Assign metadata to each document
        for doc in filtered_docs:
            content = doc.page_content
            metadata = extract_metadata(content)
            doc.metadata.update(metadata)

        documents.extend(filtered_docs)

    except Exception as e:
        print(f"Error loading from {full_path}: {e}")

print(f"Loaded {len(documents)} documents.")

# Initialize Pinecone
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBEDDING_MODEL_NAME = "text-embedding-ada-002"

print(f"PINECONE_API_KEY- {PINECONE_API_KEY}")
print(f"PINECONE_INDEX_NAME- {PINECONE_INDEX_NAME}")
print(f"OPENAI_API_KEY- {OPENAI_API_KEY}")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure Pinecone index exists
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,  # Adjust if needed
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Target index and check status
pc_index = pc.Index(PINECONE_INDEX_NAME)
print(pc_index.describe_index_stats())

# Generate embeddings and upload to Pinecone
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
namespace = "document_search"

try:
    pc_index.delete(namespace=namespace, delete_all=True)
except NotFoundException:
    print(f"Namespace '{namespace}' not found. Not deleting.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
else:
    print("Namespace deleted successfully.")

# Upload documents
PineconeVectorStore.from_documents(
    documents,
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
    namespace=namespace
)

print("Successfully uploaded documents to Pinecone.")