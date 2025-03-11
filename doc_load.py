import os
import glob
import json

from langchain_community.document_loaders import TextLoader, DirectoryLoader

with open("exclude_files.json", "r") as f:
    exclude_files = json.load(f)

allowed_dirs = {"1. Projects", "2. Areas", "3. Resources", "4. Archives"}
base_dir = "/Users/bogle/Dev/obsidian/Bogle"

all_documents = []
failed_files = []

# Process each allowed subdirectory
for sub_dir in allowed_dirs:
    full_path = os.path.join(base_dir, sub_dir)
    pattern = os.path.join(full_path, "**/*.md")
    file_list = glob.glob(pattern, recursive=True)
    
    docs_sub = []  # Documents for the current subdirectory

    for file_path in file_list:
        # Skip files listed in exclude_files (compare relative paths)
        relative_file = os.path.relpath(file_path, base_dir)
        if relative_file in exclude_files:
            continue
        
        try:
            # Use DirectoryLoader for the individual file.
            loader = DirectoryLoader(os.path.dirname(file_path), glob=os.path.basename(file_path))
            docs = loader.load()
            docs_sub.extend(docs)
        except Exception as e:
            print(f"Error loading file {file_path} with DirectoryLoader: {e}")
            failed_files.append(file_path)
            try:
                # Fallback to TextLoader for this Markdown file.
                fallback_loader = TextLoader(file_path)
                docs = fallback_loader.load()
                docs_sub.extend(docs)
            except Exception as fallback_error:
                print(f"Fallback TextLoader also failed for file {file_path}: {fallback_error}")
    
    # Print the number of documents loaded for this subdirectory.
    print(f"docs {len(docs_sub)} ---- {sub_dir}.")
    all_documents.extend(docs_sub)

print(f"Total documents loaded: {len(all_documents)}")
if failed_files:
    print("Files that required fallback to TextLoader:")
    for f in failed_files:
        print(f)