import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.loaders.web_loader import load_web_page
from src.retrieval.text_splitter import split_documents


url = "https://example.com"

# Step 1: Load webpage
documents = load_web_page(url)

print(f"Documents loaded: {len(documents)}")

# Step 2: Split webpage into chunks
chunks = split_documents(documents)

print(f"Chunks created: {len(chunks)}")

for index, chunk in enumerate(chunks, start=1):

    print("\n" + "=" * 60)
    print(f"CHUNK {index}")
    print("=" * 60)

    print("\nContent:")
    print(chunk.page_content)

    print("\nMetadata:")
    print(chunk.metadata)