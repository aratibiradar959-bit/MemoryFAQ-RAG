import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.loaders.web_loader import load_web_page
from src.retrieval.text_splitter import split_documents
from src.embeddings.embedding_model import get_embedding_model


url = "https://example.com"

# 1. Load webpage
documents = load_web_page(url)

# 2. Split webpage
chunks = split_documents(documents)

# 3. Load embedding model
embedding_model = get_embedding_model()

# 4. Create embedding for the first web chunk
text = chunks[0].page_content

vector = embedding_model.embed_query(text)

print("\n" + "=" * 60)
print("WEB EMBEDDING TEST")
print("=" * 60)

print(f"\nNumber of chunks: {len(chunks)}")
print(f"Text length: {len(text)}")
print(f"Vector dimension: {len(vector)}")

print("\nFirst 10 vector values:")
print(vector[:10])