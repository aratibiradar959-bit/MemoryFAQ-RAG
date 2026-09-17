import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.loaders.web_loader import load_web_page


url = "https://example.com"

documents = load_web_page(url)

print("\n" + "=" * 60)
print("WEB LOADER TEST")
print("=" * 60)

print(f"\nURL: {url}")
print(f"Documents loaded: {len(documents)}")

for index, document in enumerate(documents, start=1):

    print("\n" + "-" * 60)
    print(f"Document {index}")
    print("-" * 60)

    print("\nMetadata:")
    print(document.metadata)

    print("\nContent:")
    print(document.page_content[:1000])