from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document


def load_web_page(url: str) -> list[Document]:
    """Load a webpage from a URL."""

    loader = WebBaseLoader(url)

    documents = loader.load()

    for document in documents:
        document.metadata["source"] = url
        document.metadata["file_type"] = "web"
        document.metadata["url"] = url

    return documents