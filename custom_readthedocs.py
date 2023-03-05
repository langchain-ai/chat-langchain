from pathlib import Path
from typing import List, Tuple

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class ReadTheDocsLoader(BaseLoader):
    """Loader that loads ReadTheDocs documentation directory dump."""

    def __init__(self, path: str):
        """Initialize path."""
        self.file_path = path

    def load(self) -> List[Document]:
        """Load documents."""
        from bs4 import BeautifulSoup

        def _clean_data(data: str) -> Tuple[str, str, str]:
            soup = BeautifulSoup(data, features="html.parser")
            title_tag = soup.find("meta", property="og:title")
            description_tag = soup.find("meta", property="og:description")
            title = title_tag["content"] if title_tag else ""
            description = description_tag["content"] if description_tag else ""
            text = title + "\n" + description
            return "\n".join([t for t in text.split("\n") if t])


        docs = []
        for p in Path(self.file_path).rglob("*"):
            if p.is_dir():
                continue
            with open(p) as f:
                text = _clean_data(f.read())
            metadata = {"source": str(p)}
            docs.append(Document(page_content=text, metadata=metadata))
        return docs