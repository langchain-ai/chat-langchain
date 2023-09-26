import re
import urllib.request
import xml.etree.ElementTree as ET
from multiprocessing.pool import ThreadPool
from typing import Any, Generator, List

import requests
from bs4 import BeautifulSoup, Doctype, NavigableString, SoupStrainer, Tag
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
from tenacity import retry, stop_after_attempt, wait_random_exponential


class LangchainDocsLoader(BaseLoader):
    """A loader for the Langchain documentation.

    The documentation is available at https://python.langchain.com/.
    """

    _sitemap: str = "https://python.langchain.com/sitemap.xml"
    _filter_urls: List[str] = ["https://python.langchain.com/"]

    def __init__(
        self,
        number_threads: int = 50,
        include_output_cells: bool = True,
    ) -> None:
        """Initialize the loader.

        Args:
            number_threads (int, optional): Number of threads to use
                for parallel processing. Defaults to 50.
        """
        self._number_threads = number_threads
        self._include_output_cells = include_output_cells

    def load(self) -> List[Document]:
        """Load the documentation.

        Returns:
            List[Document]: A list of documents.
        """

        urls = self._get_urls()
        docs = self._process_urls(urls)
        return docs

    def _get_urls(self) -> List[str]:
        """Get the urls from the sitemap."""
        with urllib.request.urlopen(self._sitemap) as response:
            xml = response.read()

        root = ET.fromstring(xml)

        namespaces = {"sitemap": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls = [
            url.text
            for url in root.findall(".//sitemap:loc", namespaces=namespaces)
            if url.text is not None and url.text != "https://python.langchain.com/"
        ]

        return urls

    def _process_urls(self, urls: List[str]) -> List[Document]:
        """Process the urls in parallel."""

        with ThreadPool(self._number_threads) as pool:
            docs = pool.map(self._process_url, urls)
            return docs

    @retry(
        stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10)
    )
    def _process_url(self, url: str) -> Document:
        """Process a url."""
        r = requests.get(url, allow_redirects=False)
        html = r.text
        page_content = self.langchain_docs_extractor(html, self._include_output_cells)
        metadata = self._metadata_extractor(html, url)
        return Document(page_content=page_content, metadata=metadata)

    def _metadata_extractor(self, raw_html: str, url: str) -> dict[Any, Any]:
        """Extract metadata from raw html using BeautifulSoup."""
        metadata = {"source": url}

        soup = BeautifulSoup(raw_html, "lxml")
        if title := soup.find("title"):
            metadata["title"] = title.get_text()
        if description := soup.find("meta", attrs={"name": "description"}):
            if isinstance(description, Tag):
                content = description.get("content", None)
                if isinstance(content, str):
                    metadata["description"] = content
            else:
                metadata["description"] = description.get_text()
        if html := soup.find("html"):
            if isinstance(html, Tag):
                lang = html.get("lang", None)
                if isinstance(lang, str):
                    metadata["language"] = lang

        return metadata

    @staticmethod
    def langchain_docs_extractor(html: str, include_output_cells: bool) -> str:
        soup = BeautifulSoup(
            html,
            "lxml",
            parse_only=SoupStrainer(name="article"),
        )

        # Remove all the tags that are not meaningful for the extraction.
        SCAPE_TAGS = ["nav", "footer", "aside", "script", "style"]
        [tag.decompose() for tag in soup.find_all(SCAPE_TAGS)]

        def get_text(tag: Tag) -> Generator[str, None, None]:
            for child in tag.children:
                if isinstance(child, Doctype):
                    continue

                if isinstance(child, NavigableString):
                    yield child
                elif isinstance(child, Tag):
                    if child.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                        yield f"{'#' * int(child.name[1:])} {child.get_text()}\n\n"
                    elif child.name == "a":
                        yield f"[{child.get_text(strip=False)}]({child.get('href')})"
                    elif child.name == "img":
                        yield f"![{child.get('alt', '')}]({child.get('src')})"
                    elif child.name in ["strong", "b"]:
                        yield f"**{child.get_text(strip=False)}**"
                    elif child.name in ["em", "i"]:
                        yield f"_{child.get_text(strip=False)}_"
                    elif child.name == "br":
                        yield "\n"
                    elif child.name == "code":
                        parent = child.find_parent()
                        if parent is not None and parent.name == "pre":
                            classes = parent.attrs.get("class", "")

                            language = next(
                                filter(lambda x: re.match(r"language-\w+", x), classes),
                                None,
                            )
                            if language is None:
                                language = ""
                            else:
                                language = language.split("-")[1]

                            if (
                                language in ["pycon", "text"]
                                and not include_output_cells
                            ):
                                continue

                            lines: list[str] = []
                            for span in child.find_all("span", class_="token-line"):
                                line_content = "".join(
                                    token.get_text() for token in span.find_all("span")
                                )
                                lines.append(line_content)

                            code_content = "\n".join(lines)
                            yield f"```{language}\n{code_content}\n```\n\n"
                        else:
                            yield f"`{child.get_text(strip=False)}`"

                    elif child.name == "p":
                        yield from get_text(child)
                        yield "\n\n"
                    elif child.name == "ul":
                        for li in child.find_all("li", recursive=False):
                            yield "- "
                            yield from get_text(li)
                            yield "\n\n"
                    elif child.name == "ol":
                        for i, li in enumerate(child.find_all("li", recursive=False)):
                            yield f"{i + 1}. "
                            yield from get_text(li)
                            yield "\n\n"
                    elif child.name == "div" and "tabs-container" in child.attrs.get(
                        "class", [""]
                    ):
                        tabs = child.find_all("li", {"role": "tab"})
                        tab_panels = child.find_all("div", {"role": "tabpanel"})
                        for tab, tab_panel in zip(tabs, tab_panels):
                            tab_name = tab.get_text(strip=True)
                            yield f"{tab_name}\n"
                            yield from get_text(tab_panel)
                    elif child.name == "table":
                        thead = child.find("thead")
                        header_exists = isinstance(thead, Tag)
                        if header_exists:
                            headers = thead.find_all("th")
                            if headers:
                                yield "| "
                                yield " | ".join(
                                    header.get_text() for header in headers
                                )
                                yield " |\n"
                                yield "| "
                                yield " | ".join("----" for _ in headers)
                                yield " |\n"

                        tbody = child.find("tbody")
                        tbody_exists = isinstance(tbody, Tag)
                        if tbody_exists:
                            for row in tbody.find_all("tr"):
                                yield "| "
                                yield " | ".join(
                                    cell.get_text(strip=True)
                                    for cell in row.find_all("td")
                                )
                                yield " |\n"

                        yield "\n\n"
                    elif child.name in ["button"]:
                        continue
                    else:
                        yield from get_text(child)

        return "".join(get_text(soup))
