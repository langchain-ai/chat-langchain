from typing import Any, Callable, Dict, Tuple

from chromadb import PersistentClient
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import pytest

from croptalk.document_retriever import DocumentRetriever


@pytest.fixture(scope="module")
def content_chunk_1() -> Dict[str, Any]:
    """Chunk 1 is about apples in Yakima, Washington"""
    return {
        "id": "1",
        "metadatas": {
            "title": "Special Provisions for insuring 0054 under plan 90 in state 53",
            "page": 4,
            "doc_category": "SP",
            "state": "53",
            "county": "077",
            "commodity": "0054",
            "source": "https://croptalk-spoi.s3.amazonaws.com/SPOI/53_077_90_0054_20230831.pdf",
        },
        "content": "Acreage that has been grafted following set out will have the yields adjusted by the AIP in accordance with FCIC procedures.\\n5\\nPlan:\\nDate: 8/28/2023\\nCounty: Yakima (077)\\nAPH (90)\\nState:    Washington (53)\\n2024 and Succeeding Crop Years\\nSpecial Provisions\\nCommodity: Apples (0054)\\nYear: 2024"
    }


@pytest.fixture(scope="module")
def content_chunk_2() -> Dict[str, Any]:
    """Chunk 2 is about oranges in Stanislaus, California"""
    return {
       "id": "2",
       "metadatas": {
            "title": "Special Provisions for insuring 0227 under plan 90 in state 06",
            "page": 1,
            "doc_category": "SP",
            "state": "06",
            "county": "099",
            "commodity": "0227",
            "source": "https://croptalk-spoi.s3.amazonaws.com/SPOI/06_099_90_0227_20230831.pdf",
        },
        "content": "the Regional Office, based upon evidence that acceptable supporting documentation is being maintained as required in the Crop Insurance Handbook.\\nFrost protection means acreage adequately protected by frost protection equipment.  Adequately protected means: 1) at least 40 serviceable heaters per acre; 2) \\nthe number of wind machines that provide at least 5 propeller horsepower per acre (at least one wind machine is required for every ten acres regardless of propeller \\nhorsepower); or 3) solid set sprinklers or foggers supplied by well water (the pump and well must have the capacity to supply water to all the acreage \\nsimultaneously).  We will determine the adequacy of the frost protection equipment for a unit.\\nThe four citrus fruit groups under oranges are as follows: 1) Navel; 2) Valencia; 3) Sweet; and 4) Cara Cara.\\nType\\n*6\\nCommodity type Sweet includes all varieties of sweet oranges (Citrus sinensis) except for Navel, Valencia, and Cara Cara.\\nDate"
    }


@pytest.fixture(scope="module")
def content_chunk_3() -> Dict[str, Any]:
    """Chunk 3 is about oranges in Stanislaus, California"""
    return {
        "id": "3",
        "metadatas": {
            "title": "Special Provisions for insuring 0227 under plan 90 in state 06",
            "page": 2,
            "doc_category": "SP",
            "state": "06",
            "county": "099",
            "commodity": "0227",
            "source": "https://croptalk-spoi.s3.amazonaws.com/SPOI/06_019_90_0227_20230831.pdf",
        },
        "content": "the Regional Office, based upon evidence that acceptable supporting documentation is being maintained as required in the Crop Insurance Handbook.\\nFrost protection means acreage adequately protected by frost protection equipment.  Adequately protected means: 1) at least 40 serviceable heaters per acre; 2) \\nthe number of wind machines that provide at least 5 propeller horsepower per acre (at least one wind machine is required for every ten acres regardless of propeller \\nhorsepower); or 3) solid set sprinklers or foggers supplied by well water (the pump and well must have the capacity to supply water to all the acreage \\nsimultaneously).  We will determine the adequacy of the frost protection equipment for a unit.\\nThe four citrus fruit groups under oranges are as follows: 1) Navel; 2) Valencia; 3) Sweet; and 4) Cara Cara.\\nType\\n*6\\nCommodity type Sweet includes all varieties of sweet oranges (Citrus sinensis) except for Navel, Valencia, and Cara Cara.\\nDate"
    }


@pytest.fixture(scope="module")
def create_vectorstore(
    content_chunk_1: Dict[str, Any],
    content_chunk_2: Dict[str, Any],
    content_chunk_3: Dict[str, Any],
) -> Tuple[str, str, Callable]:
    # create vectorstore
    vectorstore_dir = "/tmp/chromadb"
    collection_name = "Test"
    embedding_function = DefaultEmbeddingFunction()

    client = PersistentClient(vectorstore_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
    )

    # add content to vectorstore
    chunks = [content_chunk_1, content_chunk_2, content_chunk_3]
    collection.add(
        ids = [chunk["id"] for chunk in chunks],
        metadatas = [chunk["metadatas"] for chunk in chunks],
        documents = [chunk["content"] for chunk in chunks],
    )

    # return vectorstore params
    return (vectorstore_dir, collection_name, embedding_function)


@pytest.fixture(scope="module")
def doc_retriever(create_vectorstore: Tuple[str, str, Callable]) -> DocumentRetriever:
    return DocumentRetriever(
        vectorestore_dir=create_vectorstore[0],
        collection_name=create_vectorstore[1],
        embedding_function=create_vectorstore[2],
    )


def test_get_documents_with_query(doc_retriever: DocumentRetriever):
    # no matter what the query is, all 3 documents should be returned with default top_k
    # i.e. only the sorting will change
    res = doc_retriever.get_documents(query="Any query")
    assert len(res) == 3
    # with top_k=2 and a query about oranges, all returned documents should be about oranges
    res = doc_retriever.get_documents(query="Are cara cara oranges sold in California?", top_k=2)
    assert all("Cara Cara" in doc for doc in res)
    # with top_k=1 and a query about apples, all returned documents should be about apples
    res = doc_retriever.get_documents(query="Are apples sold in Washington state?", top_k=1)
    assert all("Apples" in doc for doc in res)


@pytest.mark.parametrize(
    "doc_category_requested, nb_of_docs_expected",
    [
        ["BP", 0],
        ["SP", 3],
    ],
)
def test_get_documents_with_doc_category_filter(
    doc_retriever: DocumentRetriever, doc_category_requested: str, nb_of_docs_expected: int,
):
    res = doc_retriever.get_documents(query="Any query", doc_category=doc_category_requested)
    assert len(res) == nb_of_docs_expected


@pytest.mark.parametrize(
    "commodity_requested, nb_of_docs_expected",
    [
        ["Almonds", 0],
        ["Apples", 1],
        ["Oranges", 2],
    ]
)
def test_get_documents_with_commodity_filter(
    doc_retriever: DocumentRetriever, commodity_requested: str, nb_of_docs_expected: int,
):
    res = doc_retriever.get_documents(query="Any query", commodity=commodity_requested)
    assert len(res) == nb_of_docs_expected


@pytest.mark.parametrize(
    "state_requested, county_requested, nb_of_docs_expected",
    [
        ["Washington", "Spokane", 0],
        ["Washington", "Yakima", 1],
        ["California", "Stanislaus", 2],
    ]
)
def test_get_documents_with_state_and_county_filter(
    doc_retriever: DocumentRetriever,
    state_requested: str,
    county_requested: str,
    nb_of_docs_expected: int,
):
    # note that state and county must be specified, otherwise no filter is considered
    res = doc_retriever.get_documents(
        query="Any query", state=state_requested, county=county_requested,
    )
    assert len(res) == nb_of_docs_expected


@pytest.mark.parametrize(
    "state_requested, nb_of_docs_expected",
    [
        ["Hawaii", 0],
        ["Washington", 1],
        ["California", 2],
    ]
)
def test_get_documents_with_state_filter(
    doc_retriever: DocumentRetriever,
    state_requested: str,
    nb_of_docs_expected: int,
):
    res = doc_retriever.get_documents(query="Any query", state=state_requested)
    assert len(res) == nb_of_docs_expected


@pytest.mark.parametrize(
    "top_k_requested, nb_of_docs_expected",
    [
        [4, 3],
        [3, 3],
        [2, 2],
        [1, 1],
    ]
)
def test_get_documents_with_top_k_filter(
    doc_retriever: DocumentRetriever,
    top_k_requested: int,
    nb_of_docs_expected: int,
):
    res = doc_retriever.get_documents(query="Any query", top_k=top_k_requested)
    assert len(res) == nb_of_docs_expected


@pytest.mark.parametrize(
    "doc_category, commodity, state, county, top_k, nb_of_docs_expected",
    [
        ["SP", "Oranges", "California", "Stanislaus", 2, 2],
        # modify one filter at a time below
        ["BP", "Oranges", "California", "Stanislaus", 2, 0],
        ["SP", "Apples", "California", "Stanislaus", 2, 0],
        ["SP", "Oranges", "Washington", "Stanislaus", 2, 0],
        ["SP", "Oranges", "California", "Fresno", 2, 0],
        ["SP", "Oranges", "California", "Stanislaus", 1, 1],
    ]
)
def test_get_documents_with_all_filters(
    doc_retriever: DocumentRetriever,
    doc_category: str,
    commodity: str,
    state: str,
    county: str,
    top_k: int,
    nb_of_docs_expected: int,
):
    res = doc_retriever.get_documents(
        query="Any query",
        doc_category=doc_category,
        commodity=commodity,
        county=county,
        state=state,
        top_k=top_k,
    )
    assert len(res) == nb_of_docs_expected
