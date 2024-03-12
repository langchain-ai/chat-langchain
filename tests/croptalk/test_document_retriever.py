from dotenv import load_dotenv
import pytest

from croptalk.document_retriever import DocumentRetriever
from dsmain.dataapi.lookups import CommodityLookup, StateLookup, CountyLookup

load_dotenv("secrets/.env.secret")
load_dotenv("secrets/.env.shared")


@pytest.fixture(scope="module")
def doc_retriever() -> DocumentRetriever:
    return DocumentRetriever(
        collection_name="croptalk1",
    )


@pytest.fixture(scope="module")
def commodity_lookup() -> CommodityLookup:
    return CommodityLookup(quiet_fail=True)


@pytest.fixture(scope="module")
def state_lookup() -> StateLookup:
    return StateLookup(quiet_fail=True)


@pytest.fixture(scope="module")
def county_lookup() -> CountyLookup:
    return CountyLookup(quiet_fail=True)


@pytest.mark.parametrize(
    "single_word_query",
    [
        "apples",
    ]
)
def test_get_documents_with_query(doc_retriever: DocumentRetriever, single_word_query: str):
    # although not 100% bullet proof, with a single word query, chances are all returned documents
    # should contain the query
    res = doc_retriever.get_documents(query=single_word_query)
    assert len(res) > 0
    assert all(single_word_query in doc for doc in res)


@pytest.mark.parametrize(
    "doc_category_requested",
    [
        "BP",
        "SP",
    ],
)
def test_get_documents_with_doc_category_filter(
    doc_retriever: DocumentRetriever,
    doc_category_requested: str,
):
    res = doc_retriever.get_documents(query="Any query", doc_category=doc_category_requested)
    assert len(res) > 0
    assert all(f"doc_category='{doc_category_requested}'" in doc for doc in res)


@pytest.mark.parametrize(
    "commodity_requested, include_common_docs",
    [
        ["Apples", False],
        ["Apples", True],
        ["Almonds", False],
        ["Almonds", True],
    ]
)
def test_get_documents_with_commodity_filter(
    doc_retriever: DocumentRetriever,
    commodity_lookup: CommodityLookup,
    commodity_requested: str,
    include_common_docs: bool,
):
    res = doc_retriever.get_documents(
        query="Any query",
        commodity=commodity_requested,
        include_common_docs=include_common_docs,
    )
    assert len(res) > 0
    commodity_code_requested = int(commodity_lookup.find(commodity_requested).code)
    if include_common_docs:
        assert all(
            (
                f"commodity='{int(commodity_code_requested)}'" in doc
                or
                "commodity='0'" in doc
            )
            for doc in res
        )
    else:
        assert all(f"commodity='{int(commodity_code_requested)}'" in doc for doc in res)


@pytest.mark.parametrize(
    "state_requested, county_requested, include_common_docs",
    [
        ["Washington", "Douglas", True],
        ["Washington", "Douglas", False],
        ["Washington", "Walla Walla", True],
        ["Washington", "Walla Walla", False],
    ]
)
def test_get_documents_with_state_and_county_filter(
    doc_retriever: DocumentRetriever,
    state_lookup: StateLookup,
    county_lookup: CountyLookup,
    state_requested: str,
    county_requested: str,
    include_common_docs: bool,
):
    # note that state and county must be specified, otherwise no filter is considered
    res = doc_retriever.get_documents(
        query="Any query",
        state=state_requested,
        county=county_requested,
        include_common_docs=include_common_docs,
    )
    assert len(res) > 0
    state_code_requested = int(state_lookup.find(state_requested).code)
    county_code_requested = int(county_lookup.find_by_name(county_requested, state_requested).code)
    if include_common_docs:
        assert all(
            (
                (
                    f"state='{int(state_code_requested)}'" in doc
                    or
                    "state='0'" in doc
                )
                and
                (
                    f"county='{int(county_code_requested)}'" in doc
                    or
                    "county='0'" in doc
                )
            )
            for doc in res
        )
    else:
        assert all(
            (
                f"state='{int(state_code_requested)}'" in doc
                and
                f"county='{int(county_code_requested)}'" in doc
            )
            for doc in res
        )


@pytest.mark.parametrize(
    "state_requested, include_common_docs",
    [
        ["California", True],
        ["California", False],
        ["Washington", True],
        ["Washington", False],
    ]
)
def test_get_documents_with_state_filter(
    doc_retriever: DocumentRetriever,
    state_lookup: StateLookup,
    state_requested: str,
    include_common_docs: bool,
):
    res = doc_retriever.get_documents(
        query="Any query",
        state=state_requested,
        include_common_docs=include_common_docs,
    )
    assert len(res) > 0
    state_code_requested = int(state_lookup.find(state_requested).code)
    if include_common_docs:
        assert all(
            (
                f"state='{int(state_code_requested)}'" in doc
                or
                "state='0'" in doc
            )
            for doc in res
        )
    else:
        assert all(f"state='{int(state_code_requested)}'" in doc for doc in res)


@pytest.mark.parametrize(
    "top_k_requested, nb_of_docs_expected",
    [
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
    "doc_category, commodity, state, county, top_k",
    [
        ["SP", "Apples", "West Virginia", "Putnam", 4],
    ]
)
def test_get_documents_with_all_filters(
    doc_retriever: DocumentRetriever,
    commodity_lookup: CommodityLookup,
    state_lookup: StateLookup,
    county_lookup: CountyLookup,
    doc_category: str,
    commodity: str,
    state: str,
    county: str,
    top_k: int,
):
    res = doc_retriever.get_documents(
        query="Any query",
        doc_category=doc_category,
        commodity=commodity,
        county=county,
        state=state,
        top_k=top_k,
        include_common_docs=False,
    )
    assert len(res) == top_k
    commodity_code_requested = int(commodity_lookup.find(commodity).code)
    state_code_requested = int(state_lookup.find(state).code)
    county_code_requested = int(county_lookup.find_by_name(county, state).code)
    assert all(
        (
            f"doc_category='{doc_category}'" in doc
            and
            f"commodity='{int(commodity_code_requested)}'" in doc
            and
            f"state='{int(state_code_requested)}'" in doc
            and
            f"county='{int(county_code_requested)}'" in doc
        )
        for doc in res
    )
