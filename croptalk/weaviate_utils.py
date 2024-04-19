import os
from typing import Any, List, Optional, Tuple

from chromadb.utils import embedding_functions
import weaviate
import weaviate.classes as wvc
from weaviate.client import WeaviateClient
from weaviate.collections.classes.internal import QueryReturn
from weaviate.collections.collection import Collection


from dsmain.dataapi.lookups import CommodityLookup, StateLookup, CountyLookup


state_lookup = StateLookup(quiet_fail=True)
county_lookup = CountyLookup(quiet_fail=True)
commodity_lookup = CommodityLookup(quiet_fail=True)


def get_weaviate_client() -> WeaviateClient:
    """
    Requires the following environment variables to be set:
    - WCS_CLUSTER_URL
    - WCS_API_KEY

    Returns:
        a weaviate could services client
    """
    client = weaviate.connect_to_wcs(
        cluster_url=os.getenv("WCS_CLUSTER_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WCS_API_KEY")),
    )
    return client


def get_client_collection(collection_name: str) -> Tuple[WeaviateClient, Collection]:
    """
    Requires the following environment variables to be set:
        - WCS_CLUSTER_URL
        - WCS_API_KEY

    Args:
        collection_name: the collection name we are looking for

    Returns:
        - a weaviate cloud services client
        - a weaviate collection matching provided collection name
    """
    client = get_weaviate_client()
    collection = client.collections.get(collection_name)
    return client, collection


def query_near_vector_with_filters(
    collection: Collection,
    query: str,
    limit: int,
    doc_category: Optional[str] = None,
    commodity: Optional[str] = None,
    county: Optional[str] = None,
    state: Optional[str] = None,
    include_common_docs: bool = True,
) -> QueryReturn:
    """
    Args:
        collection: the weaviate collection to run the query on
        query: query to use
        limit: number of retrieved documents we are aiming for
        doc_category: document category to filter on, None means no filter
        commodity: commodity to filter on, None means no filter
        county: county to filter on, None means no filter
        state: state to filter on, None means no filter
        include_common_docs: whether (default) or not to include documents that apply to all
                             states, counties or commodities... does not apply to document category
                             filter

    Returns:
        weaviate query response
    """
    query_embedded = embed_text(query=query)

    weaviate_filter = create_weaviate_filter(
        state=state,
        county=county,
        commodity=commodity,
        doc_category=doc_category,
        include_common_docs=include_common_docs,
    )

    response = collection.query.near_vector(
        near_vector=query_embedded,
        filters=weaviate_filter,
        limit=limit,
    )

    return response


def embed_text(query: str, model_name: str = "all-MiniLM-L6-v2") -> List[float]:
    """
    Args:
        query: the text to embed
        model_name: the name of the model to use for embedding, defaults to "all-MiniLM-L6-v2"

    Returns:
        a list of floats representing the document's embedding 
    """
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )

    embedded_text = emb_fn([query])
    embedded_text = embedded_text[0]

    return embedded_text


def get_equal_filter(property_name: str, property_value: Any) -> wvc.query.Filter:
    """
    Args:
        property_name: property name to filter on
        property_value: property value to filter on

    Returns:
        a weaviate filter that checks if provided property name equals provided property value
    """
    filter=wvc.query.Filter.by_property(property_name).equal(property_value)
    return filter


def create_weaviate_filter(
    state: str = None,
    county: str = None,
    commodity: str = None,
    doc_category: str = None,
    include_common_docs: bool = True,
) -> Optional[wvc.query.Filter]:
    """
    Args:
        state: state to filter on, None means no filter
        county: county to filter on, None means no filter
        commodity: commodity to filter on, None means no filter
        doc_category: document category to filter on, None means no filter
        include_common_docs: whether (default) or not to include documents that apply to all
                             states, counties or commodities... does not apply to document category
                             filter

    Returns:
        a weaviate filter
        See [the docs](https://weaviate.io/developers/weaviate/search/filters) for more details!
    """
    def _is_valid(param_value: str) -> bool:
        return param_value is not None and param_value.lower() != 'none'

    def _get_integer_equal_filter(property_name: str, property_value: int) -> wvc.query.Filter:
        filter_value = get_equal_filter(property_name, property_value)
        if not include_common_docs:
            # filter: provided field equals provided value
            return filter_value
        # filter: provided field equals provided value or 0
        filter_zero = get_equal_filter(property_name, 0)
        return filter_value | filter_zero

    filter_conditions = []

    # add numerical conditions
    # note that ins plans, states, counties and commodities are stored as integers in weaviate
    # vector store, while they are 0-padded strings in lookups
    if _is_valid(state):
        try:
            state_obj = state_lookup.find(state)
        except:
            state_obj = None
        if state_obj:
            filter_conditions.append(
                _get_integer_equal_filter("state", int(state_obj.code))
            )

    if _is_valid(county) and _is_valid(state):
        try:
            county_obj = county_lookup.find_by_name(county, state)
        except:
            county_obj = None
        if county_obj:
            filter_conditions.append(
                _get_integer_equal_filter("county", int(county_obj.code))
            )

    if _is_valid(commodity):
        try:
            commodity_obj = commodity_lookup.find(commodity)
        except:
            commodity_obj = None
        if commodity_obj:
            filter_conditions.append(
                _get_integer_equal_filter("commodity", int(commodity_obj.code))
            )

    # add text conditions
    if _is_valid(doc_category):
        filter_conditions.append(
            get_equal_filter("doc_category", doc_category)
        )

    # return filter
    if len(filter_conditions) == 0:
        return None
    elif len(filter_conditions) == 1:
        return filter_conditions[0]
    else:
        # join filter conditions by AND operator
        filter = filter_conditions[0]
        for cur_condition in filter_conditions[1:]:
            filter &= cur_condition
        return filter
