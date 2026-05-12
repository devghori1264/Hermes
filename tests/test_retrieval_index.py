from src.retrieval.index import InMemoryVectorIndex
from src.retrieval.types import RetrievalFilter, RetrievalQuery, VectorItem


def test_in_memory_index_search() -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add(VectorItem(item_id="a", vector=[1.0, 0.0]))
    index.add(VectorItem(item_id="b", vector=[0.0, 1.0]))

    result = index.search(RetrievalQuery(embedding=[1.0, 0.0], top_k=1))
    assert len(result) == 1
    assert result[0].item_id == "a"


def test_index_filters_domain() -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add(VectorItem(item_id="a", vector=[1.0, 0.0], domain="movies"))
    index.add(VectorItem(item_id="b", vector=[1.0, 0.0], domain="music"))

    result = index.search(RetrievalQuery(embedding=[1.0, 0.0], top_k=5))
    assert len(result) == 2

    filtered = index.search(
        RetrievalQuery(
            embedding=[1.0, 0.0],
            top_k=5,
            filters=RetrievalFilter(domain="movies"),
        )
    )
    assert len(filtered) == 1
    assert filtered[0].item_id == "a"
