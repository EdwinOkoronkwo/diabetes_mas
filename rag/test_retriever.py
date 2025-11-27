from rag.retriever.ChromaRetriever import ChromaRetriever


def run_retriever_test(query: str):
    """Initializes and tests the ChromaRetriever with a specific query."""
    print("=" * 60)
    print(f"TESTING QUERY: '{query}'")

    retriever = ChromaRetriever()

    # 1. Embed the query
    embedded_vector = retriever.embed_query(query)

    # 2. Perform similarity search
    context = retriever.similarity_search(embedded_vector)

    print("\n--- RETRIEVED CONTEXT ---")
    print(context)
    print("=" * 60)


if __name__ == "__main__":
    # Test case 1: High T2D Relevance (Should retrieve 3 sources)
    run_retriever_test("I have high blood sugar and feel tired all the time.")

    # Test case 2: Low T2D Relevance (Should retrieve 1 general source)
    run_retriever_test("What is the general protocol for a patient after a heart procedure?")