from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore


print("Loading embedding model...")

embedding_model = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


print("Connecting to Qdrant...")

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    collection_name="melanoma_guidelines",
    path="./qdrant_data"
)

print("Connected successfully!")


questions = [
    "What are the ABCD or ABCDE criteria used in melanoma assessment?",
    "What does asymmetry mean when evaluating a suspicious skin lesion?",
    "What is the clinical significance of an irregular border?",
    "What does color variation indicate in melanoma assessment?",
    "What does the AAD guideline recommend for management of primary cutaneous melanoma?"
]


for question in questions:

    print("\n")
    print("=" * 80)
    print("QUESTION:")
    print(question)
    print("=" * 80)

    results = vector_store.similarity_search(
        question,
        k=3
    )

    for i, result in enumerate(results, start=1):

        print(f"\n--- RESULT {i} ---")

        print("\nSource:")
        print(result.metadata.get("source", "Unknown"))

        print("\nPage:")
        print(result.metadata.get("page_label", "Unknown"))

        print("\nText:")
        print(result.page_content[:800])