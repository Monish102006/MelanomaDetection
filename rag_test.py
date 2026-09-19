import os

from dotenv import load_dotenv

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI


# --------------------------------------------------
# 1. Load Gemini API key
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")


# --------------------------------------------------
# 2. Load embedding model
# --------------------------------------------------

print("Loading embedding model...")

embedding_model = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


# --------------------------------------------------
# 3. Connect to existing Qdrant database
# --------------------------------------------------

print("Connecting to Qdrant...")

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    collection_name="melanoma_guidelines",
    path="./qdrant_data"
)

print("Connected to Qdrant.")


# --------------------------------------------------
# 4. Connect to Gemini
# --------------------------------------------------

print("Connecting to Gemini...")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=api_key,
    temperature=0.1
)

print("Gemini connected.")


# --------------------------------------------------
# 5. User question
# --------------------------------------------------

question = "What does asymmetry mean in melanoma assessment?"


# --------------------------------------------------
# 6. Retrieve relevant medical evidence
# --------------------------------------------------

print("\nSearching medical knowledge base...")

results = vector_store.similarity_search(
    question,
    k=3
)


# --------------------------------------------------
# 7. Combine retrieved evidence
# --------------------------------------------------

context_parts = []

for i, result in enumerate(results, start=1):

    context_parts.append(
        f"""
SOURCE {i}

Text:
{result.page_content}

Source:
{result.metadata.get("source", "Unknown")}

Page:
{result.metadata.get("page_label", "Unknown")}
"""
    )

context = "\n".join(context_parts)


# --------------------------------------------------
# 8. Create RAG prompt
# --------------------------------------------------

prompt = f"""
You are an evidence-grounded melanoma research assistant.

Answer the user's question using the retrieved medical
literature below.

USER QUESTION:
{question}

RETRIEVED MEDICAL EVIDENCE:
{context}

RULES:

1. Use the retrieved evidence as the primary source.
2. Do not invent information.
3. Do not make a definitive medical diagnosis.
4. If the retrieved evidence is insufficient, clearly say so.
5. Give a concise explanation.
6. Mention the source/page when possible.
"""


# --------------------------------------------------
# 9. Generate answer
# --------------------------------------------------

print("\nGenerating answer with Gemini...")

response = llm.invoke(prompt)


# --------------------------------------------------
# 10. Display result
# --------------------------------------------------

print("\n")
print("=" * 70)
print("QUESTION")
print("=" * 70)

print(question)

print("\n")
print("=" * 70)
print("RETRIEVED EVIDENCE")
print("=" * 70)

print(context)

print("\n")
print("=" * 70)
print("GEMINI ANSWER")
print("=" * 70)

print(response.content)

print("\nRAG pipeline completed successfully.")