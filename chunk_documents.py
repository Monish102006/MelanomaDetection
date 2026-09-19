from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("1. Loading PDF files...")

loader = PyPDFDirectoryLoader("./medical_literature")
documents = loader.load()

print(f"Loaded {len(documents)} pages.")

print("\n2. Splitting documents into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=120
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")

print("\n3. Showing first 3 chunks:")

for i, chunk in enumerate(chunks[:3]):
    print(f"\n========== CHUNK {i + 1} ==========")
    print(chunk.page_content)
    print("===================================")