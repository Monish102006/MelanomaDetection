from langchain_community.document_loaders import PyPDFDirectoryLoader

print("Loading PDF files...")

loader = PyPDFDirectoryLoader("./medical_literature")

documents = loader.load()

print(f"Loaded {len(documents)} pages.")

if documents:
    print("\nFirst page content:\n")
    print(documents[0].page_content[:1000])
else:
    print("No PDF files were found.")