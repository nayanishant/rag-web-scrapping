import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.firecrawl import FireCrawlLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

current_directory = os.path.dirname(os.path.abspath(__file__))
db_directory = os.path.join(current_directory, "db")
persistent_directory = os.path.join(db_directory, "db_apple")

gemini_api_key = os.getenv("GEMINI_API_KEY")


def create_vector_store():
    """Crawl the website, split the content, create embeddings and persist the vector store."""

    firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
    if not firecrawl_api_key:
        raise ValueError("FIRECRAWL_API_KEY environment variable not set.")
    
    loader = FireCrawlLoader(
        api_key=firecrawl_api_key, 
        url="https://www.apple.com", 
        mode="scrape",
        api_url="https://api.firecrawl.dev"
    )

    try:
        docs = loader.load()
        print(f"Loaded {len(docs)} documents")
        if not docs:
            raise ValueError("No documents loaded from Firecrawl. Check API key, credits, or URL.")
    except Exception as e:
        raise ValueError(f"Failed to load documents: {str(e)}")

    for doc in docs:
        for key, value in doc.metadata.items():
            if isinstance(value, list):
                doc.metadata[key] = ", ".join(map(str, value))
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True
    )
    
    split_docs = text_splitter.split_documents(docs)
    print(f"Split into {len(split_docs)} chunks")
    if not split_docs:
        raise ValueError("No document chunks created after splitting.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model = "models/gemini-embedding-001",
        google_api_key = gemini_api_key
    )

    db = Chroma.from_documents(
        split_docs,
        embeddings,
        persist_directory=persistent_directory
    )
    return db

if os.path.exists(persistent_directory):
    print(f"Vector store {persistent_directory} already exists. No need to initialize.")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=gemini_api_key
    )
    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embeddings
    )
    all_docs = db.similarity_search("Apple", k=5)
    print(f"Total documents in store: {len(all_docs)}")
    for i, doc in enumerate(all_docs, 1):
        print(f"Sample Document {i} preview: {doc.page_content[:200]}...")
else:
    db = create_vector_store()

def query_vector_store(query):
    """Query the vector store with the specified questions."""

    retriever = db.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k": 3}
    )

    relevant_doc = retriever.invoke(query)
    for i, doc in enumerate(relevant_doc, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

query = "iPhone17"

query_vector_store(query)