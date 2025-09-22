RAG Web Scraping Project
This project implements a Retrieval-Augmented Generation (RAG) system to scrape content from https://www.apple.com, store it in a Chroma vector store, and query it using natural language queries (e.g., "iPhone", "WWDC24"). It leverages LangChain with firecrawl-py==0.0.13 for web scraping and Google Gemini embeddings (models/embedding-001) for vector search.

Features

Web Scraping: Uses FireCrawlLoader to scrape content from https://www.apple.com.
Vector Store: Stores scraped content as embeddings in a Chroma database for efficient retrieval.
Similarity Search: Queries the vector store to retrieve up to 3 relevant documents based on cosine similarity.
Debugging: Includes code to inspect vector store contents for verification.

Prerequisites

Python 3.13 or higher
Firecrawl API key (free tier: ~500 credits)
Google Gemini API key for embeddings
Git for version control

Setup
1. Clone the Repository
git clone <repository-url>
cd rag-web-scrapping

2. Create and Activate Virtual Environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

3. Install Dependencies
Install packages listed in requirements.txt:
pip install -r requirements.txt

requirements.txt:
firecrawl-py==0.0.13
langchain>=0.3.0
langchain-community>=0.3.0
langchain-google-genai>=2.0.0
langchain-chroma>=0.1.4
chromadb>=0.5.7
python-dotenv>=1.0.1

4. Configure Environment Variables
Create a .env file in the project root:
touch .env

Add:
FIRECRAWL_API_KEY=fc-YOUR-FIRECRAWL-API-KEY
GEMINI_API_KEY=your-gemini-api-key
