# RAG Pipeline Project

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline using ChromaDB as the vector database and a REST API built with FastAPI. Documents are stored in ChromaDB and retrieved at query time, then passed to a language model to generate responses. This is a work-in-progress and is not yet fully functional.

## TODO
- [x] Implement ChromaDB for document storage and retrieval
- [x] Implement FastAPI for serving a REST API
- [x] Implement Sentence Transformers for embedding and semantic search
- [x] Implement RAG pipeline for document retrieval and response generation
- [x] Implement Notion integration for creating Notion files from uploaded documents and questions about them
- [x] Implement functionality for detecting duplicate questions for avoiding redundant Notion file creation
- [-] Implement LLM agent for generating structured notion files (from markdown to notion maybe) and a self evaluation mechanism after content generation
- [x] Implement a frontend (ChatBot UI) for the API
- [ ] Implement reranking of retrieved documents based on the query with cross-encoding

## Features
- **ChromaDB** for vector storage and retrieval  
- **FastAPI** for serving a REST API  
- **Sentence Transformers** for embedding and semantic search  
- **Docker** support via Dockerfile and docker-compose  
- **Environment Variables** managed by pydantic-settings and `.env`  

## Requirements
You'll find the necessary Python packages in:
- `requirements.txt` for the main dependencies

Set up environment variables in `.env` (e.g. OpenAI, Notion keys).

## Installation
1. Clone this repository.  
2. Create a `.env` file (see example in the attachments).

## Usage
1. Run the project by using docker-compose (or FastAPI directly on local, but Docker is recommended for the orchestration of services like ChromaDB):

   ```
   docker compose up --build
    ```

    Note: The FastAPI service is exposed on port `5000`; ChromaDB is on port `8000`. If any of these ports are already in use, you can change them in the `docker-compose.yml` file.

2. Use endpoints to upload documents and query them as shown below with cURL. Postman can also be used for a more user-friendly experience.

### Upload a Document
```bash
curl -X POST http://localhost:5000/upload \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/document.pdf"
```

### Query the RAG Pipeline
```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"A question about the document."}'
```