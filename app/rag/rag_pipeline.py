from app.agents.notion_agent import NotionRAGAgent
from app.utils.duplicate_checker import DuplicateChecker
# from app.agents.notion_agent import NotionPageAgent
from app.database.notion_integration import NotionManager
from app.embeddings.embedding import OpenAIEmbeddingModel


class RAGPipeline:
    def __init__(self, embedding_model, vector_store, response_generator):
        self.duplicate_checker = DuplicateChecker(vector_store)
        self.notion_manager = NotionManager()
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.response_generator = response_generator

    def process_query(self, question, context):
        # Step 1: Check for duplicates
        is_duplicate, reason, notion_link = self.duplicate_checker.check_duplicates(question)
        if is_duplicate:
            return {
                # "results": results,
                "answer": f"Duplicate {reason} found - skipping generation"
                + "\n\n"
                + f"Answer stored in Notion: {notion_link}"
            }

        # Step 2: Retrieve relevant documents
        query_embed = self.embedding_model.embed_query(question)
        results = self.vector_store.query(query_embed, n_results=10)
        documents = results["documents"][0]
        distances = results["distances"][0]
        retrieved_chunks = [
            {"text": documents[i], "distance": distances[i]}
            for i in range(len(documents))
        ]

        # Step 2: Generate content
        answer = self.response_generator.generate_response(question, retrieved_chunks)
        # content = self.notion_agent.generate_content(context)

        # Step 3: Check answer duplicate
        is_duplicate, reason, notion_link = self.duplicate_checker.check_duplicates(question, answer)
        if is_duplicate:
            return {
                # "results": results,
                "answer": f"Duplicate {reason} found - skipping generation"
                + "\n\n"
                + f"Answer stored in Notion: {notion_link}"
            }

        # Step 4: Create Notion page if not a duplicate
        notion_link = self.notion_manager.create_notion_page(question, answer)
        notion_link = notion_link["url"]

        # Step 5: Store question & answer in Notion
        self.duplicate_checker.add_entry(question, answer, notion_link)

        return {
            # "results": results,
            "answer": answer
            + "\n\n"
            + f"Answer stored in Notion: {notion_link}"
        }
    
    def agentic_rag(self, question, context):
        notion_agent = NotionRAGAgent
        result = notion_agent.process_query(question)
        return result
