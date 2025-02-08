import logging
import os
from notion_client import Client
from config.settings import settings

class NotionManager:
    def __init__(self):
        self.notion = Client(auth=settings.notion_api_key, log_level=logging.DEBUG)
        self.database_id = settings.notion_database_id

    def create_notion_page(self, question_text: str, response_text: str) -> dict:
        """
        Create a new Notion page with the given RAG response text, splitting into chunks to adhere to Notion's text limits.
        """
        # Split the response text into chunks of 2000 characters each
        text_chunks = [response_text[i:i+2000] for i in range(0, len(response_text), 2000)]
        
        # Create rich_text elements for each chunk
        rich_text_elements = [
            {
                "type": "text",
                "text": {"content": chunk}
            }
            for chunk in text_chunks
        ]
        
        # Create a single paragraph block containing all rich_text elements
        children_blocks = [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": rich_text_elements
                }
            }
        ]
        
        return self.notion.pages.create(
            parent={"database_id": self.database_id},
            properties={
                "Name": {
                    "title": [
                        {
                            "text": {
                                "content": f"Q: {question_text}"
                            }
                        }
                    ]
                }
            },
            children=children_blocks
        )

    def search_notion(self, question):
        """
        Search Notion for existing answers.
        """
        query = self.notion.search(query=question, filter={"property": "object", "value": "page"}).get("results", [])
        # for page in query:
        #     page_content = self.notion.blocks.children.list(page["id"]).get("results", [])
        #     if question.lower() in page_content.lower():
        #         return page["url"]  # Return the link if the question exists
        if query:
            return query[0]["url"]