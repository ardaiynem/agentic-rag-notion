from app.database.chroma_manager import ChromaManager
import numpy as np
import hashlib

class DuplicateChecker:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DuplicateChecker, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, chroma_client=None):
        if not hasattr(self, 'initialized'):  # Ensure __init__ is only called once
            self.chroma_client = chroma_client or ChromaManager()
            self.question_collection = self.chroma_client.get_collection("questions")
            self.answer_collection = self.chroma_client.get_collection("answers")
            self.initialized = True
        
    def _check_for_duplicate(self, query, collection, threshold=0.85):
        """Check if a query is a duplicate in the specified collection and return the Notion link if found."""
        results = collection.query(
            query_texts=[query],
            n_results=1,
            include=["distances", "metadatas"]
        )
        if len(results["distances"][0]) > 0:
            distance = results["distances"][0][0]
            similarity = 1 - distance
            if similarity > threshold:
                metadata = results["metadatas"][0][0]
                notion_link = metadata.get("notion_link", None)
                return True, notion_link
        return False, None
    
    def check_duplicates(self, question, generated_answer=None, threshold=0.8):
        """
        Check for duplicate questions or answers.
        Returns a tuple (is_duplicate, duplicate_type, notion_link) if duplicate found,
        otherwise (False, None, None).
        """
        # Check question similarity
        is_dup, link = self._check_for_duplicate(question, self.question_collection, threshold)
        if is_dup:
            return (True, "duplicate_question", link)
        
        # Check answer similarity if answer is provided
        if generated_answer:
            is_dup, link = self._check_for_duplicate(generated_answer, self.answer_collection, threshold)
            if is_dup:
                return (True, "duplicate_answer", link)
        
        return (False, None, None)

    def add_entry(self, question, answer, notion_link):
        """Add a new entry with its Notion link to the Chroma collections."""
        question_id = hashlib.md5(question.encode()).hexdigest()
        answer_id = hashlib.md5(answer.encode()).hexdigest()
        
        self.question_collection.add(
            documents=[question],
            ids=[question_id],
            metadatas=[{"notion_link": notion_link}]
        )
        self.answer_collection.add(
            documents=[answer],
            ids=[answer_id],
            metadatas=[{"notion_link": notion_link}]
        )