import openai
import tiktoken
from config.settings import settings

class ResponseGenerator:
    def __init__(self):
        # Initialize the OpenAI client with the API key
        self.client = openai.OpenAI(api_key=settings.openai_api_key)

    def generate_response(
        self,
        query: str,
        chunks: list[dict],
        model: str = "gpt-4o",
        max_context_tokens: int = 4096,
        temperature: float = 0.7
    ) -> str:
        """
        Generates a response using an LLM, given a query and retrieved chunks.
        
        Args:
            query: User's question/input.
            chunks: List of chunks (each with "text" and "similarity" keys).
            model: LLM model name.
            max_context_tokens: Max tokens allowed for context + prompt.
            temperature: Controls response randomness.
        Returns:
            Generated answer.
        """
        # Sort chunks by similarity (descending order)
        chunks_sorted = sorted(chunks, key=lambda x: x["distance"], reverse=True)
        
        # Initialize tokenizer for the specified model
        tokenizer = tiktoken.encoding_for_model(model)
        
        # Build context within token limits
        system_prompt = "You are a helpful assistant. Answer the question using the provided context. If unsure, state that."
        user_prompt = f"Question: {query}\nContext:\n"
        
        # Calculate token overhead for non-context parts of the prompt
        overhead_tokens = len(tokenizer.encode(system_prompt + user_prompt))
        remaining_tokens = max_context_tokens - overhead_tokens
        
        # Accumulate context text without exceeding token limits
        context_texts = []
        total_tokens = 0
        for chunk in chunks_sorted:
            chunk_text = chunk["text"]
            chunk_token_count = len(tokenizer.encode(chunk_text))
            if total_tokens + chunk_token_count <= remaining_tokens:
                context_texts.append(chunk_text)
                total_tokens += chunk_token_count
            else:
                # Add a truncated version of the chunk if space remains
                available_tokens = remaining_tokens - total_tokens
                if available_tokens > 0:
                    truncated_text = tokenizer.decode(
                        tokenizer.encode(chunk_text)[:available_tokens]
                    )
                    context_texts.append(truncated_text)
                    total_tokens += available_tokens
                break
        
        # Combine context into a single string
        context = "\n\n".join(context_texts)
        
        # Construct the final prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{user_prompt}{context}"}
        ]
        
        # Call the LLM API using the new client
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        
        # Extract and return the generated response
        return response.choices[0].message.content