from app.database.notion_integration import NotionManager
from app.embeddings.embedding import SentenceTransformerModel
from smolagents import CodeAgent, tool, LiteLLMModel
import re
import logging
import os
from app.utils.duplicate_checker import DuplicateChecker
from app.database.chroma_manager import ChromaManager


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CODE_SYSTEM_PROMPT = """You are an expert assistant who can solve any task using code blobs. You will be given a task to solve as best you can.
  To do so, you have been given access to a list of tools: these tools are basically Python functions which you can call with code.
  To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences.

  At each step, in the 'Thought:' sequence, you should first explain your reasoning towards solving the task and the tools that you want to use.
  Then in the 'Code:' sequence, you should write the code in simple Python. The code sequence must end with '<end_code>' sequence.
  During each intermediate step, you can use 'print()' to save whatever important information you will then need.
  These print outputs will then appear in the 'Observation:' field, which will be available as input for the next step.
  In the end you have to return a final answer using the `final_answer` tool.

  Here are a few examples using notional tools:
  ---
  Task: "Generate an image of the oldest person in this document."

  Thought: I will proceed step by step and use the following tools: `document_qa` to find the oldest person in the document, then `image_generator` to generate an image according to the answer.
  Code:
  ```py
  answer = document_qa(document=document, question="Who is the oldest person mentioned?")
  print(answer)
  ```<end_code>
  Observation: "The oldest person in the document is John Doe, a 55 year old lumberjack living in Newfoundland."

  Thought: I will now generate an image showcasing the oldest person.
  Code:
  ```py
  image = image_generator("A portrait of John Doe, a 55-year-old man living in Canada.")
  final_answer(image)
  ```<end_code>

  ---
  Task: "What is the result of the following operation: 5 + 3 + 1294.678?"

  Thought: I will use python code to compute the result of the operation and then return the final answer using the `final_answer` tool
  Code:
  ```py
  result = 5 + 3 + 1294.678
  final_answer(result)
  ```<end_code>

  ---
  Task:
  "Answer the question in the variable `question` about the image stored in the variable `image`. The question is in French.
  You have been provided with these additional arguments, that you can access using the keys as variables in your python code:
  {'question': 'Quel est l'animal sur l'image?', 'image': 'path/to/image.jpg'}"

  Thought: I will use the following tools: `translator` to translate the question into English and then `image_qa` to answer the question on the input image.
  Code:
  ```py
  translated_question = translator(question=question, src_lang="French", tgt_lang="English")
  print(f"The translated question is {translated_question}.")
  answer = image_qa(image=image, question=translated_question)
  final_answer(f"The answer is {answer}")
  ```<end_code>

  ---
  Task:
  In a 1979 interview, Stanislaus Ulam discusses with Martin Sherwin about other great physicists of his time, including Oppenheimer.
  What does he say was the consequence of Einstein learning too much math on his creativity, in one word?

  Thought: I need to find and read the 1979 interview of Stanislaus Ulam with Martin Sherwin.
  Code:
  ```py
  pages = search(query="1979 interview Stanislaus Ulam Martin Sherwin physicists Einstein")
  print(pages)
  ```<end_code>
  Observation:
  No result found for query "1979 interview Stanislaus Ulam Martin Sherwin physicists Einstein".

  Thought: The query was maybe too restrictive and did not find any results. Let's try again with a broader query.
  Code:
  ```py
  pages = search(query="1979 interview Stanislaus Ulam")
  print(pages)
  ```<end_code>
  Observation:
  Found 6 pages:
  [Stanislaus Ulam 1979 interview](https://ahf.nuclearmuseum.org/voices/oral-histories/stanislaus-ulams-interview-1979/)

  [Ulam discusses Manhattan Project](https://ahf.nuclearmuseum.org/manhattan-project/ulam-manhattan-project/)

  (truncated)

  Thought: I will read the first 2 pages to know more.
  Code:
  ```py
  for url in ["https://ahf.nuclearmuseum.org/voices/oral-histories/stanislaus-ulams-interview-1979/", "https://ahf.nuclearmuseum.org/manhattan-project/ulam-manhattan-project/"]:
      whole_page = visit_webpage(url)
      print(whole_page)
      print("\n" + "="*80 + "\n")  # Print separator between pages
  ```<end_code>
  Observation:
  Manhattan Project Locations:
  Los Alamos, NM
  Stanislaus Ulam was a Polish-American mathematician. He worked on the Manhattan Project at Los Alamos and later helped design the hydrogen bomb. In this interview, he discusses his work at
  (truncated)

  Thought: I now have the final answer: from the webpages visited, Stanislaus Ulam says of Einstein: "He learned too much mathematics and sort of diminished, it seems to me personally, it seems to me his purely physics creativity." Let's answer in one word.
  Code:
  ```py
  final_answer("diminished")
  ```<end_code>

  ---
  Task: "Which city has the highest population: Guangzhou or Shanghai?"

  Thought: I need to get the populations for both cities and compare them: I will use the tool `search` to get the population of both cities.
  Code:
  ```py
  for city in ["Guangzhou", "Shanghai"]:
      print(f"Population {city}:", search(f"{city} population")
  ```<end_code>
  Observation:
  Population Guangzhou: ['Guangzhou has a population of 15 million inhabitants as of 2021.']
  Population Shanghai: '26 million (2019)'

  Thought: Now I know that Shanghai has the highest population.
  Code:
  ```py
  final_answer("Shanghai")
  ```<end_code>

  ---
  Task: "What is the current age of the pope, raised to the power 0.36?"

  Thought: I will use the tool `wiki` to get the age of the pope, and confirm that with a web search.
  Code:
  ```py
  pope_age_wiki = wiki(query="current pope age")
  print("Pope age as per wikipedia:", pope_age_wiki)
  pope_age_search = web_search(query="current pope age")
  print("Pope age as per google search:", pope_age_search)
  ```<end_code>
  Observation:
  Pope age: "The pope Francis is currently 88 years old."

  Thought: I know that the pope is 88 years old. Let's compute the result using python code.
  Code:
  ```py
  pope_current_age = 88 ** 0.36
  final_answer(pope_current_age)
  ```<end_code>

  Above example were using notional tools that might not exist for you. On top of performing computations in the Python code snippets that you create, you only have access to these tools:
  {%- for tool in tools.values() %}
  - {{ tool.name }}: {{ tool.description }}
      Takes inputs: {{tool.inputs}}
      Returns an output of type: {{tool.output_type}}
  {%- endfor %}

  {%- if managed_agents and managed_agents.values() | list %}
  You can also give tasks to team members.
  Calling a team member works the same as for calling a tool: simply, the only argument you can give in the call is 'task', a long string explaining your task.
  Given that this team member is a real human, you should be very verbose in your task.
  Here is a list of the team members that you can call:
  {%- for agent in managed_agents.values() %}
  - {{ agent.name }}: {{ agent.description }}
  {%- endfor %}
  {%- else %}
  {%- endif %}

  Here are the rules you should always follow to solve your task:
  1. Always provide a 'Thought:' sequence, and a 'Code:\n```py' sequence ending with '```<end_code>' sequence, else you will fail.
  2. Use only variables that you have defined!
  3. Always use the right arguments for the tools. DO NOT pass the arguments as a dict as in 'answer = wiki({'query': "What is the place where James Bond lives?"})', but use the arguments directly as in 'answer = wiki(query="What is the place where James Bond lives?")'.
  4. Take care to not chain too many sequential tool calls in the same code block, especially when the output format is unpredictable. For instance, a call to search has an unpredictable return format, so do not have another tool call that depends on its output in the same block: rather output results with print() to use them in the next block.
  5. Call a tool only when needed, and never re-do a tool call that you previously did with the exact same parameters.
  6. Don't name any new variable with the same name as a tool: for instance don't name a variable 'final_answer'.
  7. Never create any notional variables in our code, as having these in your logs will derail you from the true variables.
  8. You can use imports in your code, but only from the following list of modules: {{authorized_imports}}
  9. The state persists between code executions: so if in one step you've created variables or imported modules, these will all persist.
  10. Don't give up! You're in charge of solving the task, not providing directions to solve it.

  Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000."""

modified_system_prompt = CODE_SYSTEM_PROMPT + "\n" + """You are an AI assistant that helps in generating comprehensive Notion pages based on user queries.

Retrieve relevant documents using the embedding model and vector store with the tool `retrieve_documents`.

Then, generate a Notion page in markdown format with the exact headers below:
- # [Your Page Title]
- ## Summary
- ## Key Points
- ## References
- ## Action Items"""


class NotionRAGAgent:
    # Define structural requirements for generated content
    required_sections = {
        "summary": {"min_length": 100, "max_length": 500},
        "key_points": {"min_items": 3, "max_items": 10},
        "references": {"min_items": 1},
        "action_items": {
            "min_items": 2,
            "actionable_verbs": ["create", "update", "review", "implement"],
        },
    }

    vector_store = ChromaManager()
    embedding_model = SentenceTransformerModel()
    duplicate_checker = DuplicateChecker()
    notion_manager = NotionManager()
    notion_client = notion_manager.get_notion_client()

    # @tool
    def check_duplicate(question: str) -> str | None:
        """
        Check if the question-answer pair already exists in the Notion database.
        If question is duplicate returns duplication information and duplicate question's Notion link.
        Otherwise, returns None.

        Args:
            question: The question to check for duplicates.
        """
        # Check if the question-answer pair already exists in the Notion database
        is_duplicate, reason, notion_link = NotionRAGAgent.duplicate_checker.check_duplicates(question)
        if is_duplicate:
            logger.info(
                f"AGENTIC RAG (check_duplicate): Duplicate {reason} found - skipping generation"
            )
            return f"Duplicate {reason} found - skipping generation" + "\n\n" + f"Answer stored in Notion: {notion_link}"

        return None
    

    @tool
    def retrieve_documents(question: str) -> dict:
        """
        Implements your retrieval logic: embed query, query vector store, and return retrieved chunks.

        Args:
            question: The question to retrieve documents for.
        """
        query_embed = NotionRAGAgent.embedding_model.embed_query(question)
        results = NotionRAGAgent.vector_store.query(query_embed, n_results=10)
        documents = results["documents"][0]
        distances = results["distances"][0]
        retrieved_chunks = [
            {"text": documents[i], "distance": distances[i]}
            for i in range(len(documents))
        ]
        logger.info(
            f"AGENTIC RAG (retrieve_documents): {len(retrieved_chunks)} documents for question: {question}"
        )
        return {"documents": retrieved_chunks}
    
    
    model = LiteLLMModel(model_id="openai/gpt-4o", api_key=OPENAI_API_KEY)
    agent = CodeAgent(tools=[retrieve_documents], model=model)
    agent.prompt_templates["system_prompt"] = modified_system_prompt


    @staticmethod
    def _parse_markdown_sections(markdown: str) -> dict:
        EXPECTED_SECTIONS = {"summary", "key_points", "references", "action_items"}
        sections = {section: "" for section in EXPECTED_SECTIONS}
        sections["title"] = "Generated Page"

        # Extract title
        title_match = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
        if title_match:
            sections["title"] = title_match.group(1).strip()

        current_section = None
        current_content = []

        for line in markdown.splitlines():
            h2_match = re.match(r"^##\s+(.+)$", line)
            if h2_match:
                section_name = h2_match.group(1).strip().lower().replace(" ", "_")
                if section_name in EXPECTED_SECTIONS:
                    if current_section:
                        sections[current_section] = "\n".join(current_content).strip()
                    current_section = section_name
                    current_content = []
                else:
                    current_section = None
            elif current_section:
                # Clean list markers for list-based sections
                processed_line = line.strip()
                if current_section in ["key_points", "references", "action_items"]:
                    processed_line = re.sub(r'^(\s*[-*•]|\s*\d+\.)\s*', '', processed_line)
                current_content.append(processed_line)

        if current_section:
            sections[current_section] = "\n".join(current_content).strip()

        return sections

    @staticmethod
    def validate_structure(content: dict) -> dict:
        errors = {}
        requirements = NotionRAGAgent.required_sections

        for section, criteria in requirements.items():
            section_content = content.get(section, "").strip()
            if not section_content:
                errors[section] = ["Section is missing or empty"]
                continue

            if section == "summary":
                content_len = len(section_content)
                min_len = criteria["min_length"]
                max_len = criteria["max_length"]
                if content_len < min_len:
                    errors.setdefault(section, []).append(
                        f"Too short ({content_len} < {min_len} chars)"
                    )
                if content_len > max_len:
                    errors.setdefault(section, []).append(
                        f"Too long ({content_len} > {max_len} chars)"
                    )

            elif section == "key_points":
                items = [line for line in section_content.split('\n') if line.strip()]
                item_count = len(items)
                if item_count < criteria["min_items"]:
                    errors.setdefault(section, []).append(
                        f"Need {criteria['min_items']} items, got {item_count}"
                    )
                if item_count > criteria["max_items"]:
                    errors.setdefault(section, []).append(
                        f"Max {criteria['max_items']} items, got {item_count}"
                    )

            elif section == "references":
                items = [line for line in section_content.split('\n') if line.strip()]
                if len(items) < criteria["min_items"]:
                    errors.setdefault(section, []).append(
                        f"Need at least {criteria['min_items']} references"
                    )

            elif section == "action_items":
                items = [line.strip() for line in section_content.split('\n') if line.strip()]
                if len(items) < criteria["min_items"]:
                    errors.setdefault(section, []).append(
                        f"Need {criteria['min_items']} action items, got {len(items)}"
                    )
                
                invalid_actions = []
                for i, item in enumerate(items, 1):
                    if not any(item.lower().startswith(verb) for verb in criteria["actionable_verbs"]):
                        invalid_actions.append(
                            f"Item {i}: '{item}' must start with {criteria['actionable_verbs']}"
                        )
                if invalid_actions:
                    errors.setdefault(section, []).extend(invalid_actions)

        return errors

    @staticmethod
    def _create_block(block_type: str, text: str) -> dict:
        # Use the new API field "rich_text" inside each block type.
        return {
            "object": "block",
            "type": block_type,
            block_type: {
                "rich_text": [{"type": "text", "text": {"content": text}}]
            },
        }

    # @staticmethod
    # def _build_page_properties(content: dict) -> dict:
    #     # Make sure the property names match exactly your database schema.
    #     return {
    #         "Title": {
    #             "title": [{"text": {"content": content.get("title", "Generated Page")}}]
    #         },
    #         "Status": {"select": {"name": "Draft"}},
    #     }
    
    @staticmethod
    def _build_page_properties(content: dict) -> dict:
        return {
            "Name": {
                "title": [
                    {
                        "text": {
                            "content": content.get("title", "Generated Page")
                        }
                    }
                ]
            }
        }

    @staticmethod
    def _convert_to_notion_blocks(content: dict) -> list:
        blocks = []
        # Summary
        blocks.append(NotionRAGAgent._create_block("heading_2", "Summary"))
        blocks.append(NotionRAGAgent._create_block("paragraph", content.get("summary", "")))
        # Key Points
        blocks.append(NotionRAGAgent._create_block("heading_2", "Key Points"))
        for point in content.get("key_points", "").splitlines():
            if point.strip():
                blocks.append(NotionRAGAgent._create_block("numbered_list_item", point.strip()))
        # References (placed before Action Items)
        blocks.append(NotionRAGAgent._create_block("heading_2", "References"))
        for ref in content.get("references", "").splitlines():
            if ref.strip():
                blocks.append(NotionRAGAgent._create_block("bulleted_list_item", ref.strip()))
        # Action Items
        blocks.append(NotionRAGAgent._create_block("heading_2", "Action Items"))
        for item in content.get("action_items", "").splitlines():
            if item.strip():
                blocks.append(NotionRAGAgent._create_block("to_do", item.strip()))
        return blocks

    # @tool
    def create_notion_page(content: dict) -> dict:
        """
        Create a Notion page using the new Notion API.
        """
        children = NotionRAGAgent._convert_to_notion_blocks(content)
        properties = NotionRAGAgent._build_page_properties(content)
        database_id = NotionRAGAgent.notion_manager.get_database_id()
        try:
            new_page = NotionRAGAgent.notion_client.pages.create(
                parent={"database_id": database_id},
                properties=properties,
                children=children,
            )
            return {"success": True, "page_id": new_page["id"]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def process_query(question: str) -> dict:
        """
        Process the user query to generate a Notion page with the response.
        """
        # Step 1: Check for duplicates
        duplicate_response = NotionRAGAgent.check_duplicate(question)
        if duplicate_response:
            return {"answer": duplicate_response}
        
        # Step 2: Retrieve relevant documents and generate markdown content
        generated_markdown = NotionRAGAgent.agent.run(question).to_string()
        logger.info(f"Generated markdown:\n{generated_markdown}")

        # Step 2.5: Parse the markdown into structured content
        content = NotionRAGAgent._parse_markdown_sections(generated_markdown)
        
        # Step 3: Validate and create Notion page
        creation_result = NotionRAGAgent.create_notion_page(content)
        logger.info(f"Page creation result: {creation_result}")
        notion_link = notion_link["url"]

        # Step 4: Store question & answer in Notion
        NotionRAGAgent.duplicate_checker.add_entry(question, generated_markdown, notion_link)
        
        if creation_result.get("success"):
            return {
                "answer": f"Page created successfully! Notion page ID: {creation_result['page_id']}",
                "notion_link": f"https://www.notion.so/{creation_result['page_id'].replace('-', '')}"
            }
        else:
            return {
                "answer": f"Failed to create page. Error: {creation_result.get('error', 'Unknown error')}",
                "validation_errors": creation_result.get("error")
            }