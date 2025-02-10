# from smolagents import Agent, Tool
# import re
# import logging
# from typing import Dict, List, Any

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class NotionPageAgent(Agent):
#     def __init__(self, notion_client, llm_client):
#         self.notion = notion_client
#         self.llm = llm_client
#         super().__init__()
#         self.tools = [
#             Tool(
#                 name="generate_markdown_content",
#                 func=self.generate_content,
#                 description="Generates structured markdown content for Notion page with semantic validation"
#             ),
#             Tool(
#                 name="validate_page_structure",
#                 func=self.validate_structure,
#                 description="Validates structure, formatting, and content quality of Notion page"
#             ),
#             Tool(
#                 name="create_notion_page",
#                 func=self.create_page,
#                 description="Creates Notion page with automatic retries and rich content conversion"
#             ),
#             Tool(
#                 name="update_existing_page",
#                 func=self.update_page,
#                 description="Updates existing Notion page with new content blocks"
#             )
#         ]
        
#         self.required_sections = {
#             "summary": {"min_length": 100, "max_length": 500},
#             "key_points": {"min_items": 3, "max_items": 10},
#             "references": {"min_items": 1},
#             "action_items": {"min_items": 2, "actionable_verbs": ["create", "update", "review", "implement"]}
#         }

#     def generate_content(self, context: str) -> Dict:
#         """Generate and parse structured content with LLM validation"""
#         prompt = f"""Generate a comprehensive Notion page in markdown format with these exact section headers:
#         # [Your Page Title]
        
#         ## Summary
#         ## Key Points
#         ## References
#         ## Action Items
        
#         Context: {context}"""
        
#         try:
#             # Actual LLM call with temperature control for consistent structure
#             raw_markdown = self.llm.generate(
#                 prompt=prompt,
#                 temperature=0.3,
#                 max_tokens=2000
#             )
            
#             # Parse generated content into structured format
#             parsed_content = self._parse_markdown_sections(raw_markdown)
            
#             # Add semantic validation
#             validation = self.validate_structure(parsed_content)
#             if not validation["valid"]:
#                 logger.warning(f"Content validation failed: {validation['feedback']}")
#                 return self._regenerate_content(context, validation["feedback"])
            
#             return parsed_content
            
#         except Exception as e:
#             logger.error(f"Content generation failed: {str(e)}")
#             return {"error": str(e)}

#     def _parse_markdown_sections(self, markdown: str) -> Dict:
#         """Parse markdown into structured sections using header detection"""
#         sections = {}
#         current_section = None
#         current_content = []
        
#         # Extract title from first h1 header
#         title_match = re.search(r'^#\s+(.+)$', markdown, re.MULTILINE)
#         sections["title"] = title_match.group(1).strip() if title_match else "Generated Page"
        
#         for line in markdown.split('\n'):
#             header_match = re.match(r'^##+\s+(.+)$', line)
#             if header_match:
#                 if current_section:
#                     sections[current_section] = '\n'.join(current_content).strip()
#                 current_section = header_match.group(1).lower().replace(' ', '_')
#                 current_content = []
#             elif current_section:
#                 current_content.append(line)
        
#         if current_section:
#             sections[current_section] = '\n'.join(current_content).strip()
            
#         return sections

#     def validate_structure(self, content: Dict) -> Dict:
#         """Comprehensive content validation with LLM quality checks"""
#         # Structural validation
#         missing_sections = [section for section in self.required_sections 
#                           if section not in content]
#         if missing_sections:
#             return {
#                 "valid": False,
#                 "feedback": f"Missing required sections: {', '.join(missing_sections)}"
#             }

#         # Content quality validation using LLM
#         validation_prompt = f"""Evaluate this content for a Notion page. Check for:
#         - Summary clarity and conciseness
#         - Key points relevance and logical ordering
#         - Action items specificity and action verbs
#         - Reference credibility
        
#         Content: {content}
        
#         Provide feedback as: VALID|INVALID|FEEDBACK:..."""
        
#         validation_result = self.llm.generate(validation_prompt, temperature=0)
#         if "INVALID" in validation_result:
#             return {
#                 "valid": False,
#                 "feedback": validation_result.split("FEEDBACK:")[-1].strip()
#             }
            
#         return {"valid": True, "feedback": "Content passed all quality checks"}

#     def create_page(self, database_id: str, content: Dict) -> Dict:
#         """Create page with retries and rich content conversion"""
#         try:
#             children = self._convert_to_notion_blocks(content)
#             properties = self._build_page_properties(content)
            
#             new_page = self.notion.pages.create(
#                 parent={"database_id": database_id},
#                 properties=properties,
#                 children=children
#             )
            
#             logger.info(f"Created page {new_page['id']} in database {database_id}")
#             return {"success": True, "page_id": new_page["id"]}
            
#         except Exception as e:
#             logger.error(f"Page creation failed: {str(e)}")
#             return {"success": False, "error": str(e)}

#     def _convert_to_notion_blocks(self, content: Dict) -> List[Dict]:
#         """Convert parsed content to Notion blocks with rich text formatting"""
#         blocks = []
        
#         # Add summary as paragraph block
#         blocks.append(self._create_text_block("Summary", "heading_2"))
#         blocks.append(self._create_text_block(content["summary"]))
        
#         # Key Points as numbered list
#         blocks.append(self._create_text_block("Key Points", "heading_2"))
#         for point in content.get("key_points", []):
#             blocks.append(self._create_list_item(point, "numbered"))
        
#         # Action Items as todo list
#         blocks.append(self._create_text_block("Action Items", "heading_2"))
#         for item in content.get("action_items", []):
#             blocks.append(self._create_todo_item(item))
        
#         # References as bulleted list with links
#         blocks.append(self._create_text_block("References", "heading_2"))
#         for ref in content.get("references", []):
#             blocks.append(self._create_link_item(ref))
        
#         return blocks

#     def _create_text_block(self, text: str, block_type: str = "paragraph") -> Dict:
#         """Create a Notion text block with markdown formatting support"""
#         return {
#             "object": "block",
#             "type": block_type,
#             block_type: {
#                 "rich_text": [self._parse_inline_formatting(text)]
#             }
#         }

#     def _parse_inline_formatting(self, text: str) -> Dict:
#         """Convert markdown formatting to Notion rich text annotations"""
#         # This is a simplified version - consider using a proper markdown parser
#         return {
#             "type": "text",
#             "text": {"content": re.sub(r'[*_`]', '', text)},
#             "annotations": {
#                 "bold": "**" in text,
#                 "italic": "_" in text,
#                 "code": "`" in text
#             }
#         }

#     def _build_page_properties(self, content: Dict) -> Dict:
#         """Construct page properties including custom metadata"""
#         return {
#             "Title": {
#                 "title": [{"text": {"content": content.get("title", "Generated Page")}}]
#             },
#             "Status": {
#                 "select": {"name": content.get("status", "Draft")}
#             },
#             "Tags": {
#                 "multi_select": [{"name": tag} for tag in content.get("tags", [])]
#             }
#         }

#     def _regenerate_content(self, context: str, feedback: str) -> Dict:
#         """Retry content generation with validation feedback"""
#         logger.info("Regenerating content with validation feedback")
#         retry_prompt = f"""Previous content failed validation: {feedback}
#         Please regenerate the content with these improvements.
#         Original context: {context}"""
        
#         return self.generate_content(retry_prompt)

#     def update_page(self, page_id: str, new_content: Dict) -> Dict:
#         """Update existing page with new content blocks"""
#         try:
#             children = self._convert_to_notion_blocks(new_content)
#             self.notion.blocks.append(block_id=page_id, children=children)
#             return {"success": True, "page_id": page_id}
#         except Exception as e:
#             return {"success": False, "error": str(e)}