"""Translation service for natural language to GraphQL conversion."""

import json
import logging
import re
import time
import httpx
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from services.ollama_service import OllamaService
from config.settings import get_settings
from models.translation import TranslationResult
from config.icl_examples import get_initial_icl_examples

logger = logging.getLogger(__name__)


class TranslationService:
    """Service for translating natural language to GraphQL queries."""

    def __init__(self):
        self.settings = get_settings()
        self.ollama_service = OllamaService()
        
    async def _broadcast_interaction(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        raw_response: str,
        processed_response: str,
        processing_time: float,
        confidence: float,
        warnings: List[str],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Broadcast model interaction to live stream."""
        try:
            interaction_data = {
                'id': f"trans_{int(time.time() * 1000)}",
                'timestamp': time.time(),
                'model': model,
                'type': 'translation',
                'processing_time': processing_time,
                'user_id': user_id,
                'session_id': session_id,
                
                # Full prompt details
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'full_prompt': f"System: {system_prompt[:200]}...\n\nUser: {user_prompt}",
                'parameters': {
                    'temperature': 0.3,
                    'max_tokens': 2048,
                    'model': model
                },
                
                # Response details
                'raw_response': raw_response,
                'processed_response': processed_response,
                'confidence': confidence,
                'warnings': warnings,
                'response_metadata': {},
                
                'status': 'completed',
                'error': None,
                'tokens_used': len(user_prompt.split()) + len(raw_response.split()),
                'response_tokens': len(raw_response.split()),
                'total_tokens': len(user_prompt.split()) + len(raw_response.split())
            }
            
            # Send to interactions broadcast endpoint
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"http://localhost:{self.settings.api.port}/api/interactions/broadcast",
                    json=interaction_data,
                    timeout=5.0
                )
                
        except Exception as e:
            logger.warning(f"Failed to broadcast interaction: {e}")

    def _build_system_prompt(self, schema_context: str = "", icl_examples: List[str] = None) -> str:
        """Build system prompt for GraphQL translation."""
        base_prompt = """You are a GraphQL expert. Translate natural language to GraphQL queries.

Rules:
1. Use correct GraphQL syntax
2. Match fields to query intent
3. Return only necessary data

Respond with JSON:
{
  "graphql": "query string",
  "confidence": 0.0-1.0,
  "explanation": "reasoning",
  "warnings": ["issues"],
  "suggestions": ["improvements"]
}

Only return JSON, no extra text."""

        if schema_context:
            base_prompt += f"\n\nSchema Context:\n{schema_context}"
            
        if icl_examples:
            base_prompt += "\n\nExamples:\n"
            for i, example in enumerate(icl_examples, 1):
                base_prompt += f"{i}. {example}\n"
        
        return base_prompt

    def _extract_json_from_response(self, response: str) -> Dict:
        """Extract JSON object from model response."""
        # Remove markdown code blocks
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        
        # Try to find JSON object in the response - improved pattern for nested structures
        # This pattern looks for balanced braces
        def find_balanced_json(text):
            stack = []
            start = -1
            for i, char in enumerate(text):
                if char == '{':
                    if not stack:
                        start = i
                    stack.append(char)
                elif char == '}':
                    if stack:
                        stack.pop()
                        if not stack and start != -1:
                            try:
                                json_str = text[start:i+1]
                                parsed = json.loads(json_str)
                                if "graphql" in parsed:
                                    return parsed
                            except json.JSONDecodeError:
                                pass
            return None
        
        # Try to find balanced JSON
        result = find_balanced_json(response)
        if result:
            return result
        
        # Fallback: try parsing the entire response
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            # If all else fails, return a structured error with helpful message
            return {
                "graphql": "# Error: Invalid or unclear query",
                "confidence": 0.0,
                "explanation": f"The query '{response[:100]}...' could not be understood. Please provide a clear natural language query about what data you want to retrieve.",
                "warnings": ["Query parsing failed", "Please provide a clearer query"],
                "suggestions": [
                    "Try asking for specific data like 'get all users'",
                    "Use clear language like 'find posts by author'", 
                    "Specify what fields you want like 'show user names and emails'"
                ]
            }

    def _validate_graphql_syntax(self, query: str) -> Tuple[bool, List[str]]:
        """Basic GraphQL syntax validation."""
        warnings = []
        
        # Check for basic GraphQL structure
        if not query.strip():
            return False, ["Empty query"]
        
        # Check for basic query structure
        if not any(keyword in query for keyword in ['query', 'mutation', 'subscription', '{']):
            warnings.append("Query may be missing proper GraphQL structure")
        
        # Check for balanced braces
        open_braces = query.count('{')
        close_braces = query.count('}')
        if open_braces != close_braces:
            warnings.append(f"Unbalanced braces: {open_braces} open, {close_braces} close")
        
        # Check for common issues
        if '...' in query and 'fragment' not in query.lower():
            warnings.append("Spread operator found but no fragment definition")
        
        return len(warnings) == 0, warnings

    async def translate_to_graphql(
        self,
        natural_query: str,
        schema_context: str = "",
        model: Optional[str] = None,
        icl_examples: Optional[List[str]] = None
    ) -> TranslationResult:
        """Translate natural language query to GraphQL."""
        start_time = time.time()
        
        if not natural_query.strip():
            raise ValueError("Natural language query cannot be empty")
        
        model = model or self.settings.ollama.default_model
        # Use provided ICL examples or default to initial examples
        selected_icl_examples = icl_examples if icl_examples is not None else get_initial_icl_examples()[:3]  # Limit to 3 examples for brevity
        system_prompt = self._build_system_prompt(schema_context, selected_icl_examples)
        
        # Build the user prompt
        user_prompt = f"""Convert this natural language query to GraphQL:

"{natural_query}"

Remember to return only the JSON object with the specified structure."""

        try:
            # Use chat completion for better context handling
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = await self.ollama_service.chat_completion(
                messages=messages,
                model=model,
                temperature=0.3,  # Lower temperature for more consistent output
                max_tokens=2048
            )
            
            # Parse the response
            result_data = self._extract_json_from_response(response.text)
            
            # Extract components
            graphql_query = result_data.get("graphql", "")
            confidence = float(result_data.get("confidence", 0.0))
            explanation = result_data.get("explanation", "")
            warnings = result_data.get("warnings", [])
            suggestions = result_data.get("suggestions", [])
            
            # Validate the generated GraphQL
            is_valid, syntax_warnings = self._validate_graphql_syntax(graphql_query)
            warnings.extend(syntax_warnings)
            
            # Adjust confidence based on validation
            if not is_valid:
                confidence = max(0.0, confidence - 0.3)
            
            processing_time = time.time() - start_time
            
            await self._broadcast_interaction(
                model,
                system_prompt,
                user_prompt,
                response.text,
                json.dumps(result_data),
                processing_time,
                confidence,
                warnings
            )
            
            return TranslationResult(
                graphql_query=graphql_query,
                confidence=confidence,
                explanation=explanation,
                model_used=model,
                processing_time=processing_time,
                original_query=natural_query,
                suggested_improvements=suggestions,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            processing_time = time.time() - start_time
            
            await self._broadcast_interaction(
                model,
                system_prompt,
                user_prompt,
                "",
                "",
                processing_time,
                0.0,
                [f"Error: {str(e)}"],
                None,
                None
            )
            
            return TranslationResult(
                graphql_query="",
                confidence=0.0,
                explanation="Unable to translate the query due to a connection issue or internal error.",
                model_used=model,
                processing_time=processing_time,
                original_query=natural_query,
                suggested_improvements=["Please try again. If the issue persists, check the server status or contact support.", "Simplify your query or provide more context."],
                warnings=[f"Error: {str(e)}"]
            )

    async def batch_translate(
        self,
        queries: List[str],
        schema_context: str = "",
        model: Optional[str] = None
    ) -> List[TranslationResult]:
        """Translate multiple natural language queries to GraphQL."""
        results = []
        
        for query in queries:
            try:
                result = await self.translate_to_graphql(query, schema_context, model)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to translate query '{query}': {e}")
                results.append(TranslationResult(
                    graphql_query=f"# Error: {str(e)}",
                    confidence=0.0,
                    explanation=f"Batch translation failed: {str(e)}",
                    model_used=model or self.settings.ollama.default_model,
                    processing_time=0.0,
                    original_query=query,
                    suggested_improvements=[],
                    warnings=[str(e)]
                ))
        
        return results

    def get_translation_examples(self) -> List[Dict[str, str]]:
        """Get example translations for training/demonstration."""
        return [
            {
                "natural": "Get all users with their email addresses",
                "graphql": "query GetUsers {\n  users {\n    id\n    email\n  }\n}"
            },
            {
                "natural": "Find posts by a specific author with comments",
                "graphql": "query GetPostsByAuthor($authorId: ID!) {\n  posts(authorId: $authorId) {\n    id\n    title\n    content\n    comments {\n      id\n      text\n      author {\n        name\n      }\n    }\n  }\n}"
            },
            {
                "natural": "Create a new post with title and content",
                "graphql": "mutation CreatePost($title: String!, $content: String!) {\n  createPost(input: {title: $title, content: $content}) {\n    id\n    title\n    content\n    createdAt\n  }\n}"
            },
            {
                "natural": "Update user profile information",
                "graphql": "mutation UpdateProfile($userId: ID!, $input: UserUpdateInput!) {\n  updateUser(id: $userId, input: $input) {\n    id\n    name\n    email\n    profile {\n      bio\n      avatar\n    }\n  }\n}"
            }
        ]

async def chat_with_model(query: str, model: str, context: List[Dict[str, Any]]) -> str:
    """
    Handle a chat interaction with the specified model using conversation context.
    
    Args:
        query (str): The user's chat message.
        model (str): The model to use for generating the response.
        context (List[Dict[str, Any]]): The conversation history/context.
    
    Returns:
        str: The model's response.
    """
    start_time = time.time()
    logger.info(f"Initiating chat with model {model} for query: {query}")
    
    try:
        ollama_service = OllamaService()
        
        # Format the context into a conversation history for the model
        messages = []
        for msg in context:
            role = 'user' if msg.get('sender') == 'user' else 'assistant'
            messages.append({'role': role, 'content': msg.get('content', '')})
        
        # Add the current user query
        messages.append({'role': 'user', 'content': query})
        
        # Call Ollama API for chat response
        response = await ollama_service.chat(model, messages)
        logger.info(f"Chat response received from {model}")
        
        return response.get('message', {}).get('content', 'No response content available.')
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise 