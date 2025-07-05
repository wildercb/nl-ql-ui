"""Translation service for natural language to GraphQL conversion."""

import json
import logging
import re
import time
import httpx
import asyncio
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

    def _build_system_prompt(self, schema_context: str = "", icl_examples: Optional[List[str]] = None) -> str:
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
    ):
        """
        Translate natural language query to GraphQL, yielding events for streaming.
        """
        if not natural_query.strip():
            raise ValueError("Natural language query cannot be empty")

        model = model or self.settings.ollama.default_model
        logger.info(f"🦙 Translation service using model: {model}")
        
        selected_icl_examples = icl_examples if icl_examples is not None else get_initial_icl_examples()[:3]
        system_prompt = self._build_system_prompt(schema_context, selected_icl_examples)
        
        user_prompt = f"""Convert this natural language query to GraphQL:

"{natural_query}"

Remember to return only the JSON object with the specified structure."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        yield {"event": "prompt_generated", "prompt": messages}

        full_response_text = ""
        try:
            logger.info(f"🦙 Starting Ollama chat completion with model: {model}")
            stream = self.ollama_service.stream_chat_completion(
                messages=messages,
                model=model,
                temperature=0.3,
                max_tokens=2048
            )
            async for chunk in stream:
                token = chunk.get("message", {}).get("content", "")
                if token:
                    full_response_text += token
                    yield {"event": "agent_token", "token": token}
            
            logger.info(f"🦙 Ollama chat completion completed with model: {model}")
            
            # Final processing after stream is complete
            result_data = self._extract_json_from_response(full_response_text)
            
            graphql_query = result_data.get("graphql", "")
            confidence = float(result_data.get("confidence", 0.0))
            is_valid, syntax_warnings = self._validate_graphql_syntax(graphql_query)
            
            if not is_valid:
                confidence = max(0.0, confidence - 0.3)
                result_data["warnings"] = result_data.get("warnings", []) + syntax_warnings

            final_result = {
                "graphql_query": graphql_query,
                "confidence": confidence,
                "explanation": result_data.get("explanation", ""),
                "warnings": result_data.get("warnings", []),
                "suggestions": result_data.get("suggestions", [])
            }

            yield {"event": "translation_complete", "result": final_result}

        except Exception as e:
            logger.error(f"Translation streaming failed with model {model}: {e}")
            yield {"event": "error", "message": str(e)}

    async def batch_translate(
        queries: List[str],
        schema_context: str = "",
        model: Optional[str] = None
    ) -> List[TranslationResult]:
        """Translate multiple natural language queries to GraphQL in parallel."""
        
        service = TranslationService()
        tasks = [
            service.translate_to_graphql(
                natural_query=query,
                schema_context=schema_context,
                model=model
            ) 
            for query in queries
        ]
        
        results_with_prompts = await asyncio.gather(*tasks)
        # Return only the TranslationResult object from each tuple
        return [result for _prompt, result in results_with_prompts]

def get_translation_examples() -> List[str]:
    """
    Returns a list of few-shot examples for ICL.
    
    This function demonstrates how to provide dynamic or static examples
    to improve translation accuracy for specific domains or query patterns.
    """
    return get_initial_icl_examples()


async def chat_with_model(query: str, model: str, context: List[Dict[str, Any]]) -> str:
    """
    A simple chat interface for direct interaction with an Ollama model.
    This is useful for debugging, testing prompts, or direct model interaction.
    """
    service = OllamaService()
    try:
        response = await service.chat_completion(
            messages=context + [{"role": "user", "content": query}],
            model=model or service.default_model
        )
        return response.text
    except Exception as e:
        logger.error(f"Chat with model failed: {e}")
        return f"Error: {e}" 