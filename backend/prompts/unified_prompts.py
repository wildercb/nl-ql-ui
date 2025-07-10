"""
Unified Prompt System for MPPW-MCP

This module provides a streamlined prompt management system that integrates
with the unified configuration. It focuses on:

1. Clear separation by agent type
2. Reusable template components
3. Strategy-based prompt selection
4. Easy extensibility for new agents and strategies
5. Type-safe prompt definitions
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import json
import logging
from abc import ABC, abstractmethod
from jinja2 import Template, Environment, BaseLoader
import re

from config.unified_config import AgentType, PromptStrategy, AgentCapability, get_unified_config

logger = logging.getLogger(__name__)


# =============================================================================
# Core Prompt Classes
# =============================================================================

@dataclass
class PromptTemplate:
    """A unified prompt template with metadata and rendering capabilities."""
    name: str
    content: str
    agent_type: Optional[AgentType] = None
    strategy: PromptStrategy = PromptStrategy.DETAILED
    variables: List[str] = field(default_factory=list)
    required_variables: List[str] = field(default_factory=list)
    description: str = ""
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    examples: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def render(self, **kwargs) -> str:
        """Render the template with provided variables."""
        # Check for required variables
        missing = set(self.required_variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        try:
            template = Template(self.content)
            rendered = template.render(**kwargs)
            
            # Clean up extra whitespace and newlines
            rendered = re.sub(r'\n\s*\n\s*\n', '\n\n', rendered)
            return rendered.strip()
        except Exception as e:
            logger.error(f"Failed to render template {self.name}: {e}")
            raise
    
    def validate_variables(self, variables: Dict[str, Any]) -> List[str]:
        """Validate provided variables and return missing required ones."""
        missing = []
        for var in self.required_variables:
            if var not in variables:
                missing.append(var)
        return missing
    
    def get_example_variables(self) -> Dict[str, Any]:
        """Get example variables for testing the template."""
        example_vars = {}
        for var in self.variables:
            if var == "query":
                example_vars[var] = "Get all users with their names"
            elif var == "schema_context":
                example_vars[var] = "type User { id: ID! name: String! email: String! }"
            elif var == "domain":
                example_vars[var] = "general"
            elif var == "examples":
                example_vars[var] = [{"natural": "Get users", "graphql": "query { users { id name } }"}]
            else:
                example_vars[var] = f"example_{var}"
        return example_vars


# =============================================================================
# Prompt Template Definitions
# =============================================================================

class PromptTemplates:
    """Central repository of all prompt templates organized by agent and strategy."""
    
    @staticmethod
    def get_rewriter_prompts() -> Dict[str, PromptTemplate]:
        """Get all prompt templates for the rewriter agent."""
        return {
            "rewriter_detailed": PromptTemplate(
                name="rewriter_detailed",
                agent_type=AgentType.REWRITER,
                strategy=PromptStrategy.DETAILED,
                description="Comprehensive query rewriting with context enhancement",
                required_variables=["query"],
                variables=["query", "domain", "context"],
                content="""You are an expert query rewriter specialized in clarifying and enhancing natural language queries for better translation to GraphQL.

Your task is to rewrite the following query to be clearer, more specific, and better structured:

**Original Query:** {{ query }}
{% if domain %}**Domain:** {{ domain }}{% endif %}
{% if context %}**Additional Context:** {{ context }}{% endif %}

## Rewriting Guidelines:
1. **Clarify ambiguous terms** - Replace vague words with specific ones
2. **Expand abbreviations** - Write out shortened terms
3. **Add missing context** - Include implied information
4. **Structure logically** - Organize the request clearly
5. **Remove ambiguity** - Ensure single clear interpretation
6. **Preserve intent** - Keep the original meaning intact

## Output Format:
Return ONLY the rewritten query as plain text. Do not include explanations, formatting, or additional text.

**Rewritten Query:**"""
            ),
            
            "rewriter_minimal": PromptTemplate(
                name="rewriter_minimal",
                agent_type=AgentType.REWRITER,
                strategy=PromptStrategy.MINIMAL,
                description="Quick query clarification for simple cases",
                required_variables=["query"],
                variables=["query"],
                content="""Clarify this query: {{ query }}

Make it more specific and clear. Return only the improved query:"""
            ),
            
            "rewriter_chain_of_thought": PromptTemplate(
                name="rewriter_chain_of_thought",
                agent_type=AgentType.REWRITER,
                strategy=PromptStrategy.CHAIN_OF_THOUGHT,
                description="Step-by-step query analysis and rewriting",
                required_variables=["query"],
                variables=["query", "domain", "context"],
                content="""You are an expert query analyst. Rewrite the following query using step-by-step reasoning.

**Original Query:** {{ query }}
{% if domain %}**Domain:** {{ domain }}{% endif %}

## Analysis Process:
1. **Parse the query** - What is being asked?
2. **Identify entities** - What objects/data types are involved?
3. **Identify relationships** - How are entities connected?
4. **Identify missing details** - What context should be added?
5. **Reconstruct** - Build a clearer version

## Your Analysis:
1. **Query Analysis:** 
   - Main request: [what is being asked]
   - Entities involved: [list entities]
   - Relationships: [describe connections]
   - Missing context: [what should be clarified]

2. **Rewritten Query:** [your improved version]

Return your response in this exact format."""
            )
        }
    
    @staticmethod
    def get_translator_prompts() -> Dict[str, PromptTemplate]:
        """Get all prompt templates for the translator agent."""
        return {
            "translator_detailed": PromptTemplate(
                name="translator_detailed",
                agent_type=AgentType.TRANSLATOR,
                strategy=PromptStrategy.DETAILED,
                description="Comprehensive GraphQL translation with full context",
                required_variables=["query"],
                variables=["query", "schema_context", "domain", "examples"],
                content="""You are a GraphQL expert specializing in translating natural language into accurate, efficient GraphQL queries.

## Task
Convert this natural language query into proper GraphQL syntax:

**Query:** {{ query }}
{% if domain %}**Domain:** {{ domain }}{% endif %}

{% if schema_context %}
## Schema Context
```graphql
{{ schema_context }}
```
{% endif %}

{% if examples %}
## Reference Examples
{% for example in examples %}
- **Natural:** {{ example.natural }}
  **GraphQL:** `{{ example.graphql }}`
{% endfor %}
{% endif %}

## Translation Guidelines
1. **Accuracy** - Generate syntactically correct GraphQL
2. **Efficiency** - Request only necessary fields
3. **Best Practices** - Use proper naming and structure
4. **Schema Compliance** - Follow provided schema constraints
5. **Error Prevention** - Avoid common GraphQL pitfalls

## Response Format
Return a JSON object with this exact structure:
```json
{
  "graphql": "your GraphQL query here",
  "confidence": 0.95,
  "explanation": "Brief explanation of translation decisions",
  "warnings": ["any potential issues"],
  "suggestions": ["optional improvements"]
}
```

**Important:** Return ONLY the JSON object, no additional text.""",
                examples=[
                    {"natural": "Get all users", "graphql": "query { users { id name } }"},
                    {"natural": "Find products under $50", "graphql": "query { products(where: { price: { lt: 50 } }) { id name price } }"}
                ]
            ),
            
            "translator_few_shot": PromptTemplate(
                name="translator_few_shot", 
                agent_type=AgentType.TRANSLATOR,
                strategy=PromptStrategy.FEW_SHOT,
                description="Example-driven GraphQL translation",
                required_variables=["query", "examples"],
                variables=["query", "examples", "schema_context"],
                content="""Convert natural language to GraphQL using these examples as reference:

## Examples
{% for example in examples %}
**Natural:** {{ example.natural }}
**GraphQL:** {{ example.graphql }}

{% endfor %}

## Your Task
**Natural:** {{ query }}
**GraphQL:** 

Return JSON: {"graphql": "your query", "confidence": 0.9}"""
            ),
            
            "translator_minimal": PromptTemplate(
                name="translator_minimal",
                agent_type=AgentType.TRANSLATOR,
                strategy=PromptStrategy.MINIMAL,
                description="Fast GraphQL translation for simple queries",
                required_variables=["query"],
                variables=["query"],
                content="""Convert to GraphQL: {{ query }}

Return JSON: {"graphql": "query here", "confidence": 0.9}"""
            )
        }
    
    @staticmethod
    def get_reviewer_prompts() -> Dict[str, PromptTemplate]:
        """Get all prompt templates for the reviewer agent."""
        return {
            "reviewer_detailed": PromptTemplate(
                name="reviewer_detailed",
                agent_type=AgentType.REVIEWER,
                strategy=PromptStrategy.DETAILED,
                description="Comprehensive GraphQL query review and validation",
                required_variables=["graphql_query", "original_query"],
                variables=["graphql_query", "original_query", "schema_context"],
                content="""You are a GraphQL expert specializing in query review and validation.

## Review Task
Analyze this GraphQL query for correctness, efficiency, and best practices.

**Original Request:** {{ original_query }}
**GraphQL Query:**
```graphql
{{ graphql_query }}
```

{% if schema_context %}
**Schema Context:**
```graphql
{{ schema_context }}
```
{% endif %}

## Review Criteria
1. **Syntax Correctness** - Valid GraphQL syntax
2. **Semantic Accuracy** - Matches original intent
3. **Efficiency** - Optimal field selection
4. **Security** - No dangerous patterns
5. **Best Practices** - Follows GraphQL conventions

## Analysis Areas
- Query structure and syntax
- Field selection appropriateness
- Argument usage and validation
- Performance implications
- Security considerations
- Adherence to schema

## Response Format
```json
{
  "passed": true,
  "score": 85,
  "issues": [
    {
      "type": "warning",
      "category": "performance", 
      "message": "Consider adding pagination",
      "suggestion": "Add first: 10 argument"
    }
  ],
  "improvements": [
    "Add specific fields instead of selecting all",
    "Consider using fragments for repeated field sets"
  ],
  "compliments": [
    "Correct syntax and structure",
    "Appropriate filtering logic"
  ]
}
```

Return ONLY the JSON response."""
            ),
            
            "reviewer_security": PromptTemplate(
                name="reviewer_security",
                agent_type=AgentType.REVIEWER,
                strategy=PromptStrategy.DETAILED,
                description="Security-focused GraphQL query review",
                required_variables=["graphql_query"],
                variables=["graphql_query", "original_query"],
                content="""Security review for GraphQL query:

```graphql
{{ graphql_query }}
```

Check for:
- Depth limits
- Field limits  
- Authentication needs
- Data exposure risks
- Injection vulnerabilities

Return JSON: {"secure": true, "risks": [], "recommendations": []}"""
            )
        }
    
    @staticmethod
    def get_analyzer_prompts() -> Dict[str, PromptTemplate]:
        """Get all prompt templates for the analyzer agent."""
        return {
            "analyzer_detailed": PromptTemplate(
                name="analyzer_detailed",
                agent_type=AgentType.ANALYZER,
                strategy=PromptStrategy.DETAILED,
                description="Comprehensive query and data analysis",
                required_variables=["query_data"],
                variables=["query_data", "graphql_executed", "original_query", "analysis_type"],
                content="""You are a data analysis expert. Analyze the provided query results and extract meaningful insights.

## Query Results to Analyze
{{ query_data }}

{% if graphql_executed %}
## GraphQL Query Executed
```graphql
{{ graphql_executed }}
```
{% endif %}

{% if original_query %}
## Original Query
{{ original_query }}
{% endif %}

{% if analysis_type %}
## Analysis Type
{{ analysis_type }}
{% endif %}

## Analysis Framework
1. **Data Overview** - Structure and content summary
2. **Key Insights** - Important patterns and findings
3. **Anomalies** - Unusual or unexpected data points
4. **Trends** - Observable patterns over time/categories
5. **Recommendations** - Actionable insights

## Response Format
```json
{
  "summary": "Brief overview of the data",
  "insights": [
    {
      "category": "performance",
      "finding": "Key insight description",
      "significance": "high",
      "evidence": "Supporting data"
    }
  ],
  "anomalies": ["List of unusual findings"],
  "trends": ["Observable patterns"],
  "recommendations": ["Actionable suggestions"],
  "confidence": 0.9
}
```

Return ONLY the JSON response."""
            )
        }


# =============================================================================
# Unified Prompt Manager
# =============================================================================

class UnifiedPromptManager:
    """
    Unified prompt manager that integrates with the configuration system
    and provides intelligent prompt selection based on agent capabilities.
    """
    
    def __init__(self):
        """Initialize the prompt manager."""
        self.templates: Dict[str, PromptTemplate] = {}
        self.agent_templates: Dict[AgentType, Dict[PromptStrategy, List[str]]] = {}
        self.config = get_unified_config()
        
        # Load built-in templates
        self._load_builtin_templates()
        
        # Build indexes for fast lookup
        self._build_indexes()
        
        logger.info(f"Prompt manager initialized with {len(self.templates)} templates")
    
    def _load_builtin_templates(self):
        """Load all built-in prompt templates."""
        template_groups = [
            PromptTemplates.get_rewriter_prompts(),
            PromptTemplates.get_translator_prompts(),
            PromptTemplates.get_reviewer_prompts(),
            PromptTemplates.get_analyzer_prompts()
        ]
        
        for group in template_groups:
            for name, template in group.items():
                self.templates[name] = template
    
    def _build_indexes(self):
        """Build indexes for fast template lookup."""
        for template in self.templates.values():
            if template.agent_type:
                if template.agent_type not in self.agent_templates:
                    self.agent_templates[template.agent_type] = {}
                
                if template.strategy not in self.agent_templates[template.agent_type]:
                    self.agent_templates[template.agent_type][template.strategy] = []
                
                self.agent_templates[template.agent_type][template.strategy].append(template.name)
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self.templates.get(name)
    
    def find_template(
        self,
        agent_type: AgentType,
        strategy: Optional[PromptStrategy] = None,
        name_pattern: Optional[str] = None
    ) -> Optional[PromptTemplate]:
        """Find the best template for given criteria."""
        
        # Default to detailed strategy
        strategy = strategy or PromptStrategy.DETAILED
        
        # Look for exact match first
        if agent_type in self.agent_templates:
            if strategy in self.agent_templates[agent_type]:
                template_names = self.agent_templates[agent_type][strategy]
                if template_names:
                    # If name pattern provided, try to match
                    if name_pattern:
                        for name in template_names:
                            if name_pattern in name:
                                return self.templates[name]
                    
                    # Return first available template
                    return self.templates[template_names[0]]
            
            # Fall back to any strategy for this agent
            for avail_strategy, template_names in self.agent_templates[agent_type].items():
                if template_names:
                    return self.templates[template_names[0]]
        
        return None
    
    def get_prompt_for_agent(
        self,
        agent_name: str,
        context: Dict[str, Any],
        strategy_override: Optional[PromptStrategy] = None
    ) -> Optional[str]:
        """Get a rendered prompt for an agent with the given context."""
        
        # Get agent configuration
        agent_config = self.config.get_agent(agent_name)
        if not agent_config:
            logger.error(f"No configuration found for agent: {agent_name}")
            return None
        
        # Determine strategy
        strategy = strategy_override or agent_config.prompt_strategy
        
        # Find appropriate template
        template = self.find_template(agent_config.agent_type, strategy)
        if not template:
            logger.error(f"No template found for agent {agent_name} with strategy {strategy}")
            return None
        
        try:
            # Render template with context
            return template.render(**context)
        except Exception as e:
            logger.error(f"Failed to render prompt for agent {agent_name}: {e}")
            return None
    
    def register_template(self, template: PromptTemplate) -> None:
        """Register a new custom template."""
        self.templates[template.name] = template
        
        # Update indexes
        if template.agent_type:
            if template.agent_type not in self.agent_templates:
                self.agent_templates[template.agent_type] = {}
            
            if template.strategy not in self.agent_templates[template.agent_type]:
                self.agent_templates[template.agent_type][template.strategy] = []
            
            self.agent_templates[template.agent_type][template.strategy].append(template.name)
        
        logger.info(f"Registered custom template: {template.name}")
    
    def list_templates(
        self,
        agent_type: Optional[AgentType] = None,
        strategy: Optional[PromptStrategy] = None
    ) -> List[PromptTemplate]:
        """List templates with optional filtering."""
        templates = list(self.templates.values())
        
        if agent_type:
            templates = [t for t in templates if t.agent_type == agent_type]
        
        if strategy:
            templates = [t for t in templates if t.strategy == strategy]
        
        return templates
    
    def validate_template(self, template: PromptTemplate) -> List[str]:
        """Validate a template and return any issues."""
        issues = []
        
        if not template.name:
            issues.append("Template must have a name")
        
        if not template.content:
            issues.append("Template must have content")
        
        # Test rendering with example variables
        try:
            example_vars = template.get_example_variables()
            template.render(**example_vars)
        except Exception as e:
            issues.append(f"Template rendering failed: {e}")
        
        return issues
    
    def get_stats(self) -> Dict[str, Any]:
        """Get prompt manager statistics."""
        agent_counts = {}
        strategy_counts = {}
        
        for template in self.templates.values():
            if template.agent_type:
                agent_type = template.agent_type.value
                agent_counts[agent_type] = agent_counts.get(agent_type, 0) + 1
            
            strategy = template.strategy.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            "total_templates": len(self.templates),
            "by_agent": agent_counts,
            "by_strategy": strategy_counts,
            "indexed_agents": len(self.agent_templates)
        }


# =============================================================================
# Global Instance and Utilities
# =============================================================================

# Global prompt manager instance
prompt_manager = UnifiedPromptManager()


def get_prompt_manager() -> UnifiedPromptManager:
    """Get the global prompt manager instance."""
    return prompt_manager


def get_prompt_for_agent(
    agent_name: str,
    context: Dict[str, Any],
    strategy: Optional[PromptStrategy] = None
) -> Optional[str]:
    """Convenience function to get a rendered prompt for an agent."""
    return prompt_manager.get_prompt_for_agent(agent_name, context, strategy)


def register_custom_template(template: PromptTemplate) -> None:
    """Register a custom prompt template."""
    prompt_manager.register_template(template)


# =============================================================================
# Template Creation Helpers
# =============================================================================

def create_template(
    name: str,
    content: str,
    agent_type: AgentType,
    strategy: PromptStrategy = PromptStrategy.DETAILED,
    **kwargs
) -> PromptTemplate:
    """Helper to create a new prompt template."""
    return PromptTemplate(
        name=name,
        content=content,
        agent_type=agent_type,
        strategy=strategy,
        **kwargs
    )


def load_template_from_file(file_path: Path) -> PromptTemplate:
    """Load a prompt template from a YAML file."""
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    
    return PromptTemplate(
        name=data['name'],
        content=data['content'],
        agent_type=AgentType(data.get('agent_type')) if data.get('agent_type') else None,
        strategy=PromptStrategy(data.get('strategy', 'detailed')),
        variables=data.get('variables', []),
        required_variables=data.get('required_variables', []),
        description=data.get('description', ''),
        version=data.get('version', '1.0'),
        tags=data.get('tags', []),
        examples=data.get('examples', []),
        metadata=data.get('metadata', {})
    )


# Example usage for testing
if __name__ == "__main__":
    manager = get_prompt_manager()
    
    # Test getting a prompt for translation
    context = {
        "query": "Get all users with their names and emails",
        "schema_context": "type User { id: ID! name: String! email: String! }",
        "domain": "general"
    }
    
    prompt = get_prompt_for_agent("translator", context)
    if prompt:
        print("Generated prompt:")
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
    
    # Print stats
    print("\nPrompt Manager Stats:")
    print(json.dumps(manager.get_stats(), indent=2)) 