"""
Advanced Context Engineering System

Provides sophisticated prompt engineering and context management:
- Dynamic prompt strategy selection
- Template-based prompt generation
- A/B testing framework for prompts
- Context optimization and caching
- Domain-specific prompt libraries
"""

import json
import logging
import hashlib
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

try:
    from jinja2 import Environment, FileSystemLoader, Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    logging.warning("Jinja2 not available - template features will be limited")

from .base import AgentContext, AgentCapability

logger = logging.getLogger(__name__)


class PromptStrategy(Enum):
    """Different prompt engineering strategies."""
    MINIMAL = "minimal"                 # Bare minimum prompts
    DETAILED = "detailed"              # Comprehensive instructions
    CHAIN_OF_THOUGHT = "chain_of_thought"  # Step-by-step reasoning
    FEW_SHOT = "few_shot"             # Example-based learning
    ZERO_SHOT = "zero_shot"           # No examples
    DOMAIN_SPECIFIC = "domain_specific"    # Specialized for domain
    PERFORMANCE_OPTIMIZED = "performance_optimized"  # Speed-focused
    ACCURACY_OPTIMIZED = "accuracy_optimized"        # Quality-focused


class ContextOptimization(Enum):
    """Context optimization strategies."""
    NONE = "none"
    COMPRESS = "compress"              # Compress context for efficiency
    SUMMARIZE = "summarize"           # Summarize long contexts
    PRIORITIZE = "prioritize"         # Keep most important parts
    CHUNK = "chunk"                   # Split into smaller chunks


class PromptTemplate:
    """Template for generating prompts with variables."""
    
    def __init__(
        self, 
        name: str,
        template: str,
        strategy: PromptStrategy,
        variables: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.template = template
        self.strategy = strategy
        self.variables = variables or []
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
        
        # Compile template if Jinja2 is available
        if JINJA2_AVAILABLE:
            self._jinja_template = Template(template)
        else:
            self._jinja_template = None
    
    def render(self, **kwargs) -> str:
        """Render the template with provided variables."""
        if self._jinja_template:
            return self._jinja_template.render(**kwargs)
        else:
            # Simple string formatting fallback
            return self.template.format(**kwargs)
    
    def validate_variables(self, variables: Dict[str, Any]) -> List[str]:
        """Validate that all required variables are provided."""
        missing = []
        for var in self.variables:
            if var not in variables:
                missing.append(var)
        return missing
    
    def get_hash(self) -> str:
        """Get hash of template for caching."""
        content = f"{self.name}:{self.template}:{self.strategy.value}"
        return hashlib.md5(content.encode()).hexdigest()


class PromptLibrary:
    """Library of prompt templates organized by domain and capability."""
    
    def __init__(self, templates_dir: Optional[str] = None):
        self.templates: Dict[str, PromptTemplate] = {}
        self.templates_by_strategy: Dict[PromptStrategy, List[str]] = {}
        self.templates_by_capability: Dict[AgentCapability, List[str]] = {}
        self.templates_dir = Path(templates_dir) if templates_dir else None
        
        # Initialize with built-in templates
        self._load_builtin_templates()
        
        # Load from directory if provided
        if self.templates_dir and self.templates_dir.exists():
            self._load_directory_templates()
    
    def _load_builtin_templates(self):
        """Load built-in prompt templates."""
        builtin_templates = [
            PromptTemplate(
                name="minimal_rewrite",
                template="Rewrite this query clearly: {{ query }}",
                strategy=PromptStrategy.MINIMAL,
                variables=["query"]
            ),
            PromptTemplate(
                name="detailed_translation",
                template="""You are an expert GraphQL translator. 
                
Task: Convert natural language to GraphQL query.
Domain: {{ domain or "general" }}
Schema Context: {{ schema_context or "No schema provided" }}

Query: {{ query }}

Requirements:
1. Use correct GraphQL syntax
2. Include only necessary fields
3. Apply appropriate filters
4. Follow best practices

{% if examples %}
Examples:
{% for example in examples %}
Natural: {{ example.natural }}
GraphQL: {{ example.graphql }}
{% endfor %}
{% endif %}

Return JSON: {"graphql": "...", "confidence": 0.0-1.0, "explanation": "..."}""",
                strategy=PromptStrategy.DETAILED,
                variables=["query", "domain", "schema_context", "examples"]
            ),
            PromptTemplate(
                name="chain_of_thought_review",
                template="""Review this GraphQL translation step by step:

Original Query: {{ original_query }}
Generated GraphQL: {{ graphql_query }}

Step 1: Does the GraphQL match the intent?
Step 2: Are there any syntax errors?
Step 3: Are there security concerns?
Step 4: Can it be optimized?

Final Assessment: {{ assessment_prompt }}""",
                strategy=PromptStrategy.CHAIN_OF_THOUGHT,
                variables=["original_query", "graphql_query", "assessment_prompt"]
            ),
            PromptTemplate(
                name="few_shot_ecommerce",
                template="""Translate natural language to GraphQL for e-commerce:

Examples:
Natural: "Find products under $50"
GraphQL: query { products(where: { price: { lt: 50 } }) { id name price } }

Natural: "Get user orders from last month"  
GraphQL: query { orders(where: { createdAt: { gte: "{{ last_month }}" } }) { id total items { product { name } } } }

Your turn:
Natural: {{ query }}
GraphQL:""",
                strategy=PromptStrategy.FEW_SHOT,
                variables=["query", "last_month"]
            )
        ]
        
        for template in builtin_templates:
            self.add_template(template)
    
    def _load_directory_templates(self):
        """Load templates from directory."""
        if not JINJA2_AVAILABLE:
            logger.warning("Jinja2 not available - skipping directory template loading")
            return
        
        try:
            for template_file in self.templates_dir.glob("*.j2"):
                self._load_template_file(template_file)
        except Exception as e:
            logger.error(f"Error loading directory templates: {e}")
    
    def _load_template_file(self, template_path: Path):
        """Load a single template file."""
        try:
            # Look for metadata file
            metadata_path = template_path.with_suffix('.json')
            metadata = {}
            
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
            
            # Load template content
            with open(template_path) as f:
                content = f.read()
            
            template = PromptTemplate(
                name=metadata.get('name', template_path.stem),
                template=content,
                strategy=PromptStrategy(metadata.get('strategy', 'detailed')),
                variables=metadata.get('variables', []),
                metadata=metadata
            )
            
            self.add_template(template)
            
        except Exception as e:
            logger.error(f"Error loading template {template_path}: {e}")
    
    def add_template(self, template: PromptTemplate):
        """Add a template to the library."""
        self.templates[template.name] = template
        
        # Index by strategy
        if template.strategy not in self.templates_by_strategy:
            self.templates_by_strategy[template.strategy] = []
        self.templates_by_strategy[template.strategy].append(template.name)
        
        # Index by capability (inferred from name)
        for capability in AgentCapability:
            if capability.value in template.name.lower():
                if capability not in self.templates_by_capability:
                    self.templates_by_capability[capability] = []
                self.templates_by_capability[capability].append(template.name)
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get template by name."""
        return self.templates.get(name)
    
    def find_templates(
        self,
        strategy: Optional[PromptStrategy] = None,
        capability: Optional[AgentCapability] = None,
        domain: Optional[str] = None
    ) -> List[PromptTemplate]:
        """Find templates matching criteria."""
        candidates = list(self.templates.values())
        
        if strategy:
            candidates = [t for t in candidates if t.strategy == strategy]
        
        if capability:
            capability_templates = self.templates_by_capability.get(capability, [])
            candidates = [t for t in candidates if t.name in capability_templates]
        
        if domain:
            candidates = [t for t in candidates if domain.lower() in t.name.lower()]
        
        return candidates
    
    def get_best_template(
        self,
        capability: AgentCapability,
        strategy: Optional[PromptStrategy] = None,
        domain: Optional[str] = None,
        context: Optional[AgentContext] = None
    ) -> Optional[PromptTemplate]:
        """Get the best template for given criteria."""
        templates = self.find_templates(strategy, capability, domain)
        
        if not templates:
            return None
        
        # Simple ranking - prefer domain-specific, then strategy match
        scored_templates = []
        for template in templates:
            score = 0
            
            if domain and domain.lower() in template.name.lower():
                score += 10
            
            if strategy and template.strategy == strategy:
                score += 5
            
            # Performance-based scoring could be added here
            
            scored_templates.append((score, template))
        
        # Return highest scoring template
        scored_templates.sort(key=lambda x: x[0], reverse=True)
        return scored_templates[0][1]


class ContextManager:
    """Manages context optimization and caching."""
    
    def __init__(self):
        self.context_cache: Dict[str, Any] = {}
        self.optimization_stats: Dict[str, int] = {}
    
    def optimize_context(
        self, 
        context: AgentContext, 
        strategy: ContextOptimization = ContextOptimization.NONE,
        max_length: Optional[int] = None
    ) -> AgentContext:
        """Optimize context based on strategy."""
        if strategy == ContextOptimization.NONE:
            return context
        
        optimized = context.copy()
        
        if strategy == ContextOptimization.COMPRESS:
            optimized = self._compress_context(optimized, max_length)
        elif strategy == ContextOptimization.SUMMARIZE:
            optimized = self._summarize_context(optimized, max_length)
        elif strategy == ContextOptimization.PRIORITIZE:
            optimized = self._prioritize_context(optimized, max_length)
        elif strategy == ContextOptimization.CHUNK:
            # For chunking, we'd return multiple contexts
            pass
        
        self.optimization_stats[strategy.value] = self.optimization_stats.get(strategy.value, 0) + 1
        return optimized
    
    def _compress_context(self, context: AgentContext, max_length: Optional[int]) -> AgentContext:
        """Compress context by removing unnecessary data."""
        # Remove verbose metadata
        context.metadata = {k: v for k, v in context.metadata.items() if len(str(v)) < 100}
        
        # Truncate long examples
        if context.examples:
            context.examples = context.examples[:3]  # Keep only first 3 examples
        
        return context
    
    def _summarize_context(self, context: AgentContext, max_length: Optional[int]) -> AgentContext:
        """Summarize long context elements."""
        # This would use an LLM to summarize long content
        # For now, simple truncation
        if context.schema_context and len(context.schema_context) > 1000:
            context.schema_context = context.schema_context[:1000] + "..."
        
        return context
    
    def _prioritize_context(self, context: AgentContext, max_length: Optional[int]) -> AgentContext:
        """Keep only the most important context elements."""
        # Priority order: original_query > recent outputs > examples > metadata
        # Keep core fields, reduce others based on importance
        
        # Keep only most recent agent outputs
        if len(context.agent_outputs) > 5:
            recent_keys = list(context.agent_outputs.keys())[-5:]
            context.agent_outputs = {k: context.agent_outputs[k] for k in recent_keys}
        
        return context
    
    def get_context_key(self, context: AgentContext) -> str:
        """Generate a cache key for the context."""
        key_parts = [
            context.original_query,
            context.domain_context or "",
            context.prompt_strategy or "",
            str(len(context.examples))
        ]
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()
    
    def cache_context(self, key: str, context: AgentContext):
        """Cache optimized context."""
        self.context_cache[key] = context
    
    def get_cached_context(self, key: str) -> Optional[AgentContext]:
        """Get cached context."""
        return self.context_cache.get(key)


class ContextEngineering:
    """
    Main context engineering orchestrator.
    
    Combines prompt templates, optimization strategies, and caching
    to provide the best possible context for each agent.
    """
    
    def __init__(self, templates_dir: Optional[str] = None):
        self.prompt_library = PromptLibrary(templates_dir)
        self.context_manager = ContextManager()
        self.ab_test_configs: Dict[str, Dict[str, Any]] = {}
        
    def prepare_agent_context(
        self,
        context: AgentContext,
        agent_capability: AgentCapability,
        strategy: Optional[PromptStrategy] = None,
        optimization: ContextOptimization = ContextOptimization.NONE,
        ab_test_group: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Prepare optimized context and prompts for an agent.
        
        Returns:
            Dictionary with 'context' and 'prompts' keys
        """
        # Apply A/B testing if specified
        if ab_test_group and ab_test_group in self.ab_test_configs:
            config = self.ab_test_configs[ab_test_group]
            strategy = strategy or PromptStrategy(config.get('strategy', 'detailed'))
            optimization = ContextOptimization(config.get('optimization', 'none'))
        
        # Optimize context
        optimized_context = self.context_manager.optimize_context(context, optimization)
        
        # Select best prompt template
        template = self.prompt_library.get_best_template(
            capability=agent_capability,
            strategy=strategy,
            domain=context.domain_context,
            context=optimized_context
        )
        
        # Generate prompts
        prompts = {}
        if template:
            template_vars = {
                'query': context.original_query,
                'domain': context.domain_context,
                'schema_context': context.schema_context,
                'examples': context.examples,
                'agent_outputs': context.agent_outputs
            }
            
            # Validate variables
            missing_vars = template.validate_variables(template_vars)
            if missing_vars:
                logger.warning(f"Missing template variables: {missing_vars}")
            
            try:
                prompts['main'] = template.render(**template_vars)
                prompts['template_name'] = template.name
                prompts['strategy'] = template.strategy.value
            except Exception as e:
                logger.error(f"Error rendering template {template.name}: {e}")
                prompts['main'] = f"Process this query: {context.original_query}"
                prompts['template_name'] = "fallback"
        else:
            # Fallback prompt
            prompts['main'] = f"Process this query: {context.original_query}"
            prompts['template_name'] = "fallback"
        
        return {
            'context': optimized_context,
            'prompts': prompts,
            'template_used': template.name if template else None,
            'optimization_applied': optimization.value
        }
    
    def register_ab_test(
        self, 
        test_name: str, 
        groups: Dict[str, Dict[str, Any]]
    ):
        """Register A/B test configuration."""
        self.ab_test_configs[test_name] = groups
        logger.info(f"Registered A/B test: {test_name} with {len(groups)} groups")
    
    def add_custom_template(self, template: PromptTemplate):
        """Add a custom prompt template."""
        self.prompt_library.add_template(template)
    
    def get_engineering_stats(self) -> Dict[str, Any]:
        """Get context engineering statistics."""
        return {
            'total_templates': len(self.prompt_library.templates),
            'templates_by_strategy': {
                strategy.value: len(templates) 
                for strategy, templates in self.prompt_library.templates_by_strategy.items()
            },
            'optimization_stats': self.context_manager.optimization_stats,
            'cache_size': len(self.context_manager.context_cache),
            'ab_tests': list(self.ab_test_configs.keys())
        } 