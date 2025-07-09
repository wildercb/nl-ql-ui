"""
Prompt Manager - Main interface for prompt management.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from functools import lru_cache

from .types import (
    PromptTemplate, PromptContext, PromptResult, PromptStrategy, 
    AgentType, DomainType
)
from .loader import PromptLoader

logger = logging.getLogger(__name__)


class PromptManager:
    """Main manager for prompt templates and generation."""
    
    def __init__(self, prompts_dir: Optional[Path] = None):
        self.prompts_dir = prompts_dir or Path(__file__).parent
        self.loader = PromptLoader(self.prompts_dir)
        self.templates: Dict[str, PromptTemplate] = {}
        self.last_reload = 0
        self.auto_reload = True
        
        # Load initial templates
        self.reload_templates()
    
    def reload_templates(self) -> int:
        """Reload all templates from disk. Returns number of templates loaded."""
        logger.info("Reloading prompt templates...")
        self.templates = self.loader.load_all_prompts()
        self.last_reload = time.time()
        logger.info(f"Loaded {len(self.templates)} templates")
        return len(self.templates)
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        self._check_auto_reload()
        return self.templates.get(name)
    
    def get_templates_by_agent(self, agent_type: AgentType) -> List[PromptTemplate]:
        """Get all templates for a specific agent type."""
        self._check_auto_reload()
        return [t for t in self.templates.values() if t.agent_type == agent_type]
    
    def get_templates_by_domain(self, domain: DomainType) -> List[PromptTemplate]:
        """Get all templates for a specific domain."""
        self._check_auto_reload()
        return [t for t in self.templates.values() if t.domain == domain]
    
    def get_templates_by_strategy(self, strategy: PromptStrategy) -> List[PromptTemplate]:
        """Get all templates for a specific strategy."""
        self._check_auto_reload()
        return [t for t in self.templates.values() if t.strategy == strategy]
    
    def find_best_template(
        self,
        agent_type: Optional[AgentType] = None,
        domain: Optional[DomainType] = None,
        strategy: Optional[PromptStrategy] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[PromptTemplate]:
        """Find the best matching template based on criteria."""
        self._check_auto_reload()
        
        candidates = list(self.templates.values())
        
        # Filter by agent type
        if agent_type:
            candidates = [t for t in candidates if t.agent_type == agent_type]
        
        # Filter by domain  
        if domain:
            candidates = [t for t in candidates if t.domain == domain or t.domain is None]
        
        # Filter by strategy
        if strategy:
            candidates = [t for t in candidates if t.strategy == strategy]
        
        # Filter by tags
        if tags:
            candidates = [t for t in candidates if any(tag in t.tags for tag in tags)]
        
        if not candidates:
            return None
        
        # Score templates based on specificity
        def score_template(template: PromptTemplate) -> int:
            score = 0
            if template.agent_type == agent_type:
                score += 10
            if template.domain == domain:
                score += 10
            if template.strategy == strategy:
                score += 10
            if tags and any(tag in template.tags for tag in tags):
                score += 5
            return score
        
        # Return the highest scoring template
        return max(candidates, key=score_template)
    
    def generate_prompt(
        self,
        template_name: str,
        context: PromptContext,
        **kwargs
    ) -> PromptResult:
        """Generate a prompt using a template and context."""
        start_time = time.time()
        
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Prepare variables for rendering
        variables = {
            'query': context.query,
            'domain': context.domain.value if context.domain else 'general',
            'schema_context': context.schema_context,
            'examples': context.examples,
            'model_name': context.model_name,
            'temperature': context.temperature,
            'max_tokens': context.max_tokens,
            **context.metadata,
            **kwargs
        }
        
        # Render the template
        content = template.render(**variables)
        
        generation_time = time.time() - start_time
        
        return PromptResult(
            content=content,
            template_used=template_name,
            strategy=template.strategy,
            variables_used=variables,
            generation_time=generation_time,
            metadata={
                'template_version': template.version,
                'template_description': template.description
            }
        )
    
    def generate_smart_prompt(
        self,
        context: PromptContext,
        agent_type: Optional[AgentType] = None,
        preferred_strategy: Optional[PromptStrategy] = None,
        **kwargs
    ) -> PromptResult:
        """Generate a prompt by automatically selecting the best template."""
        
        # Determine strategy if not provided
        if not preferred_strategy:
            preferred_strategy = self._infer_strategy(context)
        
        # Find the best template
        template = self.find_best_template(
            agent_type=agent_type,
            domain=context.domain,
            strategy=preferred_strategy
        )
        
        if not template:
            # Fallback to any template with the same strategy
            template = self.find_best_template(strategy=preferred_strategy)
        
        if not template:
            # Final fallback to default template
            template = self.find_best_template(strategy=PromptStrategy.DETAILED)
        
        if not template:
            raise ValueError("No suitable template found")
        
        return self.generate_prompt(template.name, context, **kwargs)
    
    def create_template(
        self,
        name: str,
        content: str,
        strategy: PromptStrategy,
        agent_type: Optional[AgentType] = None,
        domain: Optional[DomainType] = None,
        **kwargs
    ) -> PromptTemplate:
        """Create a new template and optionally save it."""
        template = PromptTemplate(
            name=name,
            content=content,
            strategy=strategy,
            agent_type=agent_type,
            domain=domain,
            **kwargs
        )
        
        # Add to memory
        self.templates[name] = template
        
        # Save to disk
        self.loader.save_template(template)
        
        return template
    
    def update_template(self, name: str, **updates) -> bool:
        """Update an existing template."""
        template = self.get_template(name)
        if not template:
            return False
        
        # Update fields
        for key, value in updates.items():
            if hasattr(template, key):
                setattr(template, key, value)
        
        # Save to disk
        return self.loader.save_template(template)
    
    def delete_template(self, name: str) -> bool:
        """Delete a template."""
        template = self.templates.pop(name, None)
        if not template:
            return False
        
        # Remove file if it exists
        file_path = template.metadata.get('file_path')
        if file_path:
            try:
                Path(file_path).unlink()
                return True
            except Exception as e:
                logger.error(f"Failed to delete template file: {e}")
                return False
        
        return True
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List all templates with metadata."""
        self._check_auto_reload()
        
        result = []
        for template in self.templates.values():
            result.append({
                'name': template.name,
                'description': template.description,
                'strategy': template.strategy.value,
                'agent_type': template.agent_type.value if template.agent_type else None,
                'domain': template.domain.value if template.domain else None,
                'version': template.version,
                'tags': template.tags,
                'variables': template.variables
            })
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded templates."""
        self._check_auto_reload()
        
        by_strategy = {}
        by_agent = {}
        by_domain = {}
        
        for template in self.templates.values():
            # Count by strategy
            strategy = template.strategy.value
            by_strategy[strategy] = by_strategy.get(strategy, 0) + 1
            
            # Count by agent type
            if template.agent_type:
                agent = template.agent_type.value
                by_agent[agent] = by_agent.get(agent, 0) + 1
            
            # Count by domain
            if template.domain:
                domain = template.domain.value
                by_domain[domain] = by_domain.get(domain, 0) + 1
        
        return {
            'total_templates': len(self.templates),
            'by_strategy': by_strategy,
            'by_agent_type': by_agent,
            'by_domain': by_domain,
            'last_reload': self.last_reload
        }
    
    def _check_auto_reload(self) -> None:
        """Check if templates should be auto-reloaded."""
        if not self.auto_reload:
            return
        
        # Check every 30 seconds max
        if time.time() - self.last_reload < 30:
            return
        
        # Check for modified files
        reloaded = self.loader.reload_if_changed()
        if reloaded:
            logger.info(f"Auto-reloaded {len(reloaded)} templates: {reloaded}")
            self.last_reload = time.time()
    
    def _infer_strategy(self, context: PromptContext) -> PromptStrategy:
        """Infer the best strategy based on context."""
        # Simple heuristics for strategy selection
        query_length = len(context.query)
        
        if query_length < 50:
            return PromptStrategy.MINIMAL
        elif context.examples:
            return PromptStrategy.FEW_SHOT
        elif context.schema_context:
            return PromptStrategy.DETAILED
        else:
            return PromptStrategy.STRUCTURED 