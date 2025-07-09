"""
Prompt loader for loading templates from files and directories.
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime

from .types import PromptTemplate, PromptFile, PromptStrategy, AgentType, DomainType

logger = logging.getLogger(__name__)


class PromptLoader:
    """Loads and manages prompt templates from various sources."""
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path(__file__).parent
        self.templates: Dict[str, PromptTemplate] = {}
        self.file_cache: Dict[Path, float] = {}  # path -> last_modified
        
    def load_all_prompts(self) -> Dict[str, PromptTemplate]:
        """Load all prompts from the directory structure."""
        self.templates.clear()
        
        # Load from subdirectories
        for subdir in ['agents', 'domains', 'strategies']:
            subdir_path = self.base_path / subdir
            if subdir_path.exists():
                self._load_directory(subdir_path)
        
        # Load from root directory
        self._load_directory(self.base_path)
        
        logger.info(f"Loaded {len(self.templates)} prompt templates")
        return self.templates.copy()
    
    def load_agent_prompts(self, agent_type: AgentType) -> Dict[str, PromptTemplate]:
        """Load prompts for a specific agent type."""
        agent_dir = self.base_path / 'agents' / agent_type.value
        templates = {}
        
        if agent_dir.exists():
            for file_path in agent_dir.glob('*.yaml'):
                template = self._load_template_file(file_path)
                if template:
                    template.agent_type = agent_type
                    templates[template.name] = template
        
        return templates
    
    def load_domain_prompts(self, domain: DomainType) -> Dict[str, PromptTemplate]:
        """Load prompts for a specific domain."""
        domain_dir = self.base_path / 'domains' / domain.value
        templates = {}
        
        if domain_dir.exists():
            for file_path in domain_dir.glob('*.yaml'):
                template = self._load_template_file(file_path)
                if template:
                    template.domain = domain
                    templates[template.name] = template
        
        return templates
    
    def load_strategy_prompts(self, strategy: PromptStrategy) -> Dict[str, PromptTemplate]:
        """Load prompts for a specific strategy."""
        strategy_dir = self.base_path / 'strategies' / strategy.value
        templates = {}
        
        if strategy_dir.exists():
            for file_path in strategy_dir.glob('*.yaml'):
                template = self._load_template_file(file_path)
                if template:
                    template.strategy = strategy
                    templates[template.name] = template
        
        return templates
    
    def reload_if_changed(self) -> List[str]:
        """Reload templates that have been modified. Returns list of reloaded templates."""
        reloaded = []
        
        for template_name, template in list(self.templates.items()):
            # Find the original file path (stored in metadata if available)
            file_path = template.metadata.get('file_path')
            if not file_path:
                continue
                
            file_path = Path(file_path)
            if not file_path.exists():
                continue
                
            # Check if file has been modified
            current_mtime = file_path.stat().st_mtime
            cached_mtime = self.file_cache.get(file_path, 0)
            
            if current_mtime > cached_mtime:
                logger.info(f"Reloading modified template: {template_name}")
                new_template = self._load_template_file(file_path)
                if new_template:
                    self.templates[template_name] = new_template
                    self.file_cache[file_path] = current_mtime
                    reloaded.append(template_name)
        
        return reloaded
    
    def _load_directory(self, directory: Path) -> None:
        """Load all templates from a directory."""
        if not directory.exists():
            return
            
        for file_path in directory.glob('*.yaml'):
            template = self._load_template_file(file_path)
            if template:
                self.templates[template.name] = template
                self.file_cache[file_path] = file_path.stat().st_mtime
    
    def _load_template_file(self, file_path: Path) -> Optional[PromptTemplate]:
        """Load a single template file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # Validate required fields
            if not all(key in data for key in ['name', 'content', 'strategy']):
                logger.warning(f"Template file {file_path} missing required fields")
                return None
            
            # Parse strategy enum
            try:
                strategy = PromptStrategy(data['strategy'])
            except ValueError:
                logger.warning(f"Invalid strategy '{data['strategy']}' in {file_path}")
                return None
            
            # Parse optional enums
            agent_type = None
            if 'agent_type' in data:
                try:
                    agent_type = AgentType(data['agent_type'])
                except ValueError:
                    logger.warning(f"Invalid agent_type '{data['agent_type']}' in {file_path}")
            
            domain = None
            if 'domain' in data:
                try:
                    domain = DomainType(data['domain'])
                except ValueError:
                    logger.warning(f"Invalid domain '{data['domain']}' in {file_path}")
            
            # Create template
            template = PromptTemplate(
                name=data['name'],
                content=data['content'],
                strategy=strategy,
                agent_type=agent_type,
                domain=domain,
                variables=data.get('variables', []),
                description=data.get('description', ''),
                version=data.get('version', '1.0'),
                tags=data.get('tags', []),
                examples=data.get('examples', []),
                metadata={
                    **data.get('metadata', {}),
                    'file_path': str(file_path),
                    'loaded_at': datetime.now().isoformat()
                }
            )
            
            return template
            
        except Exception as e:
            logger.error(f"Failed to load template from {file_path}: {e}")
            return None
    
    def save_template(self, template: PromptTemplate, file_path: Optional[Path] = None) -> bool:
        """Save a template to a YAML file."""
        if not file_path:
            # Generate file path based on template metadata
            filename = f"{template.name}.yaml"
            if template.agent_type:
                file_path = self.base_path / 'agents' / template.agent_type.value / filename
            elif template.domain:
                file_path = self.base_path / 'domains' / template.domain.value / filename
            elif template.strategy:
                file_path = self.base_path / 'strategies' / template.strategy.value / filename
            else:
                file_path = self.base_path / filename
        
        try:
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for saving
            data = {
                'name': template.name,
                'content': template.content,
                'strategy': template.strategy.value,
                'description': template.description,
                'version': template.version,
                'variables': template.variables,
                'tags': template.tags,
                'examples': template.examples,
                'metadata': {k: v for k, v in template.metadata.items() 
                           if k not in ['file_path', 'loaded_at']}
            }
            
            # Add optional fields
            if template.agent_type:
                data['agent_type'] = template.agent_type.value
            if template.domain:
                data['domain'] = template.domain.value
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            
            # Update cache
            self.file_cache[file_path] = file_path.stat().st_mtime
            template.metadata['file_path'] = str(file_path)
            
            logger.info(f"Saved template '{template.name}' to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save template '{template.name}': {e}")
            return False
    
    def discover_prompt_files(self) -> List[PromptFile]:
        """Discover all prompt files in the directory structure."""
        files = []
        
        for pattern in ['*.yaml', '*.yml']:
            for file_path in self.base_path.rglob(pattern):
                if file_path.is_file():
                    files.append(PromptFile.from_path(file_path))
        
        return files 