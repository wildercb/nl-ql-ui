"""
Type definitions for the modular prompt system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pathlib import Path


class PromptStrategy(Enum):
    """Different prompt engineering strategies."""
    MINIMAL = "minimal"
    DETAILED = "detailed" 
    CHAIN_OF_THOUGHT = "chain_of_thought"
    FEW_SHOT = "few_shot"
    ROLE_PLAYING = "role_playing"
    STRUCTURED = "structured"


class AgentType(Enum):
    """Types of agents in the system."""
    TRANSLATOR = "translator"
    REVIEWER = "reviewer"
    REWRITER = "rewriter"
    OPTIMIZER = "optimizer"
    DATA_REVIEWER = "data_reviewer"
    VALIDATOR = "validator"


class DomainType(Enum):
    """Different application domains."""
    GENERAL = "general"
    ECOMMERCE = "ecommerce"
    SOCIAL_MEDIA = "social_media"
    ANALYTICS = "analytics"
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    EDUCATION = "education"


@dataclass
class PromptTemplate:
    """A prompt template with metadata and content."""
    name: str
    content: str
    strategy: PromptStrategy
    agent_type: Optional[AgentType] = None
    domain: Optional[DomainType] = None
    variables: List[str] = field(default_factory=list)
    description: str = ""
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    examples: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def render(self, **kwargs) -> str:
        """Render the template with provided variables."""
        try:
            from jinja2 import Template
            template = Template(self.content)
            return template.render(**kwargs)
        except ImportError:
            # Fallback to simple string replacement if Jinja2 not available
            content = self.content
            for key, value in kwargs.items():
                content = content.replace(f"{{{{{key}}}}}", str(value))
            return content


@dataclass
class PromptContext:
    """Context information for prompt generation."""
    query: str
    domain: Optional[DomainType] = None
    schema_context: str = ""
    examples: List[Dict[str, str]] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    session_history: List[str] = field(default_factory=list)
    model_name: str = ""
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptResult:
    """Result of prompt generation."""
    content: str
    template_used: str
    strategy: PromptStrategy
    variables_used: Dict[str, Any]
    generation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptFile:
    """Represents a prompt file on disk."""
    path: Path
    name: str
    agent_type: Optional[AgentType] = None
    domain: Optional[DomainType] = None
    strategy: Optional[PromptStrategy] = None
    last_modified: Optional[float] = None
    
    @classmethod
    def from_path(cls, path: Path) -> 'PromptFile':
        """Create PromptFile from file path."""
        parts = path.stem.split('_')
        agent_type = None
        domain = None
        strategy = None
        
        # Try to parse file name for metadata
        for part in parts:
            try:
                agent_type = AgentType(part)
            except ValueError:
                pass
            try:
                domain = DomainType(part)
            except ValueError:
                pass
            try:
                strategy = PromptStrategy(part)
            except ValueError:
                pass
        
        return cls(
            path=path,
            name=path.stem,
            agent_type=agent_type,
            domain=domain,
            strategy=strategy,
            last_modified=path.stat().st_mtime if path.exists() else None
        ) 