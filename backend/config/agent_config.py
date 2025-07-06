"""
Agent Configuration System

This module provides a centralized configuration system for agent interactions,
models, and pipeline behaviors. This makes it easy to modify agent behaviors
without touching the core orchestration logic.
"""

from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, field

class AgentRole(Enum):
    """Available agent roles in the system"""
    REWRITER = "rewriter"
    TRANSLATOR = "translator"
    REVIEWER = "reviewer"
    OPTIMIZER = "optimizer"
    DATA_REVIEWER = "data_reviewer"
    VALIDATOR = "validator"
    ANALYZER = "analyzer"

class ModelSize(Enum):
    """Model size categories for automatic selection"""
    SMALL = "small"      # Fast, low memory (phi3:mini)
    MEDIUM = "medium"    # Balanced (gemma3:4b)
    LARGE = "large"      # High quality (gemma3n:e4b)

@dataclass
class AgentConfig:
    """Configuration for a single agent"""
    name: str
    role: AgentRole
    model: str
    fallback_model: str
    timeout: float = 30.0
    max_retries: int = 2
    required_capabilities: List[str] = field(default_factory=list)
    optional_capabilities: List[str] = field(default_factory=list)
    
@dataclass
class PipelineConfig:
    """Configuration for a complete pipeline"""
    name: str
    description: str
    agents: List[AgentConfig]
    max_parallel: int = 1
    timeout: float = 60.0
    optimization_level: str = "balanced"
    use_cases: List[str] = field(default_factory=list)
    
class AgentConfigManager:
    """Manages agent configurations and provides easy customization"""
    
    def __init__(self):
        self.model_mappings = {
            ModelSize.SMALL: "phi3:mini",
            ModelSize.MEDIUM: "gemma3:4b", 
            ModelSize.LARGE: "gemma3n:e4b"
        }
        
        self.agent_configs = self._initialize_agent_configs()
        self.pipeline_configs = self._initialize_pipeline_configs()
    
    def _initialize_agent_configs(self) -> Dict[str, AgentConfig]:
        """Initialize individual agent configurations"""
        return {
            "rewriter": AgentConfig(
                name="rewriter",
                role=AgentRole.REWRITER,
                model=self.model_mappings[ModelSize.SMALL],
                fallback_model=self.model_mappings[ModelSize.SMALL],
                timeout=20.0,
                required_capabilities=["text_processing", "query_understanding"]
            ),
            "translator": AgentConfig(
                name="translator",
                role=AgentRole.TRANSLATOR,
                model=self.model_mappings[ModelSize.SMALL],
                fallback_model=self.model_mappings[ModelSize.SMALL],
                timeout=30.0,
                required_capabilities=["graphql_generation", "schema_understanding"]
            ),
            "reviewer": AgentConfig(
                name="reviewer",
                role=AgentRole.REVIEWER,
                model=self.model_mappings[ModelSize.SMALL],
                fallback_model=self.model_mappings[ModelSize.SMALL],
                timeout=25.0,
                required_capabilities=["code_review", "quality_analysis"]
            ),
            "data_reviewer": AgentConfig(
                name="data_reviewer",
                role=AgentRole.DATA_REVIEWER,
                model=self.model_mappings[ModelSize.MEDIUM],  # Needs multimodal support
                fallback_model=self.model_mappings[ModelSize.SMALL],
                timeout=45.0,
                required_capabilities=["multimodal_analysis", "data_validation"],
                optional_capabilities=["iterative_refinement"]
            ),
            "optimizer": AgentConfig(
                name="optimizer",
                role=AgentRole.OPTIMIZER,
                model=self.model_mappings[ModelSize.SMALL],
                fallback_model=self.model_mappings[ModelSize.SMALL],
                timeout=20.0,
                required_capabilities=["performance_optimization", "query_analysis"]
            )
        }
    
    def _initialize_pipeline_configs(self) -> Dict[str, PipelineConfig]:
        """Initialize pipeline configurations"""
        return {
            "fast": PipelineConfig(
                name="fast",
                description="Speed-optimized pipeline with minimal processing",
                agents=[self.agent_configs["translator"]],
                timeout=15.0,
                optimization_level="speed",
                use_cases=["simple queries", "high-throughput scenarios"]
            ),
            "standard": PipelineConfig(
                name="standard",
                description="Standard rewrite→translate→review pipeline",
                agents=[
                    self.agent_configs["rewriter"],
                    self.agent_configs["translator"],
                    self.agent_configs["reviewer"]
                ],
                timeout=45.0,
                optimization_level="balanced",
                use_cases=["general queries", "production workloads"]
            ),
            "comprehensive": PipelineConfig(
                name="comprehensive",
                description="Full pipeline with data review and iterative refinement",
                agents=[
                    self.agent_configs["rewriter"],
                    self.agent_configs["translator"],
                    self.agent_configs["reviewer"],
                    self.agent_configs["data_reviewer"]
                ],
                timeout=90.0,
                optimization_level="quality",
                use_cases=["complex queries", "critical applications", "multimodal data"]
            )
        }
    
    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """Get configuration for a specific agent"""
        return self.agent_configs.get(agent_name)
    
    def get_pipeline_config(self, pipeline_name: str) -> Optional[PipelineConfig]:
        """Get configuration for a specific pipeline"""
        return self.pipeline_configs.get(pipeline_name)
    
    def update_agent_model(self, agent_name: str, model: str) -> bool:
        """Update the model for a specific agent"""
        if agent_name in self.agent_configs:
            self.agent_configs[agent_name].model = model
            return True
        return False
    
    def add_custom_agent(self, config: AgentConfig) -> None:
        """Add a custom agent configuration"""
        self.agent_configs[config.name] = config
    
    def create_custom_pipeline(self, name: str, agent_names: List[str], **kwargs) -> PipelineConfig:
        """Create a custom pipeline from existing agents"""
        agents = [self.agent_configs[name] for name in agent_names if name in self.agent_configs]
        
        config = PipelineConfig(
            name=name,
            description=kwargs.get('description', f'Custom pipeline with {len(agents)} agents'),
            agents=agents,
            **{k: v for k, v in kwargs.items() if k != 'description'}
        )
        
        self.pipeline_configs[name] = config
        return config
    
    def get_model_for_size(self, size: ModelSize) -> str:
        """Get the model name for a given size category"""
        return self.model_mappings[size]
    
    def set_model_for_size(self, size: ModelSize, model: str) -> None:
        """Set the model for a given size category"""
        self.model_mappings[size] = model
        
        # Update all agent configs using this size
        for agent_config in self.agent_configs.values():
            if agent_config.model == self.model_mappings[size]:
                agent_config.model = model

# Global instance for easy access
agent_config_manager = AgentConfigManager() 