"""
Advanced Agent Registry System

Provides automatic agent discovery, registration, and management capabilities:
- Auto-discovery of agent classes in the agents directory
- Metadata-driven agent registration
- Dependency resolution and validation
- Plugin architecture support
- Performance monitoring integration
"""

import importlib
import inspect
import logging
import pkgutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union

from .base import BaseAgent, AgentMetadata, AgentCapability

logger = logging.getLogger(__name__)


class AgentRegistrationError(Exception):
    """Raised when agent registration fails."""
    pass


class AgentRegistry:
    """
    Central registry for all agents in the system.
    
    Handles:
    - Agent discovery and registration
    - Metadata management
    - Dependency resolution
    - Capability-based agent lookup
    - Performance tracking integration
    """
    
    def __init__(self):
        self._agents: Dict[str, Type[BaseAgent]] = {}
        self._metadata: Dict[str, AgentMetadata] = {}
        self._capabilities_map: Dict[AgentCapability, Set[str]] = {}
        self._dependency_graph: Dict[str, Set[str]] = {}
        
    def register_agent(
        self, 
        agent_class: Type[BaseAgent], 
        metadata: Optional[AgentMetadata] = None,
        name: Optional[str] = None
    ) -> str:
        """
        Register an agent class with optional metadata.
        
        Args:
            agent_class: The agent class to register
            metadata: Optional metadata (will be auto-generated if not provided)
            name: Optional name override
            
        Returns:
            The registered agent name
            
        Raises:
            AgentRegistrationError: If registration fails
        """
        # Generate metadata if not provided
        if metadata is None:
            metadata = self._generate_metadata(agent_class)
        
        # Use provided name or fall back to metadata name
        agent_name = name or metadata.name
        
        # Validate agent class
        if not issubclass(agent_class, BaseAgent):
            raise AgentRegistrationError(f"Agent {agent_name} must inherit from BaseAgent")
        
        # Check for name conflicts
        if agent_name in self._agents:
            logger.warning(f"Agent {agent_name} already registered, overwriting")
        
        # Register the agent
        self._agents[agent_name] = agent_class
        self._metadata[agent_name] = metadata
        
        # Update capabilities mapping
        for capability in metadata.capabilities:
            if capability not in self._capabilities_map:
                self._capabilities_map[capability] = set()
            self._capabilities_map[capability].add(agent_name)
        
        # Update dependency graph
        self._dependency_graph[agent_name] = set(metadata.depends_on)
        
        logger.info(f"✅ Registered agent: {agent_name} with capabilities: {metadata.capabilities}")
        return agent_name
    
    def _generate_metadata(self, agent_class: Type[BaseAgent]) -> AgentMetadata:
        """Auto-generate metadata for an agent class."""
        name = agent_class.__name__
        description = agent_class.__doc__ or f"Auto-generated agent: {name}"
        
        # Try to infer capabilities from class name or methods
        capabilities = self._infer_capabilities(agent_class)
        
        return AgentMetadata(
            name=name,
            description=description.strip(),
            capabilities=capabilities
        )
    
    def _infer_capabilities(self, agent_class: Type[BaseAgent]) -> Set[AgentCapability]:
        """Infer agent capabilities from class name and methods."""
        capabilities = set()
        
        name_lower = agent_class.__name__.lower()
        
        # Name-based inference
        capability_keywords = {
            AgentCapability.REWRITE: ['rewrite', 'rewriter', 'preprocess', 'clarify'],
            AgentCapability.TRANSLATE: ['translate', 'translator', 'convert'],
            AgentCapability.REVIEW: ['review', 'reviewer', 'validate', 'check'],
            AgentCapability.OPTIMIZE: ['optimize', 'optimizer', 'improve'],
            AgentCapability.ANALYZE: ['analyze', 'analyzer', 'introspect'],
            AgentCapability.VALIDATE: ['validate', 'validator', 'verify'],
            AgentCapability.GENERATE: ['generate', 'generator', 'create'],
            AgentCapability.EXTRACT: ['extract', 'extractor', 'parse'],
            AgentCapability.TRANSFORM: ['transform', 'transformer', 'convert']
        }
        
        for capability, keywords in capability_keywords.items():
            if any(keyword in name_lower for keyword in keywords):
                capabilities.add(capability)
        
        # Method-based inference (look for specific method patterns)
        methods = [method for method in dir(agent_class) if not method.startswith('_')]
        for method in methods:
            if 'rewrite' in method.lower():
                capabilities.add(AgentCapability.REWRITE)
            elif 'translate' in method.lower():
                capabilities.add(AgentCapability.TRANSLATE)
            elif 'review' in method.lower() or 'validate' in method.lower():
                capabilities.add(AgentCapability.REVIEW)
        
        # Default capability if none inferred
        if not capabilities:
            capabilities.add(AgentCapability.TRANSFORM)
        
        return capabilities
    
    def get_agent(self, name: str) -> Optional[Type[BaseAgent]]:
        """Get an agent class by name."""
        return self._agents.get(name)
    
    def get_metadata(self, name: str) -> Optional[AgentMetadata]:
        """Get agent metadata by name."""
        return self._metadata.get(name)
    
    def list_agents(self) -> List[str]:
        """Get list of all registered agent names."""
        return list(self._agents.keys())
    
    def find_agents_by_capability(self, capability: AgentCapability) -> List[str]:
        """Find all agents with a specific capability."""
        return list(self._capabilities_map.get(capability, set()))
    
    def find_agents_by_capabilities(self, capabilities: List[AgentCapability]) -> List[str]:
        """Find agents that have ALL the specified capabilities."""
        if not capabilities:
            return self.list_agents()
        
        agent_sets = [self._capabilities_map.get(cap, set()) for cap in capabilities]
        # Find intersection of all sets
        result = agent_sets[0]
        for agent_set in agent_sets[1:]:
            result = result.intersection(agent_set)
        
        return list(result)
    
    def get_execution_order(self, agent_names: List[str]) -> List[str]:
        """
        Resolve agent dependencies and return execution order.
        
        Args:
            agent_names: List of agent names to order
            
        Returns:
            Ordered list of agent names respecting dependencies
            
        Raises:
            AgentRegistrationError: If circular dependencies detected
        """
        # Simple topological sort
        ordered = []
        remaining = set(agent_names)
        visiting = set()
        
        def visit(agent: str):
            if agent in visiting:
                raise AgentRegistrationError(f"Circular dependency detected involving {agent}")
            if agent in ordered:
                return
            
            visiting.add(agent)
            
            # Visit dependencies first
            dependencies = self._dependency_graph.get(agent, set())
            for dep in dependencies:
                if dep in remaining:  # Only consider requested agents
                    visit(dep)
            
            visiting.remove(agent)
            ordered.append(agent)
        
        # Visit all remaining agents
        for agent in list(remaining):
            if agent in remaining:
                visit(agent)
                remaining.remove(agent)
        
        return ordered
    
    def discover_agents(self, package_path: Optional[str] = None):
        """
        Auto-discover and register all agent classes in the agents package.
        
        Args:
            package_path: Optional package path to search (defaults to agents package)
        """
        if package_path is None:
            # Default to current package
            current_dir = Path(__file__).parent
            package_path = str(current_dir)
        
        logger.info(f"🔍 Discovering agents in: {package_path}")
        
        # Import all modules in the package
        package_name = "backend.agents"
        discovered_count = 0
        
        try:
            # Walk through the package directory
            for module_info in pkgutil.walk_packages([package_path], f"{package_name}."):
                if module_info.name.endswith('.base') or module_info.name.endswith('.registry'):
                    continue  # Skip base and registry modules
                
                try:
                    module = importlib.import_module(module_info.name)
                    
                    # Find all BaseAgent subclasses in the module
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, BaseAgent) and 
                            obj != BaseAgent and 
                            obj.__module__ == module_info.name):
                            
                            self.register_agent(obj)
                            discovered_count += 1
                            
                except Exception as e:
                    logger.warning(f"Failed to import {module_info.name}: {e}")
        
        except Exception as e:
            logger.error(f"Agent discovery failed: {e}")
        
        logger.info(f"✅ Discovered and registered {discovered_count} agents")
    
    def validate_agent_dependencies(self) -> List[str]:
        """
        Validate all agent dependencies.
        
        Returns:
            List of validation errors (empty if all valid)
        """
        errors = []
        
        for agent_name, dependencies in self._dependency_graph.items():
            for dep in dependencies:
                if dep not in self._agents:
                    errors.append(f"Agent {agent_name} depends on unknown agent: {dep}")
        
        # Check for circular dependencies
        try:
            self.get_execution_order(list(self._agents.keys()))
        except AgentRegistrationError as e:
            errors.append(str(e))
        
        return errors
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics."""
        stats = {
            'total_agents': len(self._agents),
            'capabilities_distribution': {
                cap.value: len(agents) 
                for cap, agents in self._capabilities_map.items()
            },
            'agents_with_dependencies': len([
                name for name, deps in self._dependency_graph.items() if deps
            ]),
            'validation_errors': self.validate_agent_dependencies()
        }
        
        # Performance stats would come from actual agent usage
        stats['performance_summary'] = {
            agent_name: self.get_agent(agent_name)().get_performance_summary()
            for agent_name in self._agents.keys()
        }
        
        return stats


# Global registry instance
agent_registry = AgentRegistry()


def agent(
    name: Optional[str] = None,
    capabilities: Optional[List[AgentCapability]] = None,
    depends_on: Optional[List[str]] = None,
    **metadata_kwargs
):
    """
    Decorator for automatic agent registration.
    
    Usage:
        @agent(name="my_rewriter", capabilities=[AgentCapability.REWRITE])
        class MyRewriterAgent(BaseAgent):
            async def run(self, ctx: AgentContext, **kwargs):
                # Implementation
                pass
    """
    def decorator(agent_class: Type[BaseAgent]) -> Type[BaseAgent]:
        # Get description from metadata_kwargs first, then docstring, then default
        explicit_description = metadata_kwargs.pop('description', None)
        if explicit_description:
            agent_description = explicit_description
        elif agent_class.__doc__:
            agent_description = agent_class.__doc__.strip()
        else:
            agent_description = "No description provided"
        
        metadata = AgentMetadata(
            name=name or agent_class.__name__,
            description=agent_description,
            capabilities=set(capabilities or []),
            depends_on=depends_on or [],
            **metadata_kwargs
        )
        
        agent_registry.register_agent(agent_class, metadata)
        return agent_class
    
    return decorator 