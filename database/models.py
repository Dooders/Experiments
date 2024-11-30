"""SQLAlchemy models for the simulation database.

This module defines the database schema using SQLAlchemy ORM models.
Each class represents a table in the database and defines its structure and relationships.

Main Models:
- Agent: Represents simulation agents with their core attributes
- AgentState: Tracks agent state changes over time
- ResourceState: Tracks resource states in the environment
- SimulationStep: Stores simulation-wide metrics per step
- AgentAction: Records actions taken by agents
- LearningExperience: Stores agent learning data
- HealthIncident: Tracks changes in agent health
- SimulationConfig: Stores simulation configuration data

Each model includes appropriate indexes for query optimization and relationships
between related tables.
"""

import logging
from typing import Any, Dict

from sqlalchemy import Boolean, Column, Float, ForeignKey, Index, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

logger = logging.getLogger(__name__)

Base = declarative_base()


# Define SQLAlchemy Models
class Agent(Base):
    __tablename__ = "agents"
    __table_args__ = (
        Index("idx_agents_agent_type", "agent_type"),
        Index("idx_agents_birth_time", "birth_time"),
        Index("idx_agents_death_time", "death_time"),
    )

    agent_id = Column(Integer, primary_key=True)
    birth_time = Column(Integer)
    death_time = Column(Integer)
    agent_type = Column(String(50))
    position_x = Column(Float(precision=6))
    position_y = Column(Float(precision=6))
    initial_resources = Column(Float(precision=6))
    max_health = Column(Float(precision=4))
    starvation_threshold = Column(Integer)
    genome_id = Column(String(64))
    parent_id = Column(Integer, ForeignKey("agents.agent_id"))
    generation = Column(Integer)

    # Relationships
    states = relationship("AgentState", back_populates="agent")
    actions = relationship("AgentAction", back_populates="agent")
    health_incidents = relationship("HealthIncident", back_populates="agent")
    learning_experiences = relationship("LearningExperience", back_populates="agent")


class AgentState(Base):
    __tablename__ = "agent_states"
    __table_args__ = (
        Index("idx_agent_states_agent_id", "agent_id"),
        Index("idx_agent_states_step_number", "step_number"),
        Index("idx_agent_states_composite", "step_number", "agent_id"),
    )

    id = Column(Integer, primary_key=True)
    step_number = Column(Integer)
    agent_id = Column(Integer, ForeignKey("agents.agent_id"))
    position_x = Column(Float)
    position_y = Column(Float)
    resource_level = Column(Float)
    current_health = Column(Float)
    max_health = Column(Float)
    starvation_threshold = Column(Integer)
    is_defending = Column(Boolean)
    total_reward = Column(Float)
    age = Column(Integer)

    agent = relationship("Agent", back_populates="states")

    def as_dict(self) -> Dict[str, Any]:
        """Convert agent state to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent.agent_type,
            "position": (self.position_x, self.position_y),
            "resource_level": self.resource_level,
            "current_health": self.current_health,
            "max_health": self.max_health,
            "starvation_threshold": self.starvation_threshold,
            "is_defending": self.is_defending,
            "total_reward": self.total_reward,
            "age": self.age,
        }


# Additional SQLAlchemy Models
class ResourceState(Base):
    __tablename__ = "resource_states"
    __table_args__ = (
        Index("idx_resource_states_step_number", "step_number"),
        Index("idx_resource_states_resource_id", "resource_id"),
    )

    id = Column(Integer, primary_key=True)
    step_number = Column(Integer)
    resource_id = Column(Integer)
    amount = Column(Float)
    position_x = Column(Float)
    position_y = Column(Float)

    def as_dict(self) -> Dict[str, Any]:
        """Convert resource state to dictionary."""
        return {
            "resource_id": self.resource_id,
            "amount": self.amount,
            "position": (self.position_x, self.position_y),
        }


class SimulationStep(Base):
    __tablename__ = "simulation_steps"
    __table_args__ = (Index("idx_simulation_steps_step_number", "step_number"),)

    step_number = Column(Integer, primary_key=True)
    total_agents = Column(Integer)
    system_agents = Column(Integer)
    independent_agents = Column(Integer)
    control_agents = Column(Integer)
    total_resources = Column(Float)
    average_agent_resources = Column(Float)
    births = Column(Integer)
    deaths = Column(Integer)
    current_max_generation = Column(Integer)
    resource_efficiency = Column(Float)
    resource_distribution_entropy = Column(Float)
    average_agent_health = Column(Float)
    average_agent_age = Column(Integer)
    average_reward = Column(Float)
    combat_encounters = Column(Integer)
    successful_attacks = Column(Integer)
    resources_shared = Column(Float)
    genetic_diversity = Column(Float)
    dominant_genome_ratio = Column(Float)

    def as_dict(self) -> Dict[str, Any]:
        """Convert simulation step to dictionary."""
        return {
            "total_agents": self.total_agents,
            "system_agents": self.system_agents,
            "independent_agents": self.independent_agents,
            "control_agents": self.control_agents,
            "total_resources": self.total_resources,
            "average_agent_resources": self.average_agent_resources,
            "births": self.births,
            "deaths": self.deaths,
            "current_max_generation": self.current_max_generation,
            "resource_efficiency": self.resource_efficiency,
            "resource_distribution_entropy": self.resource_distribution_entropy,
            "average_agent_health": self.average_agent_health,
            "average_agent_age": self.average_agent_age,
            "average_reward": self.average_reward,
            "combat_encounters": self.combat_encounters,
            "successful_attacks": self.successful_attacks,
            "resources_shared": self.resources_shared,
            "genetic_diversity": self.genetic_diversity,
            "dominant_genome_ratio": self.dominant_genome_ratio,
        }


class AgentAction(Base):
    __tablename__ = "agent_actions"
    __table_args__ = (
        Index("idx_agent_actions_step_number", "step_number"),
        Index("idx_agent_actions_agent_id", "agent_id"),
        Index("idx_agent_actions_action_type", "action_type"),
    )

    action_id = Column(Integer, primary_key=True)
    step_number = Column(Integer, nullable=False)
    agent_id = Column(Integer, ForeignKey("agents.agent_id"), nullable=False)
    action_type = Column(String(20), nullable=False)
    action_target_id = Column(Integer)
    position_before = Column(String(32))
    position_after = Column(String(32))
    resources_before = Column(Float(precision=6))
    resources_after = Column(Float(precision=6))
    reward = Column(Float(precision=6))
    details = Column(String(1024))

    agent = relationship("Agent", back_populates="actions")


class LearningExperience(Base):
    __tablename__ = "learning_experiences"
    __table_args__ = (
        Index("idx_learning_experiences_step_number", "step_number"),
        Index("idx_learning_experiences_agent_id", "agent_id"),
        Index("idx_learning_experiences_module_type", "module_type"),
    )

    experience_id = Column(Integer, primary_key=True)
    step_number = Column(Integer)
    agent_id = Column(Integer, ForeignKey("agents.agent_id"))
    module_type = Column(String(50))
    state_before = Column(String(512))
    action_taken = Column(Integer)
    reward = Column(Float(precision=6))
    state_after = Column(String(512))
    loss = Column(Float(precision=6))

    agent = relationship("Agent", back_populates="learning_experiences")


class HealthIncident(Base):
    __tablename__ = "health_incidents"
    __table_args__ = (
        Index("idx_health_incidents_step_number", "step_number"),
        Index("idx_health_incidents_agent_id", "agent_id"),
    )

    incident_id = Column(Integer, primary_key=True)
    step_number = Column(Integer, nullable=False)
    agent_id = Column(Integer, ForeignKey("agents.agent_id"), nullable=False)
    health_before = Column(Float(precision=4))
    health_after = Column(Float(precision=4))
    cause = Column(String(50), nullable=False)
    details = Column(String(512))

    agent = relationship("Agent", back_populates="health_incidents")


class SimulationConfig(Base):
    __tablename__ = "simulation_config"

    config_id = Column(Integer, primary_key=True)
    timestamp = Column(Integer, nullable=False)
    config_data = Column(String(4096), nullable=False)
