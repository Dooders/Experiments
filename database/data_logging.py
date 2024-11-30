"""Data logging module for simulation database.

This module handles all data logging operations including buffered writes
and transaction management for simulation data. It provides methods for
logging various types of data like agent actions, learning experiences,
and health incidents.

Features:
- Buffered batch operations for performance
- Transaction safety
- Automatic buffer flushing
- Comprehensive error handling
"""

import json
import logging
from typing import Dict, Optional, Tuple

from sqlalchemy.exc import SQLAlchemyError

from .models import AgentAction, HealthIncident, LearningExperience

logger = logging.getLogger(__name__)


class DataLogger:
    """Handles data logging operations for the simulation database."""

    def __init__(self, database, buffer_size: int = 1000):
        """Initialize the data logger.

        Parameters
        ----------
        database : SimulationDatabase
            Database instance to use for logging
        buffer_size : int, optional
            Maximum size of buffers before auto-flush, by default 1000
        """
        self.db = database
        self._buffer_size = buffer_size
        self._action_buffer = []
        self._learning_exp_buffer = []
        self._health_incident_buffer = []

    def log_agent_action(
        self,
        step_number: int,
        agent_id: int,
        action_type: str,
        action_target_id: Optional[int] = None,
        position_before: Optional[Tuple[float, float]] = None,
        position_after: Optional[Tuple[float, float]] = None,
        resources_before: Optional[float] = None,
        resources_after: Optional[float] = None,
        reward: Optional[float] = None,
        details: Optional[Dict] = None,
    ) -> None:
        """Buffer an agent action with enhanced validation and error handling."""
        try:
            # Input validation
            if step_number < 0:
                raise ValueError("step_number must be non-negative")
            if agent_id < 0:
                raise ValueError("agent_id must be non-negative")
            if not isinstance(action_type, str):
                action_type = str(action_type)

            action_data = {
                "step_number": step_number,
                "agent_id": agent_id,
                "action_type": action_type,
                "action_target_id": action_target_id,
                "position_before": str(position_before) if position_before else None,
                "position_after": str(position_after) if position_after else None,
                "resources_before": resources_before,
                "resources_after": resources_after,
                "reward": reward,
                "details": json.dumps(details) if details else None,
            }

            self._action_buffer.append(action_data)

            if len(self._action_buffer) >= self._buffer_size:
                self.flush_action_buffer()

        except ValueError as e:
            logger.error(f"Invalid input for agent action: {e}")
            raise
        except TypeError as e:
            logger.error(f"Type error in agent action data: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error logging agent action: {e}")
            raise

    def log_learning_experience(
        self,
        step_number: int,
        agent_id: int,
        module_type: str,
        state_before: str,
        action_taken: int,
        reward: float,
        state_after: str,
        loss: Optional[float] = None,
    ) -> None:
        """Buffer a learning experience."""
        try:
            exp_data = {
                "step_number": step_number,
                "agent_id": agent_id,
                "module_type": module_type,
                "state_before": state_before,
                "action_taken": action_taken,
                "reward": reward,
                "state_after": state_after,
                "loss": loss,
            }

            self._learning_exp_buffer.append(exp_data)

            if len(self._learning_exp_buffer) >= self._buffer_size:
                self.flush_learning_buffer()

        except Exception as e:
            logger.error(f"Error logging learning experience: {e}")
            raise

    def log_health_incident(
        self,
        step_number: int,
        agent_id: int,
        health_before: float,
        health_after: float,
        cause: str,
        details: Optional[Dict] = None,
    ) -> None:
        """Buffer a health incident."""
        try:
            incident_data = {
                "step_number": step_number,
                "agent_id": agent_id,
                "health_before": health_before,
                "health_after": health_after,
                "cause": cause,
                "details": json.dumps(details) if details else None,
            }

            self._health_incident_buffer.append(incident_data)

            if len(self._health_incident_buffer) >= self._buffer_size:
                self.flush_health_buffer()

        except Exception as e:
            logger.error(f"Error logging health incident: {e}")
            raise

    def flush_action_buffer(self) -> None:
        """Flush the action buffer by batch inserting all buffered actions."""
        if not self._action_buffer:
            return

        buffer_copy = list(self._action_buffer)
        try:

            def _insert(session):
                session.bulk_insert_mappings(AgentAction, buffer_copy)

            self.db._execute_in_transaction(_insert)
            self._action_buffer.clear()
        except SQLAlchemyError as e:
            logger.error(f"Failed to flush action buffer: {e}")
            raise

    def flush_learning_buffer(self) -> None:
        """Flush the learning experience buffer with transaction safety."""
        if not self._learning_exp_buffer:
            return

        buffer_copy = list(self._learning_exp_buffer)
        try:

            def _insert(session):
                session.bulk_insert_mappings(LearningExperience, buffer_copy)

            self.db._execute_in_transaction(_insert)
            self._learning_exp_buffer.clear()
        except SQLAlchemyError as e:
            logger.error(f"Failed to flush learning buffer: {e}")
            raise

    def flush_health_buffer(self) -> None:
        """Flush the health incident buffer with transaction safety."""
        if not self._health_incident_buffer:
            return

        buffer_copy = list(self._health_incident_buffer)
        try:

            def _insert(session):
                session.bulk_insert_mappings(HealthIncident, buffer_copy)

            self.db._execute_in_transaction(_insert)
            self._health_incident_buffer.clear()
        except SQLAlchemyError as e:
            logger.error(f"Failed to flush health buffer: {e}")
            raise

    def flush_all_buffers(self) -> None:
        """Flush all data buffers to the database.

        Safely writes all buffered data (actions, learning experiences, health incidents)
        to the database in transactions. Maintains data integrity by only clearing
        buffers after successful writes.

        Raises
        ------
        SQLAlchemyError
            If any buffer flush fails, with details about which buffer(s) failed
        """
        buffers = {
            "action": (self._action_buffer, self.flush_action_buffer),
            "learning": (self._learning_exp_buffer, self.flush_learning_buffer),
            "health": (self._health_incident_buffer, self.flush_health_buffer),
        }

        errors = []
        for buffer_name, (buffer, flush_func) in buffers.items():
            if buffer:  # Only attempt flush if buffer has data
                try:
                    flush_func()
                except SQLAlchemyError as e:
                    logger.error(f"Error flushing {buffer_name} buffer: {e}")
                    errors.append((buffer_name, str(e)))
                except Exception as e:
                    logger.error(f"Unexpected error flushing {buffer_name} buffer: {e}")
                    errors.append((buffer_name, str(e)))

        if errors:
            error_msg = "; ".join(f"{name}: {err}" for name, err in errors)
            raise SQLAlchemyError(f"Failed to flush buffers: {error_msg}")
