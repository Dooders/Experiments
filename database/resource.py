from typing import Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import func

from database.utilities import execute_query
from database.models import ResourceState, SimulationStep

class ResourceRetriever:
    """Handles retrieval and analysis of resource-related data.

    This class encapsulates methods for analyzing resource distribution,
    consumption patterns, and efficiency metrics across the simulation.
    """

    def __init__(self, database):
        """Initialize with database connection.

        Parameters
        ----------
        database : SimulationDatabase
            Database instance to use for queries
        """
        self.db = database

    @execute_query
    def resource_distribution(self, session) -> Dict[str, List[float]]:
        """Get resource distribution over time.

        Returns
        -------
        Dict[str, List[float]]
            Dictionary containing:
            - steps: List of step numbers
            - total_resources: Total resources at each step
            - average_per_cell: Average resources per grid cell
            - distribution_entropy: Resource distribution entropy
        """
        distribution_data = (
            session.query(
                SimulationStep.step_number,
                SimulationStep.total_resources,
                SimulationStep.resource_density,
                SimulationStep.resource_distribution_entropy,
            )
            .order_by(SimulationStep.step_number)
            .all()
        )

        df = pd.DataFrame(
            distribution_data,
            columns=["step", "total", "density", "entropy"]
        )

        return {
            "steps": df["step"].tolist(),
            "total_resources": df["total"].tolist(),
            "average_per_cell": df["density"].tolist(),
            "distribution_entropy": df["entropy"].tolist(),
        }

    @execute_query
    def consumption_patterns(self, session) -> Dict[str, float]:
        """Analyze resource consumption patterns.

        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - total_consumed: Total resources consumed
            - avg_consumption_rate: Average consumption per step
            - peak_consumption: Maximum consumption in a step
            - consumption_variance: Variance in consumption rate
        """
        consumption_stats = (
            session.query(
                func.sum(SimulationStep.resources_consumed).label("total"),
                func.avg(SimulationStep.resources_consumed).label("average"),
                func.max(SimulationStep.resources_consumed).label("peak"),
                func.variance(SimulationStep.resources_consumed).label("variance"),
            )
            .first()
        )

        return {
            "total_consumed": float(consumption_stats[0] or 0),
            "avg_consumption_rate": float(consumption_stats[1] or 0),
            "peak_consumption": float(consumption_stats[2] or 0),
            "consumption_variance": float(consumption_stats[3] or 0),
        }

    @execute_query
    def resource_hotspots(self, session) -> List[Tuple[float, float, float]]:
        """Identify resource concentration hotspots.

        Returns
        -------
        List[Tuple[float, float, float]]
            List of tuples containing:
            - position_x: X coordinate
            - position_y: Y coordinate
            - concentration: Resource concentration
        """
        return (
            session.query(
                ResourceState.position_x,
                ResourceState.position_y,
                func.avg(ResourceState.amount).label("concentration"),
            )
            .group_by(ResourceState.position_x, ResourceState.position_y)
            .having(func.avg(ResourceState.amount) > 0)
            .order_by(func.avg(ResourceState.amount).desc())
            .all()
        )

    @execute_query
    def efficiency_metrics(self, session) -> Dict[str, float]:
        """Calculate resource efficiency metrics.

        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - utilization_rate: Resource utilization rate
            - distribution_efficiency: Resource distribution efficiency
            - consumption_efficiency: Resource consumption efficiency
            - regeneration_rate: Resource regeneration rate
        """
        metrics = (
            session.query(
                func.avg(SimulationStep.resource_efficiency).label("utilization"),
                func.avg(SimulationStep.distribution_efficiency).label("distribution"),
                func.avg(SimulationStep.consumption_efficiency).label("consumption"),
                func.avg(SimulationStep.regeneration_rate).label("regeneration"),
            )
            .first()
        )

        return {
            "utilization_rate": float(metrics[0] or 0),
            "distribution_efficiency": float(metrics[1] or 0),
            "consumption_efficiency": float(metrics[2] or 0),
            "regeneration_rate": float(metrics[3] or 0),
        }

    @execute_query
    def execute(self, session) -> Dict[str, Dict]:
        """Calculate comprehensive resource statistics.

        Returns
        -------
        Dict[str, Dict]
            Dictionary containing:
            - distribution: Resource distribution data
            - consumption: Consumption pattern statistics
            - hotspots: Resource concentration points
            - efficiency: Resource efficiency metrics
        """
        return {
            "distribution": self.resource_distribution(),
            "consumption": self.consumption_patterns(),
            "hotspots": self.resource_hotspots(),
            "efficiency": self.efficiency_metrics(),
        }
