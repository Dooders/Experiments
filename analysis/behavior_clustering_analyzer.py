from typing import Dict, List

from database.data_types import AgentActionData, BehaviorClustering
from database.repositories.agent_action_repository import AgentActionRepository


class BehaviorClusteringAnalyzer:
    """
    Analyzes agent behaviors and clusters them based on their action patterns.

    This analyzer processes agent actions to identify behavioral patterns and group
    agents into meaningful clusters based on their interaction styles and performance metrics.
    """

    def __init__(self, repository: AgentActionRepository):
        """
        Initialize the analyzer with a repository for accessing agent action data.

        Args:
            repository (AgentActionRepository): Repository to fetch agent action data.
        """
        self.repository = repository

    def analyze(
        self,
        scope: str,
        agent_id: int = None,
        step: int = None,
        step_range: tuple = None,
    ) -> BehaviorClustering:
        """
        Analyze agent behaviors and create behavioral clusters.

        Args:
            scope (str): The scope of analysis (e.g., 'episode', 'training').
            agent_id (int, optional): Specific agent ID to analyze.
            step (int, optional): Specific step to analyze.
            step_range (tuple, optional): Range of steps to analyze.

        Returns:
            BehaviorClustering: Object containing cluster assignments, characteristics, and performance metrics.
        """
        actions = self.repository.get_actions_by_scope(
            scope, agent_id, step, step_range
        )
        agent_metrics = self._calculate_agent_metrics(actions)
        clusters = self._cluster_agents(agent_metrics)
        characteristics, performance = self._calculate_cluster_characteristics(
            clusters, agent_metrics
        )
        return BehaviorClustering(
            clusters=clusters,
            cluster_characteristics=characteristics,
            cluster_performance=performance,
        )

    def _calculate_agent_metrics(
        self, actions: List[AgentActionData]
    ) -> Dict[int, Dict[str, float]]:
        """
        Calculate behavioral metrics for each agent based on their actions.

        Args:
            actions (List[AgentActionData]): List of agent actions to analyze.

        Returns:
            Dict[int, Dict[str, float]]: Dictionary mapping agent IDs to their metrics,
            including attack counts, share counts, interaction counts, success rates, and rewards.
        """
        agent_metrics = {}
        for action in actions:
            agent_id = action.agent_id
            if agent_id not in agent_metrics:
                agent_metrics[agent_id] = {
                    "attack_count": 0,
                    "share_count": 0,
                    "interaction_count": 0,
                    "success_count": 0,
                    "total_reward": 0,
                    "total_actions": 0,
                }
            metrics = agent_metrics[agent_id]
            metrics["total_actions"] += 1
            metrics["total_reward"] += action.reward or 0
            if action.action_type == "attack":
                metrics["attack_count"] += 1
            if action.action_type == "share":
                metrics["share_count"] += 1
            if action.action_target_id:
                metrics["interaction_count"] += 1
            if action.reward and action.reward > 0:
                metrics["success_count"] += 1
        return agent_metrics

    def _cluster_agents(
        self, agent_metrics: Dict[int, Dict[str, float]]
    ) -> Dict[str, List[int]]:
        """
        Cluster agents based on their behavioral metrics.

        Args:
            agent_metrics (Dict[int, Dict[str, float]]): Dictionary of agent metrics.

        Returns:
            Dict[str, List[int]]: Dictionary mapping cluster names to lists of agent IDs.
            Clusters include: 'aggressive', 'cooperative', 'efficient', and 'balanced'.
        """
        clusters = {
            "aggressive": [],
            "cooperative": [],
            "efficient": [],
            "balanced": [],
        }
        for agent_id, metrics in agent_metrics.items():
            if metrics["total_actions"] == 0:
                continue
            attack_rate = metrics["attack_count"] / metrics["total_actions"]
            share_rate = metrics["share_count"] / metrics["total_actions"]
            interaction_rate = metrics["interaction_count"] / metrics["total_actions"]
            success_rate = metrics["success_count"] / metrics["total_actions"]
            avg_reward = metrics["total_reward"] / metrics["total_actions"]
            if attack_rate > 0.3:
                clusters["aggressive"].append(agent_id)
            elif share_rate > 0.3 or interaction_rate > 0.4:
                clusters["cooperative"].append(agent_id)
            elif success_rate > 0.7 and avg_reward > 1.0:
                clusters["efficient"].append(agent_id)
            else:
                clusters["balanced"].append(agent_id)
        return clusters

    def _calculate_cluster_characteristics(
        self, clusters: Dict[str, List[int]], agent_metrics: Dict[int, Dict[str, float]]
    ) -> tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
        """
        Calculate characteristic metrics and performance for each cluster.

        Args:
            clusters (Dict[str, List[int]]): Dictionary mapping cluster names to agent IDs.
            agent_metrics (Dict[int, Dict[str, float]]): Dictionary of agent metrics.

        Returns:
            tuple[Dict[str, Dict[str, float]], Dict[str, float]]: A tuple containing:
                - Dictionary of cluster characteristics (attack rate, cooperation, etc.)
                - Dictionary of cluster performance metrics (average rewards)
        """
        characteristics = {}
        performance = {}
        for cluster_name, agent_ids in clusters.items():
            if not agent_ids:
                continue
            cluster_metrics = {
                "attack_rate": 0,
                "cooperation": 0,
                "risk_taking": 0,
                "success_rate": 0,
                "resource_efficiency": 0,
            }
            total_reward = 0
            for agent_id in agent_ids:
                metrics = agent_metrics[agent_id]
                total_actions = metrics["total_actions"]
                cluster_metrics["attack_rate"] += (
                    metrics["attack_count"] / total_actions
                )
                cluster_metrics["cooperation"] += (
                    metrics["share_count"] + metrics["interaction_count"]
                ) / total_actions
                cluster_metrics["success_rate"] += (
                    metrics["success_count"] / total_actions
                )
                cluster_metrics["resource_efficiency"] += (
                    metrics["total_reward"] / total_actions
                )
                total_reward += metrics["total_reward"] / total_actions
            n_agents = len(agent_ids)
            for metric in cluster_metrics:
                cluster_metrics[metric] /= n_agents
            characteristics[cluster_name] = cluster_metrics
            performance[cluster_name] = total_reward / n_agents
        return characteristics, performance
