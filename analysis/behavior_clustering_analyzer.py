from collections import defaultdict
from typing import Dict, List

import numpy as np
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.preprocessing import StandardScaler

from database.data_types import AgentActionData, BehaviorClustering
from database.repositories.agent_action_repository import AgentActionRepository


class BehaviorClusteringAnalyzer:
    """
    Analyzes agent behaviors and clusters them based on their action patterns using
    dynamic clustering algorithms (DBSCAN and Spectral Clustering).
    """

    def __init__(self, repository: AgentActionRepository):
        """
        Initialize the analyzer with a repository for accessing agent action data.

        Args:
            repository (AgentActionRepository): Repository to fetch agent action data.
        """
        self.repository = repository
        self.clustering_method = "dbscan"  # or "spectral"

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
        Cluster agents using dynamic clustering algorithms based on behavioral metrics.

        Args:
            agent_metrics (Dict[int, Dict[str, float]]): Dictionary of agent metrics.

        Returns:
            Dict[str, List[int]]: Dictionary mapping cluster names to lists of agent IDs.
        """
        # Convert metrics to feature matrix
        agent_ids = list(agent_metrics.keys())
        features = []

        for agent_id in agent_ids:
            metrics = agent_metrics[agent_id]
            if metrics["total_actions"] == 0:
                continue

            feature_vector = [
                metrics["attack_count"] / metrics["total_actions"],
                metrics["share_count"] / metrics["total_actions"],
                metrics["interaction_count"] / metrics["total_actions"],
                metrics["success_count"] / metrics["total_actions"],
                metrics["total_reward"] / metrics["total_actions"],
            ]
            features.append(feature_vector)

        if not features:
            return {"unclustered": []}

        # Standardize features
        X = StandardScaler().fit_transform(features)

        # Perform clustering
        if self.clustering_method == "dbscan":
            clustering = DBSCAN(eps=0.5, min_samples=2).fit(X)
            labels = clustering.labels_
        else:  # spectral
            n_clusters = min(
                len(features), 4
            )  # Adjust number of clusters based on data
            clustering = SpectralClustering(
                n_clusters=n_clusters, affinity="nearest_neighbors"
            ).fit(X)
            labels = clustering.labels_

        # Group agents by cluster
        clusters = defaultdict(list)
        for agent_idx, cluster_label in enumerate(labels):
            if cluster_label == -1:  # Noise points in DBSCAN
                cluster_name = "outliers"
            else:
                # Analyze cluster characteristics to assign meaningful names
                cluster_features = X[labels == cluster_label].mean(axis=0)
                cluster_name = self._get_cluster_name(cluster_features)
            clusters[cluster_name].append(agent_ids[agent_idx])

        return dict(clusters)

    def _get_cluster_name(self, cluster_features: np.ndarray) -> str:
        """
        Determine cluster name based on average feature characteristics.

        Args:
            cluster_features (np.ndarray): Average feature values for the cluster

        Returns:
            str: Descriptive name for the cluster
        """
        # Features order: [attack_rate, share_rate, interaction_rate, success_rate, avg_reward]
        if cluster_features[0] > 0.5:  # High attack rate
            return "aggressive"
        elif (
            cluster_features[1] + cluster_features[2] > 0.6
        ):  # High share + interaction
            return "cooperative"
        elif (
            cluster_features[3] > 0.6 and cluster_features[4] > 0.5
        ):  # High success + reward
            return "efficient"
        else:
            return "balanced"

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
