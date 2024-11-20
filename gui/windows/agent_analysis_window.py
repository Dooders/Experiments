import sqlite3
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from gui.components.tooltips import ToolTip
from gui.windows.base_window import BaseWindow


class AgentAnalysisWindow(BaseWindow):
    """
    Window for detailed analysis of individual agents.

    Provides detailed visualization and analysis of:
    - Basic agent information
    - Current statistics
    - Performance metrics
    - Time series data
    - Action distributions
    """

    def __init__(self, parent: tk.Tk, db_path: str):
        super().__init__(
            parent,
            title="Agent Analysis",
            size=(1200, 800)
        )
        self.db_path = db_path
        self.chart_canvas = None
        
        self._load_agents()

    def _setup_ui(self):
        """Setup the main UI components with a grid layout."""
        # Agent Selection Area (Top)
        selection_frame = ttk.LabelFrame(
            self.window, text="Agent Selection", padding=10
        )
        selection_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)

        # Configure agent selection
        ttk.Label(selection_frame, text="Select Agent:").pack(
            side=tk.LEFT, padx=(0, 10)
        )
        self.agent_var = tk.StringVar()
        self.agent_combobox = ttk.Combobox(
            selection_frame, textvariable=self.agent_var, width=60
        )
        self.agent_combobox.pack(side=tk.LEFT)
        self.agent_combobox.bind("<<ComboboxSelected>>", self._on_agent_selected)

        # Main Content Area (Bottom)
        content_frame = ttk.Frame(self.window)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        # Configure column weights
        content_frame.grid_columnconfigure(0, weight=2)  # Info panel
        content_frame.grid_columnconfigure(1, weight=3)  # Charts panel
        content_frame.grid_rowconfigure(0, weight=1)

        # Left Side - Agent Information
        self._setup_info_panel(content_frame)

        # Right Side - Charts
        self._setup_charts_panel(content_frame)

    def _setup_info_panel(self, parent):
        """Setup the left panel containing agent information."""
        info_frame = ttk.LabelFrame(parent, text="Agent Information", padding=10)
        info_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        # Basic Info Section
        self.basic_info = ttk.LabelFrame(info_frame, text="Basic Details", padding=5)
        self.basic_info.pack(fill=tk.X, pady=(0, 10))

        # Create labels for basic info
        self.info_labels: Dict[str, ttk.Label] = {}
        basic_fields = [
            "Type",
            "Birth Time",
            "Death Time",
            "Generation",
            "Parent ID",
            "Initial Resources",
            "Max Health",
            "Starvation Threshold",
            "Genome ID",
        ]
        for field in basic_fields:
            container = ttk.Frame(self.basic_info)
            container.pack(fill=tk.X, pady=2)
            ttk.Label(container, text=f"{field}:", width=25).pack(side=tk.LEFT)
            self.info_labels[field] = ttk.Label(container, text="-", width=30)
            self.info_labels[field].pack(side=tk.LEFT)

        # Current Stats Section
        stats_frame = ttk.LabelFrame(info_frame, text="Current Statistics", padding=5)
        stats_frame.pack(fill=tk.X)

        self.stat_labels: Dict[str, ttk.Label] = {}
        stats = [
            "Health",
            "Resources",
            "Total Reward",
            "Age",
            "Is Defending",
            "Current Position",
        ]
        for stat in stats:
            container = ttk.Frame(stats_frame)
            container.pack(fill=tk.X, pady=2)
            ttk.Label(container, text=f"{stat}:", width=25).pack(side=tk.LEFT)
            self.stat_labels[stat] = ttk.Label(container, text="-", width=30)
            self.stat_labels[stat].pack(side=tk.LEFT)

        # Performance Metrics Section
        metrics_frame = ttk.LabelFrame(info_frame, text="Performance Metrics", padding=5)
        metrics_frame.pack(fill=tk.X, pady=(10, 0))

        self.metric_labels: Dict[str, ttk.Label] = {}
        metrics = [
            "Survival Time",
            "Peak Health",
            "Peak Resources",
            "Total Actions",
        ]
        for metric in metrics:
            container = ttk.Frame(metrics_frame)
            container.pack(fill=tk.X, pady=2)
            ttk.Label(container, text=f"{metric}:", width=25).pack(side=tk.LEFT)
            self.metric_labels[metric] = ttk.Label(container, text="-", width=30)
            self.metric_labels[metric].pack(side=tk.LEFT)

    def _setup_charts_panel(self, parent):
        """Setup the right panel containing charts."""
        charts_frame = ttk.LabelFrame(parent, text="Agent Analytics", padding=10)
        charts_frame.grid(row=0, column=1, sticky="nsew")

        # Configure grid for charts
        charts_frame.grid_columnconfigure(0, weight=1)
        charts_frame.grid_rowconfigure(0, weight=1)
        charts_frame.grid_rowconfigure(1, weight=1)

        # Create frames for different charts
        self.metrics_chart_frame = ttk.Frame(charts_frame)
        self.metrics_chart_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))

        self.actions_chart_frame = ttk.Frame(charts_frame)
        self.actions_chart_frame.grid(row=1, column=0, sticky="nsew", pady=(5, 0))

    def _load_agents(self):
        """Load available agents from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT a.agent_id, a.agent_type, a.birth_time 
                FROM Agents a 
                ORDER BY a.birth_time DESC
            """
            df = pd.read_sql_query(query, conn)
            conn.close()

            # Format combobox values
            self.agent_combobox["values"] = [
                f"Agent {row['agent_id']} ({row['agent_type']}) - Born: {row['birth_time']}"
                for _, row in df.iterrows()
            ]
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load agents: {e}")

    def _on_agent_selected(self, event):
        """Handle agent selection."""
        if not self.agent_var.get():
            return

        # Extract agent_id from selection string
        agent_id = int(self.agent_var.get().split()[1])
        self._load_agent_data(agent_id)

    def _load_agent_data(self, agent_id: int):
        """Load and display all data for selected agent."""
        try:
            conn = sqlite3.connect(self.db_path)

            # Load and update all data components
            self._update_info_labels(self._load_basic_info(conn, agent_id))
            self._update_stat_labels(self._load_agent_stats(conn, agent_id))
            self._update_metric_labels(self._load_performance_metrics(conn, agent_id))
            self._update_metrics_chart(conn, agent_id)
            self._update_actions_chart(conn, agent_id)

            conn.close()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load agent data: {e}")

    def _load_basic_info(self, conn, agent_id) -> Dict:
        """Load basic agent information from database."""
        query = """
            SELECT 
                agent_type,
                birth_time,
                death_time,
                generation,
                parent_id,
                initial_resources,
                max_health,
                starvation_threshold,
                genome_id
            FROM Agents 
            WHERE agent_id = ?
        """
        return pd.read_sql_query(query, conn, params=(agent_id,)).iloc[0].to_dict()

    def _load_agent_stats(self, conn, agent_id) -> Dict:
        """Load current agent statistics from database."""
        query = """
            WITH LatestState AS (
                SELECT 
                    s.current_health,
                    s.resource_level,
                    s.total_reward,
                    (s.step_number - a.birth_time) as age,
                    s.is_defending,
                    s.position_x,
                    s.position_y,
                    s.step_number,
                    ROW_NUMBER() OVER (ORDER BY s.step_number DESC) as rn
                FROM AgentStates s
                JOIN Agents a ON s.agent_id = a.agent_id
                WHERE s.agent_id = ?
            )
            SELECT 
                current_health,
                resource_level,
                total_reward,
                age,
                is_defending,
                position_x || ', ' || position_y as current_position
            FROM LatestState
            WHERE rn = 1
        """
        try:
            return pd.read_sql_query(query, conn, params=(agent_id,)).iloc[0].to_dict()
        except (IndexError, pd.errors.EmptyDataError):
            return {
                "current_health": 0,
                "resource_level": 0,
                "total_reward": 0,
                "age": 0,
                "is_defending": 0,
                "current_position": "0, 0",
            }

    def _load_performance_metrics(self, conn, agent_id) -> Dict:
        """Load agent performance metrics from database."""
        query = """
            SELECT 
                MAX(s.step_number) - MIN(s.step_number) as survival_time,
                MAX(s.current_health) as peak_health,
                MAX(s.resource_level) as peak_resources,
                COUNT(DISTINCT a.action_id) as total_actions
            FROM AgentStates s
            LEFT JOIN AgentActions a ON s.agent_id = a.agent_id 
                AND s.step_number = a.step_number
            WHERE s.agent_id = ?
            GROUP BY s.agent_id
        """
        try:
            return pd.read_sql_query(query, conn, params=(agent_id,)).iloc[0].to_dict()
        except IndexError:
            return {
                "survival_time": 0,
                "peak_health": 0,
                "peak_resources": 0,
                "total_actions": 0,
            }

    def _update_metrics_chart(self, conn, agent_id):
        """Update the metrics chart with agent's time series data."""
        query = """
            SELECT 
                step_number,
                current_health,
                resource_level,
                total_reward,
                is_defending
            FROM AgentStates
            WHERE agent_id = ?
            ORDER BY step_number
        """
        df = pd.read_sql_query(query, conn, params=(agent_id,))

        # Clear previous chart
        for widget in self.metrics_chart_frame.winfo_children():
            widget.destroy()

        if not df.empty:
            self._create_metrics_plot(df)

    def _create_metrics_plot(self, df):
        """Create the metrics time series plot."""
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Plot metrics
        ax.plot(df["step_number"], df["current_health"], 
                label="Health", color="#2ecc71", linewidth=2)
        ax.plot(df["step_number"], df["resource_level"], 
                label="Resources", color="#3498db", linewidth=2)
        ax.plot(df["step_number"], df["total_reward"], 
                label="Reward", color="#e74c3c", linewidth=2)

        # Add defense indicators
        defense_steps = df[df["is_defending"] == 1]["step_number"]
        if not defense_steps.empty:
            ax.scatter(defense_steps, [0] * len(defense_steps),
                      marker="^", color="red", label="Defending", alpha=0.5)

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.set_title("Agent Metrics Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, master=self.metrics_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _update_actions_chart(self, conn, agent_id):
        """Update the actions distribution and rewards charts."""
        query = """
            SELECT 
                action_type,
                COUNT(*) as count,
                AVG(CASE WHEN reward IS NOT NULL THEN reward ELSE 0 END) as avg_reward
            FROM AgentActions
            WHERE agent_id = ?
            GROUP BY action_type
            ORDER BY count DESC
        """
        df = pd.read_sql_query(query, conn, params=(agent_id,))

        # Clear previous chart
        for widget in self.actions_chart_frame.winfo_children():
            widget.destroy()

        if not df.empty:
            self._create_actions_plot(df)

    def _create_actions_plot(self, df):
        """Create the actions distribution and rewards plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Action counts
        bars = ax1.bar(df["action_type"], df["count"])
        ax1.set_xlabel("Action Type")
        ax1.set_ylabel("Count")
        ax1.set_title("Action Distribution")
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f"{int(height)}", ha="center", va="bottom")

        # Average rewards
        bars = ax2.bar(df["action_type"], df["avg_reward"])
        ax2.set_xlabel("Action Type")
        ax2.set_ylabel("Average Reward")
        ax2.set_title("Action Rewards")
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f"{height:.2f}", ha="center", va="bottom")

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.actions_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _update_info_labels(self, info: Dict):
        """Update the basic information labels."""
        for key, label in self.info_labels.items():
            value = info.get(key.lower().replace(" ", "_"), "-")
            label.config(text=str(value))

    def _update_stat_labels(self, stats: Dict):
        """Update the current statistics labels."""
        for key, label in self.stat_labels.items():
            value = stats.get(key.lower().replace(" ", "_"), "-")
            label.config(text=f"{value:.2f}" if isinstance(value, float) else str(value))

    def _update_metric_labels(self, metrics: Dict):
        """Update the performance metrics labels."""
        for key, label in self.metric_labels.items():
            value = metrics.get(key.lower().replace(" ", "_"), "-")
            label.config(text=f"{value:.2f}" if isinstance(value, float) else str(value)) 