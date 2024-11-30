import tkinter as tk
from tkinter import ttk
from typing import Dict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from gui.windows.base_window import BaseWindow
from core.database import SimulationDatabase

class StatisticsWindow(BaseWindow):
    """Window for displaying detailed simulation statistics."""

    def __init__(self, parent: tk.Tk, db_path: str):
        # Initialize database before calling super().__init__
        self.db_path = db_path
        self.db = SimulationDatabase(db_path)
        
        # Initialize tooltips dictionary
        self.tooltips = {
            "Population Momentum": "Product of survival time and peak population",
            "Average Population": "Mean number of agents across all simulation steps",
            "Peak Population": "Maximum number of agents at any point",
            "Simulation Length": "Number of steps until last agent died",
            "Resource Utilization": "Proportion of available resources consumed",
            "Resources Consumed": "Total resources used by all agents",
            "Resources Available": "Total resources generated in simulation",
            "Utilization per Agent": "Average resources consumed per agent per step",
            "Population Variance": "Measure of population size fluctuation",
            "Coefficient of Variation": "Standardized measure of population stability"
        }
        
        # Now call parent's __init__
        super().__init__(
            parent,
            title="Simulation Statistics",
            size=(800, 600)
        )

    def _setup_ui(self):
        """Setup the statistics window UI."""
        # Create notebook for different stat views
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Summary tab
        summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(summary_frame, text="Summary")
        self._setup_summary_tab(summary_frame)

        # Population tab
        population_frame = ttk.Frame(self.notebook)
        self.notebook.add(population_frame, text="Population")
        self._setup_population_tab(population_frame)

        # Resources tab
        resources_frame = ttk.Frame(self.notebook)
        self.notebook.add(resources_frame, text="Resources")
        self._setup_resources_tab(resources_frame)

    def _setup_summary_tab(self, parent):
        """Setup the summary statistics tab."""
        # Create scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Population Dynamics section
        self._add_stat_section(scrollable_frame, "Population Dynamics")
        
        # Get statistics using SQLAlchemy
        def _get_stats(session):
            stats = self.db.get_population_statistics()
            momentum = self.db.get_population_momentum()
            advanced_stats = self.db.get_advanced_statistics()
            return stats, momentum, advanced_stats
            
        stats, momentum, advanced_stats = self.db._execute_in_transaction(_get_stats)
        
        # Add population metrics
        self._add_stat_row("Population Momentum", f"{momentum:,.0f}")
        self._add_stat_row(
            "Average Population",
            f"{stats.get('average_population', 0):,.2f}"
        )
        self._add_stat_row(
            "Peak Population",
            f"{stats.get('peak_population', 0):,d}"
        )
        self._add_stat_row(
            "Simulation Length",
            f"{stats.get('death_step', 0):,d} steps"
        )
        
        # Resource Utilization section
        self._add_stat_section(scrollable_frame, "Resource Metrics")
        self._add_stat_row(
            "Resource Utilization",
            f"{stats.get('resource_utilization', 0):.2%}"
        )
        self._add_stat_row(
            "Resources Consumed",
            f"{stats.get('resources_consumed', 0):,.0f}"
        )
        self._add_stat_row(
            "Resources Available",
            f"{stats.get('resources_available', 0):,.0f}"
        )
        self._add_stat_row(
            "Utilization per Agent",
            f"{stats.get('utilization_per_agent', 0):.2f}"
        )
        
        # Stability Metrics section
        self._add_stat_section(scrollable_frame, "Stability Metrics")
        self._add_stat_row(
            "Population Variance",
            f"{stats.get('population_variance', 0):,.2f}"
        )
        self._add_stat_row(
            "Coefficient of Variation",
            f"{stats.get('coefficient_variation', 0):.2%}"
        )
        
        # Get advanced statistics
        advanced_stats = self.db.get_advanced_statistics()
        
        # Population Dynamics (Additional)
        self._add_stat_section(scrollable_frame, "Advanced Population Metrics")
        self._add_stat_row(
            "Peak-to-End Ratio",
            f"{advanced_stats.get('peak_to_end_ratio', 0):,.2f}"
        )
        self._add_stat_row(
            "Growth Rate",
            f"{advanced_stats.get('growth_rate', 0):,.2f} agents/step"
        )
        self._add_stat_row(
            "Extinction Threshold",
            f"Step {advanced_stats.get('extinction_threshold_time', 'N/A')}"
        )
        
        # Health and Survival
        self._add_stat_section(scrollable_frame, "Health & Survival")
        self._add_stat_row(
            "Average Health",
            f"{advanced_stats.get('average_health', 0):,.2%}"
        )
        self._add_stat_row(
            "Survivor Ratio",
            f"{advanced_stats.get('survivor_ratio', 0):,.2%}"
        )
        
        # Diversity and Interaction
        self._add_stat_section(scrollable_frame, "Diversity & Interaction")
        self._add_stat_row(
            "Agent Diversity",
            f"{advanced_stats.get('agent_diversity', 0):,.3f}"
        )
        self._add_stat_row(
            "Interaction Rate",
            f"{advanced_stats.get('interaction_rate', 0):,.2f} per agent/step"
        )
        self._add_stat_row(
            "Conflict/Cooperation Ratio",
            f"{advanced_stats.get('conflict_cooperation_ratio', 0):,.2f}"
        )
        
        # Resource Dynamics
        self._add_stat_section(scrollable_frame, "Resource Dynamics")
        self._add_stat_row(
            "Scarcity Index",
            f"{advanced_stats.get('scarcity_index', 0):,.2%}"
        )
        
        # Update tooltips dictionary with new metrics
        self.tooltips.update({
            "Peak-to-End Ratio": "Ratio of peak population to final population size",
            "Growth Rate": "Average change in population size per step",
            "Extinction Threshold": "Step when population fell below 10% of peak",
            "Average Health": "Mean health level across all agents over time",
            "Survivor Ratio": "Proportion of created agents that survived to the end",
            "Agent Diversity": "Shannon entropy of agent type distribution",
            "Interaction Rate": "Average interactions per agent per step",
            "Conflict/Cooperation Ratio": "Ratio of hostile to friendly interactions",
            "Scarcity Index": "Proportion of time resources were scarce"
        })
        
        # Add tooltips for each metric
        self._add_tooltips(scrollable_frame)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

    def _add_tooltips(self, frame):
        """Add tooltips explaining each metric."""
        tooltips = {
            "Population Momentum": "Product of survival time and peak population",
            "Average Population": "Mean number of agents across all simulation steps",
            "Peak Population": "Maximum number of agents at any point",
            "Simulation Length": "Number of steps until last agent died",
            "Resource Utilization": "Proportion of available resources consumed",
            "Resources Consumed": "Total resources used by all agents",
            "Resources Available": "Total resources generated in simulation",
            "Utilization per Agent": "Average resources consumed per agent per step",
            "Population Variance": "Measure of population size fluctuation",
            "Coefficient of Variation": "Standardized measure of population stability",
            "Peak-to-End Ratio": "Ratio of peak population to final population size",
            "Growth Rate": "Average change in population size per step",
            "Extinction Threshold": "Step when population fell below 10% of peak",
            "Average Health": "Mean health level across all agents over time",
            "Survivor Ratio": "Proportion of created agents that survived to the end",
            "Agent Diversity": "Shannon entropy of agent type distribution",
            "Interaction Rate": "Average interactions per agent per step",
            "Conflict/Cooperation Ratio": "Ratio of hostile to friendly interactions",
            "Scarcity Index": "Proportion of time resources were scarce"
        }
        
        def _find_labels(widget):
            """Recursively find all Label widgets."""
            labels = []
            if isinstance(widget, tk.Label) or isinstance(widget, ttk.Label):
                labels.append(widget)
            for child in widget.winfo_children():
                labels.extend(_find_labels(child))
            return labels
        
        # Find all labels recursively
        for label in _find_labels(frame):
            label_text = label.cget("text").replace(":", "")
            if label_text in tooltips:
                from gui.components.tooltips import ToolTip
                ToolTip(label, tooltips[label_text])

    def _add_stat_section(self, parent, title: str):
        """Add a new section header."""
        section_frame = ttk.LabelFrame(parent, text=title, padding=10)
        section_frame.pack(fill="x", padx=5, pady=5)
        self.current_section = section_frame

    def _add_stat_row(self, label: str, value: str):
        """Add a row of statistics."""
        row = ttk.Frame(self.current_section)
        row.pack(fill="x", pady=2)
        
        ttk.Label(row, text=label + ":", width=30, anchor="w").pack(side="left")
        ttk.Label(row, text=value).pack(side="left", padx=10)

    def _setup_population_tab(self, parent):
        """Setup the population statistics tab."""
        # Create figure and canvas
        fig = plt.figure(figsize=(10, 6))
        canvas = FigureCanvasTkAgg(fig, master=parent)
        
        def _get_data(session):
            # Use SQLAlchemy to get population data
            return self.db.get_historical_data()
            
        data = self.db._execute_in_transaction(_get_data)
        
        # Plot population trends
        ax = fig.add_subplot(111)
        ax.plot(data["steps"], data["metrics"]["total_agents"], label="Total")
        ax.plot(data["steps"], data["metrics"]["system_agents"], label="System")
        ax.plot(data["steps"], data["metrics"]["independent_agents"], label="Independent")
        
        ax.set_xlabel("Step")
        ax.set_ylabel("Population")
        ax.legend()
        
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _setup_resources_tab(self, parent):
        """Setup the resource statistics tab."""
        # Create figure and canvas
        fig = plt.figure(figsize=(10, 6))
        canvas = FigureCanvasTkAgg(fig, master=parent)
        
        def _get_data(session):
            # Use SQLAlchemy to get resource data
            return self.db.get_historical_data()
            
        data = self.db._execute_in_transaction(_get_data)
        
        # Plot resource trends
        ax = fig.add_subplot(111)
        ax.plot(data["steps"], data["metrics"]["total_resources"], label="Total Resources")
        ax.plot(data["steps"], data["metrics"]["average_agent_resources"], 
               label="Avg Resources/Agent")
        
        ax.set_xlabel("Step")
        ax.set_ylabel("Resources")
        ax.legend()
        
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        