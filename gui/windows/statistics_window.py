import tkinter as tk
from tkinter import ttk
from typing import Dict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from gui.windows.base_window import BaseWindow
from database import SimulationDatabase

class StatisticsWindow(BaseWindow):
    """Window for displaying detailed simulation statistics."""

    def __init__(self, parent: tk.Tk, db_path: str):
        # Initialize database before calling super().__init__
        self.db_path = db_path
        self.db = SimulationDatabase(db_path)
        
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

        # Add statistics
        self._add_stat_section(scrollable_frame, "Population Dynamics")
        momentum = self.db.get_population_momentum()
        self._add_stat_row("Population Momentum", f"{momentum:,.0f}")
        
        # Add other summary statistics...
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

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
        # Add population charts and metrics
        pass

    def _setup_resources_tab(self, parent):
        """Setup the resource statistics tab."""
        # Add resource usage charts and metrics
        pass 