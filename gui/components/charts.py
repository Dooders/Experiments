import tkinter as tk
from tkinter import ttk
from typing import Dict, Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class SimulationChart(ttk.Frame):
    """Component for displaying simulation metrics charts."""

    def __init__(self, parent):
        super().__init__(parent)
        self.is_dragging = False
        self.was_playing = False
        
        # Store historical data
        self.history = {
            "steps": [],
            "system_agents": [],
            "independent_agents": [],
            "control_agents": [],
            "total_resources": []
        }
        
        # Store full simulation data
        self.full_data = {
            "steps": [],
            "system_agents": [],
            "independent_agents": [],
            "control_agents": [],
            "total_resources": []
        }
        
        self._setup_chart()
        self._setup_interactions()

    def _setup_chart(self):
        """Initialize the chart figure and axes."""
        # Create figure with two y-axes
        self.fig = Figure(figsize=(8, 4))
        self.fig.subplots_adjust(right=0.85)
        self.ax1 = self.fig.add_subplot(111)
        self.ax2 = self.ax1.twinx()

        # Initialize empty line objects with colors
        self.lines = {
            "system_agents": self.ax1.plot(
                [], [], "b-", label="System Agents", alpha=1.0)[0],
            "independent_agents": self.ax1.plot(
                [], [], "r-", label="Independent Agents", alpha=1.0)[0],
            "control_agents": self.ax1.plot(
                [], [], color="#FFD700", label="Control Agents", alpha=1.0)[0],
            "resources": self.ax2.plot(
                [], [], "g-", label="Resources", alpha=1.0)[0],
            "current_step": self.ax1.axvline(
                x=0, color="gray", linestyle="--", alpha=0.5
            )
        }
        
        # Initialize future data lines (semi-transparent)
        self.future_lines = {
            "system_agents": self.ax1.plot(
                [], [], "b-", alpha=0.3)[0],
            "independent_agents": self.ax1.plot(
                [], [], "r-", alpha=0.3)[0],
            "control_agents": self.ax1.plot(
                [], [], color="#FFD700", alpha=0.3)[0],
            "resources": self.ax2.plot(
                [], [], "g-", alpha=0.3)[0]
        }

        # Setup axis labels and colors
        self.ax1.set_xlabel("Step")
        self.ax1.set_ylabel("Agent Count", color="black")
        self.ax2.set_ylabel("Resource Count", color="green", rotation=270, labelpad=20)

        # Configure axis positions and colors
        self.ax2.yaxis.set_label_position("right")
        self.ax2.yaxis.set_ticks_position("right")
        self.ax1.tick_params(axis="y", labelcolor="black")
        self.ax2.tick_params(axis="y", labelcolor="green")

        # Setup legends
        self.ax1.legend(loc="upper left")
        self.ax2.legend(loc="upper right")

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update(self, data: Optional[Dict] = None):
        """Update chart with new simulation data."""
        if not data or "metrics" not in data:
            return

        try:
            # Get metrics
            metrics = data.get("metrics", {})
            if not metrics:
                return

            # Get current step from history length
            current_step = len(self.history["steps"])
            
            # Update history (current data)
            self.history["steps"].append(current_step)
            self.history["system_agents"].append(metrics.get("system_agents", 0))
            self.history["independent_agents"].append(metrics.get("independent_agents", 0))
            self.history["control_agents"].append(metrics.get("control_agents", 0))
            self.history["total_resources"].append(metrics.get("total_resources", 0))

            # Store full data if this is new data
            if current_step >= len(self.full_data["steps"]):
                self.full_data["steps"].append(current_step)
                self.full_data["system_agents"].append(metrics.get("system_agents", 0))
                self.full_data["independent_agents"].append(metrics.get("independent_agents", 0))
                self.full_data["control_agents"].append(metrics.get("control_agents", 0))
                self.full_data["total_resources"].append(metrics.get("total_resources", 0))

            # Update current data lines
            for key in ["system_agents", "independent_agents", "control_agents"]:
                self.lines[key].set_data(
                    self.history["steps"], 
                    self.history[key]
                )

            # Update current resources line
            self.lines["resources"].set_data(
                self.history["steps"], 
                self.history["total_resources"]
            )

            # Update future data lines
            for key in ["system_agents", "independent_agents", "control_agents"]:
                self.future_lines[key].set_data(
                    self.full_data["steps"][current_step:], 
                    self.full_data[key][current_step:]
                )

            # Update future resources line
            self.future_lines["resources"].set_data(
                self.full_data["steps"][current_step:], 
                self.full_data["total_resources"][current_step:]
            )

            # Update current step line
            self.lines["current_step"].set_xdata([current_step, current_step])
            
            # Calculate max y value for step line and axis limits
            max_agents = max(
                max(self.full_data["system_agents"] or [0]),
                max(self.full_data["independent_agents"] or [0]),
                max(self.full_data["control_agents"] or [0])
            )
            max_resources = max(self.full_data["total_resources"] or [0])
            
            # Set y-limits with padding
            self.ax1.set_ylim(0, max(1, max_agents * 1.1))
            self.ax2.set_ylim(0, max(1, max_resources * 1.1))
            
            # Set x-limits with padding
            max_step = max(self.full_data["steps"]) if self.full_data["steps"] else 100
            self.ax1.set_xlim(0, max(max_step + 10, 100))
            
            # Update step line height
            overall_max = max(max_agents, max_resources)
            self.lines["current_step"].set_ydata([0, overall_max * 1.1])

            # Force redraw
            self.canvas.draw()

        except Exception as e:
            import logging
            logging.error(f"Failed to update chart: {str(e)}")
            logging.error(f"Data received: {data}")

    def _setup_interactions(self):
        """Setup mouse interaction handlers."""
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

    def _on_press(self, event):
        """Handle mouse press on the chart."""
        if event.inaxes in (self.ax1, self.ax2):
            self.is_dragging = True
            self.was_playing = getattr(self, "playing", False)
            self._notify_timeline_click(event.xdata)

    def _on_motion(self, event):
        """Handle mouse motion while dragging."""
        if self.is_dragging and event.inaxes in (self.ax1, self.ax2):
            self._notify_timeline_click(event.xdata)

    def _on_release(self, event):
        """Handle mouse release to end dragging."""
        if self.is_dragging:
            self.is_dragging = False
            if self.was_playing:
                self._notify_playback_resume()

    def _notify_timeline_click(self, x_coord: float):
        """Notify parent of timeline navigation."""
        if hasattr(self, "on_timeline_click") and x_coord is not None:
            step = int(round(x_coord))
            self.on_timeline_click(step)

    def _notify_playback_resume(self):
        """Notify parent to resume playback."""
        if hasattr(self, "on_playback_resume"):
            self.on_playback_resume()

    def set_timeline_callback(self, callback):
        """Set callback for timeline navigation."""
        self.on_timeline_click = callback

    def set_playback_callback(self, callback):
        """Set callback for playback resume."""
        self.on_playback_resume = callback

    def clear(self):
        """Clear all chart data."""
        self.history = {key: [] for key in self.history}
        self.full_data = {key: [] for key in self.full_data}
        
        # Clear all lines
        for line in self.lines.values():
            line.set_data([], [])
        for line in self.future_lines.values():
            line.set_data([], [])
            
        # Reset axes
        self.ax1.set_xlim(0, 100)
        self.ax1.set_ylim(0, 1)
        self.ax2.set_ylim(0, 1)
        
        self.canvas.draw() 

    def set_full_data(self, data: Dict):
        """Store the full simulation data without displaying it."""
        if not data or "metrics" not in data:
            return
        
        try:
            steps = data["steps"]
            metrics = data["metrics"]
            
            # Store full data
            self.full_data = {
                "steps": steps,
                "system_agents": metrics["system_agents"],
                "independent_agents": metrics["independent_agents"],
                "control_agents": metrics["control_agents"],
                "total_resources": metrics["total_resources"]
            }
            
            # Clear history (current display)
            self.history = {key: [] for key in self.history}
            
            # Update axes limits based on full data
            max_agents = max(
                max(metrics["system_agents"] or [0]),
                max(metrics["independent_agents"] or [0]),
                max(metrics["control_agents"] or [0])
            )
            max_resources = max(metrics["total_resources"] or [0])
            
            # Set y-limits with padding
            self.ax1.set_ylim(0, max(1, max_agents * 1.1))
            self.ax2.set_ylim(0, max(1, max_resources * 1.1))
            
            # Set x-limits with padding
            max_step = max(steps) if steps else 100
            self.ax1.set_xlim(0, max(max_step + 10, 100))
            
            # Clear all lines
            for line in self.lines.values():
                line.set_data([], [])
            for line in self.future_lines.values():
                line.set_data([], [])
            
            # Show future data in semi-transparent
            for key in ["system_agents", "independent_agents", "control_agents"]:
                self.future_lines[key].set_data(steps, metrics[key])
            
            self.future_lines["resources"].set_data(steps, metrics["total_resources"])
            
            # Force redraw
            self.canvas.draw()
            
        except Exception as e:
            import logging
            logging.error(f"Failed to set full data: {str(e)}") 