import logging
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from gui.components.charts import SimulationChart
from gui.components.controls import ControlPanel
from gui.components.environment import EnvironmentView
from gui.components.stats import StatsPanel
from gui.components.tooltips import ToolTip
from gui.utils.styles import configure_ttk_styles
from gui.windows.agent_analysis_window import AgentAnalysisWindow

from config import SimulationConfig
from database import SimulationDatabase
from simulation import run_simulation


class SimulationGUI:
    """Main GUI application for running and visualizing agent-based simulations."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Agent-Based Simulation")
        self.root.geometry("1200x800")

        # Initialize variables
        self.current_db_path = None
        self.current_step = 0
        self.components = {}
        
        # Configure styles
        configure_ttk_styles()

        # Setup main components
        self._setup_menu()
        self._setup_main_frame()
        self._setup_logging()
        self._show_welcome_screen()

    def _setup_main_frame(self) -> None:
        """Setup the main container frame."""
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Configure grid weights
        self.main_frame.grid_columnconfigure(0, weight=2)  # Left pane
        self.main_frame.grid_columnconfigure(1, weight=3)  # Right pane
        self.main_frame.grid_rowconfigure(0, weight=1)

    def _show_welcome_screen(self):
        """Show the welcome screen."""
        # Clear any existing components
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        # Create welcome frame
        welcome_frame = ttk.Frame(self.main_frame)
        welcome_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")
        welcome_frame.grid_columnconfigure(0, weight=1)
        welcome_frame.grid_rowconfigure(0, weight=1)

        # Welcome message
        welcome_text = (
            "Welcome to Agent-Based Simulation\n\n"
            "Use the menu to:\n"
            "• Start a new simulation\n"
            "• Open an existing simulation\n"
            "• Configure simulation parameters"
        )
        
        welcome_label = ttk.Label(
            welcome_frame,
            text=welcome_text,
            justify=tk.CENTER,
            font=("Arial", 14)
        )
        welcome_label.grid(row=0, column=0)

        # Quick action buttons
        button_frame = ttk.Frame(welcome_frame)
        button_frame.grid(row=1, column=0, pady=20)

        ttk.Button(
            button_frame,
            text="New Simulation",
            command=self._new_simulation,
            style="Action.TButton"
        ).pack(side=tk.LEFT, padx=10)

        ttk.Button(
            button_frame,
            text="Open Simulation",
            command=self._open_simulation,
            style="Action.TButton"
        ).pack(side=tk.LEFT, padx=10)

    def _setup_simulation_view(self):
        """Setup the simulation visualization components."""
        # Clear existing components
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        # Create components
        self.components["stats"] = StatsPanel(self.main_frame)
        self.components["stats"].grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Hide progress and log sections
        if hasattr(self.components["stats"], "progress_frame"):
            self.components["stats"].progress_frame.grid_remove()
        if hasattr(self.components["stats"], "log_frame"):
            self.components["stats"].log_frame.grid_remove()
        self.components["stats"].hide_progress()
        self.components["stats"].clear_log()

        self.components["chart"] = SimulationChart(self.main_frame)
        self.components["chart"].grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        self.components["environment"] = EnvironmentView(self.main_frame)
        self.components["environment"].grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.components["controls"] = ControlPanel(
            self.main_frame,
            play_callback=self._toggle_playback,
            step_callback=self._step_to,
            export_callback=self._export_data
        )
        self.components["controls"].grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=5
        )

        # Configure grid weights for proper layout
        self.main_frame.grid_columnconfigure(0, weight=2)  # Stats panel
        self.main_frame.grid_columnconfigure(1, weight=3)  # Chart
        self.main_frame.grid_rowconfigure(0, weight=1)     # Top row
        self.main_frame.grid_rowconfigure(1, weight=1)     # Bottom row

    def _new_simulation(self) -> None:
        """Start a new simulation."""
        try:
            config = SimulationConfig.from_yaml("config.yaml")
            
            # Create new database
            self.current_db_path = "simulations/simulation.db"
            os.makedirs("simulations", exist_ok=True)

            # Show progress screen
            self._show_progress_screen("Running simulation...")

            # Run simulation in separate thread
            import threading
            sim_thread = threading.Thread(
                target=self._run_simulation,
                args=(config,)
            )
            sim_thread.start()

        except Exception as e:
            self.show_error("Error", f"Failed to start simulation: {str(e)}")

    def _clear_progress_screen(self):
        """Clear the progress screen and all its components."""
        # First stop the progress bar if it exists
        if hasattr(self, 'progress_bar'):
            try:
                self.progress_bar.stop()
                self.progress_bar.grid_remove()
            except Exception:
                pass
            delattr(self, 'progress_bar')
        
        # Remove the progress frame
        if hasattr(self, 'progress_frame'):
            try:
                self.progress_frame.grid_remove()
                self.progress_frame.destroy()
            except Exception:
                pass
            delattr(self, 'progress_frame')
        
        # Clear all widgets from main frame
        for widget in self.main_frame.winfo_children():
            try:
                widget.grid_remove()
                widget.destroy()
            except Exception:
                pass
        
        # Update the display
        self.main_frame.update()

    def _show_progress_screen(self, message: str):
        """Show progress screen while simulation is running."""
        # Clear existing screen
        self._clear_progress_screen()

        # Create progress frame
        self.progress_frame = ttk.Frame(self.main_frame)
        self.progress_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.progress_frame.grid_columnconfigure(0, weight=1)
        self.progress_frame.grid_rowconfigure(0, weight=1)

        # Progress message
        ttk.Label(
            self.progress_frame,
            text=message,
            font=("Arial", 12)
        ).grid(row=0, column=0, pady=10)

        # Progress bar
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            mode="indeterminate",
            length=300
        )
        self.progress_bar.grid(row=1, column=0, pady=10)
        self.progress_bar.start()

    def _setup_menu(self) -> None:
        """Create the application menu bar."""
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)

        # File Menu
        file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=file_menu)
        
        file_menu.add_command(
            label="New Simulation",
            command=self._new_simulation,
            accelerator="Ctrl+N"
        )
        file_menu.add_command(
            label="Open Simulation",
            command=self._open_simulation,
            accelerator="Ctrl+O"
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Export Data",
            command=self._export_data,
            accelerator="Ctrl+E"
        )
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_exit)

        # Simulation Menu
        sim_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Simulation", menu=sim_menu)
        sim_menu.add_command(label="Run Batch", command=self._run_batch)
        sim_menu.add_command(label="Configure", command=self._configure_simulation)

        # Analysis Menu
        analysis_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Generate Report", command=self._generate_report)
        analysis_menu.add_command(label="View Statistics", command=self._view_statistics)
        analysis_menu.add_command(
            label="Agent Analysis",
            command=self._open_agent_analysis_window
        )

        # Help Menu
        help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self._show_documentation)
        help_menu.add_command(label="About", command=self._show_about)

        # Bind keyboard shortcuts
        self.root.bind("<Control-n>", lambda e: self._new_simulation())
        self.root.bind("<Control-o>", lambda e: self._open_simulation())
        self.root.bind("<Control-e>", lambda e: self._export_data())

    def _run_simulation(self, config: SimulationConfig) -> None:
        """Run simulation in background thread."""
        try:
            run_simulation(
                num_steps=config.simulation_steps,
                config=config,
                db_path=self.current_db_path
            )
            self.root.after(0, self._simulation_complete)
        except Exception as e:
            self.root.after(0, self._simulation_error, str(e))

    def _start_visualization(self) -> None:
        """Initialize visualization components with simulation data."""
        if not self.current_db_path:
            return

        try:
            # Initialize database connection
            db = SimulationDatabase(self.current_db_path)
            
            # Get historical data for the chart
            historical_data = db.get_historical_data()
            
            # Store the full data in the chart but don't display it yet
            if historical_data and "metrics" in historical_data:
                logging.debug("Setting full data in chart")
                self.components["chart"].set_full_data({
                    "steps": historical_data["steps"],
                    "metrics": {
                        "system_agents": historical_data["metrics"]["system_agents"],
                        "independent_agents": historical_data["metrics"]["independent_agents"],
                        "control_agents": historical_data["metrics"]["control_agents"],
                        "total_resources": historical_data["metrics"]["total_resources"],
                    }
                })
            
            # Reset to initial state (step 0)
            initial_data = db.get_simulation_data(0)
            self.current_step = 0
            
            # Set up timeline interaction callbacks first
            logging.debug("Setting up timeline callbacks")
            self.components["chart"].set_timeline_callback(self._step_to)
            self.components["chart"].set_playback_callback(
                lambda: self.components["controls"].set_playing(True)
            )
            
            # Update components with initial data
            for name, component in self.components.items():
                if name != "controls" and hasattr(component, "update"):
                    component.update(initial_data)

        except Exception as e:
            logging.error(f"Error starting visualization: {str(e)}", exc_info=True)
            self.show_error("Visualization Error", f"Failed to initialize visualization: {str(e)}")

    def _open_simulation(self) -> None:
        """Open existing simulation database."""
        filepath = filedialog.askopenfilename(
            title="Open Simulation",
            initialdir="simulations",
            filetypes=[("Database files", "*.db"), ("All files", "*.*")]
        )
        if filepath:
            self.current_db_path = filepath
            self._setup_simulation_view()
            self._start_visualization()

    def _run_batch(self) -> None:
        """Run batch simulation."""
        messagebox.showinfo("Not Implemented", "Batch simulation not yet implemented.")

    def _configure_simulation(self) -> None:
        """Open configuration dialog."""
        messagebox.showinfo("Not Implemented", "Configuration dialog not yet implemented.")

    def _generate_report(self) -> None:
        """Generate analysis report."""
        messagebox.showinfo("Not Implemented", "Report generation not yet implemented.")

    def _view_statistics(self) -> None:
        """Show statistics window."""
        messagebox.showinfo("Not Implemented", "Statistics view not yet implemented.")

    def _open_agent_analysis_window(self) -> None:
        """Open agent analysis window."""
        if not self.current_db_path:
            messagebox.showwarning("No Data", "Please open or run a simulation first.")
            return
        AgentAnalysisWindow(self.root, self.current_db_path)

    def _show_documentation(self) -> None:
        """Show documentation window."""
        messagebox.showinfo("Not Implemented", "Documentation not yet implemented.")

    def _show_about(self) -> None:
        """Show about dialog."""
        messagebox.showinfo("About", 
            "Agent-Based Simulation\n\n"
            "A tool for running and analyzing agent-based simulations."
        )

    def _on_exit(self) -> None:
        """Handle application exit."""
        if messagebox.askokcancel("Exit", "Do you want to exit the application?"):
            self.root.quit()

    def _export_data(self) -> None:
        """Export simulation data."""
        if not self.current_db_path:
            messagebox.showwarning("No Data", "Please run or open a simulation first.")
            return
            
        filepath = filedialog.asksaveasfilename(
            title="Export Data",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                db = SimulationDatabase(self.current_db_path)
                db.export_data(filepath)
                messagebox.showinfo("Success", "Data exported successfully!")
            except Exception as e:
                self.show_error("Export Error", f"Failed to export data: {str(e)}")

    def _setup_logging(self) -> None:
        """Configure application logging system."""
        if not os.path.exists("logs"):
            os.makedirs("logs")

        class GUIHandler(logging.Handler):
            def __init__(self, gui):
                super().__init__()
                self.gui = gui

            def emit(self, record):
                msg = self.format(record)
                if hasattr(self.gui, "components") and "stats" in self.gui.components:
                    self.gui.components["stats"].log_message(msg)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add GUI handler
        gui_handler = GUIHandler(self)
        gui_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        root_logger.addHandler(gui_handler)

    def _toggle_playback(self, playing: bool) -> None:
        """Handle playback state change."""
        if playing:
            # Start playback
            self._play_simulation()
        else:
            # Stop playback
            self._stop_simulation()

    def _play_simulation(self):
        """Start simulation playback."""
        if not self.current_db_path:
            return
        
        try:
            db = SimulationDatabase(self.current_db_path)
            data = db.get_simulation_data(self.current_step + 1)
            
            # Check if we've reached the end of the data
            if not data or not data.get("metrics"):
                # Stop playback
                self.components["controls"].set_playing(False)
                return
            
            self.current_step += 1
            # Update each component with the data, except controls
            for name, component in self.components.items():
                if name != "controls" and hasattr(component, "update"):
                    try:
                        # Ensure data is passed as a dictionary
                        if not isinstance(data, dict):
                            logging.warning(f"Invalid data format for {name}: {type(data)}")
                            continue
                        component.update(data)
                    except Exception as comp_error:
                        logging.error(f"Error updating {name}: {str(comp_error)}")
                    
            # Schedule next update if still playing
            if self.components["controls"].playing:
                delay = self.components["controls"].get_delay()
                self.root.after(delay, self._play_simulation)
                
        except Exception as e:
            self.show_error("Playback Error", f"Failed to update simulation: {str(e)}")
            self.components["controls"].set_playing(False)

    def _stop_simulation(self):
        """Stop simulation playback."""
        # Nothing needs to be done here since we're using after() for timing
        pass

    def _step_to(self, step: int) -> None:
        """Move to specific simulation step."""
        if not self.current_db_path:
            return
            
        try:
            db = SimulationDatabase(self.current_db_path)
            
            # Ensure step is within valid range
            if step < 0:
                step = 0
            
            # Get max step from chart's full data
            max_step = len(self.components["chart"].full_data["steps"]) - 1
            if step > max_step:
                step = max_step
            
            data = db.get_simulation_data(step)
            
            if not isinstance(data, dict):
                raise ValueError(f"Invalid data format: expected dict, got {type(data)}")
            
            # Update current step
            self.current_step = step
            
            # Reset chart history to current step
            self.components["chart"].reset_history_to_step(step)
            
            # Update other components
            for name, component in self.components.items():
                if name not in ["controls", "chart"] and hasattr(component, "update"):
                    try:
                        component.update(data)
                    except Exception as comp_error:
                        logging.error(f"Error updating {name}: {str(comp_error)}")
                    
        except Exception as e:
            self.show_error("Navigation Error", f"Failed to move to step {step}: {str(e)}")

    def _simulation_complete(self) -> None:
        """Handle simulation completion."""
        try:
            # Clear the progress screen first
            self._clear_progress_screen()
            
            # Setup and start visualization
            self._setup_simulation_view()
            self._start_visualization()
            
        except Exception as e:
            logging.error(f"Error during simulation completion: {str(e)}")
            self.show_error("Error", "Failed to complete simulation setup")

    def _simulation_error(self, error_msg: str) -> None:
        """Handle simulation error."""
        self.progress_bar.stop()
        self._show_welcome_screen()
        self.show_error("Simulation Error", error_msg)

    def show_error(self, title: str, message: str):
        """Display error message."""
        messagebox.showerror(title, message, parent=self.root)