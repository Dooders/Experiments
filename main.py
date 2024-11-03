import logging
import os
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

from agents import main as run_simulation
from analysis import SimulationAnalyzer
from batch_runner import BatchRunner
from config import SimulationConfig
from gui_config import *
from visualization import SimulationVisualizer


class SimulationGUI:
    """
    Main GUI application for running and visualizing agent-based simulations.

    This class provides a graphical interface for:
    - Running new simulations
    - Loading existing simulations
    - Configuring simulation parameters
    - Visualizing simulation results
    - Analyzing simulation data
    - Generating reports

    Attributes
    ----------
    root (tk.Tk):
        The main window of the application
    current_db_path (str):
        Path to the current simulation database
    visualizer (SimulationVisualizer):
        Instance of visualization component

    Methods
    -------
    log_message(message: str) -> None
        Add a message to the log display.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_SIZE)

        # Initialize variables
        self.current_db_path = None
        self.visualizer = None

        self._setup_menu()
        self._setup_main_frame()
        self._setup_logging()

    def _setup_main_frame(self) -> None:
        """
        Setup the main container frame and layout of the application.

        Creates a two-pane layout:
        - Left pane: Main visualization/welcome area
        - Right pane: Log display and progress indicators

        Configures grid weights and scrollbars for proper resizing behavior.
        """
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Configure grid weights for main frame
        self.main_frame.grid_columnconfigure(0, weight=LEFT_PANE_WEIGHT)
        self.main_frame.grid_columnconfigure(1, weight=RIGHT_PANE_WEIGHT)
        self.main_frame.grid_rowconfigure(0, weight=1)  # Allow vertical expansion

        # Create left and right panes
        self.left_pane = ttk.Frame(self.main_frame)
        self.right_pane = ttk.Frame(self.main_frame)
        self.left_pane.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.right_pane.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # Configure right pane grid weights
        self.right_pane.grid_columnconfigure(0, weight=1)
        self.right_pane.grid_rowconfigure(0, weight=1)  # Log frame expands
        self.right_pane.grid_rowconfigure(1, weight=0)  # Progress frame stays small

        # Setup log frame in right pane
        self.log_frame = ttk.LabelFrame(self.right_pane, text="Simulation Log")
        self.log_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))

        # Configure log frame grid
        self.log_frame.grid_columnconfigure(0, weight=1)
        self.log_frame.grid_rowconfigure(0, weight=1)

        # Create log text widget with scrollbar
        self.log_text = tk.Text(self.log_frame, **LOG_TEXT_CONFIG)
        self.log_text.grid(row=0, column=0, sticky="nsew")

        self.log_scrollbar = ttk.Scrollbar(self.log_frame, command=self.log_text.yview)
        self.log_scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=self.log_scrollbar.set)

        # Setup progress frame
        self.progress_frame = ttk.LabelFrame(self.right_pane, text="Progress")
        self.progress_frame.grid(row=1, column=0, sticky="ew")

        # Configure progress frame grid
        self.progress_frame.grid_columnconfigure(0, weight=1)

        self.progress_label = ttk.Label(self.progress_frame, text="Ready")
        self.progress_label.grid(row=0, column=0, pady=5)

        self.progress_bar = ttk.Progressbar(
            self.progress_frame, mode="indeterminate", length=200
        )
        self.progress_bar.grid(row=1, column=0, pady=5, sticky="ew")

        # Configure left pane grid
        self.left_pane.grid_columnconfigure(0, weight=1)
        self.left_pane.grid_rowconfigure(0, weight=1)

        # Welcome message in left pane
        self.welcome_label = ttk.Label(
            self.left_pane,
            text="Welcome to Agent-Based Simulation\n\nUse the menu to start a new simulation or open an existing one.",
            justify=tk.CENTER,
        )
        self.welcome_label.grid(row=0, column=0, sticky="nsew")

    def log_message(self, message: str) -> None:
        """
        Add a message to the log display.

        Parameters
        ----------
        message : str
            Message to be displayed in log

        Updates the log text widget and auto-scrolls to the bottom.
        """
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)  # Auto-scroll to bottom
        self.log_text.update()

    def _new_simulation(self) -> None:
        """
        Start a new simulation run.

        - Creates a new database with timestamp
        - Loads configuration from YAML
        - Sets up logging handlers
        - Runs simulation in separate thread
        - Updates UI with progress
        """
        try:
            # Restore default layout
            self._restore_default_layout()

            # Clear log
            self.log_text.delete(1.0, tk.END)

            # Update progress
            self.progress_label.config(text="Running simulation...")
            self.progress_bar.start()

            # Create new database path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_db_path = f"{SIMULATIONS_DIR}/simulation_{timestamp}.db"
            os.makedirs(SIMULATIONS_DIR, exist_ok=True)

            # Load default configuration
            config = SimulationConfig.from_yaml(CONFIG_FILE)

            # Remove all existing handlers from root logger to prevent double logging
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)

            # Setup logging handler to capture logs
            class LogHandler(logging.Handler):
                def __init__(self, gui):
                    super().__init__()
                    self.gui = gui

                def emit(self, record):
                    msg = self.format(record)
                    self.gui.root.after(0, self.gui.log_message, msg)

            # Add handler to root logger
            log_handler = LogHandler(self)
            log_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logging.getLogger().addHandler(log_handler)

            # Run simulation in a separate thread
            def run_sim():
                try:
                    run_simulation(
                        num_steps=DEFAULT_SIMULATION_STEPS,
                        config=config,
                        db_path=self.current_db_path,
                    )

                    # Update UI in main thread
                    self.root.after(0, self._simulation_complete)
                except Exception as e:
                    self.root.after(0, self._simulation_error, str(e))

            import threading

            sim_thread = threading.Thread(target=run_sim)
            sim_thread.start()

        except Exception as e:
            self._simulation_error(str(e))

    def _simulation_complete(self) -> None:
        """Handle simulation completion."""
        self.progress_bar.stop()
        self.progress_label.config(text="Simulation completed")

        # Hide log and progress frames
        self.log_frame.grid_remove()
        self.progress_frame.grid_remove()

        # Start visualizer
        self._start_visualizer()

    def _simulation_error(self, error_msg: str) -> None:
        """Handle simulation error."""
        self.progress_bar.stop()
        self.progress_label.config(text="Simulation failed")
        self.log_message(f"ERROR: {error_msg}")

    def _start_visualizer(self) -> None:
        """
        Initialize and display the simulation visualizer.

        - Clears existing content
        - Reconfigures layout for visualization
        - Creates new SimulationVisualizer instance
        - Loads simulation data from database
        """
        # Clear left pane
        for widget in self.left_pane.winfo_children():
            widget.destroy()

        # Hide right pane completely
        self.right_pane.grid_remove()

        # Configure main frame to use full width
        self.main_frame.grid_columnconfigure(0, weight=1)  # Left pane gets full width
        self.main_frame.grid_columnconfigure(1, weight=0)  # Right pane hidden

        # Reconfigure left pane to use full width
        self.left_pane.grid(sticky="nsew", padx=5, pady=5, columnspan=2)

        # Create visualizer in left pane
        self.visualizer = SimulationVisualizer(
            self.left_pane, db_path=self.current_db_path
        )

    def _export_data(self) -> None:
        """
        Export simulation data to CSV format.

        Opens a file dialog for the user to choose save location.
        Uses SimulationAnalyzer to export the current simulation data.
        Shows success/error messages via messagebox.

        Raises
        ------
        Warning
            If no simulation data exists
        Error
            If export fails
        """
        if not self.current_db_path:
            messagebox.showwarning("Warning", "No simulation data to export.")
            return

        try:
            filepath = filedialog.asksaveasfilename(
                title="Export Data",
                defaultextension=".csv",
                filetypes=FILE_TYPES["csv"],
            )

            if filepath:
                analyzer = SimulationAnalyzer(self.current_db_path)
                analyzer.db.export_data(filepath)
                messagebox.showinfo("Success", "Data exported successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to export data: {str(e)}")
            logging.error(f"Failed to export data: {str(e)}", exc_info=True)

    def _run_batch(self) -> None:
        """
        Run multiple simulations with varying parameters.

        Creates a BatchRunner instance with the base configuration,
        adds parameter variations for system and individual agents,
        and executes the batch simulation.

        Default variations:
        - System agents: [20, 30, 40]
        - Individual agents: [20, 30, 40]
        """
        try:
            # Load base configuration
            config = SimulationConfig.from_yaml(CONFIG_FILE)

            # Create batch runner
            runner = BatchRunner(config)

            # Add parameter variations from config
            for param, values in DEFAULT_BATCH_VARIATIONS.items():
                runner.add_parameter_variation(param, values)

            # Run batch
            runner.run("batch_experiment", num_steps=DEFAULT_SIMULATION_STEPS)
            self.log_message("Batch simulation completed!")

        except Exception as e:
            self._simulation_error(f"Failed to run batch simulation: {str(e)}")

    def _configure_simulation(self) -> None:
        """
        Open configuration dialog for simulation parameters.

        Creates a scrollable window containing editable fields for all
        configuration parameters from config.yaml. Changes are saved
        back to the configuration file when applied.
        """
        config_window = tk.Toplevel(self.root)
        config_window.title("Simulation Configuration")
        config_window.geometry("400x600")

        # Load current configuration
        config = SimulationConfig.from_yaml("config.yaml")

        # Create scrollable frame
        canvas = tk.Canvas(config_window)
        scrollbar = ttk.Scrollbar(
            config_window, orient="vertical", command=canvas.yview
        )
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Add configuration fields
        self._create_config_fields(scrollable_frame, config)

        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

    def _create_config_fields(
        self, parent: tk.Widget, config: SimulationConfig
    ) -> None:
        """
        Create input fields for configuration parameters.

        Parameters
        ----------
        parent : tk.Widget
            Parent widget to contain the fields
        config : SimulationConfig
            Current configuration object

        Creates labeled entry fields for each configuration parameter,
        excluding visualization settings. Includes save button with
        validation.
        """
        row = 0
        entries = {}

        # Environment settings
        ttk.Label(parent, text="Environment Settings", font=("Arial", 10, "bold")).grid(
            row=row, column=0, columnspan=2, pady=10
        )
        row += 1

        for key, value in config.__dict__.items():
            if key != "visualization":  # Skip visualization config for now
                ttk.Label(parent, text=key).grid(row=row, column=0, padx=5, pady=2)
                entry = ttk.Entry(parent)
                entry.insert(0, str(value))
                entry.grid(row=row, column=1, padx=5, pady=2)
                entries[key] = entry
                row += 1

        # Add save button
        def save_config():
            try:
                # Update config with new values
                for key, entry in entries.items():
                    value = entry.get()
                    # Convert to appropriate type
                    if isinstance(getattr(config, key), int):
                        setattr(config, key, int(value))
                    elif isinstance(getattr(config, key), float):
                        setattr(config, key, float(value))
                    else:
                        setattr(config, key, value)

                # Save to file
                config.to_yaml("config.yaml")
                messagebox.showinfo("Success", "Configuration saved successfully!")
                parent.master.master.destroy()  # Close config window

            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")

        ttk.Button(parent, text="Save", command=save_config).grid(
            row=row, column=0, columnspan=2, pady=20
        )

    def _generate_report(self) -> None:
        """
        Generate HTML analysis report of simulation results.

        Opens file dialog for save location, then uses SimulationAnalyzer
        to generate a comprehensive report including:
        - Survival rates
        - Resource efficiency
        - Population dynamics
        - Other key metrics

        Raises:
            Warning if no simulation data exists
            Error if report generation fails
        """
        if not self.current_db_path:
            messagebox.showwarning("Warning", "No simulation data to analyze.")
            return

        try:
            filepath = filedialog.asksaveasfilename(
                title="Save Report",
                defaultextension=".html",
                filetypes=[("HTML files", "*.html"), ("All files", "*.*")],
            )

            if filepath:
                analyzer = SimulationAnalyzer(self.current_db_path)
                analyzer.generate_report(filepath)
                self.log_message("Report generated successfully!")

        except Exception as e:
            self._simulation_error(f"Failed to generate report: {str(e)}")

    def _view_statistics(self) -> None:
        """
        Display window showing key simulation statistics.

        Shows:
        - Survival rates by agent type
        - Resource efficiency metrics
        - Statistical summaries

        Creates a new window with scrollable text display.

        Raises
        ------
        Warning
            If no simulation data exists
        Error
            If statistics calculation fails
        """
        if not self.current_db_path:
            messagebox.showwarning("Warning", "No simulation data to analyze.")
            return

        # Create statistics window
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Simulation Statistics")
        stats_window.geometry("600x400")

        # Add statistics content
        try:
            analyzer = SimulationAnalyzer(self.current_db_path)
            survival_rates = analyzer.calculate_survival_rates()
            efficiency_data = analyzer.analyze_resource_efficiency()

            # Display statistics
            text = tk.Text(stats_window, wrap=tk.WORD)
            text.pack(fill=tk.BOTH, expand=True)

            text.insert(tk.END, "Survival Rates:\n\n")
            for agent_type, rate in survival_rates.items():
                text.insert(tk.END, f"{agent_type}: {rate:.2%}\n")

            text.insert(tk.END, "\nEfficiency Statistics:\n\n")
            text.insert(tk.END, efficiency_data.describe().to_string())

            text.config(state=tk.DISABLED)

        except Exception as e:
            self._simulation_error(f"Failed to load statistics: {str(e)}")

    def _show_documentation(self) -> None:
        """
        Display simulation documentation from agents.md file.

        Creates a new window with scrollable text display of the
        markdown documentation. Documentation includes:
        - Agent behaviors
        - System mechanics
        - Configuration options
        - Usage instructions
        """
        try:
            with open(DOCS_FILE, "r") as f:
                content = f.read()

            doc_window = tk.Toplevel(self.root)
            doc_window.title("Documentation")
            doc_window.geometry("800x600")

            text = tk.Text(doc_window, wrap=tk.WORD)
            text.pack(fill=tk.BOTH, expand=True)
            text.insert(tk.END, content)
            text.config(state=tk.DISABLED)

        except Exception as e:
            self._simulation_error(f"Failed to load documentation: {str(e)}")

    def _show_about(self) -> None:
        """
        Display about dialog with application information.

        Shows:
        - Application name
        - Version number
        - Brief description
        - Basic usage information
        """
        messagebox.showinfo("About", ABOUT_TEXT)

    def _on_exit(self) -> None:
        """
        Handle application exit request.

        - Prompts for confirmation
        - Closes visualizer if active
        - Terminates application
        """
        if messagebox.askokcancel("Exit", "Do you want to exit the application?"):
            if self.visualizer:
                self.visualizer.close()
            self.root.quit()

    def _setup_menu(self) -> None:
        """
        Create the application menu bar with tooltips.

        Creates menus for:
        - File operations (new/open/export)
        - Simulation controls (batch/configure)
        - Analysis tools (reports/statistics)
        - Help documentation

        Each menu item includes hover tooltips.
        """
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)

        # File Menu
        file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=file_menu)

        file_menu_items = [
            ("New Simulation", self._new_simulation, "Start a new simulation"),
            ("Open Simulation", self._open_simulation, "Open an existing simulation"),
            (None, None, None),  # Separator
            ("Export Data", self._export_data, "Export simulation data to CSV"),
            (None, None, None),  # Separator
            ("Exit", self._on_exit, "Exit the application"),
        ]

        for label, command, tooltip in file_menu_items:
            if label is None:
                file_menu.add_separator()
            else:
                file_menu.add_command(label=label, command=command)
                if tooltip:
                    self._add_menu_tooltip(file_menu, label, tooltip)

        # Simulation Menu
        sim_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Simulation", menu=sim_menu)

        sim_menu_items = [
            (
                "Run Batch",
                self._run_batch,
                "Run multiple simulations with different parameters",
            ),
            (
                "Configure",
                self._configure_simulation,
                "Configure simulation parameters",
            ),
        ]

        for label, command, tooltip in sim_menu_items:
            sim_menu.add_command(label=label, command=command)
            self._add_menu_tooltip(sim_menu, label, tooltip)

        # Analysis Menu
        analysis_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Analysis", menu=analysis_menu)

        analysis_menu_items = [
            (
                "Generate Report",
                self._generate_report,
                "Generate detailed analysis report",
            ),
            ("View Statistics", self._view_statistics, "View simulation statistics"),
        ]

        for label, command, tooltip in analysis_menu_items:
            analysis_menu.add_command(label=label, command=command)
            self._add_menu_tooltip(analysis_menu, label, tooltip)

        # Help Menu
        help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=help_menu)

        help_menu_items = [
            (
                "Documentation",
                self._show_documentation,
                "View simulation documentation",
            ),
            ("About", self._show_about, "About this application"),
        ]

        for label, command, tooltip in help_menu_items:
            help_menu.add_command(label=label, command=command)
            self._add_menu_tooltip(help_menu, label, tooltip)

    def _add_menu_tooltip(self, menu: tk.Menu, label: str, tooltip_text: str) -> None:
        """
        Add hover tooltip to a menu item.

        Parameters
        ----------
        menu : tk.Menu
            Menu widget containing the item
        label : str
            Label of the menu item
        tooltip_text : str
            Text to display in tooltip

        Creates a temporary label that appears when hovering
        over the menu item and auto-destroys after 2 seconds.
        """

        def show_tooltip(event):
            tooltip = tk.Label(
                self.root,
                text=tooltip_text,
                background="#FFFFEA",
                foreground="black",
                relief="solid",
                borderwidth=1,
                font=("Arial", 9),
            )
            tooltip.place(x=event.x_root, y=event.y_root + 20)
            self.root.after(TOOLTIP_DURATION, tooltip.destroy)

        menu.bind("<Enter>", show_tooltip)

    def _open_simulation(self) -> None:
        """
        Open an existing simulation database file.

        Opens file dialog in simulations directory,
        loads selected database, and initializes visualizer
        with the loaded data.

        Updates:
            - current_db_path
            - log display
            - visualization

        Raises
        ------
        Error
            If file loading fails
        """
        try:
            filepath = filedialog.askopenfilename(
                title="Open Simulation",
                initialdir="simulations",
                filetypes=[("Database files", "*.db"), ("All files", "*.*")],
            )

            if filepath:
                # Clear log
                self.log_text.delete(1.0, tk.END)
                self.log_message(f"Opening simulation: {filepath}")

                # Store current database path
                self.current_db_path = filepath

                # Start visualizer
                self._start_visualizer()

                self.log_message("Simulation loaded successfully")

        except Exception as e:
            self._simulation_error(f"Failed to open simulation: {str(e)}")

    def _setup_logging(self) -> None:
        """
        Configure application logging system.

        Sets up:
        - File logging to dated log files
        - GUI logging to text widget
        - Formatting for log messages
        - Log level filtering
        """
        if not os.path.exists("logs"):
            os.makedirs("logs")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create file handler
        file_handler = logging.FileHandler(f"logs/simulation_gui_{timestamp}.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        # Create console handler that writes to our log window
        class GUIHandler(logging.Handler):
            def __init__(self, gui):
                super().__init__()
                self.gui = gui

            def emit(self, record):
                msg = self.format(record)
                self.gui.root.after(0, self.gui.log_message, msg)

        gui_handler = GUIHandler(self)
        gui_handler.setLevel(logging.INFO)
        gui_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        # Get root logger and remove any existing handlers
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add our handlers
        root_logger.addHandler(file_handler)
        root_logger.addHandler(gui_handler)

        logging.info("Logging system initialized")

    def _restore_default_layout(self) -> None:
        """
        Reset the GUI layout to its default state.

        Restores:
        - Two-pane layout
        - Log and progress frames
        - Original grid weights
        """
        # Show right pane
        self.right_pane.grid()

        # Reset left pane to original position
        self.left_pane.grid(
            row=0, column=0, sticky="nsew", padx=5, pady=5, columnspan=1
        )

        # Restore original grid weights
        self.main_frame.grid_columnconfigure(0, weight=LEFT_PANE_WEIGHT)
        self.main_frame.grid_columnconfigure(1, weight=RIGHT_PANE_WEIGHT)

        # Show log and progress frames
        self.log_frame.grid()
        self.progress_frame.grid()

    def _setup_controls(self) -> None:
        """
        Setup simulation playback control panel.

        Creates:
        - Play/pause button
        - Step controls (forward/backward)
        - Speed adjustment slider
        - Export button

        All controls include tooltips and consistent styling.
        """
        # Configure styles for controls
        style = ttk.Style()
        style.configure("Control.TButton", padding=5)

        # Configure Scale style
        style.layout(
            "Horizontal.TScale",
            [
                (
                    "Horizontal.Scale.trough",
                    {
                        "sticky": "nswe",
                        "children": [
                            ("Horizontal.Scale.slider", {"side": "left", "sticky": ""})
                        ],
                    },
                )
            ],
        )
        style.configure("Horizontal.TScale", background="white")

        # Control buttons frame
        buttons_frame = ttk.Frame(self.controls_frame)
        buttons_frame.pack(side="left", fill="x", expand=True)

        # Playback controls
        self.play_button = ttk.Button(
            buttons_frame,
            text="▶ Play/Pause",
            command=self._toggle_playback,
            style="Control.TButton",
        )
        self.play_button.pack(side="left", padx=5)
        self._add_tooltip(self.play_button, "Start or pause the simulation playback")

        # Step controls with consistent styling and tooltips
        step_controls = [
            ("⏪", lambda: self._step_to(self.current_step - 10), "Go back 10 steps"),
            ("◀", lambda: self._step_to(self.current_step - 1), "Previous step"),
            ("▶", lambda: self._step_to(self.current_step + 1), "Next step"),
            (
                "⏩",
                lambda: self._step_to(self.current_step + 10),
                "Skip forward 10 steps",
            ),
        ]

        for text, command, tooltip in step_controls:
            btn = ttk.Button(
                buttons_frame, text=text, command=command, style="Control.TButton"
            )
            btn.pack(side="left", padx=2)
            self._add_tooltip(btn, tooltip)

        # Speed control frame
        speed_frame = ttk.Frame(self.controls_frame)
        speed_frame.pack(side="left", fill="x", expand=True, padx=10)

        speed_label = ttk.Label(speed_frame, text="Playback Speed:")
        speed_label.pack(side="left", padx=5)
        self._add_tooltip(speed_label, "Adjust the simulation playback speed")

        # Create scale with tooltip
        self.speed_scale = ttk.Scale(speed_frame, from_=1, to=50, orient="horizontal")
        self.speed_scale.set(10)
        self.speed_scale.pack(side="left", padx=5, fill="x", expand=True)
        self._add_tooltip(self.speed_scale, "1 = Slowest, 50 = Fastest")

        # Export button with tooltip
        export_btn = ttk.Button(
            self.controls_frame,
            text="Export Data",
            command=self._export_data,
            style="Control.TButton",
        )
        export_btn.pack(side="right", padx=5)
        self._add_tooltip(export_btn, "Export simulation data to CSV file")

    def _add_tooltip(self, widget: tk.Widget, text: str) -> None:
        """
        Add hover tooltip to any widget.

        Parameters
        ----------
        widget : tk.Widget
            Widget to add tooltip to
        text : str
            Text to display in tooltip

        Creates a label that appears below the widget on hover
        and disappears when mouse leaves the widget area.
        Handles proper positioning and timing.
        """
        tooltip = tk.Label(
            widget,
            text=text,
            background="#FFFFEA",
            foreground="black",
            relief="solid",
            borderwidth=1,
            font=("Arial", 9),
        )
        tooltip.bind(
            "<Enter>", lambda e: tooltip.place_forget()
        )  # Hide if mouse enters tooltip

        def enter(event):
            # Position tooltip below the widget
            x = widget.winfo_rootx()
            y = widget.winfo_rooty() + widget.winfo_height() + 2
            tooltip.place(x=x, y=y)

        def leave(event):
            tooltip.place_forget()

        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)


def main():
    """
    Main entry point for the simulation GUI application.

    Creates the root window and initializes the SimulationGUI
    instance. Starts the Tkinter main event loop.
    """
    root = tk.Tk()
    app = SimulationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()