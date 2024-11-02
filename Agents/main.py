import logging
import os
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

from analysis import SimulationAnalyzer
from batch_runner import BatchRunner
from config import SimulationConfig
from visualization import SimulationVisualizer

from agents import main as run_simulation


class SimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Agent-Based Simulation")
        self.root.geometry("1200x800")

        # Initialize variables
        self.current_db_path = None
        self.visualizer = None

        self._setup_menu()
        self._setup_main_frame()
        self._setup_logging()

    def _setup_main_frame(self):
        """Setup the main container frame."""
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Configure grid weights for main frame
        self.main_frame.grid_columnconfigure(0, weight=3)  # Left pane gets more space
        self.main_frame.grid_columnconfigure(1, weight=1)  # Right pane
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
        self.log_text = tk.Text(
            self.log_frame,
            wrap=tk.WORD,
            bg="black",
            fg="#00FF00",
            font=("Courier", 10),
            height=20,
            width=50,
        )
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

    def log_message(self, message):
        """Add a message to the log."""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)  # Auto-scroll to bottom
        self.log_text.update()

    def _new_simulation(self):
        """Start a new simulation."""
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
            self.current_db_path = f"simulations/simulation_{timestamp}.db"
            os.makedirs("simulations", exist_ok=True)

            # Load default configuration
            config = SimulationConfig.from_yaml("Agents/config.yaml")

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
                        num_steps=500,  # Default steps
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

    def _simulation_complete(self):
        """Handle simulation completion."""
        self.progress_bar.stop()
        self.progress_label.config(text="Simulation completed")

        # Hide log and progress frames
        self.log_frame.grid_remove()
        self.progress_frame.grid_remove()

        # Start visualizer
        self._start_visualizer()

    def _simulation_error(self, error_msg):
        """Handle simulation error."""
        self.progress_bar.stop()
        self.progress_label.config(text="Simulation failed")
        self.log_message(f"ERROR: {error_msg}")

    def _start_visualizer(self):
        """Start the simulation visualizer."""
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

    def _export_data(self):
        """Export simulation data to CSV."""
        if not self.current_db_path:
            messagebox.showwarning("Warning", "No simulation data to export.")
            return

        try:
            filepath = filedialog.asksaveasfilename(
                title="Export Data",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )

            if filepath:
                analyzer = SimulationAnalyzer(self.current_db_path)
                analyzer.db.export_data(filepath)
                messagebox.showinfo("Success", "Data exported successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to export data: {str(e)}")
            logging.error(f"Failed to export data: {str(e)}", exc_info=True)

    def _run_batch(self):
        """Run batch simulations."""
        try:
            # Load base configuration
            config = SimulationConfig.from_yaml("Agents/config.yaml")

            # Create batch runner
            runner = BatchRunner(config)

            # Add some default parameter variations
            runner.add_parameter_variation("system_agents", [20, 30, 40])
            runner.add_parameter_variation("individual_agents", [20, 30, 40])

            # Run batch
            runner.run("batch_experiment", num_steps=500)
            self.log_message("Batch simulation completed!")

        except Exception as e:
            self._simulation_error(f"Failed to run batch simulation: {str(e)}")

    def _configure_simulation(self):
        """Open configuration dialog."""
        config_window = tk.Toplevel(self.root)
        config_window.title("Simulation Configuration")
        config_window.geometry("400x600")

        # Load current configuration
        config = SimulationConfig.from_yaml("Agents/config.yaml")

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

    def _create_config_fields(self, parent, config):
        """Create input fields for configuration parameters."""
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
                config.to_yaml("Agents/config.yaml")
                messagebox.showinfo("Success", "Configuration saved successfully!")
                parent.master.master.destroy()  # Close config window

            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")

        ttk.Button(parent, text="Save", command=save_config).grid(
            row=row, column=0, columnspan=2, pady=20
        )

    def _generate_report(self):
        """Generate analysis report."""
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

    def _view_statistics(self):
        """Show simulation statistics window."""
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

    def _show_documentation(self):
        """Show documentation window."""
        try:
            with open("Agents/agents.md", "r") as f:
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

    def _show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About",
            "Agent-Based Simulation\n\n"
            "A simulation environment for studying emergent behaviors "
            "in populations of system and individual agents.\n\n"
            "Version 1.0",
        )

    def _on_exit(self):
        """Handle application exit."""
        if messagebox.askokcancel("Exit", "Do you want to exit the application?"):
            if self.visualizer:
                self.visualizer.close()
            self.root.quit()

    def _setup_menu(self):
        """Create the menu bar."""
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)

        # File Menu
        file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Simulation", command=self._new_simulation)
        file_menu.add_command(label="Open Simulation", command=self._open_simulation)
        file_menu.add_separator()
        file_menu.add_command(label="Export Data", command=self._export_data)
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
        analysis_menu.add_command(
            label="Generate Report", command=self._generate_report
        )
        analysis_menu.add_command(
            label="View Statistics", command=self._view_statistics
        )

        # Help Menu
        help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self._show_documentation)
        help_menu.add_command(label="About", command=self._show_about)

    def _open_simulation(self):
        """Open an existing simulation database."""
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

    def _setup_logging(self):
        """Setup logging configuration."""
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

    def _restore_default_layout(self):
        """Restore the default layout with log and progress frames."""
        # Show right pane
        self.right_pane.grid()

        # Reset left pane to original position
        self.left_pane.grid(
            row=0, column=0, sticky="nsew", padx=5, pady=5, columnspan=1
        )

        # Restore original grid weights
        self.main_frame.grid_columnconfigure(0, weight=3)
        self.main_frame.grid_columnconfigure(1, weight=1)

        # Show log and progress frames
        self.log_frame.grid()
        self.progress_frame.grid()


def main():
    root = tk.Tk()
    app = SimulationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
