import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import ImageTk


class SimulationVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Agent-Based Simulation Visualizer")

        # Create main containers
        self.stats_frame = ttk.LabelFrame(
            root, text="Simulation Statistics", padding=10
        )
        self.stats_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.chart_frame = ttk.LabelFrame(
            root, text="Population & Resources", padding=10
        )
        self.chart_frame.grid(row=0, column=1, rowspan=2, padx=5, pady=5, sticky="nsew")

        self.env_frame = ttk.LabelFrame(root, text="Environment View", padding=10)
        self.env_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.controls_frame = ttk.Frame(root, padding=10)
        self.controls_frame.grid(
            row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew"
        )

        # Configure grid weights
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)

        # Add update frequency control
        self.update_frequency = 5  # Only update visualization every N steps
        self.current_step = 0

        self._setup_stats_panel()
        self._setup_chart()
        self._setup_environment_view()
        self._setup_controls()

    def _setup_stats_panel(self):
        # Create a canvas with scrollbar for stats
        canvas = tk.Canvas(self.stats_frame)
        scrollbar = ttk.Scrollbar(
            self.stats_frame, orient="vertical", command=canvas.yview
        )
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Basic Stats
        self.stats_vars = {
            "cycle": tk.StringVar(value="Cycle: 0"),
            "system_agents": tk.StringVar(value="System Agents: 0"),
            "individual_agents": tk.StringVar(value="Individual Agents: 0"),
            "total_resources": tk.StringVar(value="Total Resources: 0"),
            # Population Dynamics
            "births": tk.StringVar(value="Births: 0"),
            "deaths": tk.StringVar(value="Deaths: 0"),
            "population_stability": tk.StringVar(value="Population Stability: 0%"),
            "avg_lifespan": tk.StringVar(value="Avg Lifespan: 0"),
            # Resource Metrics
            "avg_resources": tk.StringVar(value="Avg Resources/Agent: 0.0"),
            "resource_efficiency": tk.StringVar(value="Resource Efficiency: 0%"),
            "resource_density": tk.StringVar(value="Resource Density: 0.0"),
            # Territory Control
            "system_territory": tk.StringVar(value="System Territory: 0%"),
            "individual_territory": tk.StringVar(value="Individual Territory: 0%"),
        }

        # Create labels with section headers
        ttk.Label(
            scrollable_frame, text="Basic Statistics", font=("Arial", 10, "bold")
        ).pack(anchor="w", pady=(5, 0))
        for var in ["cycle", "system_agents", "individual_agents", "total_resources"]:
            ttk.Label(scrollable_frame, textvariable=self.stats_vars[var]).pack(
                anchor="w"
            )

        ttk.Label(
            scrollable_frame, text="Population Dynamics", font=("Arial", 10, "bold")
        ).pack(anchor="w", pady=(10, 0))
        for var in ["births", "deaths", "population_stability", "avg_lifespan"]:
            ttk.Label(scrollable_frame, textvariable=self.stats_vars[var]).pack(
                anchor="w"
            )

        ttk.Label(
            scrollable_frame, text="Resource Metrics", font=("Arial", 10, "bold")
        ).pack(anchor="w", pady=(10, 0))
        for var in ["avg_resources", "resource_efficiency", "resource_density"]:
            ttk.Label(scrollable_frame, textvariable=self.stats_vars[var]).pack(
                anchor="w"
            )

        ttk.Label(
            scrollable_frame, text="Territory Control", font=("Arial", 10, "bold")
        ).pack(anchor="w", pady=(10, 0))
        for var in ["system_territory", "individual_territory"]:
            ttk.Label(scrollable_frame, textvariable=self.stats_vars[var]).pack(
                anchor="w"
            )

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _setup_chart(self):
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 4))
        self.ax1 = self.fig.add_subplot(111)
        self.ax2 = self.ax1.twinx()

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _setup_environment_view(self):
        self.env_canvas = tk.Canvas(self.env_frame, width=400, height=400)
        self.env_canvas.pack(fill=tk.BOTH, expand=True)

    def _setup_controls(self):
        ttk.Button(
            self.controls_frame, text="Pause/Resume", command=self._toggle_simulation
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            self.controls_frame, text="Step Forward", command=self._step_forward
        ).pack(side=tk.LEFT, padx=5)

        # Speed control with wider range
        ttk.Label(self.controls_frame, text="Simulation Speed:").pack(
            side=tk.LEFT, padx=5
        )
        self.speed_scale = ttk.Scale(
            self.controls_frame, from_=1, to=50, orient=tk.HORIZONTAL
        )
        self.speed_scale.set(10)  # Default to faster speed
        self.speed_scale.pack(side=tk.LEFT, padx=5)

        # Update frequency control
        ttk.Label(self.controls_frame, text="Update Frequency:").pack(
            side=tk.LEFT, padx=5
        )
        self.freq_scale = ttk.Scale(
            self.controls_frame, from_=1, to=20, orient=tk.HORIZONTAL
        )
        self.freq_scale.set(5)  # Update every 5 steps by default
        self.freq_scale.pack(side=tk.LEFT, padx=5)

        # Save button
        ttk.Button(
            self.controls_frame, text="Save State", command=self._save_state
        ).pack(side=tk.RIGHT, padx=5)

    def update(self, simulation_data):
        """Update the visualization with new simulation data"""
        self.current_step += 1

        # Only update visualization every N steps
        if self.current_step % int(self.freq_scale.get()) != 0:
            return

        # Update all statistics
        self.stats_vars["cycle"].set(f"Cycle: {simulation_data['cycle']}")
        self.stats_vars["system_agents"].set(
            f"System Agents: {simulation_data['system_agents']}"
        )
        self.stats_vars["individual_agents"].set(
            f"Individual Agents: {simulation_data['individual_agents']}"
        )
        self.stats_vars["total_resources"].set(
            f"Total Resources: {simulation_data['total_resources']}"
        )
        self.stats_vars["births"].set(f"Births this cycle: {simulation_data['births']}")
        self.stats_vars["deaths"].set(f"Deaths this cycle: {simulation_data['deaths']}")
        self.stats_vars["population_stability"].set(
            f"Population Stability: {simulation_data['population_stability']:.1%}"
        )
        self.stats_vars["avg_lifespan"].set(
            f"Avg Lifespan: {simulation_data['average_lifespan']:.1f} cycles"
        )
        self.stats_vars["avg_resources"].set(
            f"Avg Resources/Agent: {simulation_data['avg_resources']:.2f}"
        )
        self.stats_vars["resource_efficiency"].set(
            f"Resource Efficiency: {simulation_data['resource_efficiency']:.1%}"
        )
        self.stats_vars["resource_density"].set(
            f"Resource Density: {simulation_data['resource_density']:.2f}"
        )
        self.stats_vars["system_territory"].set(
            f"System Territory: {simulation_data['system_agent_territory']:.1%}"
        )
        self.stats_vars["individual_territory"].set(
            f"Individual Territory: {simulation_data['individual_agent_territory']:.1%}"
        )

        # Update chart and environment view
        self._update_chart(simulation_data)
        self._update_environment_view(simulation_data["environment_image"])

    def _update_chart(self, data):
        self.ax1.clear()
        self.ax2.clear()

        history_df = data["history"]

        if len(history_df) > 0:  # Only plot if we have data
            # Plot agent counts on left axis
            self.ax1.plot(
                history_df.index,
                history_df["system_agent_count"],
                "b-",
                label="System Agents",
            )
            self.ax1.plot(
                history_df.index,
                history_df["individual_agent_count"],
                "r-",
                label="Individual Agents",
            )
            self.ax1.set_xlabel("Cycle")
            self.ax1.set_ylabel("Agent Count", color="b")

            # Plot resources on right axis
            self.ax2.plot(
                history_df.index, history_df["total_resources"], "g-", label="Resources"
            )
            self.ax2.set_ylabel("Resource Count", color="g")

            # Add legends
            lines1, labels1 = self.ax1.get_legend_handles_labels()
            lines2, labels2 = self.ax2.get_legend_handles_labels()
            self.ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        self.canvas.draw()

    def _update_environment_view(self, image):
        # Convert PIL image to PhotoImage and update canvas
        photo = ImageTk.PhotoImage(image)
        self.env_canvas.create_image(0, 0, image=photo, anchor="nw")
        self.env_canvas.image = photo  # Keep a reference

    def _toggle_simulation(self):
        # Implement pause/resume functionality
        pass

    def _step_forward(self):
        # Implement single step functionality
        pass

    def _save_state(self):
        # Implement save functionality
        pass
