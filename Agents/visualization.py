import tkinter as tk
import tkinter.messagebox as messagebox
from tkinter import ttk
from typing import Dict, List, Tuple

import numpy as np
import yaml
from database import SimulationDatabase
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont, ImageTk


class SimulationVisualizer:
    # Default visualization settings - can be overridden through config
    DEFAULT_SETTINGS = {
        # Canvas settings
        "canvas_size": (400, 400),
        "padding": 20,
        "background_color": "black",
        # Animation settings
        "max_animation_frames": 5,
        "animation_min_delay": 50,
        # Resource visualization
        "max_resource_amount": 30,
        "resource_colors": {"glow_red": 150, "glow_green": 255, "glow_blue": 50},
        "resource_size": 2,
        # Agent visualization
        "agent_radius_scale": 2,
        "birth_radius_scale": 4,
        "death_mark_scale": 1.5,
        "agent_colors": {"SystemAgent": "blue", "IndividualAgent": "red"},
        # Font settings
        "min_font_size": 10,
        "font_scale_factor": 40,
        "font_family": "arial",
        # Marker colors
        "death_mark_color": (255, 0, 0),
        "birth_mark_color": (255, 255, 255),
        # Chart colors
        "metric_colors": {
            "total_agents": "#4a90e2",  # Blue
            "system_agents": "#50c878",  # Emerald green
            "individual_agents": "#e74c3c",  # Red
            "total_resources": "#f39c12",  # Orange
            "average_agent_resources": "#9b59b6",  # Purple
        },
    }

    def __init__(self, root, db_path="simulation_results.db", settings=None):
        """Initialize visualizer with optional custom settings."""
        # Load settings
        self.settings = self.DEFAULT_SETTINGS.copy()
        if settings:
            self.settings.update(settings)

        # Core attributes
        self.root = root
        self.root.title("Agent-Based Simulation Visualizer")
        self.db = SimulationDatabase(db_path)
        self.current_step = 0
        self.playing = False
        self.was_playing = False

        # Animation and state tracking
        self.birth_animations = {}
        self.death_animations = {}
        self.max_animation_frames = self.settings["max_animation_frames"]
        self.previous_agent_ids = set()
        self.previous_agent_states = []
        self.is_dragging = False
        self.canvas_size = self.settings["canvas_size"]

        # Initialize UI components
        self._setup_ui()

        # Get total simulation length
        self.total_steps = self._get_total_steps()

        # Initialize visualization
        self.root.update_idletasks()
        self._step_to(0)
        self.root.update()

        # Track scheduled tasks
        self.play_job = None
        self.animation_job = None

    def _setup_ui(self):
        """Setup all UI components with current settings."""
        # Create main containers
        self.stats_frame = ttk.LabelFrame(
            self.root, text="Simulation Statistics", padding=10
        )
        self.stats_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.chart_frame = ttk.LabelFrame(
            self.root, text="Population & Resources", padding=10
        )
        self.chart_frame.grid(row=0, column=1, rowspan=2, padx=5, pady=5, sticky="nsew")

        self.env_frame = ttk.LabelFrame(self.root, text="Environment View", padding=10)
        self.env_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.controls_frame = ttk.Frame(self.root, padding=10)
        self.controls_frame.grid(
            row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew"
        )

        # Configure grid weights
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Initialize components
        self._setup_stats_panel()
        self._setup_chart()
        self._setup_environment_view()
        self._setup_controls()

    def _draw_agents(self, draw, agent_states, params):
        """Draw agents using configured colors and sizes."""
        for agent in agent_states:
            x, y = self._transform_coords(agent[2], agent[3], params)
            agent_type = agent[1]
            color = self.settings["agent_colors"].get(
                agent_type, "white"
            )  # Default to white if type unknown
            radius = max(1, int(self.settings["agent_radius_scale"] * params["scale"]))

            draw.ellipse(
                [(x - radius, y - radius), (x + radius, y + radius)], fill=color
            )

    def _draw_resources(self, draw, resource_states, params):
        """Draw resources using configured colors and sizes."""
        for resource in resource_states:
            amount = resource[1]
            if amount > 0:
                x, y = self._transform_coords(resource[2], resource[3], params)

                # Calculate color intensity using configured colors
                intensity = amount / self.settings["max_resource_amount"]
                resource_color = (
                    int(self.settings["resource_colors"]["glow_red"] * intensity),
                    int(self.settings["resource_colors"]["glow_green"] * intensity),
                    int(self.settings["resource_colors"]["glow_blue"] * intensity),
                )

                size = max(1, int(self.settings["resource_size"] * params["scale"]))
                radius = int(size * 0.2)

                self._draw_rounded_rectangle(draw, x, y, size, radius, resource_color)

    @classmethod
    def from_config_file(cls, root, config_path, db_path="simulation_results.db"):
        """Create visualizer instance from configuration file."""
        try:
            with open(config_path, "r") as f:
                settings = yaml.safe_load(f).get("visualization", {})
            return cls(root, db_path=db_path, settings=settings)
        except Exception as e:
            messagebox.showerror(
                "Configuration Error", f"Failed to load configuration: {str(e)}"
            )
            return cls(root, db_path=db_path)  # Fall back to defaults

    def save_settings(self, config_path):
        """Save current visualization settings to file."""
        try:
            config = {"visualization": self.settings}
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            messagebox.showinfo("Success", "Settings saved successfully!")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save settings: {str(e)}")

    def _setup_stats_panel(self):
        """Setup statistics panel with card-like metrics in two columns."""
        # Create container for two columns using ttk
        columns_frame = ttk.Frame(self.stats_frame)
        columns_frame.pack(fill="both", expand=True, padx=2)

        # Create left and right column frames
        left_column = ttk.Frame(columns_frame)
        right_column = ttk.Frame(columns_frame)
        left_column.pack(side="left", fill="both", expand=True, padx=2)
        right_column.pack(side="left", fill="both", expand=True, padx=2)

        # Configure styles for cards
        style = ttk.Style()
        style.configure("Card.TFrame", background="white", relief="solid")
        style.configure("CardLabel.TLabel", background="white", font=("Arial", 10))
        style.configure(
            "CardValue.TLabel", background="white", font=("Arial", 14, "bold")
        )

        # Stats variables with labels and colors
        self.stats_vars = {
            "total_agents": {
                "var": tk.StringVar(value="0"),
                "label": "Total Agents",
                "color": self.settings["metric_colors"]["total_agents"],
                "column": left_column,
            },
            "system_agents": {
                "var": tk.StringVar(value="0"),
                "label": "System Agents",
                "color": self.settings["metric_colors"]["system_agents"],
                "column": right_column,
            },
            "individual_agents": {
                "var": tk.StringVar(value="0"),
                "label": "Individual Agents",
                "color": self.settings["metric_colors"]["individual_agents"],
                "column": left_column,
            },
            "total_resources": {
                "var": tk.StringVar(value="0"),
                "label": "Total Resources",
                "color": self.settings["metric_colors"]["total_resources"],
                "column": right_column,
            },
            "average_agent_resources": {
                "var": tk.StringVar(value="0.0"),
                "label": "Avg Resources/Agent",
                "color": self.settings["metric_colors"]["average_agent_resources"],
                "column": left_column,
            },
        }

        # Create card-like frames for each stat
        for stat_id, stat_info in self.stats_vars.items():
            # Create card frame with custom style
            card = ttk.Frame(stat_info["column"], style="Card.TFrame")
            card.pack(fill="x", padx=3, pady=3)

            # Configure specific style for this stat
            style.configure(
                f"{stat_id}.CardLabel.TLabel",
                foreground=stat_info["color"],
                background="white",
                font=("Arial", 10),
            )
            style.configure(
                f"{stat_id}.CardValue.TLabel",
                foreground="black",
                background="white",
                font=("Arial", 14, "bold"),
            )

            # Inner padding frame
            padding_frame = ttk.Frame(card, style="Card.TFrame")
            padding_frame.pack(fill="x", padx=1, pady=1)

            # Label
            ttk.Label(
                padding_frame,
                text=stat_info["label"],
                style=f"{stat_id}.CardLabel.TLabel",
            ).pack(anchor="w", padx=8, pady=(5, 0))

            # Value
            ttk.Label(
                padding_frame,
                textvariable=stat_info["var"],
                style=f"{stat_id}.CardValue.TLabel",
            ).pack(anchor="e", padx=8, pady=(0, 5))

    def _setup_chart(self):
        """Setup the chart with click interaction."""
        self.fig = Figure(figsize=(8, 4))
        self.fig.subplots_adjust(right=0.85)
        self.ax1 = self.fig.add_subplot(111)
        self.ax2 = self.ax1.twinx()

        # Initialize empty line objects
        self.lines = {
            "system_agents": self.ax1.plot(
                [],
                [],
                "-",
                color=self.settings["metric_colors"]["system_agents"],
                label="System Agents",
            )[0],
            "individual_agents": self.ax1.plot(
                [],
                [],
                "-",
                color=self.settings["metric_colors"]["individual_agents"],
                label="Individual Agents",
            )[0],
            "resources": self.ax2.plot(
                [],
                [],
                "-",
                color=self.settings["metric_colors"]["total_resources"],
                label="Resources",
            )[0],
            "system_agents_future": self.ax1.plot(
                [],
                [],
                "-",
                color=self.settings["metric_colors"]["system_agents"],
                alpha=0.3,
            )[0],
            "individual_agents_future": self.ax1.plot(
                [],
                [],
                "-",
                color=self.settings["metric_colors"]["individual_agents"],
                alpha=0.3,
            )[0],
            "resources_future": self.ax2.plot(
                [],
                [],
                "-",
                color=self.settings["metric_colors"]["total_resources"],
                alpha=0.3,
            )[0],
            "current_step": self.ax1.axvline(
                x=0, color="gray", linestyle="--", alpha=0.5
            ),
        }

        # Setup axis labels and colors
        self.ax1.set_xlabel("Step")
        self.ax1.set_ylabel(
            "Agent Count", color=self.settings["metric_colors"]["system_agents"]
        )
        self.ax2.set_ylabel(
            "Resource Count",
            color=self.settings["metric_colors"]["total_resources"],
            rotation=270,
            labelpad=20,
        )

        # Configure axis positions and colors
        self.ax2.yaxis.set_label_position("right")
        self.ax2.yaxis.set_ticks_position("right")
        self.ax1.tick_params(
            axis="y", labelcolor=self.settings["metric_colors"]["system_agents"]
        )
        self.ax2.tick_params(
            axis="y", labelcolor=self.settings["metric_colors"]["total_resources"]
        )

        # Setup legends
        self.ax1.legend(loc="upper left")
        self.ax2.legend(loc="upper right")

        # Create canvas without toolbar
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.draw()

        # Add mouse event handlers
        self.canvas.mpl_connect("button_press_event", self._on_chart_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_chart_motion)
        self.canvas.mpl_connect("button_release_event", self._on_chart_release)

        # Pack canvas
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _on_chart_press(self, event):
        """Handle mouse press on the chart."""
        if event.inaxes in (self.ax1, self.ax2):
            self.is_dragging = True
            # Store the current playback state
            self.was_playing = self.playing
            # Temporarily pause while dragging
            self.playing = False
            # Update to the clicked position
            self._update_step_from_x(event.xdata)

    def _on_chart_motion(self, event):
        """Handle mouse motion while dragging."""
        if self.is_dragging and event.inaxes in (self.ax1, self.ax2):
            self._update_step_from_x(event.xdata)

    def _on_chart_release(self, event):
        """Handle mouse release to end dragging."""
        self.is_dragging = False
        # Restore playback state if it was playing before
        if hasattr(self, "was_playing") and self.was_playing:
            self.playing = True
            self._play_simulation()

    def _update_step_from_x(self, x_coord):
        """Update the simulation step based on x coordinate."""
        if x_coord is not None:
            # Get the step number from x coordinate
            step = int(round(x_coord))
            # Ensure step is within bounds
            step = max(0, min(step, self.total_steps))
            # Jump to the selected step
            self._step_to(step)

    def _setup_environment_view(self):
        """Setup the environment view with ttk styling."""
        style = ttk.Style()
        style.configure(
            "Environment.TFrame", background=self.settings["background_color"]
        )

        # Create frame for canvas
        canvas_frame = ttk.Frame(self.env_frame, style="Environment.TFrame")
        canvas_frame.pack(fill="both", expand=True)

        # Create canvas with black background
        self.env_canvas = tk.Canvas(
            canvas_frame, bg=self.settings["background_color"], highlightthickness=0
        )
        self.env_canvas.pack(fill="both", expand=True)

        # Bind resize event
        self.env_canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas_size = self.settings["canvas_size"]

    def _on_canvas_resize(self, event):
        """Handle canvas resize events."""
        try:
            self.canvas_size = (event.width, event.height)
            # Trigger redraw if we have current data
            data = self.db.get_simulation_data(self.current_step)
            if data["agent_states"] or data["resource_states"]:
                self._update_visualization(data)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to handle resize: {str(e)}")

    def _calculate_transform_params(self, resource_states, width, height):
        """Calculate scaling and offset parameters for coordinate transformation."""
        env_width = max(x for _, _, x, _ in resource_states + [(0, 0, 100, 0)])
        env_height = max(y for _, _, _, y in resource_states + [(0, 0, 0, 100)])

        scale_x = (width - 2 * self.settings["padding"]) / env_width
        scale_y = (height - 2 * self.settings["padding"]) / env_height
        scale = min(scale_x, scale_y)

        offset_x = max(self.settings["padding"], (width - (env_width * scale)) / 2)
        offset_y = max(self.settings["padding"], (height - (env_height * scale)) / 2)

        return {
            "scale": scale,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "padding": self.settings["padding"],
            "width": width,
            "height": height,
        }

    def _transform_coords(self, x, y, params):
        """Transform environment coordinates to screen coordinates."""
        return (
            params["offset_x"] + x * params["scale"],
            params["offset_y"] + y * params["scale"],
        )

    def _draw_rounded_rectangle(self, draw, x, y, size, radius, color):
        """Helper method to draw a rounded rectangle."""
        x1, y1 = x - size, y - size
        x2, y2 = x + size, y + size

        # Main rectangle
        draw.rectangle([x1, y1, x2, y2], fill=color)

        # Corner circles
        corners = [
            (x1, y1),  # Top-left
            (x2 - radius * 2, y1),  # Top-right
            (x1, y2 - radius * 2),  # Bottom-left
            (x2 - radius * 2, y2 - radius * 2),  # Bottom-right
        ]

        for corner_x, corner_y in corners:
            draw.ellipse(
                [corner_x, corner_y, corner_x + radius * 2, corner_y + radius * 2],
                fill=color,
            )

    def _setup_controls(self):
        """Setup playback controls with validation and consistent ttk styling."""
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

        # Playback controls with validation
        self.play_button = ttk.Button(
            buttons_frame,
            text="Play/Pause",
            command=self._validate_and_toggle_playback,
            style="Control.TButton",
        )
        self.play_button.pack(side="left", padx=5)

        # Step controls with validation
        step_controls = [
            ("<<", lambda: self._validate_and_step_to(self.current_step - 10)),
            ("<", lambda: self._validate_and_step_to(self.current_step - 1)),
            (">", lambda: self._validate_and_step_to(self.current_step + 1)),
            (">>", lambda: self._validate_and_step_to(self.current_step + 10)),
        ]

        for text, command in step_controls:
            ttk.Button(
                buttons_frame, text=text, command=command, style="Control.TButton"
            ).pack(side="left", padx=2)

        # Speed control frame
        speed_frame = ttk.Frame(self.controls_frame)
        speed_frame.pack(side="left", fill="x", expand=True, padx=10)

        ttk.Label(
            speed_frame,
            text="Playback Speed:",
        ).pack(side="left", padx=5)

        # Create scale with validation
        self.speed_scale = ttk.Scale(speed_frame, from_=1, to=50, orient="horizontal")
        self.speed_scale.set(10)  # Default speed
        self.speed_scale.bind("<ButtonRelease-1>", self._validate_speed_scale)
        self.speed_scale.pack(side="left", padx=5, fill="x", expand=True)

        # Export button with validation
        ttk.Button(
            self.controls_frame,
            text="Export Data",
            command=self._validate_and_export_data,
            style="Control.TButton",
        ).pack(side="right", padx=5)

    def _validate_speed_scale(self, event):
        """Validate and adjust the playback speed if needed."""
        try:
            speed = self.speed_scale.get()
            if speed < 1:
                self.speed_scale.set(1)
                messagebox.showwarning(
                    "Invalid Speed", "Playback speed cannot be less than 1"
                )
            elif speed > 50:
                self.speed_scale.set(50)
                messagebox.showwarning(
                    "Invalid Speed", "Playback speed cannot exceed 50"
                )
        except Exception as e:
            messagebox.showerror(
                "Validation Error", f"Invalid playback speed: {str(e)}"
            )
            self.speed_scale.set(10)  # Reset to default

    def _validate_and_step_to(self, step):
        """Validate step number before moving to it."""
        try:
            if step < 0:
                step = 0
            elif step > self.total_steps:
                step = self.total_steps
            self._step_to(step)
        except Exception as e:
            messagebox.showerror(
                "Step Error", f"Failed to move to step {step}: {str(e)}"
            )

    def _validate_and_toggle_playback(self):
        """Validate simulation state before toggling playback."""
        try:
            if self.current_step >= self.total_steps and self.playing:
                messagebox.showinfo(
                    "End of Simulation", "Reached the end of simulation"
                )
                self.playing = False
                # Clean up any pending jobs
                if self.play_job is not None:
                    self.root.after_cancel(self.play_job)
                    self.play_job = None
                return

            self._toggle_playback()
        except Exception as e:
            messagebox.showerror(
                "Playback Error", f"Failed to toggle playback: {str(e)}"
            )
            self.playing = False
            # Clean up on error
            if self.play_job is not None:
                self.root.after_cancel(self.play_job)
                self.play_job = None

    def _validate_and_export_data(self):
        """Validate before exporting data."""
        try:
            if not hasattr(self, "db") or self.db is None:
                messagebox.showerror(
                    "Export Error", "No simulation data available to export"
                )
                return
            self._export_data()
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")

    def _get_total_steps(self) -> int:
        """Get the total number of steps in the simulation."""
        try:
            self.db.cursor.execute("SELECT MAX(step_number) FROM SimulationSteps")
            result = self.db.cursor.fetchone()
            return result[0] if result and result[0] is not None else 0
        except Exception as e:
            messagebox.showerror(
                "Database Error", f"Failed to get total steps: {str(e)}"
            )
            return 0

    def _step_to(self, step_number):
        """Move to a specific step in the simulation."""
        try:
            data = self.db.get_simulation_data(step_number)
            if data["agent_states"] or data["resource_states"]:
                self.current_step = step_number
                self._update_visualization(data)

                # Show message when agents die out, but continue playback
                if not data["agent_states"] and self.playing:
                    messagebox.showinfo(
                        "Population Extinct",
                        f"All agents have died at step {step_number}",
                    )
        except Exception as e:
            messagebox.showerror(
                "Database Error", f"Failed to load step {step_number}: {str(e)}"
            )
            self.playing = False  # Stop playback on error

    def _update_visualization(self, data):
        """Update all visualization components with new data."""
        # Update statistics
        metrics = data["metrics"]
        for metric_name, value in metrics.items():
            if metric_name in self.stats_vars:
                formatted_value = (
                    f"{value:.1f}" if isinstance(value, float) else str(value)
                )
                self.stats_vars[metric_name]["var"].set(formatted_value)

        # Update environment view
        self._draw_environment(data["agent_states"], data["resource_states"])

        # Update charts
        self._update_charts()

    def _draw_environment(self, agent_states, resource_states):
        """Draw the current state of the environment with auto-scaling."""
        # Get canvas dimensions and create image
        width, height = self.canvas_size
        img = Image.new("RGB", (width, height), self.settings["background_color"])
        draw = ImageDraw.Draw(img)

        # Calculate transformation parameters
        transform_params = self._calculate_transform_params(
            resource_states, width, height
        )

        # Track current agents for birth/death detection
        self._update_animation_states(agent_states)

        # Draw environment elements
        self._draw_resources(draw, resource_states, transform_params)
        self._draw_agents(draw, agent_states, transform_params)
        self._draw_birth_animations(draw, transform_params)
        self._draw_death_animations(draw, transform_params)
        self._draw_step_number(draw, transform_params)

        # Update canvas
        photo = ImageTk.PhotoImage(img)
        self.env_canvas.create_image(0, 0, image=photo, anchor="nw")
        self.env_canvas.image = photo  # Keep reference to prevent garbage collection

    def _update_animation_states(self, agent_states):
        """Update birth and death animation states."""
        current_agent_ids = {agent[0] for agent in agent_states}

        # Check for new births
        new_births = current_agent_ids - self.previous_agent_ids
        for agent_id in new_births:
            agent_data = next(a for a in agent_states if a[0] == agent_id)
            pos = (agent_data[2], agent_data[3])
            self.birth_animations[agent_id] = (pos, 0)

        # Check for deaths
        deaths = self.previous_agent_ids - current_agent_ids
        for agent_id in deaths:
            if agent_data := next(
                (a for a in self.previous_agent_states if a[0] == agent_id), None
            ):
                pos = (agent_data[2], agent_data[3])
                self.death_animations[agent_id] = (pos, 0)

        self.previous_agent_ids = current_agent_ids
        self.previous_agent_states = agent_states

    def _draw_birth_animations(self, draw, params):
        """Draw expanding circle animations for new agents."""
        births_to_remove = []
        for agent_id, (pos, frame) in self.birth_animations.items():
            if frame < self.max_animation_frames:
                x, y = self._transform_coords(pos[0], pos[1], params)
                radius = (
                    max(2, int(self.settings["birth_radius_scale"] * params["scale"]))
                    * (frame + 1)
                    / self.max_animation_frames
                )
                opacity = int(255 * (1 - frame / self.max_animation_frames))
                draw.ellipse(
                    [(x - radius, y - radius), (x + radius, y + radius)],
                    outline=(*self.settings["birth_mark_color"], opacity),
                )
                self.birth_animations[agent_id] = (pos, frame + 1)
            else:
                births_to_remove.append(agent_id)

        for agent_id in births_to_remove:
            del self.birth_animations[agent_id]

    def _draw_death_animations(self, draw, params):
        """Draw fading X mark animations for dying agents."""
        deaths_to_remove = []
        for agent_id, (pos, frame) in self.death_animations.items():
            if frame < self.max_animation_frames:
                x, y = self._transform_coords(pos[0], pos[1], params)
                size = max(1, int(self.settings["death_mark_scale"] * params["scale"]))
                opacity = int(128 * (1 - frame / self.max_animation_frames))
                color = (*self.settings["death_mark_color"], opacity)

                draw.line(
                    [(x - size, y - size), (x + size, y + size)], fill=color, width=1
                )
                draw.line(
                    [(x - size, y + size), (x + size, y - size)], fill=color, width=1
                )

                self.death_animations[agent_id] = (pos, frame + 1)
            else:
                deaths_to_remove.append(agent_id)

        for agent_id in deaths_to_remove:
            del self.death_animations[agent_id]

    def _draw_step_number(self, draw, params):
        """Draw the current step number on the visualization."""
        font_size = max(
            self.settings["min_font_size"],
            int(
                min(params["width"], params["height"])
                / self.settings["font_scale_factor"]
            ),
        )
        try:
            font = ImageFont.truetype(self.settings["font_family"], font_size)
        except:
            font = ImageFont.load_default()

        draw.text(
            (self.settings["padding"], self.settings["padding"]),
            f"Step: {self.current_step}",
            fill=self.settings["birth_mark_color"],  # Using white color
            font=font,
        )
