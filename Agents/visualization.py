import tkinter as tk
import tkinter.messagebox as messagebox
from tkinter import ttk
from typing import Dict, List, Tuple

import numpy as np
from database import SimulationDatabase
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont, ImageTk


class SimulationVisualizer:
    def __init__(self, root, db_path="simulation_results.db"):
        # Core attributes
        self.root = root
        self.root.title("Agent-Based Simulation Visualizer")
        self.db = SimulationDatabase(db_path)
        self.current_step = 0
        self.playing = False
        self.was_playing = False
        
        # Animation and state tracking
        self.birth_animations = {}  # {agent_id: (position, frame)}
        self.death_animations = {}  # {agent_id: (position, frame)}
        self.max_animation_frames = 5
        self.previous_agent_ids = set()
        self.previous_agent_states = []
        self.is_dragging = False
        self.canvas_size = (400, 400)  # Default size
        
        # Create main containers
        self.stats_frame = ttk.LabelFrame(root, text="Simulation Statistics", padding=10)
        self.stats_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.chart_frame = ttk.LabelFrame(root, text="Population & Resources", padding=10)
        self.chart_frame.grid(row=0, column=1, rowspan=2, padx=5, pady=5, sticky="nsew")

        self.env_frame = ttk.LabelFrame(root, text="Environment View", padding=10)
        self.env_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.controls_frame = ttk.Frame(root, padding=10)
        self.controls_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # Configure grid weights
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)

        # Initialize UI components
        self._setup_stats_panel()
        self._setup_chart()
        self._setup_environment_view()
        self._setup_controls()

        # Get total simulation length
        self.total_steps = self._get_total_steps()

        # Initialize visualization
        self.root.update_idletasks()
        self._step_to(0)
        self.root.update()

    def _get_total_steps(self) -> int:
        """Get the total number of steps in the simulation."""
        try:
            self.db.cursor.execute("SELECT MAX(step_number) FROM SimulationSteps")
            result = self.db.cursor.fetchone()
            return result[0] if result and result[0] is not None else 0
        except Exception as e:
            messagebox.showerror("Database Error", f"Failed to get total steps: {str(e)}")
            return 0

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
        style.configure('Card.TFrame', background='white', relief='solid')
        style.configure('CardLabel.TLabel', background='white', font=('Arial', 10))
        style.configure('CardValue.TLabel', background='white', font=('Arial', 14, 'bold'))

        # Stats variables with labels and colors
        self.stats_vars = {
            "total_agents": {
                "var": tk.StringVar(value="0"),
                "label": "Total Agents",
                "color": "#4a90e2",  # Blue
                "column": left_column,
            },
            "system_agents": {
                "var": tk.StringVar(value="0"),
                "label": "System Agents",
                "color": "#50c878",  # Emerald green
                "column": right_column,
            },
            "individual_agents": {
                "var": tk.StringVar(value="0"),
                "label": "Individual Agents",
                "color": "#e74c3c",  # Red
                "column": left_column,
            },
            "total_resources": {
                "var": tk.StringVar(value="0"),
                "label": "Total Resources",
                "color": "#f39c12",  # Orange
                "column": right_column,
            },
            "average_agent_resources": {
                "var": tk.StringVar(value="0.0"),
                "label": "Avg Resources/Agent",
                "color": "#9b59b6",  # Purple
                "column": left_column,
            },
        }

        # Create card-like frames for each stat
        for stat_id, stat_info in self.stats_vars.items():
            # Create card frame with custom style
            card = ttk.Frame(stat_info["column"], style='Card.TFrame')
            card.pack(fill="x", padx=3, pady=3)

            # Configure specific style for this stat
            style.configure(
                f'{stat_id}.CardLabel.TLabel',
                foreground=stat_info["color"],
                background='white',
                font=('Arial', 10)
            )
            style.configure(
                f'{stat_id}.CardValue.TLabel',
                foreground='black',
                background='white',
                font=('Arial', 14, 'bold')
            )

            # Inner padding frame
            padding_frame = ttk.Frame(card, style='Card.TFrame')
            padding_frame.pack(fill="x", padx=1, pady=1)

            # Label
            ttk.Label(
                padding_frame,
                text=stat_info["label"],
                style=f'{stat_id}.CardLabel.TLabel'
            ).pack(anchor="w", padx=8, pady=(5, 0))

            # Value
            ttk.Label(
                padding_frame,
                textvariable=stat_info["var"],
                style=f'{stat_id}.CardValue.TLabel'
            ).pack(anchor="e", padx=8, pady=(0, 5))

    def _setup_chart(self):
        """Setup the chart with click interaction."""
        self.fig = Figure(figsize=(8, 4))
        self.fig.subplots_adjust(right=0.85)
        self.ax1 = self.fig.add_subplot(111)
        self.ax2 = self.ax1.twinx()

        # Create canvas without toolbar
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.draw()

        # Add mouse event handlers
        self.canvas.mpl_connect("button_press_event", self._on_chart_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_chart_motion)
        self.canvas.mpl_connect("button_release_event", self._on_chart_release)

        # Initialize dragging state
        self.is_dragging = False

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
        style.configure('Environment.TFrame', background='black')
        
        # Create frame for canvas
        canvas_frame = ttk.Frame(self.env_frame, style='Environment.TFrame')
        canvas_frame.pack(fill="both", expand=True)
        
        # Create canvas with black background
        self.env_canvas = tk.Canvas(
            canvas_frame,
            bg='black',
            highlightthickness=0
        )
        self.env_canvas.pack(fill="both", expand=True)

        # Bind resize event
        self.env_canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas_size = (400, 400)  # Default size

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

    def _setup_controls(self):
        """Setup playback controls with consistent ttk styling."""
        # Configure styles for controls
        style = ttk.Style()
        style.configure('Control.TButton', padding=5)
        
        # Configure Scale style - need to use the correct style name
        style.layout('Horizontal.TScale',
                    [('Horizontal.Scale.trough',
                      {'sticky': 'nswe',
                       'children': [('Horizontal.Scale.slider',
                                   {'side': 'left', 'sticky': ''})]})])
        style.configure('Horizontal.TScale', background='white')
        
        # Control buttons frame
        buttons_frame = ttk.Frame(self.controls_frame)
        buttons_frame.pack(side="left", fill="x", expand=True)

        # Playback controls
        self.play_button = ttk.Button(
            buttons_frame,
            text="Play/Pause",
            command=self._toggle_playback,
            style='Control.TButton'
        )
        self.play_button.pack(side="left", padx=5)

        # Step controls with consistent styling
        for text, command in [
            ("<<", lambda: self._step_to(self.current_step - 10)),
            ("<", lambda: self._step_to(self.current_step - 1)),
            (">", lambda: self._step_to(self.current_step + 1)),
            (">>", lambda: self._step_to(self.current_step + 10))
        ]:
            ttk.Button(
                buttons_frame,
                text=text,
                command=command,
                style='Control.TButton'
            ).pack(side="left", padx=2)

        # Speed control frame
        speed_frame = ttk.Frame(self.controls_frame)
        speed_frame.pack(side="left", fill="x", expand=True, padx=10)

        ttk.Label(
            speed_frame,
            text="Playback Speed:",
        ).pack(side="left", padx=5)

        # Create scale without custom style
        self.speed_scale = ttk.Scale(
            speed_frame,
            from_=1,
            to=50,
            orient="horizontal"
        )
        self.speed_scale.set(10)
        self.speed_scale.pack(side="left", padx=5, fill="x", expand=True)

        # Export button
        ttk.Button(
            self.controls_frame,
            text="Export Data",
            command=self._export_data,
            style='Control.TButton'
        ).pack(side="right", padx=5)

    def _toggle_playback(self):
        self.playing = not self.playing
        if self.playing:
            self._play_simulation()

    def _play_simulation(self):
        """Handle simulation playback."""
        if self.playing:
            try:
                data = self.db.get_simulation_data(self.current_step + 1)
                if not data["agent_states"] and not data["resource_states"]:
                    self.playing = False
                    return

                self._step_to(self.current_step + 1)
                # Reduce delay if there are active animations
                delay = int(1000 / self.speed_scale.get())
                if self.birth_animations or self.death_animations:
                    delay = min(delay, 50)  # Ensure smooth animations
                self.root.after(delay, self._play_simulation)
            except Exception as e:
                messagebox.showerror("Playback Error", f"Failed to advance simulation: {str(e)}")
                self.playing = False

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
                        "Population Extinct", f"All agents have died at step {step_number}"
                    )
        except Exception as e:
            messagebox.showerror("Database Error", f"Failed to load step {step_number}: {str(e)}")
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
        img = Image.new("RGB", (width, height), "black")
        draw = ImageDraw.Draw(img)

        # Calculate transformation parameters
        transform_params = self._calculate_transform_params(resource_states, width, height)
        
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

    def _calculate_transform_params(self, resource_states, width, height):
        """Calculate scaling and offset parameters for coordinate transformation."""
        padding = 20
        env_width = max(x for _, _, x, _ in resource_states + [(0, 0, 100, 0)])
        env_height = max(y for _, _, _, y in resource_states + [(0, 0, 0, 100)])

        scale_x = (width - 2 * padding) / env_width
        scale_y = (height - 2 * padding) / env_height
        scale = min(scale_x, scale_y)

        offset_x = max(padding, (width - (env_width * scale)) / 2)
        offset_y = max(padding, (height - (env_height * scale)) / 2)

        return {
            'scale': scale,
            'offset_x': offset_x,
            'offset_y': offset_y,
            'padding': padding,
            'width': width,
            'height': height
        }

    def _transform_coords(self, x, y, params):
        """Transform environment coordinates to screen coordinates."""
        return (
            params['offset_x'] + x * params['scale'],
            params['offset_y'] + y * params['scale']
        )

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
            if agent_data := next((a for a in self.previous_agent_states if a[0] == agent_id), None):
                pos = (agent_data[2], agent_data[3])
                self.death_animations[agent_id] = (pos, 0)

        self.previous_agent_ids = current_agent_ids
        self.previous_agent_states = agent_states

    def _draw_resources(self, draw, resource_states, params):
        """Draw resources as rounded squares with intensity-based coloring."""
        for resource in resource_states:
            amount = resource[1]
            if amount > 0:
                x, y = self._transform_coords(resource[2], resource[3], params)
                
                # Calculate color intensity and size
                intensity = amount / 30  # Assuming max amount is 30
                resource_color = (
                    int(150 * intensity),  # Red component for glow
                    int(255 * intensity),  # Green component
                    int(50 * intensity)    # Blue component
                )
                
                size = max(1, int(2 * params['scale']))
                radius = int(size * 0.2)

                # Draw rounded rectangle
                self._draw_rounded_rectangle(draw, x, y, size, radius, resource_color)

    def _draw_rounded_rectangle(self, draw, x, y, size, radius, color):
        """Helper method to draw a rounded rectangle."""
        x1, y1 = x - size, y - size
        x2, y2 = x + size, y + size

        # Main rectangle
        draw.rectangle([x1, y1, x2, y2], fill=color)
        
        # Corner circles
        corners = [
            (x1, y1),           # Top-left
            (x2 - radius*2, y1), # Top-right
            (x1, y2 - radius*2), # Bottom-left
            (x2 - radius*2, y2 - radius*2)  # Bottom-right
        ]
        
        for corner_x, corner_y in corners:
            draw.ellipse(
                [corner_x, corner_y, corner_x + radius*2, corner_y + radius*2],
                fill=color
            )

    def _draw_agents(self, draw, agent_states, params):
        """Draw agents as colored circles."""
        for agent in agent_states:
            x, y = self._transform_coords(agent[2], agent[3], params)
            color = "blue" if agent[1] == "SystemAgent" else "red"
            radius = max(1, int(2 * params['scale']))
            draw.ellipse(
                [(x - radius, y - radius), (x + radius, y + radius)],
                fill=color
            )

    def _draw_birth_animations(self, draw, params):
        """Draw expanding circle animations for new agents."""
        births_to_remove = []
        for agent_id, (pos, frame) in self.birth_animations.items():
            if frame < self.max_animation_frames:
                x, y = self._transform_coords(pos[0], pos[1], params)
                radius = max(2, int(4 * params['scale'])) * (frame + 1) / self.max_animation_frames
                opacity = int(255 * (1 - frame / self.max_animation_frames))
                draw.ellipse(
                    [(x - radius, y - radius), (x + radius, y + radius)],
                    outline=(255, 255, 255, opacity)
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
                size = max(1, int(1.5 * params['scale']))
                opacity = int(128 * (1 - frame / self.max_animation_frames))
                color = (255, 0, 0, opacity)  # Red X

                # Draw X
                draw.line([(x - size, y - size), (x + size, y + size)], fill=color, width=1)
                draw.line([(x - size, y + size), (x + size, y - size)], fill=color, width=1)

                self.death_animations[agent_id] = (pos, frame + 1)
            else:
                deaths_to_remove.append(agent_id)
        
        for agent_id in deaths_to_remove:
            del self.death_animations[agent_id]

    def _draw_step_number(self, draw, params):
        """Draw the current step number on the visualization."""
        font_size = max(10, int(min(params['width'], params['height']) / 40))
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        draw.text(
            (params['padding'], params['padding']),
            f"Step: {self.current_step}",
            fill="white",
            font=font
        )

    def _update_charts(self):
        """Update the population and resource charts with historical data."""
        try:
            # Clear previous plots
            self.ax1.clear()
            self.ax2.clear()

            # Fetch all historical data
            history = self.db.get_historical_data()

            if not history["steps"]:
                return  # No data to plot

            # Plot agent counts with faded future data
            steps = history["steps"]
            system_agents = history["metrics"]["system_agents"]
            individual_agents = history["metrics"]["individual_agents"]
            total_resources = history["metrics"]["total_resources"]

            # Split data into past and future
            past_mask = [step <= self.current_step for step in steps]
            future_mask = [not x for x in past_mask]

            # Plot past data (solid lines)
            if any(past_mask):
                past_steps = [s for s, m in zip(steps, past_mask) if m]
                past_system = [v for v, m in zip(system_agents, past_mask) if m]
                past_individual = [v for v, m in zip(individual_agents, past_mask) if m]
                past_resources = [v for v, m in zip(total_resources, past_mask) if m]

                self.ax1.plot(past_steps, past_system, "b-", label="System Agents")
                self.ax1.plot(
                    past_steps, past_individual, "r-", label="Individual Agents"
                )
                self.ax2.plot(past_steps, past_resources, "g-", label="Resources")

            # Plot future data (faded lines)
            if any(future_mask):
                future_steps = [s for s, m in zip(steps, future_mask) if m]
                future_system = [v for v, m in zip(system_agents, future_mask) if m]
                future_individual = [
                    v for v, m in zip(individual_agents, future_mask) if m
                ]
                future_resources = [
                    v for v, m in zip(total_resources, future_mask) if m
                ]

                self.ax1.plot(future_steps, future_system, "b-", alpha=0.3)
                self.ax1.plot(future_steps, future_individual, "r-", alpha=0.3)
                self.ax2.plot(future_steps, future_resources, "g-", alpha=0.3)

            # Add vertical line for current step
            self.ax1.axvline(
                x=self.current_step, color="gray", linestyle="--", alpha=0.5
            )

            # Set axis limits to show full simulation
            self.ax1.set_xlim(0, max(steps))

            # Labels and legend
            self.ax1.set_xlabel("Step")
            self.ax1.set_ylabel("Agent Count", color="b")
            # Position the resource count label on the right with proper spacing
            self.ax2.yaxis.set_label_position("right")
            self.ax2.set_ylabel("Resource Count", color="g", rotation=270, labelpad=20)

            # Adjust tick positions
            self.ax2.yaxis.set_ticks_position("right")

            # Adjust tick colors to match the lines
            self.ax1.tick_params(axis="y", labelcolor="b")
            self.ax2.tick_params(axis="y", labelcolor="g")

            # Add legends - separate for each axis
            lines1, labels1 = self.ax1.get_legend_handles_labels()
            self.ax1.legend(lines1, labels1, loc="upper left")

            lines2, labels2 = self.ax2.get_legend_handles_labels()
            self.ax2.legend(lines2, labels2, loc="upper right")

        except Exception as e:
            messagebox.showerror("Chart Error", f"Failed to update charts: {str(e)}")

        # Update the canvas
        self.canvas.draw()

    def run(self):
        """Start the visualization."""
        # Make sure we're showing the first frame
        if self.current_step == 0:
            self._step_to(0)
        self.root.mainloop()

    def close(self):
        """Clean up resources."""
        try:
            self.db.close()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to close database connection: {str(e)}")

    def _export_data(self):
        """Export simulation data to CSV."""
        try:
            from tkinter import filedialog

            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )
            if filename:
                try:
                    self.db.export_data(filename)
                    messagebox.showinfo("Success", "Data exported successfully!")
                except Exception as e:
                    messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file dialog: {str(e)}")

    def _initialize_visualization(self):
        """Initialize the visualization with the first frame of data."""
        initial_data = self.db.get_simulation_data(0)
        if initial_data["agent_states"] or initial_data["resource_states"]:
            self._update_visualization(initial_data)
            # Force an update of the window
            self.root.update_idletasks()
