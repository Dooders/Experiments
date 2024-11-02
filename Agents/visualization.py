import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk, ImageDraw, ImageFont
from database import SimulationDatabase
import numpy as np
from typing import Dict, List, Tuple

class SimulationVisualizer:
    def __init__(self, root, db_path='simulation_results.db'):
        self.root = root
        self.root.title("Agent-Based Simulation Visualizer")
        self.db = SimulationDatabase(db_path)
        self.current_step = 0
        self.playing = False

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

        self.birth_animations = {}  # {agent_id: (position, frame)}
        self.death_animations = {}  # {agent_id: (position, frame)}
        self.max_animation_frames = 5

        self._setup_stats_panel()
        self._setup_chart()
        self._setup_environment_view()
        self._setup_controls()

        # Get total simulation length
        self.total_steps = self._get_total_steps()
        
        # Initialize with first frame
        self.root.update_idletasks()
        self._step_to(0)
        self.root.update()

    def _get_total_steps(self) -> int:
        """Get the total number of steps in the simulation."""
        self.db.cursor.execute('SELECT MAX(step_number) FROM SimulationSteps')
        result = self.db.cursor.fetchone()
        return result[0] if result and result[0] is not None else 0

    def _setup_stats_panel(self):
        # Create a canvas with scrollbar for stats
        canvas = tk.Canvas(self.stats_frame)
        scrollbar = ttk.Scrollbar(self.stats_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", 
                            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Stats variables
        self.stats_vars = {
            'total_agents': tk.StringVar(value="Total Agents: 0"),
            'system_agents': tk.StringVar(value="System Agents: 0"),
            'individual_agents': tk.StringVar(value="Individual Agents: 0"),
            'total_resources': tk.StringVar(value="Total Resources: 0"),
            'average_agent_resources': tk.StringVar(value="Avg Resources/Agent: 0.0"),
        }

        # Create labels for each stat
        for var in self.stats_vars.values():
            ttk.Label(scrollable_frame, textvariable=var).pack(anchor="w")

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _setup_chart(self):
        self.fig = Figure(figsize=(8, 4))
        # Add spacing on the right for the label
        self.fig.subplots_adjust(right=0.85)
        self.ax1 = self.fig.add_subplot(111)
        self.ax2 = self.ax1.twinx()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _setup_environment_view(self):
        """Setup the environment view with auto-scaling canvas."""
        self.env_canvas = tk.Canvas(self.env_frame)
        self.env_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind resize event
        self.env_canvas.bind('<Configure>', self._on_canvas_resize)
        self.canvas_size = (400, 400)  # Default size

    def _on_canvas_resize(self, event):
        """Handle canvas resize events."""
        self.canvas_size = (event.width, event.height)
        # Trigger redraw if we have current data
        data = self.db.get_simulation_data(self.current_step)
        if data['agent_states'] or data['resource_states']:
            self._update_visualization(data)

    def _setup_controls(self):
        # Playback controls
        self.play_button = ttk.Button(self.controls_frame, text="Play/Pause", 
                                    command=self._toggle_playback)
        self.play_button.pack(side=tk.LEFT, padx=5)

        # Step controls
        ttk.Button(self.controls_frame, text="<<", 
                  command=lambda: self._step_to(self.current_step - 10)).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.controls_frame, text="<", 
                  command=lambda: self._step_to(self.current_step - 1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.controls_frame, text=">", 
                  command=lambda: self._step_to(self.current_step + 1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.controls_frame, text=">>", 
                  command=lambda: self._step_to(self.current_step + 10)).pack(side=tk.LEFT, padx=2)

        # Speed control
        ttk.Label(self.controls_frame, text="Playback Speed:").pack(side=tk.LEFT, padx=5)
        self.speed_scale = ttk.Scale(self.controls_frame, from_=1, to=50, orient=tk.HORIZONTAL)
        self.speed_scale.set(10)
        self.speed_scale.pack(side=tk.LEFT, padx=5)

        # Add export button
        ttk.Button(self.controls_frame, text="Export Data", 
                   command=self._export_data).pack(side=tk.RIGHT, padx=5)

    def _toggle_playback(self):
        self.playing = not self.playing
        if self.playing:
            self._play_simulation()

    def _play_simulation(self):
        if self.playing:
            self._step_to(self.current_step + 1)
            # Reduce delay if there are active animations
            delay = int(1000 / self.speed_scale.get())
            if self.birth_animations or self.death_animations:
                delay = min(delay, 50)  # Ensure smooth animations
            self.root.after(delay, self._play_simulation)

    def _step_to(self, step_number):
        """Move to a specific step in the simulation."""
        data = self.db.get_simulation_data(step_number)
        if data['agent_states'] or data['resource_states']:  # If we have data for this step
            self.current_step = step_number
            self._update_visualization(data)

    def _update_visualization(self, data):
        """Update all visualization components with new data."""
        # Update statistics
        metrics = data['metrics']
        for metric_name, value in metrics.items():
            if metric_name in self.stats_vars:
                self.stats_vars[metric_name].set(f"{metric_name}: {value:.2f}")

        # Update environment view
        self._draw_environment(data['agent_states'], data['resource_states'])
        
        # Update charts
        self._update_charts()

    def _draw_environment(self, agent_states, resource_states):
        """Draw the current state of the environment with auto-scaling."""
        # Get canvas dimensions
        width, height = self.canvas_size

        # Create a new image at canvas size
        img = Image.new('RGB', (width, height), 'black')
        draw = ImageDraw.Draw(img)

        # Calculate scaling factors
        padding = 20
        env_width = max(x for _, _, x, _ in resource_states + [(0, 0, 100, 0)])
        env_height = max(y for _, _, _, y in resource_states + [(0, 0, 0, 100)])
        
        scale_x = (width - 2*padding) / env_width
        scale_y = (height - 2*padding) / env_height
        scale = min(scale_x, scale_y)

        # Calculate offset to center the environment
        offset_x = (width - (env_width * scale)) / 2
        offset_y = (height - (env_height * scale)) / 2

        # Ensure minimum padding
        offset_x = max(padding, offset_x)
        offset_y = max(padding, offset_y)

        def transform(x, y):
            """Transform environment coordinates to screen coordinates."""
            return (offset_x + x * scale, offset_y + y * scale)

        # Draw resources as rounded squares
        for resource in resource_states:
            amount = resource[1]  # amount
            if amount > 0:  # Only draw if resource has any amount left
                x, y = transform(resource[2], resource[3])  # position_x, position_y
                
                # Calculate color based on amount (black to radioactive green)
                # Using a more vibrant green for the "radioactive" effect
                max_amount = 30  # Assuming max amount is 30
                intensity = amount / max_amount
                
                # Create a gradient from black to radioactive green
                # RGB values for a more "radioactive" green effect
                green = int(255 * intensity)  # Base green
                red = int(150 * intensity)    # Add some red for glow effect
                blue = int(50 * intensity)    # Small amount of blue
                
                resource_color = (red, green, blue)
                
                # Calculate square size to match agent size
                size = max(1, int(2 * scale))  # Same as agent radius
                radius = int(size * 0.2)  # 20% of size for corner radius
                
                # Define square corners
                x1, y1 = x - size, y - size
                x2, y2 = x + size, y + size
                
                # Draw rounded rectangle
                # First draw the main rectangle
                draw.rectangle([x1, y1, x2, y2], fill=resource_color)
                
                # Then draw four circles at corners for rounding
                draw.ellipse([x1, y1, x1 + radius * 2, y1 + radius * 2], fill=resource_color)
                draw.ellipse([x2 - radius * 2, y1, x2, y1 + radius * 2], fill=resource_color)
                draw.ellipse([x1, y2 - radius * 2, x1 + radius * 2, y2], fill=resource_color)
                draw.ellipse([x2 - radius * 2, y2 - radius * 2, x2, y2], fill=resource_color)

        # Track current agents for birth/death detection
        current_agent_ids = {agent[0] for agent in agent_states}  # agent[0] is agent_id
        
        # Check for new births (agents that weren't in the previous frame)
        if hasattr(self, 'previous_agent_ids'):
            new_births = current_agent_ids - self.previous_agent_ids
            for agent_id in new_births:
                agent_data = next(a for a in agent_states if a[0] == agent_id)
                pos = (agent_data[2], agent_data[3])  # x, y position
                self.birth_animations[agent_id] = (pos, 0)  # Start animation
                
            # Check for deaths (agents that were in previous frame but not current)
            deaths = self.previous_agent_ids - current_agent_ids
            for agent_id in deaths:
                # Find the last known position from previous frame
                if hasattr(self, 'previous_agent_states'):
                    agent_data = next((a for a in self.previous_agent_states if a[0] == agent_id), None)
                    if agent_data:
                        pos = (agent_data[2], agent_data[3])  # x, y position
                        self.death_animations[agent_id] = (pos, 0)  # Start animation
        
        self.previous_agent_ids = current_agent_ids
        self.previous_agent_states = agent_states

        # Draw agents
        for agent in agent_states:
            x, y = transform(agent[2], agent[3])
            agent_type = agent[1]
            color = 'blue' if agent_type == 'SystemAgent' else 'red'
            radius = max(1, int(2 * scale))
            draw.ellipse([(x-radius, y-radius), (x+radius, y+radius)], 
                        fill=color)

        # Draw birth animations (expanding white circle)
        births_to_remove = []
        for agent_id, (pos, frame) in self.birth_animations.items():
            if frame < self.max_animation_frames:
                x, y = transform(pos[0], pos[1])
                radius = max(2, int(4 * scale)) * (frame + 1) / self.max_animation_frames
                # Draw white circle with decreasing opacity
                opacity = int(255 * (1 - frame / self.max_animation_frames))
                draw.ellipse(
                    [(x-radius, y-radius), (x+radius, y+radius)],
                    outline=(255, 255, 255, opacity)
                )
                self.birth_animations[agent_id] = (pos, frame + 1)
            else:
                births_to_remove.append(agent_id)
                
        # Draw death animations (fading X mark)
        deaths_to_remove = []
        for agent_id, (pos, frame) in self.death_animations.items():
            if frame < self.max_animation_frames:
                x, y = transform(pos[0], pos[1])
                size = max(1, int(1.5 * scale))
                # Draw X mark with decreasing opacity
                opacity = int(128 * (1 - frame / self.max_animation_frames))
                color = (255, 0, 0, opacity)  # Red X
                
                # Draw X
                draw.line([(x-size, y-size), (x+size, y+size)], fill=color, width=1)
                draw.line([(x-size, y+size), (x+size, y-size)], fill=color, width=1)
                
                self.death_animations[agent_id] = (pos, frame + 1)
            else:
                deaths_to_remove.append(agent_id)

        # Clean up completed animations
        for agent_id in births_to_remove:
            del self.birth_animations[agent_id]
        for agent_id in deaths_to_remove:
            del self.death_animations[agent_id]

        # Add step number
        font_size = max(10, int(min(width, height) / 40))
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
            
        draw.text((padding, padding), f"Step: {self.current_step}", 
                 fill='white', font=font)

        # Update canvas
        photo = ImageTk.PhotoImage(img)
        self.env_canvas.create_image(0, 0, image=photo, anchor="nw")
        self.env_canvas.image = photo  # Keep reference to prevent garbage collection

    def _update_charts(self):
        """Update the population and resource charts with historical data."""
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()

        # Fetch all historical data
        history = self.db.get_historical_data()
        
        if history['steps']:
            # Plot agent counts with faded future data
            steps = history['steps']
            system_agents = history['metrics']['system_agents']
            individual_agents = history['metrics']['individual_agents']
            total_resources = history['metrics']['total_resources']
            
            # Split data into past and future
            past_mask = [step <= self.current_step for step in steps]
            future_mask = [not x for x in past_mask]
            
            # Plot past data (solid lines)
            if any(past_mask):
                past_steps = [s for s, m in zip(steps, past_mask) if m]
                past_system = [v for v, m in zip(system_agents, past_mask) if m]
                past_individual = [v for v, m in zip(individual_agents, past_mask) if m]
                past_resources = [v for v, m in zip(total_resources, past_mask) if m]
                
                self.ax1.plot(past_steps, past_system, 'b-', label='System Agents')
                self.ax1.plot(past_steps, past_individual, 'r-', label='Individual Agents')
                self.ax2.plot(past_steps, past_resources, 'g-', label='Resources')
            
            # Plot future data (faded lines)
            if any(future_mask):
                future_steps = [s for s, m in zip(steps, future_mask) if m]
                future_system = [v for v, m in zip(system_agents, future_mask) if m]
                future_individual = [v for v, m in zip(individual_agents, future_mask) if m]
                future_resources = [v for v, m in zip(total_resources, future_mask) if m]
                
                self.ax1.plot(future_steps, future_system, 'b-', alpha=0.3)
                self.ax1.plot(future_steps, future_individual, 'r-', alpha=0.3)
                self.ax2.plot(future_steps, future_resources, 'g-', alpha=0.3)
            
            # Add vertical line for current step
            self.ax1.axvline(x=self.current_step, color='gray', linestyle='--', alpha=0.5)
            
            # Set axis limits to show full simulation
            self.ax1.set_xlim(0, max(steps))
            
            # Labels and legend
            self.ax1.set_xlabel('Step')
            self.ax1.set_ylabel('Agent Count', color='b')
            # Position the resource count label on the right with proper spacing
            self.ax2.yaxis.set_label_position('right')
            self.ax2.set_ylabel('Resource Count', color='g', rotation=270, labelpad=20)
            
            # Adjust tick positions
            self.ax2.yaxis.set_ticks_position('right')
            
            # Adjust tick colors to match the lines
            self.ax1.tick_params(axis='y', labelcolor='b')
            self.ax2.tick_params(axis='y', labelcolor='g')
            
            # Add legends - separate for each axis
            lines1, labels1 = self.ax1.get_legend_handles_labels()
            self.ax1.legend(lines1, labels1, loc='upper left')
            
            lines2, labels2 = self.ax2.get_legend_handles_labels()
            self.ax2.legend(lines2, labels2, loc='upper right')

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
        self.db.close()

    def _export_data(self):
        """Export simulation data to CSV."""
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.db.export_data(filename)

    def _initialize_visualization(self):
        """Initialize the visualization with the first frame of data."""
        initial_data = self.db.get_simulation_data(0)
        if initial_data['agent_states'] or initial_data['resource_states']:
            self._update_visualization(initial_data)
            # Force an update of the window
            self.root.update_idletasks()
