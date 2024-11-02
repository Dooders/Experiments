import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk, ImageDraw, ImageFont
from database import SimulationDatabase
import numpy as np

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

        self._setup_stats_panel()
        self._setup_chart()
        self._setup_environment_view()
        self._setup_controls()

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
            self.root.after(int(1000 / self.speed_scale.get()), self._play_simulation)

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

        # Calculate scaling factors - increased padding from 10px to 20px on each side
        padding = 20
        env_width = max(x for _, _, x, _ in resource_states + [(0, 0, 100, 0)])  # Default to 100 if no resources
        env_height = max(y for _, _, _, y in resource_states + [(0, 0, 0, 100)])
        
        scale_x = (width - 2*padding) / env_width  # Add padding on both sides
        scale_y = (height - 2*padding) / env_height
        scale = min(scale_x, scale_y)  # Use smaller scale to maintain aspect ratio

        # Calculate offset to center the environment
        offset_x = (width - (env_width * scale)) / 2
        offset_y = (height - (env_height * scale)) / 2

        # Ensure minimum padding
        offset_x = max(padding, offset_x)
        offset_y = max(padding, offset_y)

        def transform(x, y):
            """Transform environment coordinates to screen coordinates."""
            return (offset_x + x * scale, offset_y + y * scale)

        # Draw resources
        for resource in resource_states:
            amount = resource[1]  # amount
            if amount > 0:  # Only draw if resource has any amount left
                x, y = transform(resource[2], resource[3])  # position_x, position_y
                
                # Calculate color based on amount (fade from green to black)
                # Assuming max amount is 30
                color_intensity = int((amount / 30) * 255)
                resource_color = (0, color_intensity, 0)  # RGB: (0, 0-255, 0)
                
                radius = max(3, int(5 * (amount / 30) * scale))  # Scale size with amount and canvas
                draw.ellipse([(x-radius, y-radius), (x+radius, y+radius)], 
                            fill=resource_color, outline=resource_color)

        # Draw agents
        for agent in agent_states:
            x, y = transform(agent[2], agent[3])  # position_x, position_y
            agent_type = agent[1]  # agent_type
            color = 'blue' if agent_type == 'SystemAgent' else 'red'
            radius = max(1, int(2 * scale))  # Reduced from max(2, int(3 * scale))
            draw.ellipse([(x-radius, y-radius), (x+radius, y+radius)], 
                        fill=color)

        # Add step number - scale font size with canvas and move it inside the padding
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

        # Fetch historical data for the last 100 steps
        start_step = max(0, self.current_step - 100)
        history = self.db.get_historical_data(start_step, self.current_step)
        
        if history['steps']:
            # Plot agent counts
            self.ax1.plot(history['steps'], history['metrics']['system_agents'], 
                         'b-', label='System Agents')
            self.ax1.plot(history['steps'], history['metrics']['individual_agents'], 
                         'r-', label='Individual Agents')

            # Plot resources
            self.ax2.plot(history['steps'], history['metrics']['total_resources'], 
                         'g-', label='Resources')

            # Set axis limits
            self.ax1.set_xlim(start_step, self.current_step)
            
            # Labels and legend
            self.ax1.set_xlabel('Step')
            self.ax1.set_ylabel('Agent Count', color='b')
            self.ax2.set_ylabel('Resource Count', color='g')
            
            # Add legends
            lines1, labels1 = self.ax1.get_legend_handles_labels()
            lines2, labels2 = self.ax2.get_legend_handles_labels()
            self.ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        # Update the canvas
        self.canvas.draw()

    def run(self):
        """Start the visualization."""
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
