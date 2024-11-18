"""Configuration settings for the simulation GUI."""

# Window settings
WINDOW_TITLE = "Agent-Based Simulation"
WINDOW_SIZE = "1200x800"

# Layout weights
LEFT_PANE_WEIGHT = 3
RIGHT_PANE_WEIGHT = 1

# Log display settings
LOG_TEXT_CONFIG = {
    "wrap": "word",
    "bg": "black",
    "fg": "#00FF00",
    "font": ("Courier", 10),
    "height": 20,
    "width": 50,
}

# File paths and directories
LOG_DIR = "logs"
SIMULATIONS_DIR = "simulations"
CONFIG_FILE = "config.yaml"
DOCS_FILE = "agents.md"

# Default simulation settings
DEFAULT_SIMULATION_STEPS = 1000

# Batch simulation parameters
DEFAULT_BATCH_VARIATIONS = {
    "system_agents": [20, 30, 40],
    "independent_agents": [20, 30, 40],
}

# About dialog text
ABOUT_TEXT = """Agent-Based Simulation

A simulation environment for studying emergent behaviors 
in populations of system and independent agents.

Version 1.0"""

# File dialog configurations
FILE_TYPES = {
    "database": [("Database files", "*.db"), ("All files", "*.*")],
    "csv": [("CSV files", "*.csv"), ("All files", "*.*")],
    "html": [("HTML files", "*.html"), ("All files", "*.*")],
}

# Tooltip display time (milliseconds)
TOOLTIP_DURATION = 2000 