"""Shared styles and constants for GUI components"""

CARD_COLORS = {
    "total_agents": "#4a90e2",  # Blue
    "system_agents": "#50c878",  # Emerald green
    "independent_agents": "#e74c3c",  # Red
    "control_agents": "#DAA520",  # Goldenrod
    "total_resources": "#f39c12",  # Orange
    "average_agent_resources": "#9b59b6",  # Purple
}

AGENT_COLORS = {
    "SystemAgent": "blue",
    "IndependentAgent": "red",
    "ControlAgent": "#DAA520"  # Goldenrod
}

VISUALIZATION_CONSTANTS = {
    "DEFAULT_CANVAS_SIZE": (400, 400),
    "PADDING": 20,
    "MAX_ANIMATION_FRAMES": 5,
    "ANIMATION_MIN_DELAY": 50,
    "MAX_RESOURCE_AMOUNT": 30,
    "RESOURCE_GLOW": {
        "RED": 50,
        "GREEN": 255,
        "BLUE": 50
    },
    "AGENT_RADIUS_SCALE": 2,
    "BIRTH_RADIUS_SCALE": 4,
    "DEATH_MARK_SCALE": 1.5,
    "MIN_FONT_SIZE": 10,
    "FONT_SCALE_FACTOR": 40
}

def configure_ttk_styles():
    """Configure common ttk styles used across the application"""
    from tkinter import ttk
    
    style = ttk.Style()
    
    # Control button style
    style.configure("Control.TButton", padding=5)
    
    # Card styles
    style.configure("Card.TFrame", background="white", relief="solid")
    style.configure("CardLabel.TLabel", background="white", font=("Arial", 10))
    style.configure("CardValue.TLabel", background="white", font=("Arial", 14, "bold"))
    
    # Scale style
    style.layout("Horizontal.TScale", [
        ("Horizontal.Scale.trough", {
            "sticky": "nswe",
            "children": [
                ("Horizontal.Scale.slider", {"side": "left", "sticky": ""})
            ]
        })
    ])
    style.configure("Horizontal.TScale", background="white") 