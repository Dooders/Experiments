import tkinter as tk
from gui import SimulationGUI
from api.server import socketio, app
import threading
import os

save_path = "results/simulation_results.db"

def main():
    """
    Main entry point for the simulation GUI application.
    """
    try:
        # Create required directories
        os.makedirs('results', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Start SocketIO server
        socketio.run(app, port=5000, debug=True)
        
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        raise

if __name__ == "__main__":
    main()