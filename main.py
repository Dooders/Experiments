import tkinter as tk
from gui import SimulationGUI

save_path = "results/simulation_results.db"

def main():
    """
    Main entry point for the simulation GUI application.
    """
    root = tk.Tk()
    app = SimulationGUI(root, save_path)
    root.mainloop()

if __name__ == "__main__":
    main()
