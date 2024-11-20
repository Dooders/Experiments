import tkinter as tk
from gui import SimulationGUI

def main():
    """
    Main entry point for the simulation GUI application.
    """
    root = tk.Tk()
    app = SimulationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
