from dataclasses import dataclass


@dataclass
class Params:
    # GUI
    window_size = (1000, 600)
    agent_radius = 6
    draw_delay = 50  # ms

    # Simulation
    move_delay = 0.01  # s
