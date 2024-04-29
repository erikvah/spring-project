from dataclasses import dataclass


@dataclass
class Params:
    # GUI
    window_size = (1600, 900)
    agent_radius = 6
    draw_delay = 17  # ms

    # Simulation
    move_delay = 0.01  # s
    margin = 200
