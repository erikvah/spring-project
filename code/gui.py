import time
import tkinter as tk

import agent as agent
from params import Params


def init(agents):
    width, height = Params.window_size
    root = tk.Tk()
    root.title("Visualization")
    canvas = tk.Canvas(root, bg="black", width=width, height=height)
    canvas.pack(fill=tk.BOTH)

    circles = [canvas.create_oval(0, 0, 0, 0, fill="white") for _ in agents]

    r = Params.agent_radius

    def update():
        t0 = time.time()
        states = []
        for ag in agents:
            lock = ag.get_lock()

            with lock:
                pos = ag.get_pos()

            states.append(pos)

        for state, circle in zip(states, circles):
            pos = state
            canvas.coords(circle, pos[0] - r, pos[1] - r, pos[0] + r, pos[1] + r)

        time_taken = int(1000 * (time.time() - t0))
        root.after(max(Params.draw_delay - time_taken, 0), update)

    root.after(Params.draw_delay, update)

    return root, canvas


def start(root):
    root.mainloop()
