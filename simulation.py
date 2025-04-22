import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def run_simulation(x, y, z):
    start = input("Start simulation? (y/n): ").strip().lower()
    if start != 'y':
        print("Simulation aborted.")
        return

    simulate_third_person(x, y, z)

def simulate_third_person(x, y, z, interval=20):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, color='forestgreen', linewidth=2)
    ax.set_axis_off()
    ax.grid(False)
    ax.view_init(elev=8, azim=-90, roll=0)
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.set_zlim(min(z), max(z))

    box, = ax.plot([], [], [], 'o', color='blue', markersize=12)

    def init():
        box.set_data([], [])
        box.set_3d_properties([])
        return box,

    def animate(i):
        idx = i % len(x)
        box.set_data([x[idx]], [y[idx]])
        box.set_3d_properties([z[idx]])
        return box,

    ani = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(x), interval=interval, blit=True
    )

    plt.tight_layout()
    plt.show()