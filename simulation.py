import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def run_simulation(x, y, z):
    start = input("Start simulation? (y/n): ").strip().lower()
    if start != 'y':
        print("Simulation aborted.")
        return

    view = input("First person or third person? (1/3): ").strip()
    if view == '3':
        simulate_third_person(x, y, z)
    elif view == '1':
        launch = input("Launch immersive 3D ride? (y/n): ").strip().lower()
        if launch == 'y':
            run_ursina_simulation(x, y, z)
        else:
            simulate_first_person(x, y, z)
    else:
        print("Invalid option. Please enter 1 or 3.")

def simulate_third_person(x, y, z, interval=20):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, color='forestgreen', linewidth=2)
    ax.set_axis_off()
    ax.grid(False)
    ax.view_init(elev=20, azim=-70)
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.set_zlim(min(z), max(z))

    box, = ax.plot([], [], [], 's', color='red', markersize=8)

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

def simulate_first_person(x, y, z, interval=20):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    ax.grid(False)

    ax.plot(x, y, z, color='forestgreen', linewidth=1.5, alpha=1.0)
    visible_track, = ax.plot([], [], [], color='blue', linewidth=4)
    camera_cube, = ax.plot([], [], [], 'o', color='red', markersize=10)
    direction_arrow, = ax.plot([], [], [], color='black', linewidth=2, linestyle='dashed', alpha=0.6)

    def init():
        visible_track.set_data([], [])
        visible_track.set_3d_properties([])
        camera_cube.set_data([], [])
        camera_cube.set_3d_properties([])
        direction_arrow.set_data([], [])
        direction_arrow.set_3d_properties([])
        return visible_track, camera_cube, direction_arrow

    def animate(i):
        idx = i % (len(x) - 10)
        path_ahead = slice(idx, idx + 10)
        xp, yp, zp = x[path_ahead], y[path_ahead], z[path_ahead]
        visible_track.set_data(xp, yp)
        visible_track.set_3d_properties(zp)

        camera_cube.set_data([xp[-1]], [yp[-1]])
        camera_cube.set_3d_properties([zp[-1]])

        p1 = np.array([x[idx], y[idx], z[idx]])
        p2 = np.array([x[idx + 1], y[idx + 1], z[idx + 1]])
        direction = p2 - p1
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = direction / norm

        arrow_end = p1 + 10 * direction
        direction_arrow.set_data([p1[0], arrow_end[0]], [p1[1], arrow_end[1]])
        direction_arrow.set_3d_properties([p1[2], arrow_end[2]])

        return visible_track, camera_cube, direction_arrow

    ani = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(x), interval=interval, blit=True
    )

    plt.tight_layout()
    plt.show()

def run_ursina_simulation(x, y, z):
    from ursina import Ursina, Entity, color, Mesh, Vec3, Grid, camera, window
    import numpy as np
    import os

    app = Ursina(borderless=False)
    window.size = (1280, 720)
    window.title = 'DinoCoaster'

    # Set dummy icon to avoid crash
    icon_path = 'blank_icon.png'  # Or path to a valid PNG/ICO you have
    if os.path.exists(icon_path):
        window.icon = icon_path

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    path = list(zip(np.round(x).astype(int), np.round(y).astype(int), np.round(z).astype(int)))

    # Create ground grid
    Entity(model=Grid(32, 32), scale=100, color=color.gray, rotation=(90, 0, 0), y=-0.1)

    # Create a path line
    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]
        Entity(
            model=Mesh(vertices=[start, end], mode='line'),
            color=color.green,
            double_sided=True
        )

    # Cart entity
    cart = Entity(model='cube', color=color.azure, scale=1)
    camera.parent = cart
    camera.position = (0, 1.5, -4)
    camera.rotation = (0, 0, 0)

    # Movement
    i = 0

    def update():
        nonlocal i
        if i < len(path) - 1:
            p1 = path[i]
            p2 = path[i + 1]
            cart.position = p1
            cart.look_at(p2)
            i += 1
        else:
            i = 0  # loop

    app.run()