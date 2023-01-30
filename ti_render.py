import taichi as ti
import numpy as np
import pickle
ti.init(arch=ti.cuda)#ti.vulkan)  # Alternatively, ti.init(arch=ti.cpu)

trajectory_duration = 1.5
gravity = 9.8
total_iterations = 1

num_of_balls = 1
r0 = 0.3
r1 = 0.1
dist = 0.25
offset = -0.1
x0 = -(r0+r1+dist)/2 + r0/2 + offset
x1 = (r0+r1+dist)/2 + r1/2 + offset

n = 128
quad_size = 1.0 / n
dt = 4e-2 / n
substeps = int(1 / 60 // dt)

gravity = ti.Vector([0, -gravity, 0])
spring_Y = 3e4
dashpot_damping = 1e4
drag_damping = 1

ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
ball_center[0] = [x0 if num_of_balls == 2 else 0, 0, 0]

ball1_center = ti.Vector.field(3, dtype=float, shape=(1, ))
ball1_center[0] = [x1, 0.1, 0]


x = ti.Vector.field(3, dtype=float, shape=(n, n)) # cloth positions
v = ti.Vector.field(3, dtype=float, shape=(n, n)) # cloth velocity

num_triangles = (n - 1) * (n - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)

bending_springs = False

@ti.kernel
def initialize_mesh_indices():
    for i, j in ti.ndrange(n - 1, n - 1):
        quad_id = (i * (n - 1)) + j
        # 1st triangle of the square
        indices[quad_id * 6 + 0] = i * n + j
        indices[quad_id * 6 + 1] = (i + 1) * n + j
        indices[quad_id * 6 + 2] = i * n + (j + 1)
        # 2nd triangle of the square
        indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
        indices[quad_id * 6 + 4] = i * n + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * n + j

    for i, j in ti.ndrange(n, n):
        if (i // 4 + j // 4) % 2 == 0:
            colors[i * n + j] = (0.22, 0.72, 0.52)
        else:
            colors[i * n + j] = (1, 0.334, 0.52)

initialize_mesh_indices()

window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

step = 0
#load data from pickle file
with open('data/cloth_data.pkl', 'rb') as f:
    np_list = pickle.load(f)
    trajectory = np_list[0]
    vertices = ti.Vector.field(3, float, n**2)

while window.running:
    if step < trajectory.shape[0]-1:
        step += 1
    else:
        step = 0
        vertices = ti.Vector.field(3, float, n**2)

    vertices.from_numpy(trajectory[step])
    camera.position(0.0, 0.0, 3)
    # camera.position(0.0, 0.0, 3)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    
    scene.mesh(vertices,
               indices=indices,
               per_vertex_color=colors,
               two_sided=True)

    # Draw a smaller ball to avoid visual penetration
    scene.particles(ball_center, radius=r0 * 0.95, color=(0.5, 0.42, 0.8))
    if num_of_balls == 2:
        scene.particles(ball1_center, radius=r1 * 0.95, color=(0.8, 0.42, 0.8))
    canvas.scene(scene)
    window.show()