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
def initialize_mass_points():
    random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1

    for i, j in x:
        x[i, j] = [
            i * quad_size - 0.5 + random_offset[0], 0.6,# + 0.002*i, # tile the cloth
            j * quad_size - 0.5 + random_offset[1]
        ]
        v[i, j] = [0, 0, 0] #[0,1,0]: move up at first


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

spring_offsets = []
if bending_springs:
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i, j) != (0, 0):
                spring_offsets.append(ti.Vector([i, j]))

else:
    for i in range(-2, 3):
        for j in range(-2, 3):
            if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
                spring_offsets.append(ti.Vector([i, j]))

@ti.kernel
def substep():
    for i in ti.grouped(x):
        v[i] += gravity * dt

    for i in ti.grouped(x):
        force = ti.Vector([0.0, 0.0, 0.0])
        for spring_offset in ti.static(spring_offsets):
            j = i + spring_offset
            if 0 <= j[0] < n and 0 <= j[1] < n:
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                d = x_ij.normalized()
                current_dist = x_ij.norm()
                original_dist = quad_size * float(i - j).norm()
                # Spring force
                force += -spring_Y * d * (current_dist / original_dist - 1)
                # Dashpot damping
                force += -v_ij.dot(d) * d * dashpot_damping * quad_size

        v[i] += force * dt

    for i in ti.grouped(x):
        v[i] *= ti.exp(-drag_damping * dt)
        offset_to_center = x[i] - ball_center[0]
        if offset_to_center.norm() <= r0:
            # Velocity projection
            normal = offset_to_center.normalized()
            v[i] -= ti.min(v[i].dot(normal), 0) * normal

        if num_of_balls == 2:
            offset_to_center = x[i] - ball1_center[0]
            if offset_to_center.norm() <= r1:
                # Velocity projection
                normal = offset_to_center.normalized()
                v[i] -= ti.min(v[i].dot(normal), 0) * normal

        x[i] += dt * v[i]

@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]

window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

current_t = 0.0
initialize_mass_points()
np_list = []
np_arr = np.empty([0, n**2, 3])
iterations = 0
while window.running:
    if current_t > trajectory_duration:
        # Reset
        initialize_mass_points()
        current_t = 0      
        np_list.append(np_arr)
        np_arr = np.empty([0, n**2, 3])

        iterations += 1
        if iterations >= total_iterations:
            #save np_list to pickle file
            with open('data/cloth_data.pkl', 'wb') as f:
                pickle.dump(np_list, f)
            break

    for i in range(substeps):
        substep()
        current_t += dt
    update_vertices()

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

    np_arr = np.insert(np_arr, np_arr.shape[0], vertices.to_numpy(), axis=0)

    # Draw a smaller ball to avoid visual penetration
    scene.particles(ball_center, radius=r0 * 0.95, color=(0.5, 0.42, 0.8))
    if num_of_balls == 2:
        scene.particles(ball1_center, radius=r1 * 0.95, color=(0.8, 0.42, 0.8))
    canvas.scene(scene)
    window.show()