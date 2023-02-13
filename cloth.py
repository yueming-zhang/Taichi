import taichi as ti
import numpy as np
import pickle
from statistics import mean
import json
from  spherical_points import generate_sphere_points, get_sphere_points

ti.init(arch=ti.cuda)#ti.vulkan)  # Alternatively, ti.init(arch=ti.cpu)
# save_to_folder = "//wsl.localhost/Ubuntu-20.04/home/ming/dev/gns/data/cloth"
save_to_folder = "data"
trajectory_duration = 2.5#1.5
mass_initial_height = 0.6
mass_initial_velocity = -2
gravity = 1#9.8
total_iterations = 300

num_of_balls = 1
r0 = 0.3
r1 = 0.1
dist = 0.25 #distance between two balls
offset = -0.1
x0 = -(r0+r1+dist)/2 + r0/2 + offset
x1 = (r0+r1+dist)/2 + r1/2 + offset

n = 128
down_sample_count = 4
quad_size = 1.0 / n
dt = 4e-2 / n
substeps = int(1 / 60 // dt)
include_ball = True
ball_density = quad_size * down_sample_count


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

def build_sample_index(sc):
    '''
    sc: sample count, 1 means all points, 2 means half points, 4 means quarter points
    '''
    tmp = np.zeros((n*n, 3), dtype=np.int32)
    for i in range(n*n):
        tmp[i] = [i, i//n, i%n]

    sample_index = [i[0] for i in tmp if i[1] % sc == 0 and i[2] % sc == 0]
    return sample_index

sample_index = build_sample_index(down_sample_count)

@ti.kernel
def initialize_mass_points():
    random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.02

    for i, j in x:
        x[i, j] = [
            i * quad_size - 0.55 + random_offset[0], mass_initial_height,# + 0.002*i, # angle the cloth
            j * quad_size - 0.5 + random_offset[1]
        ]
        v[i, j] = [0, mass_initial_velocity, 0] #[0,1,0]: move up at first


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
np_arr = np.empty([0, int(n/down_sample_count)**2, 3])
iterations = 0
spherical_surface_points = get_sphere_points(r0, ball_density)
spherical_surface_particle_type = np.full(len(spherical_surface_points), 3)

particle_type = np.full((int(n/down_sample_count)**2), 5)
if include_ball:
    particle_type = np.concatenate((particle_type, spherical_surface_particle_type))

def save_metadata(trajectory_duration, r0, ball_center, np_list):
    sequence_length = mean([len(i[1][0]) for i in np_list]) - 5
    dt = trajectory_duration / sequence_length

    mean_v = []
    std_v = []
    mean_a = []
    std_a = []
    for item in np_list:
        trajectory = item[1][0]
                # calculate displacement of x,y,z
        v_l = []
        a_l = []
        for i in range(1, trajectory.shape[0]):
            displacement = trajectory[i] - trajectory[i-1]
            velocity = displacement / dt
            if len(v_l) > 0:
                acceleration = (velocity - v_l[-1]) / dt
                a_l.append(acceleration)

            v_l.append(velocity)
        mean_v.append(np.mean(v_l, axis=(0,1)))
        mean_a.append(np.mean(a_l, axis=(0,1)))
        std_v.append(np.std(v_l, axis=(0,1)))
        std_a.append(np.std(a_l, axis=(0,1)))

    vel_mean = np.mean(mean_v, axis=(0))
    vel_std = np.mean(std_v, axis=(0))
    acc_mean = np.mean(mean_a, axis=(0))
    acc_std = np.mean(std_a, axis=(0))

    #save meta data
    meta_data = {
                "bounds": [
                    [-1, 1],
                    [-1, 1],
                    [-1, 1]
                ],
                "balls":[
                    [ball_center[0][0], ball_center[0][1], ball_center[0][2], r0],
                ],
                "dt": dt,
                "dim": 3,
                "sequence_length": sequence_length,
                "default_connectivity_radius": quad_size * down_sample_count * 1.5,
                "cloth_width": int(n / down_sample_count),
                "radius_in_quad": 1.5,
                "quad_size": quad_size * down_sample_count,
                "neighbour_search_size": 1.5,
                "vel_mean": [   
                    0,
                    0,#vel_mean[1],
                    0,         
                ],
                "vel_std": [
                    vel_std[0],
                    vel_std[1],
                    vel_std[2]
                ],
                "acc_mean": [
                    0,
                    acc_mean[1],
                    0
                ],
                "acc_std": [
                    acc_std[0],
                    acc_std[1],
                    acc_std[2]
                ]
            }

            #save np_list to pickle file
    with open(f'{save_to_folder}/metadata.json', 'w') as f:
        json.dump(meta_data, f)
    return dt

while window.running:
    if current_t > trajectory_duration:
        # Reset
        initialize_mass_points()
        current_t = 0      
        p = np.array(spherical_surface_points)
        #reshape ,repeat p, and append to np_arr
        p = np.reshape(p, (1, p.shape[0], p.shape[1]))
        p = np.repeat(p, len(np_arr), axis=0)
        if include_ball:
            np_arr = np.concatenate((np_arr, p), axis=1)

        np_list.append((f'cloth_trajectory_{iterations}', [np_arr, particle_type]))
        np_arr = np.empty([0, int(n/down_sample_count)**2, 3])

        iterations += 1
        if iterations >= total_iterations:
            dt = save_metadata(trajectory_duration, r0, ball_center, np_list)

            with open(f'{save_to_folder}/train.npz', 'wb') as f:
                pickle.dump(np_list[:int(total_iterations*0.9)], f)
            with open(f'{save_to_folder}/valid.npz', 'wb') as f:
                pickle.dump(np_list[int(total_iterations*0.9):int(total_iterations*0.95)], f)
            with open(f'{save_to_folder}/test.npz', 'wb') as f:
                pickle.dump(np_list[-int(total_iterations*0.05):], f)
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

    vertices_np = vertices.to_numpy()
    if down_sample_count != 1:
        # retrieve even indices from vertices_np
        vertices_np = np.take(vertices_np, sample_index, axis=0)

    np_arr = np.insert(np_arr, np_arr.shape[0], vertices_np, axis=0)

    # Draw a smaller ball to avoid visual penetration
    scene.particles(ball_center, radius=r0 * 0.95, color=(0.5, 0.42, 0.8))
    if num_of_balls == 2:
        scene.particles(ball1_center, radius=r1 * 0.95, color=(0.8, 0.42, 0.8))
    canvas.scene(scene)
    window.show()