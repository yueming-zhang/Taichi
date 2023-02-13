import numpy as np
import math
import matplotlib.pyplot as plt

def spherical_to_cartesian(radius, theta, phi):
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return x, y, z

def generate_sphere_points(radius, surface_distance_interval):
    # Approximate the polar angle interval based on the surface distance interval
    theta_interval = np.arcsin(surface_distance_interval / (2 * radius))

    # Approximate the azimuthal angle interval based on the polar angle interval
    phi_interval = np.arccos(1 - (surface_distance_interval / (2 * radius * np.sin(theta_interval))))

    theta = np.arange(0, np.pi + theta_interval, theta_interval)
    phi = np.arange(0, 2 * np.pi + phi_interval, phi_interval)
    points = []
    for t in theta:
        for p in phi:
            x, y, z = spherical_to_cartesian(radius, t, p)
            points.append((x, y, z))
    return points

def get_sphere_points(radius, surface_distance_interval):
    points = []

    step = (surface_distance_interval / (2 * math.pi * radius)) * 360
    for i in range(-90+int(step/1),90,int(step/2)): # -90 to 90 south pole to north pole
        alt = math.radians(i)
        c_alt = math.cos(alt)
        s_alt = math.sin(alt)

        step1 = (surface_distance_interval / (2 * math.pi * radius * c_alt)) * 360
        for j in range(0,360,int(step1)): # 360 degree (around the sphere)
            azi = math.radians(j)
            c_azi = math.cos(azi)
            s_azi = math.sin(azi)
            if (s_alt * radius > 0):
                points.append([c_azi*c_alt * radius, s_alt * radius, s_azi*c_alt * radius])

    return points

def draw_sphere(radius, surface_distance_interval):
    points = get_sphere_points(radius, surface_distance_interval)

    x = [p[0] for p in points]
    y = [p[1] for p in points]
    z = [p[2] for p in points]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='.')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show(block=True)


# draw_sphere(0.3, 0.09)
# radius = 0.3
# surface_distance_interval = (1./128)*4
# sphere_points = generate_sphere_points(radius, surface_distance_interval)
# pass