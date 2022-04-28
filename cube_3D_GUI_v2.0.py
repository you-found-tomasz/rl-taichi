import taichi as ti
import matplotlib.pyplot as plt
import numpy as np
from engine.mpm_solver import MPMSolver
import os
import math
from scipy.spatial import Delaunay
write_to_disk = False

# Try to run on GPU
#ti.init(arch=ti.gpu, device_memory_GB=4.0)
ti.init(arch=ti.gpu, device_memory_fraction=0.9)
#ti.init(arch=ti.vulkan)
#ti.init(arch=ti.cpu)
import numpy as np
import pygalmesh


factor = 4
height = 6
center = 5

s = pygalmesh.Ball([center, height + 1, center], 1.0)
mesh = pygalmesh.generate_mesh(s, max_cell_circumradius=0.05)
mesh_triangles2 = mesh.cells_dict['triangle']

mpm = MPMSolver(res=(64, 64, 64), size=10, max_num_particles=2 ** 20, use_ggui=True)
mpm.add_sphere_collider_inv(center=(center, height, center), radius=factor-0.5,surface=mpm.surface_sticky)
#mpm.set_gravity((0, -50, 0))
mpm.set_gravity((0, -350, 0))
mesh_particles = np.load('vertices_reduced.npy')
#mesh_particles = np.load('2d_mesh_1015.npy')
#mesh_particles = mesh_particles + np.array([-2.5, -2.5])
mesh_triangles = np.load('faces_reduced.npy')
mesh_nr = mesh_particles.shape[0]
#tensor_field = ti.Matrix.field(n=3, m=1, dtype=ti.f32, shape=(mesh_nr))

particles = np.array([mesh_particles[:,0]*factor+center, np.ones(len(mesh_particles))*height, mesh_particles[:,1]*factor+center]).T
#particles_reduced = np.delete(particles, np.arange(0, particles.shape[0], 5), axis=0)
mpm.add_particles(particles=particles,material=MPMSolver.material_elastic)
mpm.add_particles(particles=mesh.points,material=MPMSolver.material_elastic)
#mpm.add_ellipsoid(center=[center, 7.5, center], radius=1.2, material=MPMSolver.material_elastic, velocity=[0, 0, 0], sample_density=2**mpm.dim)

#triangluation
baseline_length = np.ones([mesh_triangles.shape[0],3])
for i in range(len(mesh_triangles)):
    baseline_length[i,0] = np.linalg.norm(particles[mesh_triangles[i,0],:] - particles[mesh_triangles[i,1],:])
    baseline_length[i,1] = np.linalg.norm(particles[mesh_triangles[i,0],:] - particles[mesh_triangles[i,2],:])
    baseline_length[i,2] = np.linalg.norm(particles[mesh_triangles[i,1],:] - particles[mesh_triangles[i,2],:])


particles = mpm.particle_info()
particle_nr = particles['position'].shape[0]


'''
#calculate strain
strain_particles = particles['position'][:mesh_nr]
tri = Delaunay(strain_particles[:,[0,2]], incremental=True)

neighbour_list = list()
for i in range(mesh_nr):
    neighbour_indices = np.nonzero(tri.simplices == i)
    neighbour_indices2 = np.unique(tri.simplices[neighbour_indices[0], :].reshape(-1,1).squeeze())
    neighbour_indices3 = np.delete(neighbour_indices2, np.where(neighbour_indices2 == i))
    neighbour_list.append(neighbour_indices3)

dist_list_baseline = list()
for i in range(len(neighbour_list)):
    neighbour_points = neighbour_list[i]
    dist_neighbour_points = np.linalg.norm(strain_particles[neighbour_indices,:] - strain_particles[i,:], axis=1)
    dist_list_baseline.append(dist_neighbour_points)


dist_list_baseline = list()
for i in range(mesh_nr):
    neighbour_indices = np.nonzero(tri.simplices == i)
    neighbour_indices2 = np.unique(tri.simplices[neighbour_indices[0], :].reshape(-1,1).squeeze())
    neighbour_indices3 = np.delete(neighbour_indices2, np.where(neighbour_indices2 == i))
    neighbour_points = strain_particles[neighbour_indices3,:]
    dist_neighbour_points = np.linalg.norm(neighbour_points - strain_particles[i,:], axis=1)
    dist_list_baseline.append(dist_neighbour_points)

fig = plt.figure(figsize =(30, 30))
fig.set_dpi(150.0)
plt.triplot(strain_particles[:,0], strain_particles[:,2], tri.simplices)
plt.plot(neighbour_points[:,0], neighbour_points[:,2], 'o')
plt.plot(strain_particles[i,0], strain_particles[i,2], 'x')
plt.show()
i=2
'''
'''
idx_list = list()
for i in range(len(mesh_triangles)):
    particle_triangle = np.array([mesh_particles[mesh_triangles[i, 0]], mesh_particles[mesh_triangles[i, 1]], mesh_particles[mesh_triangles[i, 2]]])
    #particle_triangle = np.array([mesh_particles[tri.simplices[i, 0]], mesh_particles[tri.simplices[i, 1]], mesh_particles[tri.simplices[i, 2]]])
    particle_array = particles['position'][:, [0, 2]]
    idx = (np.linalg.norm((particle_array[:, :, None] - particle_triangle.T), axis=1)).argmin(axis=0)
    idx_list.append(idx)
idx_array = np.array(idx_list).reshape(-1,1).squeeze()
'''

#triangles_final = np.concatenate([mesh_triangles2, mesh_triangles])
triangles_final = np.concatenate([mesh_triangles])
idx_array = np.array(triangles_final).reshape(-1,1).squeeze()
idx_array_tai = ti.field(ti.i32, shape=len(idx_array))
idx_array_tai.from_numpy(idx_array)

idx_array2 = np.array(mesh_triangles2).reshape(-1,1).squeeze()
idx_array_tai2 = ti.field(ti.i32, shape=len(idx_array2))
idx_array_tai2.from_numpy(idx_array2)

print('finished')

@ti.kernel
def set_color(ti_color: ti.template(), material_color: ti.ext_arr(), ti_material: ti.template()):
    for I in ti.grouped(ti_material):
        material_id = ti_material[I]
        color_4d = ti.Vector([0.0, 0.0, 0.0, 1.0])
#        if ti_material[I] != 0:
#            color_4d = ti.Vector([0.0, 0.0, 0.0, 0.0])
        for d in ti.static(range(3)):
            color_4d[d] = material_color[material_id, d]
        ti_color[I] = color_4d

@ti.kernel
def set_color2(ti_color: ti.template(), strain: ti.ext_arr()):
    i = 0
    for I in ti.grouped(ti_color):
        ti_color[I][1] = ti_color[I][1] + (strain[i] -1)
        i += 1

tensor_field = ti.Matrix.field(n=3, m=1, dtype=ti.f32, shape=(mesh_nr))
tensor_field2 = ti.Matrix.field(n=3, m=1, dtype=ti.f32, shape=(particle_nr-mesh_nr))

@ti.kernel
def split(mpm_x: ti.template()):
    for i in ti.grouped(tensor_field):
        tensor_field[i] = mpm_x[i]
    for i in ti.grouped(tensor_field2):
        tensor_field2[i] = mpm_x[i + mesh_nr]
        #print(mpm_x[i])


res = (1920, 1080)
window = ti.ui.Window("Real MPM 3D", res, vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(-6.0, 8, 5.0)
camera.lookat(5, 2, 5)
camera.fov(55)
particles_radius = 0.05


def render():
    camera.track_user_inputs(window, movement_speed=0.2, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.ambient_light((0, 0, 0))
    strain = calculate_strain()
    set_color(mpm.color_with_alpha, material_type_colors, mpm.material)
    #set_color2(mpm.color_with_alpha, strain)

    split(mpm.x)
    #scene.particles(tensor_field2, radius=particles_radius)
    scene.particles(mpm.x, radius=particles_radius, per_vertex_color=mpm.color_with_alpha)
    #scene.mesh(tensor_field2, indices=idx_array_tai2, two_sided=False)
    #scene.mesh(tensor_field, indices=idx_array_tai, two_sided=False)
    #scene.mesh(mpm.x, two_sided=True)

    scene.point_light(pos=(5.0, 6.5, 5.0), color=(1.0, 1.0, 1.0))
    scene.point_light(pos=(5.0, 0.0, 5.0), color=(1.0, 1.0, 1.0))
    scene.point_light(pos=(5.5, 5.5, 5.5), color=(1.0, 1.0, 1.0))
    scene.point_light(pos=(5.5, 0.0, 5.5), color=(1.0, 1.0, 1.0))

    canvas.scene(scene)

def calculate_strain():
    particles = mpm.particle_info()
    strain_particles = particles['position'][:mesh_nr]

    strain = np.ones([particles['position'].shape[0] ,1]).squeeze()
    '''
    for i in range(mesh_nr):
        hello = np.nonzero(tri.simplices == i)
        neighbour_points = strain_particles[tri.simplices[hello[0], :].reshape(-1, 1).squeeze()]
        dist_neighbour_points = np.linalg.norm(neighbour_points - strain_particles[i, :], axis=1)
        strain[i] = np.nanmax(dist_neighbour_points / dist_list_baseline[i])
    '''
    for i in range(mesh_nr):
        neighbour_points = neighbour_list[i]
        dist_neighbour_points = np.linalg.norm(strain_particles[neighbour_points,:] - strain_particles[i,:], axis=1)
        strain[i] = np.nanmax(dist_neighbour_points / dist_list_baseline[i].reshape(-1,1))

    return strain

def show_options():
    global particles_radius

    window.GUI.begin("Solver Property", 0.05, 0.1, 0.2, 0.10)
    window.GUI.text(f"Current particle number {mpm.n_particles[None]}")
    particles_radius = window.GUI.slider_float("particles radius ",
                                               particles_radius, 0, 0.1)
    window.GUI.end()

    window.GUI.begin("Camera", 0.05, 0.2, 0.2, 0.16)
    camera.curr_position[0] = window.GUI.slider_float("camera pos x", camera.curr_position[0], -10, 10)
    camera.curr_position[1] = window.GUI.slider_float("camera pos y", camera.curr_position[1], -10, 10)
    camera.curr_position[2] = window.GUI.slider_float("camera pos z", camera.curr_position[2], -10, 10)

    camera.curr_lookat[0] = window.GUI.slider_float("camera look at x", camera.curr_lookat[0], -10, 10)
    camera.curr_lookat[1] = window.GUI.slider_float("camera look at y", camera.curr_lookat[1], -10, 10)
    camera.curr_lookat[2] = window.GUI.slider_float("camera look at z", camera.curr_lookat[2], -10, 10)

    window.GUI.end()


material_type_colors = np.array([
    [0.1, 0.1, 1.0, 0.8],
    [236.0 / 255.0, 84.0 / 255.0, 59.0 / 255.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 0.0, 1.0]
]
)

import time
for frame in range(1500):
    '''
    start = time.time()
    end = time.time()
    print(1/(end - start))
    '''
    mpm.step(4e-3)
    osc_freq=1/10. #every 1 frames one cycle
    #mpm.set_gravity((1000*np.sin(np.pi*frame*osc_freq), -150, 1000*np.cos(np.pi*frame*osc_freq)))
    '''
    if 100 < frame < 200:
        mpm.add_cube(lower_corner=[0.1, 4 , 2],
                     cube_size=[10.0, 0.5, 0.5],
                     material=MPMSolver.material_water,
                     velocity=[0, 0, 100 ])
        #100*math.sin(frame * 0.1)
    '''
    render()
    show_options()
    window.show()