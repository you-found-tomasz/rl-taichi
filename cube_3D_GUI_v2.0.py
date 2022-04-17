import taichi as ti
import numpy as np
from engine.mpm_solver import MPMSolver
import os
import math
write_to_disk = False

# Try to run on GPU
#ti.init(arch=ti.gpu, device_memory_GB=4.0)
ti.init(arch=ti.gpu, device_memory_fraction=0.9)
#ti.init(arch=ti.vulkan)
#ti.init(arch=ti.cpu)


factor = 4
height = 6
center = 5
mpm = MPMSolver(res=(64, 64, 64), size=10, max_num_particles=2 ** 20, use_ggui=True)
mpm.add_sphere_collider_inv(center=(center, height, center), radius=factor-0.5,surface=mpm.surface_sticky)
#mpm.set_gravity((0, -50, 0))
mpm.set_gravity((0, -150, 0))
mesh_particles = np.load('vertices_reduced.npy')
mesh_triangles = np.load('faces_reduced.npy')

particles = np.array([mesh_particles[:,0]*factor+center, np.ones(len(mesh_particles))*height, mesh_particles[:,1]*factor+center]).T
#particles_reduced = np.delete(particles, np.arange(0, particles.shape[0], 5), axis=0)

mpm.add_particles(particles=particles,material=MPMSolver.material_elastic)
#mpm.add_ellipsoid(center=[center, 7.5, center], radius=1.7, material=MPMSolver.material_water, velocity=[0, 0, 0], sample_density=2**mpm.dim) good
#mpm.add_ellipsoid(center=[center, 8.5, center], radius=3.2, material=MPMSolver.material_water, velocity=[0, 0, 0], sample_density=2**mpm.dim)
#mpm.add_ellipsoid(center=[center, 9.5, center], radius=3.2, material=MPMSolver.material_water, velocity=[0, 0, 0], sample_density=2**mpm.dim)

#mpm.add_ellipsoid(center=[center, 9.5, center], radius=1.5, material=MPMSolver.material_water, velocity=[0, -10, 0], sample_density=2**mpm.dim*2)
#mpm.add_ellipsoid(center=[2.5, 4.5, 2.5], radius=0.25, material=MPMSolver.material_elastic, velocity=[0, -10, 0], sample_density=2**mpm.dim*20)

particles = mpm.particle_info()

'''
idx_list = list()
for i in range(len(mesh_triangles)):
    particle_triangle = np.array([mesh_particles[mesh_triangles[i, 0]], mesh_particles[mesh_triangles[i, 1]], mesh_particles[mesh_triangles[i, 2]]])
    particle_array = particles['position'][:, [0, 2]]
    idx = (np.linalg.norm((particle_array[:, :, None] - particle_triangle.T), axis=1)).argmin(axis=0)
    idx_list.append(idx)
idx_array = np.array(idx_list).reshape(-1,1).squeeze()
'''

idx_array = np.array(mesh_triangles).reshape(-1,1).squeeze()
idx_array_tai = ti.field(ti.i32, shape=len(idx_array))
idx_array_tai.from_numpy(idx_array)
print('finished')

@ti.kernel
def set_color(ti_color: ti.template(), material_color: ti.ext_arr(), ti_material: ti.template()):
    for I in ti.grouped(ti_material):
        material_id = ti_material[I]
        color_4d = ti.Vector([0.0, 0.0, 0.0, 1.0])
        for d in ti.static(range(3)):
            color_4d[d] = material_color[material_id, d]
        ti_color[I] = color_4d


res = (1920, 1080)
window = ti.ui.Window("Real MPM 3D", res, vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(-6.0, 8, -6.0)
camera.lookat(5, 2, 5)
#camera.position(4.389, 9.5, -9.5)
#camera.lookat(4.25, 1.89, 1.7)
camera.fov(55)
particles_radius = 0.05


def render():
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.ambient_light((0, 0, 0))
    set_color(mpm.color_with_alpha, material_type_colors, mpm.material)

    scene.particles(mpm.x, per_vertex_color=mpm.color_with_alpha, radius=particles_radius)
    #scene.mesh(mpm.x, indices=idx_array_tai, two_sided=True)
    #scene.mesh(mpm.x, two_sided=True)

    scene.point_light(pos=(2.5, 5.0, 2.5), color=(1.0, 1.0, 1.0))
    scene.point_light(pos=(2.5, 0.0, 2.5), color=(1.0, 1.0, 1.0))
    scene.point_light(pos=(3.5, 5.0, 3.5), color=(1.0, 1.0, 1.0))
    scene.point_light(pos=(3.5, 0.0, 3.5), color=(1.0, 1.0, 1.0))
    #scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))
    #scene.point_light(pos=(4.0, 5, 4.0), color=(0.5, 0.5, 0.5))


    '''
    scene.point_light(pos=(0.0, 1.0, 0.0), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.0, 1.0, 1.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.0, -1.5, 0.0), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.0, -1.5, 1.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.0, 1.0, -1.5), color=(0.5, 0.5, 0.5))
    '''

    canvas.scene(scene)


def show_options():
    global particles_radius

    window.GUI.begin("Solver Property", 0.05, 0.1, 0.2, 0.10)
    window.GUI.text(f"Current particle number {mpm.n_particles[None]}")
    particles_radius = window.GUI.slider_float("particles radius ",
                                               particles_radius, 0, 0.1)
    window.GUI.end()

    window.GUI.begin("Camera", 0.05, 0.3, 0.3, 0.16)
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
    if 20 < frame < 100:
        mpm.add_cube(lower_corner=[1, 1 , 2],
                     cube_size=[5.0, 4.0, 0.1],
                     material=MPMSolver.material_water,
                     velocity=[0, 0, 100 ])
        #100*math.sin(frame * 0.1)
    render()
    show_options()
    print()
    window.show()
