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
#mpm.set_gravity((0, -350, 0)) correct
mpm.set_gravity((0, -550, 0))

mesh_particles = np.load('mesh_data/vertices_reduced.npy')
mesh_triangles = np.load('mesh_data/faces_reduced.npy')
#mesh_particles = np.load('2d_mesh_1015.npy')
#mesh_particles = mesh_particles + np.array([-2.5, -2.5])
mesh_nr = mesh_particles.shape[0]
triangles_nr = mesh_triangles.shape[0]

neighbour_list = np.zeros([mesh_nr,1])
for i in range(mesh_nr):
    neighbour_indices = np.nonzero(mesh_triangles == i)
    try:
        len(neighbour_indices) != 0
        #neighbour_list.append(neighbour_indices[0][0])
        neighbour_list[i] = neighbour_indices[0][0]
    except:
        #neighbour_list.append(0)
        neighbour_list[i] = 0

neighbour_list_ti = ti.Matrix.field(n=1, m=1, dtype=ti.f32, shape=(mesh_nr))
neighbour_list_ti.from_numpy(neighbour_list)

neighbour_indices_ti = ti.Matrix.field(n=3, m=1, dtype=ti.f32, shape=(triangles_nr))
@ti.kernel
def base_strain_calc(mpm_x: ti.template(), mesh_triangles: ti.ext_arr()):
    for I in ti.grouped(updated_length_ti):
        baseline_length_ti[I][0] = (mpm_x[mesh_triangles[I, 0]] - mpm_x[mesh_triangles[I, 1]]).norm()
        baseline_length_ti[I][1] = (mpm_x[mesh_triangles[I, 0]] - mpm_x[mesh_triangles[I, 2]]).norm()
        baseline_length_ti[I][2] = (mpm_x[mesh_triangles[I, 1]] - mpm_x[mesh_triangles[I, 2]]).norm()


#tensor_field = ti.Matrix.field(n=3, m=1, dtype=ti.f32, shape=(mesh_nr))

particles = np.array([mesh_particles[:,0]*factor+center, np.ones(len(mesh_particles))*height, mesh_particles[:,1]*factor+center]).T
#particles_reduced = np.delete(particles, np.arange(0, particles.shape[0], 5), axis=0)
mpm.add_particles(particles=particles,material=MPMSolver.material_elastic)
mpm.add_particles(particles=mesh.points,material=MPMSolver.material_elastic)
#mpm.add_ellipsoid(center=[center, 7.5, center], radius=1.2, material=MPMSolver.material_elastic, velocity=[0, 0, 0], sample_density=2**mpm.dim)

#triangluation
baseline_length_ti = ti.Matrix.field(n=3, m=1, dtype=ti.f32, shape=(triangles_nr))
@ti.kernel
def base_strain_calc(mpm_x: ti.template(), mesh_triangles: ti.ext_arr()):
    for I in ti.grouped(updated_length_ti):
        baseline_length_ti[I][0] = (mpm_x[mesh_triangles[I, 0]] - mpm_x[mesh_triangles[I, 1]]).norm()
        baseline_length_ti[I][1] = (mpm_x[mesh_triangles[I, 0]] - mpm_x[mesh_triangles[I, 2]]).norm()
        baseline_length_ti[I][2] = (mpm_x[mesh_triangles[I, 1]] - mpm_x[mesh_triangles[I, 2]]).norm()

particles = mpm.particle_info()
particle_nr = particles['position'].shape[0]

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

#strain_ti = ti.Matrix.field(n=3, m=1, dtype=ti.f32, shape=(mesh_triangles.shape[0]))
strain_ti = ti.Matrix.field(n=3, m=1, dtype=ti.f32, shape=(triangles_nr))
updated_length_ti = ti.Matrix.field(n=3, m=1, dtype=ti.f32, shape=(triangles_nr))
strain_ti_particles = ti.Matrix.field(n=3, m=1, dtype=ti.f32, shape=(mesh_nr))
@ti.kernel
def strain_calc(mpm_x: ti.template(), mesh_triangles: ti.ext_arr(), triangles: ti.template()):
    for I in ti.grouped(triangles):
        updated_length_ti[I][0] = (mpm_x[mesh_triangles[I, 0]] - mpm_x[mesh_triangles[I, 1]]).norm()
        updated_length_ti[I][1] = (mpm_x[mesh_triangles[I, 0]] - mpm_x[mesh_triangles[I, 2]]).norm()
        updated_length_ti[I][2] = (mpm_x[mesh_triangles[I, 1]] - mpm_x[mesh_triangles[I, 2]]).norm()
        strain_ti[I] = updated_length_ti[I] / baseline_length_ti[I]

#triangle_color_ti = ti.Matrix.field(n=4, m=1, dtype=ti.f32, shape=(mesh_triangles.shape[0]))
triangle_color_ti = ti.Matrix.field(n=4, m=1, dtype=ti.f32, shape=(mesh_nr))

@ti.kernel
def set_color2(triangle_color_ti: ti.template(), strain_ti:ti.template(), triangles: ti.template()):
    for I in ti.grouped(triangle_color_ti):
        color_4d = ti.Vector([1.0, 1.0, 1.0, 1.0])
        #strain = strain_ti[neighbour_list[I]]
        color_4d[1] = color_4d[1] - ti.min(ti.max((strain_ti[neighbour_list_ti[I]][0] - 1) * 2, (strain_ti[neighbour_list_ti[I]][1] - 1) * 2, (strain_ti[neighbour_list_ti[I]][2] - 1) * 2, 0), 1.0)
        color_4d[2] = color_4d[2] - ti.min(ti.max((strain_ti[neighbour_list_ti[I]][0] - 1) * 2, (strain_ti[neighbour_list_ti[I]][1] - 1) * 2, (strain_ti[neighbour_list_ti[I]][2] - 1) * 2, 0), 1.0)
        print(ti.min(ti.max((strain_ti[neighbour_list_ti[I]][0] - 1) * 4, (strain_ti[neighbour_list_ti[I]][1] - 1) * 4, (strain_ti[neighbour_list_ti[I]][2] - 1) * 4, 0), 1.0))
        #if (ti.max(strain_ti[I][1]) - 1) > 0.05:
        #    color_4d = ti.Vector([1.0, 0.0, 0.0, 1.0])
        #    print(ti.max(strain_ti[I][1]))
        #color_4d[0] = color_4d[0] + 0.5
        #color_4d[1] = color_4d[1] - ti.min(ti.max((strain_ti[I][0]-1)*8,(strain_ti[I][1]-1)*8,(strain_ti[I][2]-1)*8,0),1.0)
        #color_4d[2] = color_4d[2] - ti.min(ti.max((strain_ti[I][0]-1)*8,(strain_ti[I][1]-1)*8,(strain_ti[I][2]-1)*8,0),1.0)
        #print(ti.min(ti.max((strain_ti[I][0]-1)*8,(strain_ti[I][1]-1)*8,(strain_ti[I][2]-1)*8,0),1.0))
        #print(ti.min((ti.max(strain_ti[I][0]) - 1)*2, 0.5))
        #print(ti.max(strain_ti[I][1]))
        #color_4d = ti.Vector([1.0, 0.0, 0.0, 1.0])
        #if ti.max(strain_ti[I][1]) > 1.2:
            #color_4d = ti.Vector([1.0, 0.0, 0.0, 1.0])
            #color_4d[1] = color_4d[1] + ti.max(strain_ti[I][1]) - 1
        #print(I)
        triangle_color_ti[I] = color_4d


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
base_strain_calc(mpm.x, mesh_triangles)

def render():
    camera.track_user_inputs(window, movement_speed=0.2, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0, 0, 0))

    split(mpm.x)
    strain_calc(mpm.x, mesh_triangles, tensor_field)
    set_color2(triangle_color_ti, strain_ti, tensor_field)

    scene.particles(tensor_field2, radius=particles_radius)
    #scene.particles(mpm.x, radius=particles_radius, per_vertex_color=mpm.color_with_alpha)
    #scene.mesh(tensor_field2, indices=idx_array_tai2, two_sided=False)
    scene.mesh(tensor_field, indices=idx_array_tai, per_vertex_color=triangle_color_ti, two_sided=False)
    #scene.mesh(tensor_field, indices=idx_array_tai, two_sided=False)
    #scene.mesh(mpm.x, two_sided=True)

    scene.point_light(pos=(5.0, 6.5, 5.0), color=(1.0, 1.0, 1.0))
    scene.point_light(pos=(5.0, 0.0, 5.0), color=(1.0, 1.0, 1.0))
    scene.point_light(pos=(5.5, 5.5, 5.5), color=(1.0, 1.0, 1.0))
    scene.point_light(pos=(5.5, 0.0, 5.5), color=(1.0, 1.0, 1.0))

    canvas.scene(scene)


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
