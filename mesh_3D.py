import taichi as ti
import numpy as np
from engine.mpm_solver import MPMSolver
from utils.utils import Mesh
from utils.utils import Ball
from utils.utils import show_options

ti.init(arch=ti.gpu, device_memory_fraction=0.9)
#ti.init(arch=ti.vulkan)
#ti.init(arch=ti.cpu)
#ti.init(arch=ti.gpu, device_memory_GB=4.0)

#initialisation
factor = 4
height = 6
center = 5
step_size = 4e-3
global particles_radius

mesh = Mesh(factor=factor, height=height, center=center)
ball = Ball(factor=factor, height=height, center=center)
mesh_ti = ti.Matrix.field(n=3, m=1, dtype=ti.f32, shape=(mesh.particles_nr))
other_ti = ti.Matrix.field(n=3, m=1, dtype=ti.f32, shape=(ball.particles_nr))

# setup mpm
mpm = MPMSolver(res=(64, 64, 64), size=10, max_num_particles=2 ** 20, use_ggui=True)
mpm.add_sphere_collider_inv(center=(center, height, center), radius=factor-0.5,surface=mpm.surface_sticky)
mpm.set_gravity((0, -50, 0))
mpm.add_particles(particles=mesh.particles,material=MPMSolver.material_elastic)
mpm.add_particles(particles=ball.particles,material=MPMSolver.material_elastic)

# Rendering settings
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

    split(mpm.x)
    strain_calc(mpm.x, mesh.triangles_ti)
    set_color(particle_color_ti, strain_ti)
    #scene.particles(mpm.x, radius=particles_radius)
    scene.particles(other_ti, radius=particles_radius)
    scene.mesh(mesh_ti, indices=mesh.triangles_reshaped_ti, per_vertex_color=particle_color_ti, two_sided=False)

    scene.point_light(pos=(5.0, 6.5, 5.0), color=(1.0, 1.0, 1.0))
    scene.point_light(pos=(5.0, 0.0, 5.0), color=(1.0, 1.0, 1.0))
    scene.point_light(pos=(5.5, 5.5, 5.5), color=(1.0, 1.0, 1.0))
    scene.point_light(pos=(5.5, 0.0, 5.5), color=(1.0, 1.0, 1.0))

    canvas.scene(scene)

@ti.kernel
def split(mpm_x: ti.template()):
    for i in ti.grouped(mesh_ti):
        mesh_ti[i] = mpm_x[i]
    for i in ti.grouped(other_ti):
        other_ti[i] = mpm_x[i + mesh.particles_nr]

baseline_length_ti = ti.Matrix.field(n=3, m=1, dtype=ti.f32, shape=(mesh.triangles_nr))
@ti.kernel
def base_strain_calc(mpm_x: ti.template(), triangles_ti: ti.template()):
    for I in ti.grouped(triangles_ti):

        baseline_length_ti[I][0] = (mpm_x[triangles_ti[I][ 0]] - mpm_x[triangles_ti[I][ 1]]).norm()
        baseline_length_ti[I][1] = (mpm_x[triangles_ti[I][ 0]] - mpm_x[triangles_ti[I][ 2]]).norm()
        baseline_length_ti[I][2] = (mpm_x[triangles_ti[I][ 1]] - mpm_x[triangles_ti[I][ 2]]).norm()

strain_ti = ti.Matrix.field(n=3, m=1, dtype=ti.f32, shape=(mesh.triangles_nr))
updated_length_ti = ti.Matrix.field(n=3, m=1, dtype=ti.f32, shape=(mesh.triangles_nr))
@ti.kernel
def strain_calc(mpm_x: ti.template(), triangles_ti: ti.template()):
    for I in ti.grouped(triangles_ti):
        updated_length_ti[I][0] = (mpm_x[triangles_ti[I][ 0]] - mpm_x[triangles_ti[I][ 1]]).norm()
        updated_length_ti[I][1] = (mpm_x[triangles_ti[I][ 0]] - mpm_x[triangles_ti[I][ 2]]).norm()
        updated_length_ti[I][2] = (mpm_x[triangles_ti[I][ 1]] - mpm_x[triangles_ti[I][ 2]]).norm()
        strain_ti[I] = updated_length_ti[I] / baseline_length_ti[I]
        #print(strain_ti[I])

neighbour_list = np.zeros([mesh.particles_nr,3])
for i in range(mesh.particles_nr):
    neighbour_indices = np.nonzero(mesh.triangles == i)
    try:
        len(neighbour_indices) != 0
        #neighbour_list.append(neighbour_indices[0][0])
        neighbour_list[i,0] = neighbour_indices[0][0]
        neighbour_list[i,1] = neighbour_indices[0][1]
        neighbour_list[i,2] = neighbour_indices[0][2]

    except:
        #neighbour_list.append(0)
        neighbour_list[i] = 0

neighbour_list_ti = ti.Matrix.field(n=3, m=1, dtype=ti.f32, shape=(mesh.particles_nr))
neighbour_list_ti.from_numpy(neighbour_list)

particle_color_ti = ti.Matrix.field(n=4, m=1, dtype=ti.f32, shape=(mesh.particles_nr))
@ti.kernel
def set_color(particle_color_ti: ti.template(), strain_ti:ti.template()):
    for I in ti.grouped(particle_color_ti):
        color_4d = ti.Vector([1.0, 1.0, 1.0, 1.0])
        strain_max_0 = ti.max(strain_ti[neighbour_list_ti[I][0]][0],strain_ti[neighbour_list_ti[I][0]][1],strain_ti[neighbour_list_ti[I][0]][2])
        strain_max_1 = ti.max(strain_ti[neighbour_list_ti[I][1]][0],strain_ti[neighbour_list_ti[I][1]][1],strain_ti[neighbour_list_ti[I][1]][2])
        strain_max_2 = ti.max(strain_ti[neighbour_list_ti[I][2]][0],strain_ti[neighbour_list_ti[I][2]][1],strain_ti[neighbour_list_ti[I][2]][2])
        strain_min = ti.min(ti.max(strain_max_0-1, strain_max_1-1,strain_max_2-1, 0), 1.0)
        color_4d[1] = color_4d[1] - strain_min
        color_4d[2] = color_4d[2] - strain_min
        particle_color_ti[I] = color_4d

if __name__ == "__main__":

    base_strain_calc(mpm.x, mesh.triangles_ti)

    for frame in range(1500):

        mpm.step(step_size)
        render()
        particles_radius = show_options(window, camera, mpm, particles_radius)
        window.show()