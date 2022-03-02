import taichi as ti
import numpy as np
from engine.mpm_solver import MPMSolver
import os

print(os.environ['VULKAN_SDK'])
print(os.environ['LD_LIBRARY_PATH'])
print(os.environ['VK_LAYER_PATH'])
print(os.environ['PATH'])
print(os.environ['TAICHI_CMAKE_ARGS'])

write_to_disk = False

# Try to run on GPU
#ti.init(arch=ti.gpu, device_memory_GB=4.0)
#ti.init(arch=ti.gpu, device_memory_fraction=0.9)
#ti.init(arch=ti.vulkan)
ti.init(arch=ti.cpu)


mpm = MPMSolver(res=(64, 64, 64), size=10, max_num_particles=2 ** 15, use_ggui=True)

mpm.add_cube(lower_corner=[0, 4, 0],
             cube_size=[5, 0.01, 5],
             material=MPMSolver.material_elastic,
             sample_density=64)


mpm.add_sphere_collider_inv(center=(2.5, 4, 2.5),
                        radius=2,
                        surface=mpm.surface_sticky)


mpm.set_gravity((0, -50, 0))
particles = mpm.particle_info()


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
camera.position(4.389, 9.5, -9.5)
camera.lookat(4.25, 1.89, 1.7)
camera.fov(55)
particles_radius = 0.05


def render():
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.ambient_light((0, 0, 0))
    set_color(mpm.color_with_alpha, material_type_colors, mpm.material)

    scene.particles(mpm.x, per_vertex_color=mpm.color_with_alpha, radius=particles_radius)

    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))

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

for frame in range(1500):
    mpm.step(4e-3)

    render()
    show_options()
    print()
    window.show()