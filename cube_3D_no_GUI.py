import matplotlib.pyplot as plt
import taichi as ti
import numpy as np
from engine.mpm_solver import MPMSolver
import os
import optimesh
import dmsh


geo = dmsh.Circle([2.5, 2.5], 1)
X, cells = dmsh.generate(geo, 0.05)

# optionally optimize the mesh
X, cells = optimesh.optimize_points_cells(X, cells, "CVT (full)", 1.0e-10, 100)


# Try to run on GPU
#ti.init(arch=ti.gpu, device_memory_GB=4.0)
#ti.init(arch=ti.gpu, device_memory_fraction=0.9)
#ti.init(arch=ti.vulkan)
write_to_disk = False
ti.init(arch=ti.cpu)
gui = ti.GUI("Taichi Elements", res=512, background_color=0x112F41)

mpm = MPMSolver(res=(64, 64, 64), size=10, max_num_particles=2 ** 15, use_ggui=False)
mpm.add_sphere_collider_inv(center=(2.5, 4, 2.5),
                        radius=1,
                        surface=mpm.surface_sticky)
mpm.set_gravity((0, -150, 0))
particles = np.array([X[:,0], np.ones(len(X))*4, X[:,1]]).T
#particles_reduced = np.delete(particles, np.arange(0, particles.shape[0], 2), axis=0)
particles_reduced = particles[:int(3*particles.shape[0]/4),:]

mpm.add_particles(particles=particles_reduced,material=MPMSolver.material_elastic)
particles = mpm.particle_info()

min_list = list()
for frame in range(1500):
    mpm.step(4e-3)
    colors = np.array([0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00],
                      dtype=np.uint32)
    particles = mpm.particle_info()
    np_x = particles['position'] / 10.0
    min_z = np.min(np_x[:,1])
    min_list.append(min_z)

    # simple camera transform
    screen_x = ((np_x[:, 0] + np_x[:, 2]) / 2**0.5) - 0.2
    screen_y = (np_x[:, 1])

    screen_pos = np.stack([screen_x, screen_y], axis=-1)

    gui.circles(screen_pos, radius=1, color=colors[particles['material']])
    gui.show(f'{frame:06d}.png' if write_to_disk else None)

plt.plot(np.asarray(min_list))
plt.show()
