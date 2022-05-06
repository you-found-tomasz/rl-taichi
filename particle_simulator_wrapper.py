import matplotlib.pyplot as plt
import taichi as ti
import numpy as np
from engine.mpm_solver import MPMSolver
import os
import optimesh
import dmsh
from os.path import exists
import time

class Particle_Simulator:

    def __init__(self):
        self.center = [2.5, 2.5]
        mesh_file = "mesh_data/2d_mesh_1015.npy"
        if not exists(mesh_file):
            geo = dmsh.Circle(self.center, 1)
            precision = 0.15
            self.X, cells = dmsh.generate(geo, precision)
            plt.scatter(self.X[:,0], self.X[:,1])
            plt.show()

            # optionally optimize the mesh
            self.X, cells = optimesh.optimize_points_cells(self.X, cells, "CVT (full)", 1.0e-10, 100)
            np.save("2d_mesh_{}.npy".format(self.X.shape[0]), self.X)

        self.X = np.load(mesh_file)
        self.simulation_show = True
        self.cloth_broken = False

        X_centered = self.X - self.center
        self.first_quarter_index = np.where(np.logical_and(X_centered[:,0] >= 0, X_centered[:,1] >= 0))[0]
        self.first_quarter = self.X[self.first_quarter_index,:]
        self.counter_max = self.first_quarter.shape[0]
        self.counter = 0
        self.particle_indices = np.ones(self.counter_max)

        #ti.init(arch=ti.gpu, device_memory_GB=2.0)
        #ti.init(arch=ti.gpu, device_memory_fraction=0.9)
        #ti.init(arch=ti.vulkan)
        self.write_to_disk = False
        ti.init(arch=ti.cpu, kernel_profiler = True, debug=True)
        self.gui = ti.GUI("Taichi Elements", res=512, background_color=0x112F41)
        self.mpm = MPMSolver(res=(64, 64, 64), size=10, max_num_particles=2 ** 15, use_ggui=False, use_g2p2g=False)
        self.mpm.add_sphere_collider_inv(center=(2.5, 4, 2.5), radius=1, surface=self.mpm.surface_sticky)
        self.mpm.set_gravity((0, -200, 0))
        self.mpm.step(4e-3, print_stat=False)

    def update(self, index, action):
        self.particle_indices[index] = action
        #print(self.first_quarter[index,:])
        state = self.first_quarter[index,:]
        return np.ndarray(shape=(2,), buffer=np.array([state[0], state[1]]))#, dtype=np.float32)

    def simulate(self):
        #first_quarter_reduced = self.first_quarter[np.where(self.particle_indices)[0],:]
        first_quarter_reduced = self.first_quarter
        first_quarter_reduced = self.first_quarter[np.where(self.particle_indices)[0],:]
        second_quarter_reduced = -first_quarter_reduced + self.center + self.center
        third_quarter_reduced = np.array([first_quarter_reduced[:,0], -first_quarter_reduced[:,1] + self.center[1] + self.center[1]]).T
        forth_quarter_reduced = np.array([-first_quarter_reduced[:,0] + self.center[1] + self.center[1], first_quarter_reduced[:,1]]).T

        particles = np.concatenate([first_quarter_reduced, second_quarter_reduced, third_quarter_reduced, forth_quarter_reduced], axis=0)
        particles_reduced = np.array([particles[:,0], np.ones(len(particles))*4, particles[:,1]]).T
        self.mpm.n_particles[None] = 0
        self.mpm.x.parent().deactivate_all()
        self.mpm.add_particles(particles=particles_reduced[:, :], material=MPMSolver.material_elastic)
        self.mpm.add_ellipsoid(center=[2.5, 4.5, 2.5], radius=0.25, material=MPMSolver.material_elastic, velocity=[0, -30, 0])
        min_list = list()
        begin_t = time.time()
        for frame in range(100):
            self.mpm.step(8e-3, print_stat=False)
            colors = np.array([0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00], dtype=np.uint32)
            particles = self.mpm.particle_info()
            np_x = particles['position'] / 10.0
            min_z = np.min(np_x[:,1])
            if min_z < 0.28:
                print("broken")
                self.cloth_broken = True
                break
            min_list.append(min_z)

            if self.simulation_show == True:
                # simple camera transform
                screen_x = ((np_x[:, 0] + np_x[:, 2]) / 2**0.5) - 0.2
                screen_y = (np_x[:, 1])

                screen_pos = np.stack([screen_x, screen_y], axis=-1)

                self.gui.circles(screen_pos, radius=1, color=colors[particles['material']])
                self.gui.show(f'{frame:06d}.png' if self.write_to_disk else None)

        print(f'  frame time {time.time() - begin_t:.3f} s')
        '''
        plt.plot(np.asarray(min_list))
        plt.show()
        plt.scatter(first_quarter_reduced[:,0], first_quarter_reduced[:,1])
        plt.scatter(second_quarter_reduced[:,0], second_quarter_reduced[:,1])
        plt.scatter(third_quarter_reduced[:,0], third_quarter_reduced[:,1])
        plt.scatter(forth_quarter_reduced[:,0], forth_quarter_reduced[:,1])
        plt.show()
        print("finished")
        '''