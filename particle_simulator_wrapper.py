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
        if not exists("2d_mesh.npy"):
            geo = dmsh.Circle(self.center, 1)
            precision = 0.035
            self.X, cells = dmsh.generate(geo, precision)

            # optionally optimize the mesh
            self.X, cells = optimesh.optimize_points_cells(self.X, cells, "CVT (full)", 1.0e-10, 100)
            np.save('2d_mesh.npy', self.X)

        self.X = np.load('2d_mesh.npy')
        self.simulation_show = False
        self.cloth_broken = False

        X_centered = self.X - self.center
        self.first_quarter_index = np.where(np.logical_and(X_centered[:,0] >= 0, X_centered[:,1] >= 0))[0]

        '''
        hello = self.X - self.center
        hello2 = np.where(np.logical_and(hello[:,0] >= 0, hello[:,1] >= 0))[0]
        hello5 = np.where(np.logical_and(hello[:,0] >= 0, hello[:,1] >= 0))[0]
        hello3 = self.X[hello2,:]
        hello4 = -hello3 + self.center + self.center
        hello6 = np.array([hello3[:,0], -hello3[:,1] + self.center[1] + self.center[1]]).T
        hello7 = np.array([-hello3[:,0] + self.center[1] + self.center[1], hello3[:,1]]).T

        #plt.scatter(self.X[:int(1926 / 2), 0], self.X[:int(1926 / 2), 1])
        #plt.scatter(self.X[int(1926/2):,0], self.X[int(1926/2):,1])
        #plt.scatter(hello2[:,0], hello2[:,1])
        dot=100
        plt.scatter(hello3[dot,0], hello3[dot,1])
        plt.scatter(hello4[dot,0], hello4[dot,1])
        plt.scatter(hello6[dot,0], hello6[dot,1])
        plt.scatter(hello7[dot,0], hello7[dot,1])
        plt.show()
        '''

        # Try to run on GPU
        #ti.init(arch=ti.gpu, device_memory_GB=4.0)
        #ti.init(arch=ti.gpu, device_memory_fraction=0.9)
        #ti.init(arch=ti.vulkan)
        self.write_to_disk = False
        ti.init(arch=ti.cpu)
        self.gui = ti.GUI("Taichi Elements", res=512, background_color=0x112F41)

    def simulate(self, particle_indices):

        mpm = MPMSolver(res=(64, 64, 64), size=10, max_num_particles=2 ** 15, use_ggui=False, use_g2p2g=False)
        mpm.add_sphere_collider_inv(center=(2.5, 4, 2.5), radius=1, surface=mpm.surface_sticky)
        mpm.set_gravity((0, -150, 0))
        '''
        X_centered = self.X - self.center
        first_quarter_index = np.where(np.logical_and(X_centered[:,0] >= 0, X_centered[:,1] >= 0))[0]
        '''

        first_quarter = self.X[self.first_quarter_index,:]
        first_quarter_reduced = first_quarter[np.where(particle_indices)[0],:]
        second_quarter_reduced = -first_quarter_reduced + self.center + self.center
        third_quarter_reduced = np.array([first_quarter_reduced[:,0], -first_quarter_reduced[:,1] + self.center[1] + self.center[1]]).T
        forth_quarter_reduced = np.array([-first_quarter_reduced[:,0] + self.center[1] + self.center[1], first_quarter_reduced[:,1]]).T
        '''
        plt.scatter(first_quarter[:,0], first_quarter[:,1])
        plt.scatter(second_quarter_reduced[:,0], second_quarter_reduced[:,1])
        plt.scatter(third_quarter_reduced[:,0], third_quarter_reduced[:,1])
        plt.scatter(forth_quarter_reduced[:,0], forth_quarter_reduced[:,1])
        plt.show()
        '''
        '''
        plt.scatter(first_quarter[:,0], first_quarter[:,1])
        plt.scatter(first_quarter_reduced[:,0], first_quarter_reduced[:,1])
        plt.show()
        '''

        '''
        particles = np.array([self.X[:,0], np.ones(len(self.X))*4, self.X[:,1]]).T
        #particles_reduced = np.delete(particles, np.arange(0, particles.shape[0], 2), axis=0)
        #particles_reduced = particles[:int(3*particles.shape[0]/4),:]
        particles_reduced = particles[np.where(particle_indices)[0],:]
        #particles_reduced = particles[:particle_indices,:]
        #particles_reduced = particles[:,:]
        '''
        particles = np.concatenate([first_quarter_reduced, second_quarter_reduced, third_quarter_reduced, forth_quarter_reduced], axis=0)
        particles_reduced = np.array([particles[:,0], np.ones(len(particles))*4, particles[:,1]]).T
        mpm.add_particles(particles=particles_reduced,material=MPMSolver.material_elastic)
        #mpm.clear_particles()


        min_list = list()
        begin_t = time.time()
        for frame in range(50):
            mpm.step(4e-3, print_stat=False)
            colors = np.array([0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00], dtype=np.uint32)
            particles = mpm.particle_info()
            np_x = particles['position'] / 10.0
            min_z = np.min(np_x[:,1])
            if min_z < 0.3:
                print(min_z)
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
        plt.plot(np.asarray(min_list))
        plt.show()
        plt.scatter(first_quarter_reduced[:,0], first_quarter_reduced[:,1])
        plt.scatter(second_quarter_reduced[:,0], second_quarter_reduced[:,1])
        plt.scatter(third_quarter_reduced[:,0], third_quarter_reduced[:,1])
        plt.scatter(forth_quarter_reduced[:,0], forth_quarter_reduced[:,1])
        plt.show()
            #if np.asarray(min_list) < 0.3:
