import matplotlib.pyplot as plt
import taichi as ti
import numpy as np
from engine.mpm_solver import MPMSolver
import os
import optimesh
import dmsh
from os.path import exists
import time

@ti.kernel
def base_strain_calc(mpm_x: ti.template(), triangles_ti: ti.template(), baseline_length_ti: ti.template(), triangles_indices_ti: ti.template()):
    for I in ti.grouped(triangles_ti):
        #print(triangles_indices_ti[I])
        if triangles_indices_ti[I][0] == 1:
            baseline_length_ti[I][0] = ti.cast((mpm_x[triangles_ti[I][0]] - mpm_x[triangles_ti[I][1]]).norm(), ti.f32)
            baseline_length_ti[I][1] = ti.cast((mpm_x[triangles_ti[I][0]] - mpm_x[triangles_ti[I][2]]).norm(), ti.f32)
            baseline_length_ti[I][2] = ti.cast((mpm_x[triangles_ti[I][1]] - mpm_x[triangles_ti[I][2]]).norm(), ti.f32)
        else:
            baseline_length_ti[I][0] = 1
            baseline_length_ti[I][1] = 1
            baseline_length_ti[I][2] = 1

@ti.kernel
def split(mpm_x: ti.template(), mesh_ti: ti.template()):
    for i in ti.grouped(mesh_ti):
        mesh_ti[i] = mpm_x[i]

@ti.kernel
def strain_calc(mpm_x: ti.template(), triangles_ti: ti.template(), strain_ti: ti.template(), baseline_length_ti: ti.template(), \
    updated_length_ti: ti.template(), triangles_indices_ti:ti.template()):
    for I in ti.grouped(triangles_ti):
        if triangles_indices_ti[I][0] == 1:
            updated_length_ti[I][0] = (mpm_x[triangles_ti[I][ 0]] - mpm_x[triangles_ti[I][ 1]]).norm()
            updated_length_ti[I][1] = (mpm_x[triangles_ti[I][ 0]] - mpm_x[triangles_ti[I][ 2]]).norm()
            updated_length_ti[I][2] = (mpm_x[triangles_ti[I][ 1]] - mpm_x[triangles_ti[I][ 2]]).norm()
            #print(updated_length_ti[I], baseline_length_ti[I], mpm_x[triangles_ti[I][ 0]], mpm_x[triangles_ti[I][ 1]])
            strain_ti[I] = updated_length_ti[I] / baseline_length_ti[I]
            #print(strain_ti[I])
        else:
            strain_ti[I].fill(1)
@ti.kernel
def set_color(particle_color_ti: ti.template(), strain_ti:ti.template(), neighbour_list_ti:ti.template()):
    for I in ti.grouped(particle_color_ti):
        strain_max_0 = ti.max(strain_ti[neighbour_list_ti[I][0]][0],strain_ti[neighbour_list_ti[I][0]][1],strain_ti[neighbour_list_ti[I][0]][2])
        strain_max_1 = ti.max(strain_ti[neighbour_list_ti[I][1]][0],strain_ti[neighbour_list_ti[I][1]][1],strain_ti[neighbour_list_ti[I][1]][2])
        strain_max_2 = ti.max(strain_ti[neighbour_list_ti[I][2]][0],strain_ti[neighbour_list_ti[I][2]][1],strain_ti[neighbour_list_ti[I][2]][2])
        strain_min = ti.min(ti.max(strain_max_0-1, strain_max_1-1,strain_max_2-1, 0)*0.2, 1.0)
        #print(color_4d[2])
        particle_color_ti[I].fill(strain_min)

@ti.kernel
def update_triangle(triangles_ti: ti.template(),  triangles_indices_ti: ti.template(),  particle_indices_ti: ti.template()):
    for k in ti.grouped(ti.ndrange(particle_indices_ti.shape[0], triangles_ti.shape[0])):
        if particle_indices_ti[k[0]][0] == triangles_ti[k[1]][0] or particle_indices_ti[k[0]][0] == triangles_ti[k[1]][1] or particle_indices_ti[k[0]][0] == triangles_ti[k[1]][2]:
            triangles_indices_ti[k[1]].fill(1)

class Particle_Simulator:

    def __init__(self):

        self.factor = 1
        self.height = 6
        self.center = 5
        self.step_size = 8e-3
        self.colour_multiplier = 0.1
        self.counter_max = 1015

        self.mesh_file = "mesh_data/2d_mesh_1015.npy"
        self.triangles_file = "mesh_data/2d_mesh_1015_faces.npy"

        if not exists(self.mesh_file):
            geo = dmsh.Circle([self.center, self.center], 1)
            precision = 0.06
            self.X, cells = dmsh.generate(geo, precision)
            plt.scatter(self.X[:,0], self.X[:,1])
            plt.show()

            # optionally optimize the mesh
            self.X, cells = optimesh.optimize_points_cells(self.X, cells, "CVT (full)", 1.0e-10, 100)
            np.save("mesh_data/2d_mesh_{}_faces.npy".format(self.X.shape[0]), cells)
            np.save("mesh_data/2d_mesh_{}.npy".format(self.X.shape[0]), self.X)

            self.neighbour_list = np.zeros([self.X.shape[0], 3])
            for i in range(self.X.shape[0]):
                neighbour_indices = np.nonzero(cells == i)
                print(neighbour_indices[0])
                try:
                    len(neighbour_indices) != 0
                    # neighbour_list.append(neighbour_indices[0][0])
                    self.neighbour_list[i, 0] = neighbour_indices[0][0]
                    self.neighbour_list[i, 1] = neighbour_indices[0][1]
                    self.neighbour_list[i, 2] = neighbour_indices[0][2]

                except:
                    # neighbour_list.append(0)
                    self.neighbour_list[i] = 0
            np.save("mesh_data/2d_mesh_{}_neighbours.npy".format(self.X.shape[0]), self.neighbour_list)

        #X = np.load(self.mesh_file)
        mesh = np.load(self.mesh_file)
        self.neighbour_list = np.load("mesh_data/2d_mesh_1015_neighbours.npy")
        self.triangles = np.load(self.triangles_file)
        self.particles = np.array([mesh[:, 0]*self.factor, np.ones(len(mesh)) * self.height, mesh[:, 1]*self.factor]).T
        self.particle_indices = np.ones([self.particles.shape[0],1], dtype=np.int) #hardcoded number, full mesh size
        self.simulation_show = True
        self.cloth_broken = False

        #ti.init(arch=ti.gpu, device_memory_GB=2.0)
        #ti.init(arch=ti.gpu, device_memory_fraction=0.9)
        #ti.init(arch=ti.vulkan)
        ti.init(arch=ti.cpu, kernel_profiler=True)
        self.gui = ti.GUI("Taichi Elements", res=512, background_color=0x112F41)
        self.mpm = MPMSolver(res=(64, 64, 64), size=10, max_num_particles=2 ** 15, use_ggui=False, use_g2p2g=False, E_scale=0.1)
        self.mpm.add_sphere_collider_inv(center=(self.center, self.height, self.center), radius=self.factor-0.1, surface=self.mpm.surface_sticky)
        self.mpm.set_gravity((0, -50, 0))
        self.mpm.step(4e-3, print_stat=False)

        self.triangles_ti = ti.Matrix.field(n=3, m=1, dtype=ti.i32, shape=self.triangles.shape[0])
        self.triangles_ti.from_numpy(self.triangles)
        self.neighbour_list_ti = ti.Matrix.field(n=3, m=1, dtype=ti.f32, shape=(self.particles.shape[0]))
        self.neighbour_list_ti.from_numpy(self.neighbour_list)
        self.baseline_length_ti = ti.Matrix.field(n=3, m=1, dtype=ti.f32, shape=(self.triangles.shape[0]))
        self.strain_ti = ti.Matrix.field(n=3, m=1, dtype=ti.f32, shape=(self.triangles.shape[0]))
        self.updated_length_ti = ti.Matrix.field(n=3, m=1, dtype=ti.f32, shape=(self.triangles.shape[0]))
        self.triangles_indices_ti = ti.Matrix.field(n=1, m=1, dtype=ti.f32, shape=(self.triangles.shape[0]))
        self.particle_indices_ti = ti.Matrix.field(n=1, m=1, dtype=ti.f32, shape=(self.particles.shape[0]))


    def update(self, index, action):
        self.particle_indices[index] = action
        state = self.particles[index,:]
        return np.ndarray(shape=(2,), buffer=np.array([state[0], state[1]]))#, dtype=np.float32)

    def reset(self):
        self.particle_indices = np.ones([self.particles.shape[0], 1], dtype=np.int)  # hardcoded number, full mesh size

    def kill(self):
        ti.reset()

    def simulate(self):
        #ti.profiler.print_memory_profiler_info()
        self.mpm.n_particles[None] = 0
        self.mpm.x.parent().deactivate_all()
        self.particle_reduced = self.particles[np.where(self.particle_indices)[0],:]
        triangle_index = np.nonzero(self.particle_indices)[0]
        triangle_index_full = np.expand_dims(np.append(triangle_index,np.ones(self.particles.shape[0]-self.particle_reduced.shape[0])*triangle_index[0]),axis=1)
        self.particle_indices_ti.from_numpy(triangle_index_full)

        neighbour_list_reduced = self.neighbour_list[np.where(self.particle_indices)[0], :]
        update_triangle(self.triangles_ti, self.triangles_indices_ti, self.particle_indices_ti)
        hello23 = self.triangles_indices_ti.to_numpy()

        #self.neighbour_list_reduced_ti = ti.Matrix.field(n=3, m=1, dtype=ti.f32, shape=(self.particle_reduced.shape[0]))
        #self.neighbour_list_reduced_ti.from_numpy(neighbour_list_reduced)
        self.mpm.add_particles(particles=self.particle_reduced, material=MPMSolver.material_elastic)
        self.mpm.add_ellipsoid(center=[self.center, self.height + 0.25, self.center], radius=(self.factor/3.0), material=MPMSolver.material_elastic, velocity=[0,-50,0])
        self.other_particles_number = self.mpm.n_particles[None] - self.particle_reduced.shape[0]
        self.mesh_ti = ti.Matrix.field(n=3, m=1, dtype=ti.f32, shape=(self.particle_reduced.shape[0]))

        #ti.profiler.print_memory_profiler_info()

        #split(self.mpm.x, self.mesh_ti)

        self.baseline_length_ti = ti.Matrix.field(n=3, m=1, dtype=ti.f32, shape=(self.triangles.shape[0]))
        self.particle_color_ti = ti.Matrix.field(n=1, m=1, dtype=ti.f32, shape=(self.particle_reduced.shape[0]))
        base_strain_calc(self.mpm.x, self.triangles_ti, self.baseline_length_ti, self.triangles_indices_ti)
        min_list = list()
        begin_t = time.time()

        for frame in range(40):
            self.mpm.step(4e-3, print_stat=False)
            particles = self.mpm.particle_info()
            #split(self.mpm.x, self.mesh_ti)
            strain_calc(self.mpm.x, self.triangles_ti, self.strain_ti, self.baseline_length_ti, self.updated_length_ti, self.triangles_indices_ti)
            #set_color(self.particle_color_ti, self.strain_ti, self.neighbour_list_ti)
            strain_np = np.nanmax(self.strain_ti.to_numpy())
            #ti.profiler.print_memory_profiler_info()
            #ti.profiler.print_memory_profiler_info()

            np_x = particles['position'] / 10.0
            min_z = np.min(np_x[:,1])
            #if min_z < 0.28:
            if strain_np > 4.0 or min_z < 0.1:
            #if min_z < 0.28:
                #print("broken")
                self.cloth_broken = True
                break
            min_list.append(np.array([frame * 0.01, strain_np]))
            self.gui.text("Strain", [0.01, 0.99])
            self.gui.text("x-y mesh", [0.5, 0.99])
            self.gui.text("x-z mesh", [0.01, 0.5])
            self.gui.lines(np.array([[0.0, 0.75]]), np.array([[0.5, 0.75]]), radius=1)

            if self.simulation_show == True:
                # simple camera transform
                #screen_x = ((np_x[:, 0] + np_x[:, 2]) / 2**0.5)
                screen_x = ((np_x[:, 0]-(self.center/10))*3+self.center/10)
                screen_y = ((np_x[:, 1]-(self.center/10))*3+self.center/10)-0.3
                screen_x_top = ((np_x[:self.particle_reduced.shape[0], 0]-(self.center/10))*2+self.center/10 + 0.3)
                screen_y_top = ((np_x[:self.particle_reduced.shape[0], 2]-(self.center/10))*2+self.center/10 + 0.3)
                screen_pos = np.stack([np.concatenate((screen_x, screen_x_top)), np.concatenate((screen_y, screen_y_top))], axis=-1)
                color_np = 0xFFFFFF
                #strain_color = 0xFFFFFF - self.particle_color_ti.to_numpy()*255# -self.particle_color_ti.to_numpy()[:,1]*255*0xFF
                #color_np = np.concatenate((strain_color.squeeze(), np.ones([self.other_particles_number])*0xFFFFFF,strain_color.squeeze()))

                #self.gui.circles(screen_pos, radius=3), color=colors[particles['material']])
                self.gui.circles(screen_pos, color=color_np, radius=3)
                #self.gui.circles(np.array([[frame*0.01,0.75+(strain_np-1)*0.01]]), radius=2)
                #self.gui.lines(np.array([[0.0,0.75+0.4*0.5]]),np.array([[0.5,0.75+0.4*0.5]]), radius=1, color=0xFF0000)
                self.gui.circles((np.asarray(min_list) + [[0, -1]])*[[0.25, 0.05]] + [[0, 0.75]], radius=2)
                self.gui.show(f'{frame:06d}.png')


        print(f'  frame time {time.time() - begin_t:.3f} s')
        #ti.profiler.print_memory_profiler_info()

        #ti.profiler.print_kernel_profiler_info('trace')
        #ti.profiler.clear_kernel_profiler_info()
        #plt.plot(np.asarray(min_list))
        #plt.show()