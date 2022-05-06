import numpy as np
import pygalmesh
import taichi as ti

class Mesh ():

    def __init__(self, factor, height, center):
        mesh_particles = np.load('mesh_data/vertices_reduced.npy')
        self.triangles = np.load('mesh_data/faces_reduced.npy')
        self.particles_nr = mesh_particles.shape[0]
        self.triangles_nr = self.triangles.shape[0]
        self.particles = np.array([mesh_particles[:, 0] * factor + center, np.ones(len(mesh_particles)) * height,
                              mesh_particles[:, 1] * factor + center]).T
        triangles_reshaped = np.array(self.triangles).reshape(-1, 1).squeeze()
        self.triangles_reshaped_ti = ti.field(ti.i32, shape=len(triangles_reshaped))
        self.triangles_reshaped_ti.from_numpy(triangles_reshaped)
        self.triangles_ti = ti.Matrix.field(n=3, m=1, dtype=ti.i32, shape=self.triangles_nr)
        self.triangles_ti.from_numpy(self.triangles)

class Ball():

    def __init__(self, factor, height, center):
        s = pygalmesh.Ball([center, height + 1, center], 1.0)
        mesh = pygalmesh.generate_mesh(s, max_cell_circumradius=0.05)
        self.particles = mesh.points
        self.particles_nr = self.particles.shape[0]
        self.triangles = mesh.cells_dict['triangle']

        triangles_reshaped = np.array(self.triangles).reshape(-1, 1).squeeze()
        self.triangles_reshaped_ti = ti.field(ti.i32, shape=len(triangles_reshaped))
        self.triangles_reshaped_ti.from_numpy(triangles_reshaped)

def show_options(window, camera, mpm, particles_radius):
    window.GUI.begin("Solver Property", 0.05, 0.1, 0.2, 0.10)
    window.GUI.text(f"Current particle number {mpm.n_particles[None]}")
    particles_radius = window.GUI.slider_float("particles radius ", particles_radius, 0, 0.1)
    window.GUI.end()

    window.GUI.begin("Camera", 0.05, 0.2, 0.2, 0.16)
    camera.curr_position[0] = window.GUI.slider_float("camera pos x", camera.curr_position[0], -10, 10)
    camera.curr_position[1] = window.GUI.slider_float("camera pos y", camera.curr_position[1], -10, 10)
    camera.curr_position[2] = window.GUI.slider_float("camera pos z", camera.curr_position[2], -10, 10)

    camera.curr_lookat[0] = window.GUI.slider_float("camera look at x", camera.curr_lookat[0], -10, 10)
    camera.curr_lookat[1] = window.GUI.slider_float("camera look at y", camera.curr_lookat[1], -10, 10)
    camera.curr_lookat[2] = window.GUI.slider_float("camera look at z", camera.curr_lookat[2], -10, 10)

    window.GUI.end()

    return particles_radius


