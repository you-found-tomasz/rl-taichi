#import dmsh
import meshio
#import optimesh
import numpy as np
import numpy as np
import pygalmesh

s = pygalmesh.Ball([2.5, 4, 2.5], 1.0)
mesh = pygalmesh.generate_mesh(s, max_cell_circumradius=0.2)


'''
geo = dmsh.Circle([2.5, 2.5], 1.0)
X, cells = dmsh.generate(geo, 0.5)

# optionally optimize the mesh
X, cells = optimesh.optimize_points_cells(X, cells, "CVT (full)", 1.0e-10, 100)

# visualize the mesh
dmsh.helpers.show(X, cells, geo)
'''

points = mesh


x = mesh.points[:,0]
y = mesh.points[:,1]
z = mesh.points[:,2]
triangles_ind = mesh.cells_dict['triangle']
num_tris = len(triangles_ind)
triangles = np.zeros((num_tris, 9), dtype=np.float32)

for i, face in enumerate(triangles_ind):
    assert len(face) == 3
    for d in range(3):
        triangles[i, d * 3 + 0] = x[face[d]]
        triangles[i, d * 3 + 1] = y[face[d]]
        triangles[i, d * 3 + 2] = z[face[d]]

print('loaded')
np.save('2d_mesh2.npy', triangles)
triangles2 = np.load('2d_mesh2.npy', allow_pickle=True)