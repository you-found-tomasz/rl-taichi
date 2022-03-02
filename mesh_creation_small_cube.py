import numpy as np

v1 = np.array([0, 0, 0])
v2 = np.array([1, 0, 0])
v3 = np.array([0, 1, 0])
v4 = np.array([1, 1, 0])
v5 = np.array([0, 0, 1])
v6 = np.array([1, 0, 1])
v7 = np.array([0, 1, 1])
v8 = np.array([1, 1, 1])

f1 = np.array([v1, v2, v6])
f2 = np.array([v1, v2, v4])
f3 = np.array([v2, v4, v6])
f4 = np.array([v1, v4, v6])



triangles = np.array([
    [f1[0][0], f1[0][1], f1[0][2], f1[1][0], f1[1][1], f1[1][2], f1[2][0], f1[2][1], f1[2][2]],
    [f2[0][0], f2[0][1], f2[0][2], f2[1][0], f2[1][1], f2[1][2], f2[2][0], f2[2][1], f2[2][2]],
    [f3[0][0], f3[0][1], f3[0][2], f3[1][0], f3[1][1], f3[1][2], f3[2][0], f3[2][1], f3[2][2]],
    [f4[0][0], f4[0][1], f4[0][2], f4[1][0], f4[1][1], f4[1][2], f4[2][0], f4[2][1], f4[2][2]]
], dtype=np.float32)

print('loaded')
np.save('2d_mesh_little_cube.npy', triangles)
triangles2 = np.load('2d_mesh_little_cube.npy', allow_pickle=True)