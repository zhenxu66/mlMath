# REF  https://github.com/ageron/handson-ml/

from __future__ import division, print_function, unicode_literals

# NumPy's ndarray provides a lot of convenient and optimized implementations of
# essential mathematical operations on vectors

#                                          1.Vectors
import numpy as np
vector0 = np.array([1, 2, 3, 4])
print(vector0.size)
print(vector0.shape)
print(vector0[0])

#                                   visualization in 2D and 3D #
import matplotlib.pyplot as plt
#%matplotlib inline

vector1 = np.array([1, 4])
vector2 = np.array([4, 1])

x_coords, y_coords = zip(vector1, vector2)
plt.scatter(x_coords, y_coords, marker='*', color=['r', 'b'])
plt.axis([0, 5, 0, 5])
plt.grid()
plt.show()

def plot_vector2d(vector2d, origin=[0, 0], **options):
    return plt.arrow(origin[0], origin[1], vector2d[0], vector2d[1], head_width=0.2, head_length=0.3,
                     length_includes_head=True, **options)

plot_vector2d(vector1, color="r")
plot_vector2d(vector2, color="b")
plt.axis([0, 9, 0, 6])
plt.grid()
plt.show()

from mpl_toolkits.mplot3d import Axes3D

vector3 = np.array([1, 4, 8])
vector4 = np.array([4, 1, 8])

subplot3d = plt.subplot(111, projection='3d')
x_coords, y_coords, z_coords = zip(vector3, vector4)
subplot3d.scatter(x_coords, y_coords, z_coords)
subplot3d.set_zlim3d([0, 9])
plt.show()

def plot_vectors3d(ax, vectors3d, z0, **options):
    for v in vectors3d:
        x, y, z = v
        ax.plot([x,x], [y,y], [z0, z], color="gray", linestyle='dotted', marker=".")
    x_coords, y_coords, z_coords = zip(*vectors3d)
    ax.scatter(x_coords, y_coords, z_coords, **options)

subplot3d = plt.subplot(111, projection='3d')
subplot3d.set_zlim([0, 9])
plot_vectors3d(subplot3d, [vector3, vector4], 0, color=("r", "b"))
plt.show()

#                                    math norm and plot

import numpy.linalg as LA
print(LA.norm(np.array([3, 4])))

radius = LA.norm(np.array([3, 4]))
plt.gca().add_artist(plt.Circle((0,0), radius, color="#DDDDDD"))
plot_vector2d(np.array([3, 4]), color="red")
plt.axis([0, 6, 0, 6])
plt.grid()
plt.show()

#          math vector addition/geometric translation/scalar zoom/Normalize and plot

plot_vector2d(vector1, color="r")
plot_vector2d(vector2, color="b")
plot_vector2d(vector1, origin=vector2, color="b", linestyle="dotted")
plot_vector2d(vector2, origin=vector1, color="r", linestyle="dotted")
plot_vector2d(vector1+vector2, color="g")
plt.axis([0, 9, 0, 7])
plt.text(0.7, 3, "vector1", color="r", fontsize=18)
plt.text(4, 3, "vector1", color="r", fontsize=18)
plt.text(1.8, 0.2, "vector2", color="b", fontsize=18)
plt.text(3.1, 5.6, "vector2", color="b", fontsize=18)
plt.text(2.4, 2.5, "vector1+vector2", color="g", fontsize=18)
plt.grid()
plt.show()

# geometric translation

t1 = np.array([2, 0.25])
t2 = np.array([2.5, 3.5])
t3 = np.array([1, 2])
geo_trans = np.array([2, 1])

x_coords, y_coords = zip(t1, t2, t3, t1)
plt.plot(x_coords, y_coords, "c--", x_coords, y_coords, "co")

plot_vector2d(geo_trans, t1, color="r", linestyle=":")
plot_vector2d(geo_trans, t2, color="r", linestyle=":")
plot_vector2d(geo_trans, t3, color="r", linestyle=":")

t1b = t1 + geo_trans
t2b = t2 + geo_trans
t3b = t3 + geo_trans

x_coords_b, y_coords_b = zip(t1b, t2b, t3b, t1b)
plt.plot(x_coords_b, y_coords_b, "b-", x_coords_b, y_coords_b, "bo")

plt.text(4, 4.2, "v", color="r", fontsize=18)
plt.text(3, 2.3, "v", color="r", fontsize=18)
plt.text(3.5, 0.4, "v", color="r", fontsize=18)

plt.axis([0, 6, 0, 5])
plt.grid()
plt.show()

#  Zoom by scalar

k = 2.5
t1c = k * t1
t2c = k * t2
t3c = k * t3

plt.plot(x_coords, y_coords, "c--", x_coords, y_coords, "co")

plot_vector2d(t1, color="r")
plot_vector2d(t2, color="r")
plot_vector2d(t3, color="r")

x_coords_c, y_coords_c = zip(t1c, t2c, t3c, t1c)
plt.plot(x_coords_c, y_coords_c, "b-", x_coords_c, y_coords_c, "bo")

plot_vector2d(k * t1, color="b", linestyle=":")
plot_vector2d(k * t2, color="b", linestyle=":")
plot_vector2d(k * t3, color="b", linestyle=":")

plt.axis([0, 9, 0, 9])
plt.grid()
plt.show()

# normalize

plt.gca().add_artist(plt.Circle((0,0),1,color='c'))
plt.plot(0, 0, "ko")
plot_vector2d(vector2 / LA.norm(vector2), color="k")
plot_vector2d(vector2, color="b", linestyle=":")
plt.text(0.3, 0.3, "$\hat{vector2}$", color="k", fontsize=18)
plt.text(1.5, 0.7, "$vector2$", color="b", fontsize=18)
plt.axis([-1.5, 5.5, -1.5, 3.5])
plt.grid()
plt.show()

#                     Dot product & angle calculation of vectors & projection in vector

print(np.dot(vector1, vector2))
print("Element wise calculation with *" + str(vector1*vector2))

def vector_angle(u, v):
    cos_theta = u.dot(v) / LA.norm(u) / LA.norm(v)
    return np.arccos(np.clip(cos_theta, -1, 1))  # make sure arccos do not receive wrong para, use clip to limit range

theta = vector_angle(vector1, vector2)
print("Angle =", theta, "radians")
print("      =", theta * 180 / np.pi, "degrees")

# project onto some other vector, dot with normalized vector and * same normalized vector
vector1_normalized = vector1 / LA.norm(vector1)
proj2on1 = vector2.dot(vector1_normalized) * vector1_normalized

plot_vector2d(vector1, color="r")
plot_vector2d(vector2, color="b")

plot_vector2d(proj2on1, color="k", linestyle=":")
plt.plot(proj2on1[0], proj2on1[1], "ko")

plt.plot([proj2on1[0], vector2[0]], [proj2on1[1], vector2[1]], "b:")

plt.text(1, 2, "$proj_1 2$", color="k", fontsize=18)
plt.text(1.8, 0.2, "$vector2$", color="b", fontsize=18)
plt.text(0.8, 3, "$vector1$", color="r", fontsize=18)

plt.axis([0, 8, 0, 5.5])
plt.grid()
plt.show()

#                                          2.Matrices (upper case preferred)


M0 = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])
print(M0.shape)

# No vertical or horizontal one-dimensional array. (ie. a 2D NumPy array), or a column vector as a one-column matrix
# then you need to use a slice instead of an integer when accessing the row or column

print(M0[1, :])
print(M0[1:2, :])

# math diagonal matrix and identity matrix
print(np.diag([10, 50, 90]))
print(np.diag(M0))

print(np.eye(4))

# math: Matrix multiplication .dot()
# * operator performs elementwise multiplication, NOT a matrix multiplication

print(M0[0:2, :].shape)
print(M0[:, 0:2].shape)
print(M0[0:2, :].dot(M0[:, 0:2]))

try:
    M0[0:2, :].dot(M0[0:2, :])
except ValueError as e:
    print("ValueError:", e)

print(np.array_equal(M0, M0.dot(np.eye(3))))


import sys
print("Python version: {}.{}.{}".format(*sys.version_info))
print("Numpy version:", np.version.version)

# Uncomment the following line if your Python version is ≥3.5
# and your NumPy version is ≥1.10:

#A @ D
print(M0[0:2, :] @ (M0[:, 0:2]))  # also work for vector dot

# Math: Matrix transpose and multiplication properties, symmetric matrix
# The product of a matrix by its transpose is always a symmetric matrix
print(M0.T)
print((M0[0:2, :].dot(M0[:, 0:2])).T)
print(M0[:, 0:2].T.dot(M0[0:2, :].T))

print(M0.dot(M0.T))

print(np.array([vector1]).T)  # convert a row vector matrix to vertical vector matrix

# Math: Inverse of matrix
# Math: determinant
# One of the main uses of the determinant is to determine whether a square matrix can be inversed or not:
# if the determinant is equal to 0, then the matrix cannot be inversed (it is a singular matrix),
# and if the determinant is not 0, then it can be inversed.

try:
    print(LA.inv(M0))
except LA.LinAlgError as e:
    print("LinAlgError:", e)

print(LA.det(M0))

# Math: Singular Value Decomposition
# It turns out that any m*n matrix M can be decomposed into the dot product of three simple matrices:
# m*m orthogonal matrix for rotation, m*n diagonal matrix for scaling & projecting, n*n orthogonal matrix for rotation

F_shear = np.array([
        [1, 1.5],
        [0, 1]
    ])

U, S_diag, V_T = LA.svd(F_shear)

print(np.array_equal(F_shear, np.around(U.dot(np.diag(S_diag)).dot(V_T), 1)))

# Math: Eignevectors and eigenvalues
# NumPy's eig function returns the list of unit eigenvectors and their corresponding eigenvalues for any square matrix.

eigenvalues2, eigenvectors2 = LA.eig(F_shear)
print(eigenvalues2) # [λ0, λ1, …]
print(np.around(eigenvectors2, 1))

# Math: Trace is the sum of the values on its main diagonal
# have a useful geometric interpretation in the case of projection matrices
# corresponds to the number of dimensions after projection

D = np.array([
        [100, 200, 300],
        [ 10,  20,  30],
        [  1,   2,   3],
    ])
print(np.trace(D))

#              Matrix Plot    Geometric applications of matrix operations

