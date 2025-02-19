import time
import math

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from cvxopt import matrix, spmatrix
from cvxopt.cholmod import linsolve

matplotlib.use('TkAgg')


class Load:
    def __init__(self, nelx, nely, E, nu):
        self.nelx = nelx
        self.nely = nely
        self.E = E
        self.nu = nu
        self.dim = 2
        print(f"Initialized Load with nelx={self.nelx}, nely={self.nely}, E={self.E}, nu={self.nu}, dim={self.dim}")

    # -------------------------- For ForceAndFix --------------------------------------------
    def nodeNum(self, elx, ely):
        node_number = (self.nely + 1) * elx + ely
        print(f"Node number for element ({elx}, {ely}): {node_number}")
        return node_number

    def nodes(self, elx, ely):
        n1 = self.nodeNum(elx, ely)
        n2 = self.nodeNum(elx + 1, ely)
        n3 = self.nodeNum(elx + 1, ely + 1)
        n4 = self.nodeNum(elx, ely + 1)
        print("Nodes of element (", elx, ",", ely, "): n1=", n1, "n2=", n2, "n3=", n3, "n4=", n4)
        return n1, n2, n3, n4

    def force(self):
        force_vector = np.zeros(self.dim * (self.nely + 1) * (self.nelx + 1))
        print(f"Initialized force vector: {force_vector}")
        return force_vector

    def print_initial_matrix(self):
        grid = np.zeros((self.nely + 1, self.nelx + 1), dtype=int)
        for i in range(self.nelx + 1):
            for j in range(self.nely + 1):
                grid[j, i] = self.nodeNum(i, j)
        print("Initial node matrix:")
        print(grid)


    # -------------------------- For CvxFEA --------------------------------------------

    def edof(self):
        elx = np.repeat(range(self.nelx), self.nely).reshape((self.nelx * self.nely, 1))
        ely = np.tile(range(self.nely), self.nelx).reshape((self.nelx * self.nely, 1))
        n1, n2, n3, n4 = self.nodes(elx, ely)
        edof = np.array([self.dim*n1, self.dim*n1+1, self.dim*n2, self.dim*n2+1,
                         self.dim*n3, self.dim*n3+1, self.dim*n4, self.dim*n4+1])
        edof = edof.T[0]
        x_list = np.repeat(edof, 8)
        y_list = np.tile(edof, 8).flatten()
        return edof, x_list, y_list

    def lk(self, E, nu):
        k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,
                      -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
        ke = E/(1-nu**2) * np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                                     [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                                     [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                                     [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                                     [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                                     [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                                     [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                                     [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
        return ke

    def alldofs(self):
        return [a for a in range(self.dim*(self.nely+1)*(self.nelx+1))]

    def fixdofs(self):
        return []

    def freedofs(self):
        return list(set(self.alldofs()) - set(self.fixdofs()))

class ForceAndFix(Load):
    def __init__(self, nelx, nely, E, nu):
        super().__init__(nelx, nely, E, nu)
        if nely % 2 != 0:
            raise ValueError('Nely needs to be even in a cantilever beam.')

    def force(self):
        f = super().force()
        n1, n2, n3, n4 = self.nodes(self.nelx, self.nely)
        f[self.dim * n1 + 1] = -1
        print(f"Applying force at node {n1} (DOF {self.dim * n1 + 1})")
        print(f"Updated force vector: {f}")
        return f

    def fixdofs(self):
        return ([x for x in range(0, self.dim*(self.nely+1))])


class FESolver:
    def displace(self, load, x, ke, penal):
        freedofs = np.array(load.freedofs())
        nely, nelx = x.shape
        f_free = load.force()[freedofs]
        k_free = self.gk_freedofs(load, x, ke, penal)
        u = np.zeros(load.dim*(nely+1)*(nelx+1))
        u[freedofs] = spsolve(k_free, f_free)
        return u

    def gk_freedofs(self, load, x, ke, penal):
        freedofs = np.array(load.freedofs())
        nelx = load.nelx
        nely = load.nely
        edof, x_list, y_list = load.edof()
        factor = x.T.reshape(nelx*nely, 1, 1) ** penal
        value_list = (np.tile(ke, (nelx*nely, 1, 1))*factor).flatten()
        dof = load.dim*(nelx+1)*(nely+1)
        k = coo_matrix((value_list, (y_list, x_list)), shape=(dof, dof)).tocsc()
        k = k[freedofs, :][:, freedofs]
        return k

class CvxFEA(FESolver):
    def displace(self, load, x, ke, penal):
        freedofs = np.array(load.freedofs())
        nely, nelx = x.shape
        f = load.force()
        Matrix_free = matrix(f[freedofs])
        k_free = self.gk_freedofs(load, x, ke, penal).tocoo()
        k_free = spmatrix(k_free.data, k_free.row, k_free.col)
        u = np.zeros(load.dim*(nely+1)*(nelx+1))
        linsolve(k_free, Matrix_free)
        print("CvxFEA.displace: ")
        print("k_free", k_free)
        print("Matrix free", Matrix_free)
        u[freedofs] = np.array(Matrix_free)[:, 0]
        return u


if __name__ == "__main__":
    # Material properties
    E = 1
    nu = 0.3

    # Mesh definition
    nelx = 3
    nely = 2


    print("Проверка ForceAndFix")
    # Applying load
    load = ForceAndFix(nelx, nely, E, nu)
    load.print_initial_matrix()  # Вывод изначальной матрицы

    # f = load.force()
    # fix = load.fixdofs()
    #
    # print("Force vector:")
    # print(f)
    # print(fix)

    penal = 3
    t = time.time()
    x = np.ones((nely, nelx))
    ke = load.lk(load.E, load.nu)

    print("Проверка CvxFEA")
    fesolver = CvxFEA()

    u = fesolver.displace(load, x, ke, penal)
    print(u)
    print('Time cost: ', time.time() - t, 'seconds.')