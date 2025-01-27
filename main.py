import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.widgets import Button
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.ndimage import zoom
import matplotlib
matplotlib.use('TkAgg')

# Метод BESO для оптимизации топологии балки
def beso_optimization(nu, length, width, volfrac, penal, rmin, fx, fy, load_value):
    Emin, Emax = 1e-9, 1.0
    nelx, nely = int(length * 50), int(width * 50)
    ndof = 2 * (nelx + 1) * (nely + 1)

    x = volfrac * np.ones(nely * nelx)
    xPhys = x.copy()
    loop, change, g = 0, 1, 0

    KE = generate_stiffness_matrix(nu)
    edofMat, iK, jK = generate_edof_and_stiffness_indices(nelx, nely)
    H, Hs = create_filter(nelx, nely, rmin)

    dofs = np.arange(ndof)
    fixed = np.union1d(dofs[0:2 * (nely + 1):2], np.array([2 * (nelx + 1) * (nely + 1) - 1]))
    free = np.setdiff1d(dofs, fixed)

    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    load_dof = 2 * (fy * (nely + 1) + fx) + 1
    f[load_dof, 0] = load_value

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(zoom(-xPhys.reshape((nely, nelx)).T, 2, order=1), cmap='gray', interpolation='bicubic')
    ax.set_title("Оптимизация топологии")
    fig.colorbar(im, ax=ax, orientation="horizontal")

    def stop_callback(event):
        nonlocal change
        change = 0

    stop_button_ax = fig.add_axes([0.4, 0.01, 0.2, 0.05])
    stop_button = Button(stop_button_ax, "Остановить")
    stop_button.on_clicked(stop_callback)

    plt.ion()
    plt.show()

    while change > 1e-3 and loop < 100:
        loop += 1

        sK = ((KE.flatten()[np.newaxis]).T * (Emin + xPhys**penal * (Emax - Emin))).flatten(order='F')
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K = K[free, :][:, free]

        u[free, 0] = spsolve(K, f[free, 0])

        ce = np.sum((u[edofMat].reshape(-1, 8) @ KE) * u[edofMat].reshape(-1, 8), axis=1)
        dc = -penal * xPhys**(penal - 1) * (Emax - Emin) * ce
        dc = H @ (dc / Hs.flatten())

        xPhys_old = xPhys.copy()
        l1, l2 = 0, 1e9
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            xNew = np.maximum(0.001, np.minimum(1.0, x * np.sqrt(-dc / lmid)))
            if xNew.sum() - volfrac * nelx * nely > 0:
                l1 = lmid
            else:
                l2 = lmid
        xPhys = xNew.copy()

        change = np.linalg.norm(xPhys - xPhys_old, np.inf)

        im.set_array(zoom(-xPhys.reshape((nely, nelx)).T, 2, order=1))
        plt.pause(0.01)

    plt.ioff()
    plt.show()

# Генерация матрицы жесткости элемента
def generate_stiffness_matrix(nu):
    E = 1.0
    a = 1/2 - nu/6
    b = 1/8 + nu/8
    c = -1/4 - nu/12
    d = -1/8 + 3*nu/8
    e = -1/4 + nu/12
    f = -1/8 - nu/8
    g = nu/6
    h = 1/8 - 3*nu/8
    KE = E / (1 - nu**2) * np.array([
        [a, b, c, d, e, f, g, h],
        [b, a, h, g, f, e, d, c],
        [c, h, a, f, g, d, e, b],
        [d, g, f, a, h, c, b, e],
        [e, f, g, h, a, b, c, d],
        [f, e, d, c, b, a, h, g],
        [g, d, e, b, c, h, a, f],
        [h, c, b, e, d, g, f, a]
    ])
    return KE

# Генерация индексов для МКЭ
def generate_edof_and_stiffness_indices(nelx, nely):
    edofMat = np.zeros((nelx * nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edofMat[el, :] = [
                2*n1, 2*n1+1, 2*n2, 2*n2+1,
                2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3
            ]
    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()
    return edofMat, iK, jK

# Создание фильтра для метода BESO
def create_filter(nelx, nely, rmin):
    nfilter = int(nelx * nely * (2 * (np.ceil(rmin) - 1) + 1)**2)
    iH, jH, sH = np.zeros(nfilter, dtype=int), np.zeros(nfilter, dtype=int), np.zeros(nfilter)
    k = 0
    for i in range(nelx):
        for j in range(nely):
            row = i * nely + j
            for k1 in range(max(i - int(np.ceil(rmin)) + 1, 0), min(i + int(np.ceil(rmin)), nelx)):
                for k2 in range(max(j - int(np.ceil(rmin)) + 1, 0), min(j + int(np.ceil(rmin)), nely)):
                    col = k1 * nely + k2
                    fac = rmin - np.sqrt((i - k1)**2 + (j - k2)**2)
                    if fac > 0:
                        iH[k], jH[k], sH[k] = row, col, fac
                        k += 1
    H = coo_matrix((sH, (iH, jH)), shape=(nelx*nely, nelx*nely)).tocsc()
    Hs = np.array(H.sum(axis=1)).flatten()
    return H, Hs

# Запуск оптимизации
if __name__ == "__main__":
    beso_optimization(nu=0.3, length=2.0, width=1.0, volfrac=0.4, penal=3.0, rmin=1.5, fx=0, fy=0, load_value=10)
