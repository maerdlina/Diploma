import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.ndimage import zoom
import matplotlib
matplotlib.use('TkAgg')  # Установка бэкенда

def beso(nu, length, width, volfrac, penal, rmin, ft):
    Emin = 1e-9
    Emax = 1.0

    # Определяем количество элементов на основе длины и ширины
    nelx = int(length * 50)  # Увеличиваем плотность сетки
    nely = int(width * 50)  # Увеличиваем плотность сетки

    # Ввод параметров нагрузки
    print("Введите координаты точки приложения нагрузки (в узлах):")
    fx = int(input(f"Координата x (от 0 до {nelx}): "))
    fy = int(input(f"Координата y (от 0 до {nely}): "))
    load_value = float(input("Введите значение нагрузки (положительное вниз): "))

    # Определяем число степеней свободы
    ndof = 2 * (nelx + 1) * (nely + 1)

    # Создаём начальные значения
    x = volfrac * np.ones(nely * nelx, dtype=float)
    xold = x.copy()
    xPhys = x.copy()
    g = 0

    # FE
    KE = lk_programmatically(nu)
    edofMat, iK, jK = generate_edof_and_stiffness(nelx, nely)

    # Фильтрация
    H, Hs = filter_matrix(nelx, nely, rmin)

    # Закрепления
    dofs = np.arange(2 * (nelx + 1) * (nely + 1))
    fixed = np.union1d(dofs[0:2 * (nely + 1):2], np.array([2 * (nelx + 1) * (nely + 1) - 1]))
    free = np.setdiff1d(dofs, fixed)

    # Нагрузки
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    load_dof = 2 * ((nely + 1) * fx + fy) + 1  # DOF для указанной точки (нагрузка по y)
    f[load_dof, 0] = load_value

    # Построение
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        zoom(-xPhys.reshape((nelx, nely)).T, 2, order=1),
        cmap='gray',
        interpolation='bicubic',
        norm=colors.Normalize(vmin=-1, vmax=0)
    )
    ax.xaxis.tick_top()
    ax.set_title("Оптимизация топологии")
    fig.colorbar(im, ax=ax, orientation="horizontal")
    fig.show()

    loop = 0
    change = 1
    dv = np.ones(nely * nelx)
    dc = np.ones(nely * nelx)
    ce = np.ones(nely * nelx)

    while change > 0.01 and loop < 100:
        loop += 1

        # Решение FE
        sK = ((KE.flatten()[np.newaxis]).T * (Emin + (xPhys) ** penal * (Emax - Emin))).flatten(order='F')
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K = K[free, :][:, free]
        u[free, 0] = spsolve(K, f[free, 0])

        # Чувствительность
        ce[:] = (np.dot(u[edofMat].reshape(nelx * nely, 8), KE) * u[edofMat].reshape(nelx * nely, 8)).sum(1)
        dc[:] = (-penal * xPhys ** (penal - 1) * (Emax - Emin)).reshape(nely * nelx) * ce
        dv[:] = np.ones(nely * nelx)

        # Фильтрация
        dc[:] = np.asarray((H * (x * dc))[np.newaxis].T / Hs)[:, 0] / np.maximum(0.001, x)

        # Критерий оптимальности
        xold[:] = x
        (x[:], g) = optimal(nelx, nely, x, volfrac, dc, dv, g)
        xPhys[:] = x

        change = np.linalg.norm(x.reshape(nelx * nely, 1) - xold.reshape(nelx * nely, 1), np.inf)

        # Вывод на экран
        im.set_array(zoom(-xPhys.reshape((nelx, nely)).T, 2, order=1))
        plt.pause(0.01)
        fig.canvas.draw()

        if loop % 5 == 0:
            plt.savefig(f'iteration_{loop}.png', dpi=300)

    plt.ioff()
    xx = x.reshape(nely, nelx, order='F')

    # Вывод результата
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.grid(False)
    im.set_array(zoom(-xx.T, 2, order=1))
    fig.canvas.draw()
    plt.show()

# Генерация матрицы жёсткости
def lk_programmatically(nu):
    E = 1
    a = 1 / 2 - nu / 6
    b = 1 / 8 + nu / 8
    c = -1 / 4 - nu / 12
    d = -1 / 8 + 3 * nu / 8
    e = -1 / 4 + nu / 12
    f = -1 / 8 - nu / 8
    g = nu / 6
    h = 1 / 8 - 3 * nu / 8
    KE = E / (1 - nu ** 2) * np.array([
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

# Генерация edof и глобальной жёсткости
def generate_edof_and_stiffness(nelx, nely):
    edofMat = np.zeros((nelx * nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edofMat[el, :] = np.array([2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n2 + 2, 2 * n2 + 3, 2 * n1 + 2, 2 * n1 + 3])
    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()
    return edofMat, iK, jK

# Фильтрация
def filter_matrix(nelx, nely, rmin):
    nfilter = int(nelx * nely * (2 * (np.ceil(rmin) - 1) + 1) ** 2)
    iH = np.zeros(nfilter, dtype=int)
    jH = np.zeros(nfilter, dtype=int)
    sH = np.zeros(nfilter)
    k = 0
    for i in range(nelx):
        for j in range(nely):
            row = i * nely + j
            for k1 in range(max(i - int(np.ceil(rmin)) + 1, 0), min(i + int(np.ceil(rmin)), nelx)):
                for k2 in range(max(j - int(np.ceil(rmin)) + 1, 0), min(j + int(np.ceil(rmin)), nely)):
                    col = k1 * nely + k2
                    fac = rmin - np.sqrt((i - k1) ** 2 + (j - k2) ** 2)
                    if fac > 0:
                        iH[k] = row
                        jH[k] = col
                        sH[k] = fac
                        k += 1
    H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsc()
    Hs = H.sum(1)
    return H, Hs

# Критерий оптимальности
def optimal(nelx, nely, x, volfrac, dc, dv, g):
    l1 = 0
    l2 = 1e9
    move = 0.2
    while (l2 - l1) / (l1 + l2) > 1e-3:
        lmid = 0.5 * (l2 + l1)
        xnew = np.maximum(0.001, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid)))))
        if np.sum(xnew) - volfrac * nelx * nely > 0:
            l1 = lmid
        else:
            l2 = lmid
    return xnew, g

if __name__ == "__main__":
    beso(nu=0.3, length=2, width=1, volfrac=0.4, penal=3.0, rmin=1.5, ft=0)
