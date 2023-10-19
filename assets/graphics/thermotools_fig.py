"""
This script was used to generate the header graphic (the nice flowy lines) it contains:

ColorGradient: A custom colormapping taking at least two colors to create a gradient effect
Colomap2D: A custom colormap class that takes several ColorGradients to create a 2d colormap

f, g, h : Functions that create fancy waves

Some plotting at the bottom to generate header.pdf

Feel free to play around with nice colors :)
Tip: Because a huge number of lines are used to create the gradient effects, this runs quite slowly. Reduce the resolution
in `xl` and the `nz` lists to get reasonable runtime while playing around.
"""
import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
from tools import ColorGradient, Colormap2D

norm_rgb = lambda r, g, b : tuple(x / 255 for x in (r, g, b))

grad1 = ColorGradient([(0, 0, 1)  , (1, 0.5, 1), (1, 1, 1), (1, 1, 0), (1, 0, 0)])
grad2 = ColorGradient([(0, 1, 1)  , (0, 1, 0.5), (1, 1, 1), (0.985, 0.06, 0.75), (1, 0, 1)])
grad3 = ColorGradient([(0.5, 0, 1), (0, 0, 1), (1, 1, 1), (0.5, 0, 1), (1, 0, 0)])

cmap2d = Colormap2D([grad1, grad2, grad3])
# cmap2d.display()
# exit(0)

def f1(x, z):
    return sin(x + z) * 0.5 * x + np.sqrt(abs(z - 0.5))

def f2(x, z):
    return (z - 0.5) * cos(x) * 0.5 * x**2 + 0.5 * sin(z)

flist = [f1, f2]
def f(x, z):
    return [fi(x, z) for fi in flist]

def g1(x, z):
    return sin(x / (z + 0.5)) * cos(x - 3 * z)**2 + (z - 0.5) * x

def g2(x, z):
    return cos(x * (z - 0.5)) + 0.5 * x * z + sin(z)

glist = [g1, g2]
def g(x, z):
    return [gi(x, z) for gi in glist]

def h1(x, z):
    return -cos(x**2 * (z - 0.5)**3) - 0.5 * (z - 0.5) * abs(x) + z**2 + 0.2 * x**2 * cos((0.5 * z + 0.5) * x)

def h2(x, z):
    return cos((x - sin(x)) + z) + 0.3 * x**2 + z

hlist = [h1, h2]
def h(x, z):
    return [hi(x, z) for hi in hlist]

if __name__ == '__main__':
    xl = np.linspace(-3, 3, 100)

    plt.figure(figsize=(10, 5))

    nz = 50
    funclist = [h]# f, h, g]
    alpha_func = lambda xi: max(np.tanh(abs(0.15 * xi)) - 0.1, 0.05)
    norm = lambda x: (x - min(xl)) / (max(xl) - min(xl))
    zl = np.linspace(0, 1, nz)
    for fi, func in enumerate(funclist):
        for z in zl:
            for i in range(len(xl) - 1):
                alpha = alpha_func(xl[i])
                funcvals = func(xl[i : i + 2], z)
                for val in funcvals:
                    plt.plot(xl[i : i + 2], val, color=cmap2d(norm(xl[i]), z)
                         , alpha=alpha)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.gca().set_facecolor('black')

    plt.xlim(min(xl), max(xl))
    plt.ylim(-2, 3)
    # plt.savefig('toolfig.pdf')
    # plt.savefig('toolfig.png', dpi=1200)
    plt.show()