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

grad1 = ColorGradient([(0, 0, 1), (1, 1, 1), (1, 0, 0)])
grad2 = ColorGradient([(0, 1, 1), (0.75, 1, 0.75), (1, 1, 0)])
grad3 = ColorGradient([(0, 1, 0), (0, 1, 1), (1, 0, 1)])

cmap2d = Colormap2D([grad1, grad2, grad3])

def mod(x):
    return 1 # np.exp(-10 * x**2) - 1

def f1(x, z):
    return np.sin(x + z) * np.cos(x - z) * np.sin(x * z) / z

def f2(x, z):
    return z * np.sin(x * z) * np.cos(x**2 * z)**2

flist = [f1, f2]
def f(x, z):
    return (sum(fi(x, z) for fi in flist) + (z - 0.5)) * mod(x)

def g1(x, z):
    return sin(x * z) * cos(x - 3 * z)**2

def g2(x, z):
    return cos(x * (z - 0.5)) * cos(x * (z - 0.5))**3 + 0.5 * (z - 0.5)

glist = [g1, g2]
def g(x, z):
    return sum(gi(x, z) for gi in glist) * 5 * (z - 0.5)

def h1(x, z):
    return -cos(x**2 * (z - 0.5)**3) * cos(x * (z - 0.5))**3 - 0.5 * (z - 0.5)

def h2(x, z):
    return cos((x - sin(x)) * z) * (sin(x + cos(x)))**2

hlist = [h1, h2]
def h(x, z):
    return sum(hi(x, z) for hi in hlist) * 2

if __name__ == '__main__':
    xl = np.linspace(-3, 3, 200)

    fig = plt.figure(figsize=(10, 5))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    plt.sca(ax)

    xlims1 = lambda z : int(len(xl) * z / 3) + int(len(xl) / 6)
    xlims2 = lambda z : int(len(xl) * z * 2 / 3)
    nz = [100, 200, 200]
    funclist = [f, h, g]
    color_xshift = [0.8, 1, 0.6]
    alpha_list = [0.05, 0.05, 0.05]
    for fi, func in enumerate(funclist):
        zl = np.linspace(0, 1, nz[fi])
        for z in zl:
            for i in range(xlims1(z)):
                plt.plot(xl[i : i + 2], func(xl[i : i + 2], z), color=cmap2d((xl[i] / max(xl)) / color_xshift[fi], z)
                         , alpha=alpha_list[fi])
    nz = [nz[2], nz[0], nz[1]]
    funclist = [h, f, g]
    color_xshift = [1, 0.8, 0.6]
    for fi, func in enumerate(funclist):
        zl = np.linspace(0, 1, nz[fi])
        for z in zl:
            for i in range(xlims1(z), xlims2(z)):
                plt.plot(xl[i : i + 2], func(xl[i : i + 2], z), color=cmap2d((xl[i] / max(xl)) / color_xshift[fi], z)
                         , alpha=alpha_list[fi])

    nz = [nz[2], nz[0], nz[1]]
    funclist = [h, g, f]
    color_xshift = [1, 0.6, 0.8]
    for fi, func in enumerate(funclist):
        zl = np.linspace(0, 1, nz[fi])
        for z in zl:
            for i in range(xlims2(z), len(xl) - 1):
                plt.plot(xl[i : i + 2], func(xl[i : i + 2], z), color=cmap2d((xl[i] / max(xl)) / color_xshift[fi], z)
                         , alpha=alpha_list[fi])

    # NOTE: Because I couldn't figure out how to completely remove the green background from the header, the background
    #       figure needs to have white backing (not be transparent). Someone that knows more CSS than me can probably
    #       just remove the green background in `style.css` or something.
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.gca().set_facecolor('black')

    plt.xlim(min(xl), max(xl))
    plt.ylim(-3, 4)
    plt.savefig('thermopack.png', dpi=96)
    plt.show()