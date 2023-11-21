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

grad1 = ColorGradient([(0, 0, 1)  , (1, 0.5, 1), (1, 1, 0.85), (1, 1, 0), (1, 0, 0)])
grad2 = ColorGradient([(0, 1, 1)  , (0, 1, 0.5), (0.85, 1, 0.85), (0.985, 0.06, 0.75), (1, 0, 1)])
grad3 = ColorGradient([(0.5, 0, 1), (0, 0, 1), (0.85, 0.85, 1), (0.5, 0, 1), (1, 0, 0)])

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

def w1(t):
    return 0.5 * (np.tanh(5 * (t - 0.3)) + 1)

def distrib(z):
    return 4 - 8 * z

def ft(x, z, offset=0):
    return distrib(z) + cos(10 * x - 5 * z) + offset

def gt(x, z, offset=0):
    return distrib(z) + 2 * sin(10 * x + 5 * z) + offset

def ht(x, z, offset=0):
    return distrib(z) + 5 * sin(15 * x * z) + offset

def tt_plot():
    xl = np.linspace(-3, 3, len(tlist) * 2)

    funclist = [tt_f, tt_h, tt_g]
    alpha_func = lambda xi: 0.15 * np.tanh(abs(0.95 * xi) - 3) + 0.16
    norm = lambda x: (x - min(xl)) / (max(xl) - min(xl))
    zl = zlist
    print(max([alpha_func(x) for x in xl]))
    for fi, func in enumerate(funclist):
        for z in zl:
            for i in range(len(xl) - 1):
                alpha = alpha_func(xl[i])
                plt.plot(xl[i: i + 2], func(xl[i: i + 2], z), color=cmap2d(norm(xl[i]), z)
                             , alpha=alpha, linewidth=linewidth)

def tt_sides():
    grad1 = ColorGradient([(0, 0.5, 1), (0, 1, 1), (0, 0, 1)])
    grad2 = ColorGradient([(0, 1, 1), (0, 0.85, 0.95), (0, 1, 1)])
    grad3 = ColorGradient([(0, 1, 0.5), (0, 1, 1), (0.5, 0, 1)])
    tt_cmap_l = Colormap2D([grad1, grad2, grad3])

    grad1 = ColorGradient([(1, 0, 0), (1, 0.6, 0), (1, 0.6, 0)][::-1])
    grad2 = ColorGradient([(1, 0, 1), (1, 1, 1), (1, 0, 0)][::-1])
    grad3 = ColorGradient([(1, 0, 0), (0.9, 1, 0), (0.985, 0.06, 0.75)][::-1])
    tt_cmap_r = Colormap2D([grad1, grad2, grad3])

    expand = 4

    tt_funclist_l = [lambda x, z : tt_f(- 6 + 3 * x, z) * w1(x) + expand * ft(x, z) * (1 - w1(x)),
                     lambda x, z : tt_h(- 6 + 3 * x, z) * w1(x) + expand * ht(x, z) * (1 - w1(x)),
                     lambda x, z : tt_g(- 6 + 3 * x, z) * w1(x) + expand * gt(x, z) * (1 - w1(x))]

    tt_funclist_r = [lambda x, z : tt_f(6 - 3 * x, z) * w1(x) + expand * ft(x, z) * (1 - w1(x)),
                     lambda x, z : tt_h(6 - 3 * x, z) * w1(x) + expand * ht(x, z) * (1 - w1(x)),
                     lambda x, z : tt_g(6 - 3 * x, z) * w1(x) + expand * gt(x, z) * (1 - w1(x))]

    for fi, (func_l, func_r) in enumerate(zip(tt_funclist_l, tt_funclist_r)):
        for z in zlist:
            for i in range(len(tlist) - 1):
                t = tlist[i: i + 2]
                fval_l = func_l(t, z)
                fval_r = func_r(t, z)
                plt.plot(-6 + 3 * t, fval_l, color=tt_cmap_l(t[0], z), alpha=0.1376, linewidth=linewidth)
                plt.plot(6 - 3 * t, fval_r, color=tt_cmap_r(t[0], z), alpha=0.1376, linewidth=linewidth)


if __name__ == '__main__':
    fig = plt.figure(figsize=(40, 5))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    plt.sca(ax)

    linewidth = 2

    tlist = np.linspace(0, 1, 150)
    zlist = np.linspace(0, 1, 200)
    salpha = 0.1

    tt_f = lambda x, z: f1(x, z) + f2(x, z) + 1
    tt_h = lambda x, z: h1(x, z) + h2(x, z)
    tt_g = lambda x, z: g1(x, z) + g2(x, z) - 1

    tt_plot()
    print('Finished plot')
    tt_sides()
    print('Finished sides')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.gca().set_facecolor('0.1')

    plt.xlim(-4, 4)
    plt.ylim(-3.5, 5)
    # plt.savefig('toolfig.pdf')
    plt.savefig('toolfig.png', dpi=96)
    plt.show()