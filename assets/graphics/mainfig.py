from thermopack_fig import f as tp_f, g as tp_g, h as tp_h
from kingas_fig import f as kg_f, g as kg_g, h as kg_h, k as kg_k
from thermotools_fig import f1 as tt_f1, f2 as tt_f2, g1 as tt_g1, g2 as tt_g2, h1 as tt_h1, h2 as tt_h2
from thermopack_fig import cmap2d as tp_cmap
from kingas_fig import cmap2d as kg_cmap
from thermotools_fig import cmap2d as tt_cmap
from tools import ColorGradient, Colormap2D
import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt

linewidth = 2

def distrib(z):
    return 4 - 8 * z

def ft(x, z, offset=0):
    return distrib(z) + cos(10 * x - 5 * z) + offset

def gt(x, z, offset=0):
    return distrib(z) + 2 * sin(10 * x + 5 * z) + offset

def ht(x, z, offset=0):
    return distrib(z) + 5 * sin(15 * x * z) + offset

def kt(x, z, offset=0):
    return distrib(z) + 3 * cos(15 * x) * sin(5 * x) + offset

def tp_plot():
    xl = np.linspace(-3, 3, len(tlist) * 2)

    f = lambda x, z: tp_f(x, z) + offsets[1]
    g = lambda x, z: tp_g(x, z) + offsets[1]
    h = lambda x, z: tp_h(x, z) + offsets[1]

    cmap2d = tp_cmap

    xlims1 = lambda z: int(len(xl) * z / 3) + int(len(xl) / 6)
    xlims2 = lambda z: int(len(xl) * z * 2 / 3)
    funclist = [f, h, g]
    color_xshift = [0.8, 1, 0.6]
    alpha_list = [0.05, 0.05, 0.05]
    for fi, func in enumerate(funclist):
        zl = zlist
        for z in zl:
            for i in range(xlims1(z)):
                plt.plot(xl[i: i + 2], func(xl[i: i + 2], z), color=cmap2d((xl[i] / max(xl)) / color_xshift[fi], z)
                         , alpha=alpha_list[fi], linewidth=linewidth)

    funclist = [h, f, g]
    color_xshift = [1, 0.8, 0.6]
    alpha_list = [0.05, 0.05, 0.05]
    for fi, func in enumerate(funclist):
        zl = zlist
        for z in zl:
            for i in range(xlims1(z), xlims2(z)):
                plt.plot(xl[i: i + 2], func(xl[i: i + 2], z), color=cmap2d((xl[i] / max(xl)) / color_xshift[fi], z)
                         , alpha=alpha_list[fi], linewidth=linewidth)

    funclist = [h, g, f]
    color_xshift = [1, 0.6, 0.8]
    alpha_list = [0.05, 0.05, 0.05]
    for fi, func in enumerate(funclist):
        zl = zlist
        for z in zl:
            for i in range(xlims2(z), len(xl) - 1):
                plt.plot(xl[i: i + 2], func(xl[i: i + 2], z), color=cmap2d((xl[i] / max(xl)) / color_xshift[fi], z)
                         , alpha=alpha_list[fi], linewidth=linewidth)

def tp_sides():
    grad1 = ColorGradient([(0, 1, 1), (1, 1, 1), (0, 0, 1)])
    grad2 = ColorGradient([(0, 1, 0), (0.5, 1, 1), (0, 1, 1)])
    grad3 = ColorGradient([(0, 0, 1), (0, 1, 1), (0, 1, 0)])
    tp_cmap_l = Colormap2D([grad1, grad2, grad3])

    grad1 = ColorGradient([(1, 0, 0), (1, 0, 1), (1, 1, 0)][::-1])
    grad2 = ColorGradient([(1, 1, 0), (1, 1, 1), (1, 0.75, 0.2)][::-1])
    grad3 = ColorGradient([(1, 0, 1), (1, 0.5, 0.5), (1, 0, 0)][::-1])
    tp_cmap_r = Colormap2D([grad1, grad2, grad3])

    tp_funclist_l = [lambda x, z: (tp_f(- 6 + 3 * x, z) + offsets[1]) * w1(x) + ft(x, z, offsets[1]) * (1 - w1(x)),
                     lambda x, z: (tp_g(- 6 + 3 * x, z) + offsets[1]) * w1(x) + gt(x, z, offsets[1]) * (1 - w1(x)),
                     lambda x, z: (tp_h(- 6 + 3 * x, z) + offsets[1]) * w1(x) + ht(x, z, offsets[1]) * (1 - w1(x))]
    tp_funclist_r = [lambda x, z: (tp_f(6 - 3 * x, z) + offsets[1]) * w1(x) + ft(x, z, offsets[1]) * (1 - w1(x)),
                     lambda x, z: (tp_g(6 - 3 * x, z) + offsets[1]) * w1(x) + gt(x, z, offsets[1]) * (1 - w1(x)),
                     lambda x, z: (tp_h(6 - 3 * x, z) + offsets[1]) * w1(x) + ht(x, z, offsets[1]) * (1 - w1(x))]

    for z in zlist:
        for i in range(len(tlist) - 1):
            t = tlist[i: i + 2]
            for func_l, func_r in zip(tp_funclist_l, tp_funclist_r):
                fval_l = func_l(t, z)
                fval_r = func_r(t, z)
                plt.plot(-6 + 3 * t, fval_l, color=tp_cmap_l(t[0], z), alpha=0.05, linewidth=linewidth)
                plt.plot(6 - 3 * t, fval_r, color=tp_cmap_r(t[0], z), alpha=0.05, linewidth=linewidth)

def kg_plot():
    xl = np.linspace(-6, 6, int(len(tlist) * (8 / 3)))

    def norm(x):
        return (x + 3) / 6

    f = lambda x, z : kg_f(0.75 * x, z) + offsets[2]
    g = lambda x, z : kg_g(0.75 * x, z) + offsets[2]
    h = lambda x, z : kg_h(0.75 * x, z) + offsets[2]
    k = lambda x, z : kg_k(0.75 * x, z) + offsets[2]

    funclist = [h, g, f, k]
    xlims = [0] + [len(xl) // (4 - N) for N in range(4)]
    alpha_list = [[0.05, 0.075, 0.075, 0.05],
                  [0.075, 0.05, 0.05, 0.075],
                  [0.05, 0.075, 0.075, 0.05],
                  [0.075, 0.05, 0.05, 0.075]]

    color_xshifts = [0.1, -0.2, 0.2, -0.1]
    alpha_list = np.array(alpha_list) + 0.05

    for fi, func in enumerate(funclist):
        zl = zlist
        for z in zl:
            for xli in range(1, len(xlims)):
                for i in range(xlims[xli - 1], xlims[xli]):
                    plt.plot(xl[i: i + 2], func(xl[i: i + 2], z), color=kg_cmap(norm(xl[i]) + color_xshifts[fi], z)
                             , alpha=alpha_list[xli - 1][fi] * min(1, abs(xl[i]) ** 2 + 0.05), linewidth=linewidth)

def kg_sides():
    grad1 = ColorGradient([(0, 0, 1), (0, 0.5, 1), (0.5, 0.5, 1), (0, 1, 1)][::-1])
    grad2 = ColorGradient([(0, 1, 1), (0, 1, 1), (0, 1, 0), (0, 0, 1)][::-1])
    grad3 = ColorGradient([(0, 1, 0.5), (0, 1, 1), (0, 1, 0.5), (1, 1, 1)][::-1])
    grad4 = ColorGradient([(0.5, 0.5, 1), (0.5, 0.5, 1), (0, 1, 1), (0, 1, 1)][::-1])
    grad5 = ColorGradient([(0, 1, 1), (0, 0, 1), (0, 1, 1), (0, 1, 0)][::-1])
    kg_cmap_l = Colormap2D([grad1, grad2, grad3, grad4, grad5])

    grad1 = ColorGradient([(1, 1, 0), (1, 0, 1), (1, 0, 0)])
    grad2 = ColorGradient([(1, 1, 0), (1, 1, 1), (1, 1, 0)])
    grad3 = ColorGradient([(1, 0, 0), (1, 1, 0), (1, 0, 0.5)])
    grad4 = ColorGradient([(1, 0.75, 0.2), (1, 0.5, 1), (1, 0.5, 0)])
    grad5 = ColorGradient([(1, 0, 0), (1, 1, 1), (1, 0, 1)])
    kg_cmap_r = Colormap2D([grad1, grad2, grad3, grad4, grad5])

    kg_funclist_l = [lambda x, z: (kg_f(- 4 + x, z) + offsets[2]) * w1(x) + ft(x, z, offsets[2]) * (1 - w1(x)),
                     lambda x, z: (kg_g(- 4 + x, z) + offsets[2]) * w1(x) + gt(x, z, offsets[2]) * (1 - w1(x)),
                     lambda x, z: (kg_h(- 4 + x, z) + offsets[2]) * w1(x) + ht(x, z, offsets[2]) * (1 - w1(x)),
                     lambda x, z: (kg_k(- 4 + x, z) + offsets[2]) * w1(x) + kt(x, z, offsets[2]) * (1 - w1(x))]

    kg_funclist_r = [lambda x, z: (kg_f(4 - x, z) + offsets[2]) * w1(x) + ft(x, z, offsets[2]) * (1 - w1(x)),
                     lambda x, z: (kg_g(4 - x, z) + offsets[2]) * w1(x) + gt(x, z, offsets[2]) * (1 - w1(x)),
                     lambda x, z: (kg_h(4 - x, z) + offsets[2]) * w1(x) + ht(x, z, offsets[2]) * (1 - w1(x)),
                     lambda x, z: (kg_k(4 - x, z) + offsets[2]) * w1(x) + kt(x, z, offsets[2]) * (1 - w1(x))]

    alpha_list = [0.1, 0.125, 0.1, 0.125]
    for z in zlist:
        for i in range(len(tlist) - 1):
            t = tlist[i: i + 2]
            for ai, (func_l, func_r) in enumerate(zip(kg_funclist_l, kg_funclist_r)):
                fval_l = func_l(t, z)
                fval_r = func_r(t, z)
                plt.plot(-6 + 3 * t, fval_l, color=kg_cmap_l(t[0], z), alpha=alpha_list[ai], linewidth=linewidth)
                plt.plot(6 - 3 * t, fval_r, color=kg_cmap_r(t[0], z), alpha=alpha_list[ai], linewidth=linewidth)

def tt_plot():
    xl = np.linspace(-3, 3, len(tlist) * 2)

    funclist = [tt_f, tt_h, tt_g]
    alpha_func = lambda xi: 0.5 * max(np.tanh(abs(0.15 * xi)) - 0.1, 0.02)
    norm = lambda x: (x - min(xl)) / (max(xl) - min(xl))
    zl = zlist
    for fi, func in enumerate(funclist):
        for z in zl:
            for i in range(len(xl) - 1):
                alpha = alpha_func(xl[i])
                plt.plot(xl[i: i + 2], func(xl[i: i + 2], z), color=tt_cmap(norm(xl[i]), z)
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
                plt.plot(-6 + 3 * t, fval_l, color=tt_cmap_l(t[0], z), alpha=0.1609, linewidth=linewidth)
                plt.plot(6 - 3 * t, fval_r, color=tt_cmap_r(t[0], z), alpha=0.1609, linewidth=linewidth)

def w1(t):
    return 0.5 * (np.tanh(5 * (t - 0.3)) + 1)

tlist = np.linspace(0, 1, 150) # 90
zlist = np.linspace(0, 1, 200) # 50
salpha = 0.1

tt_f = lambda x, z: tt_f1(x, z) + tt_f2(x, z) + 1
tt_h = lambda x, z: tt_h1(x, z) + tt_h2(x, z)
tt_g = lambda x, z: tt_g1(x, z) + tt_g2(x, z) - 1

offsets = (0, -5, -11)

fig = plt.figure(figsize=(7, 9), frameon=False)
ax = plt.Axes(fig, [0, 0, 1, 1])
# ax.set_axis_off()
fig.add_axes(ax)
plt.sca(ax)

tp_plot()
tp_sides()
print('Thermopack finished')
kg_plot()
# kg_sides()
print('Kineticgas finished')
tt_plot()
tt_sides()
print('Thermotools finished')

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
plt.gca().set_facecolor('black')

plt.xlim(-4, 4)
plt.ylim(-14, 5)
# plt.savefig('background.pdf')
plt.savefig('background.png', dpi=96)
# plt.savefig('background.svg')
# plt.savefig('background.jpg', dpi=96)
plt.show()