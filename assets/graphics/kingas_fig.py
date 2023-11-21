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

grad1 = ColorGradient([(0, 0, 1)    , (0, 0, 1)    , (0, 0, 1)    , (0.9, 0.9, 1)  , (1, 0.9, 0.9) , (1, 0, 0)  , (1, 0, 0)  , (1, 0, 0)  ])
grad2 = ColorGradient([(0, 1, 1)    , (0, 1, 1)    , (0, 1, 1)    , (0.9, 1, 1)    , (1, 1, 0.9)   , (1, 1, 0)  , (1, 1, 0)  , (1, 1, 0)  ])
grad3 = ColorGradient([(0, 1, 0.5)  , (0, 1, 0.5)  , (0, 1, 0.5)  , (0.9, 1, 0.95) , (1, 0.9, 0.95), (1, 0, 0.5), (1, 0, 0.5), (1, 0, 0.5)])
grad4 = ColorGradient([(0.5, 0.5, 1), (0.5, 0.5, 1), (0.5, 0.5, 1), (0.95, 0.95, 1), (1, 0.95, 0.9), (1, 0.5, 0), (1, 0.5, 0), (1, 0.5, 0)])
grad5 = ColorGradient([(0, 1, 1)    , (0, 1, 1)    , (0, 1, 1)    , (0.9, 1, 1)    , (1, 0.9, 1)   , (1, 0, 1)  , (1, 0, 1)  , (1, 0, 1)  ])

cmap2d = Colormap2D([grad1, grad2, grad3, grad4, grad5])

x_lst = np.linspace(0, 1, 50)
y_lst = np.linspace(0, 1, 50)
c_lst = []
for i, x in enumerate(x_lst):
    c_lst.append([])
    for j, y in enumerate(y_lst):
        c_lst[i].append(cmap2d(x, y))
# for i, x in enumerate(x_lst):
#     plt.scatter(np.ones_like(y_lst) * x, y_lst, color=c_lst[i], s=40)
# plt.show()

def mod(x):
    return 0.8 * np.cosh(0.6 * x)

def f1(x, z):
    return np.sin(x + z) * np.sin(x * z) * cos(z)

def f2(x, z):
    return cos(z) * np.cos(x**2 * z)**2

flist = [f1, f2]
def f(x, z):
    return sum(fi(x, z) for fi in flist) * mod(x)

def g1(x, z):
    return sin(x * z - cos(2 * np.pi * z))

def g2(x, z):
    return cos(x * (z - 0.5)) * cos(x * (z - 0.5))**3

glist = [g1, g2]
def g(x, z):
    return sum(gi(x, z) for gi in glist) * mod(x)

def h1(x, z):
    return cos(x**2 * (z - 0.5)**3) * cos(x * (z - 0.5))**3

def h2(x, z):
    return (sin(z + z * cos(x)))**2

hlist = [h1, h2]
def h(x, z):
    return sum(hi(x, z) for hi in hlist) * mod(x)

def k1(x, z):
    return sin(x * (z + 1)**2)

def k2(x, z):
    return cos(np.pi * (z - 0.5)) * cos(x * sin(x) * cos(x)**2)

klist = [k1, k2]
def k(x, z):
    return sum(ki(x, z) for ki in klist) * mod(x)

if __name__ == '__main__':
    fig = plt.figure(figsize=(10, 5))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    plt.sca(ax)

    xl = np.linspace(-3, 3, 50)

    def norm(x):
        return (x + 3) / 6

    nz = [50 for _ in range(4)]
    funclist = [h, g, f, k]
    xlims = [0] + [len(xl) // (4 - N) for N in range(4)]
    alpha_list = [[0.05 , 0.075, 0.075 , 0.05  ],
                  [0.075, 0.05 , 0.05  , 0.075 ],
                  [0.05 , 0.075, 0.075 , 0.05  ],
                  [0.075, 0.05 , 0.05  , 0.075 ]]

    color_xshifts = [0.1, -0.2, 0.2, -0.1]
    alpha_list = np.array(alpha_list) + 0.05

    for fi, func in enumerate(funclist):
        zl = np.linspace(0, 1, nz[fi])
        for z in zl:
            for xli in range(1, len(xlims)):
                for i in range(xlims[xli - 1], xlims[xli]):
                    plt.plot(xl[i : i + 2], func(xl[i : i + 2], z), color=cmap2d(norm(xl[i]) + color_xshifts[fi], z)
                             , alpha=alpha_list[xli - 1][fi] * min(1, abs(xl[i])**2 + 0.15))

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
    plt.ylim(-1.5, 4)
    plt.savefig('kingas.png', dpi=96)
    plt.show()