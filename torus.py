import matplotlib as mpl
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.animation as animation

def draw_torus(c, a):
    n = 32
    theta = np.linspace(0, 2.*np.pi, n)
    phi = np.linspace(0, 2.*np.pi, n)
    theta, phi = np.meshgrid(theta, phi)
    x = (c + a * np.cos(theta)) * np.cos(phi)
    y = (c + a * np.cos(theta)) * np.sin(phi)
    z = a * np.sin(theta)

    a = time.time()
    t = np.transpose(np.array([x, y, z]), (1, 2, 0))
    m = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    x, y, z = np.transpose(np.dot(t, m), (2, 0, 1))

    return ax.plot_wireframe(x, y, z, color = 'black', linewidth = 0.2, antialiased = True)

def update():
    surf = draw_torus(4, 1)
    fig.canvas.draw()
    fig.canvas.flush_events()
    surf.remove()

mpl.interactive(True)
fig = plt.figure()
fig.set_size_inches(6, 6)
ax = fig.gca(projection='3d')

ax.set_autoscale_on(False)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)

while True:
    update()
