import matplotlib as mpl
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.animation as animation
from numpy.linalg import inv

def draw_torus():
    global x, y, z, omegas, dt, ax

    elapsed = np.fmod(time.time(), (dt * len(omegas)))
    timeframe = int(elapsed)

    omega = omegas[timeframe]

    omegamag = np.sqrt(omega.dot(omega))

    alpha = omega[0] / omegamag
    beta = omega[1] / omegamag
    gamma = omega[2] / omegamag

    theta = omegamag * dt
    ctheta = np.cos(theta)
    mctheta = 1 - ctheta
    stheta = np.sin(theta)

    p = np.transpose(np.array([x, y, z]), [1, 2, 0])
    m = [[alpha**2 * mctheta + ctheta, alpha * beta * mctheta - gamma * stheta, alpha * gamma * mctheta + beta * stheta],
        [alpha * beta * mctheta + gamma * stheta, beta**2 * mctheta + ctheta, beta * gamma * mctheta - alpha * stheta],
        [alpha * gamma * mctheta - beta * stheta, beta * gamma * mctheta + alpha * stheta, gamma**2 * mctheta + ctheta]]
    x, y, z = np.transpose(np.dot(p, m), [2, 0, 1])

    return ax.plot_wireframe(x, y, z, color = 'black', linewidth = 0.2, antialiased = False)

def runga_kutta():
    global duration, fps, dt, omegas, G1, G2, G3, omegasx, omegasy, omegasz

    nmax = int(duration * fps)
    h = dt

    lastomega = np.array([1.0, 1.0, 1.0])
    lastframe = np.array([1.0, 1.0, 1.0])

    for n in range(0, nmax):
        kx = lastomega[0]
        ky = lastomega[1]
        kz = lastomega[2]

        kx1 = -G1 * h * ky * kz
        ky1 = -G2 * h * kx * kz
        kz1 = -G3 * h * kx * ky

        kx2 = -G1 * h * (ky + ky1/2) * (kz + kz1/2)
        ky2 = -G2 * h * (kx + kx1/2) * (kz + kz1/2)
        kz2 = -G3 * h * (kx + kx1/2) * (ky + ky1/2)

        kx3 = -G1 * h * (ky + ky2/2) * (kz + kz2/2)
        ky3 = -G2 * h * (kx + kx2/2) * (kz + kz2/2)
        kz3 = -G3 * h * (kx + kx2/2) * (ky + ky2/2)

        kx4 = -G1 * h * (ky + ky3) * (kz + kz3)
        ky4 = -G2 * h * (kx + kx3) * (kz + kz3)
        kz4 = -G3 * h * (kx + kx3) * (ky + ky3)

        wx = kx + (kx1 / 6) + (kx2 / 3) + (kx3 / 3) + (kx4 / 6)
        wy = ky + (ky1 / 6) + (ky2 / 3) + (ky3 / 3) + (ky4 / 6)
        wz = kz + (kz1 / 6) + (kz2 / 3) + (kz3 / 3) + (kz4 / 6)

        lastomega = np.array([wx, wy, wz])
        newomega = lastomega

        omegas.append(newomega)
        omegasx.append(newomega[0])
        omegasy.append(newomega[1])
        omegasz.append(newomega[2])

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def find_fixed_omegas():
    global omegas, dt, duration, fps, frame, fixedomegasx, fixedomegasy, fixedomegasz

    for timeframe in range(0, int(duration * fps)):
        omega = omegas[timeframe]

        omegamag = np.sqrt(omega.dot(omega))

        alpha = omega[0] / omegamag
        beta = omega[1] / omegamag
        gamma = omega[2] / omegamag

        theta = omegamag * dt
        ctheta = np.cos(theta)
        mctheta = 1 - ctheta
        stheta = np.sin(theta)

        m = [[alpha**2 * mctheta + ctheta, alpha * beta * mctheta - gamma * stheta, alpha * gamma * mctheta + beta * stheta],
            [alpha * beta * mctheta + gamma * stheta, beta**2 * mctheta + ctheta, beta * gamma * mctheta - alpha * stheta],
            [alpha * gamma * mctheta - beta * stheta, beta * gamma * mctheta + alpha * stheta, gamma**2 * mctheta + ctheta]]
        frame = np.dot(m, frame)

        worldomega = np.dot(inv(frame), omega)

        fixedomegasx.append(worldomega[0])
        fixedomegasy.append(worldomega[1])
        fixedomegasz.append(worldomega[2])

def update():
    global fig

    surf = draw_torus()
    fig.canvas.draw()
    fig.canvas.flush_events()
    surf.remove()

#mpl.interactive(True)
#fig.set_size_inches(6, 6)
#fig = plt.figure()
#ax = fig.gca(projection='3d')

#ax.set_autoscale_on(False)
#ax.set_xlim(-5, 5)
#ax.set_ylim(-5, 5)
#ax.set_zlim(-5, 5)

starttime = time.time()
fps = 16.0
dt = 1.0/fps
duration = 16.0

a = 1.0
c = 4.0
M = 3.14

n = 16
theta = np.linspace(0, 2.*np.pi, n)
phi = np.linspace(0, 2.*np.pi, n)
theta, phi = np.meshgrid(theta, phi)
x = (c + a * np.cos(theta)) * np.cos(phi)
y = (c + a * np.cos(theta)) * np.sin(phi)
z = a * np.sin(theta)

I1 = I2 = (1.0 / 8.0) * (5.0 * a**2 + 4.0 * c**2) * M
I3 = (3.0 / 4.0 * a**2 + c**2) * M
G1 = (I3 - I2) / I1
G2 = (I1 - I3) / I2
G3 = (I2 - I1) / I3

omegas = []
omegasx = []
omegasy = []
omegasz = []
fixedomegasx = []
fixedomegasy = []
fixedomegasz = []

frame = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

runga_kutta()
find_fixed_omegas()

times = []

for t in range(0, int(duration * fps)):
    times.append(t * dt)

plt.plot(times, fixedomegasx)
plt.plot(times, fixedomegasy)
plt.plot(times, fixedomegasz)
plt.show()

while True:
    #update()
    time.sleep(dt - ((time.time() - starttime) % dt))
