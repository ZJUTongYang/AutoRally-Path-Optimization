import sys

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
from matplotlib.patches import Rectangle


class AutoRallyNet:
    def __init__(self, save_path, dt=0.02):
        self.weight_dict = np.load(save_path)
        self.non_linearity = np.tanh
        self.dt = dt

    def __call__(self, x, steering, throttle):
        state = np.copy(x)  # Don't modify x, return updated state as new variable
        # First compute the state derivative
        fn_input = np.concatenate([state[3:], [steering], [throttle]])
        h1 = np.dot(self.weight_dict["dynamics_W1"], fn_input) + self.weight_dict["dynamics_b1"]
        h1 = np.tanh(h1)
        h2 = np.dot(self.weight_dict["dynamics_W2"], h1) + self.weight_dict["dynamics_b2"]
        h2 = np.tanh(h2)
        state_der = np.dot(self.weight_dict["dynamics_W3"], h2) + self.weight_dict["dynamics_b3"]
        # Now compute the actual state update
        state[0] += (np.cos(state[2]) * state[4] - np.sin(state[2]) * state[5]) * self.dt
        state[1] += (np.sin(state[2]) * state[4] + np.cos(state[2]) * state[5]) * self.dt
        state[2] += -state[6] * self.dt  # Sixth state is NEGATIVE of yaw derivative
        state[3] += state_der[0] * self.dt
        state[4] += state_der[1] * self.dt
        state[5] += state_der[2] * self.dt
        state[6] += state_der[3] * self.dt
        return state


def animate(npy_array, name):
    save_as = name + ".mp4"
    Xs = npy_array[:, 0]
    Ys = npy_array[:, 1]
    Yaw = npy_array[:, 2]
    Vx = npy_array[:, 4]
    Vy = npy_array[:, 4] * np.tan(npy_array[:, 3])
    T = range(Xs.size)
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 20), ylim=(-10, 10))
    Xs = np.array(Xs)
    Ys = np.array(Ys)
    lines = []
    line, = ax.plot([], [])
    circs = []
    circs.append(Rectangle((Xs[0] - .30, Ys[0] - .13), .6, .26, color='r'))

    def init():
        line.set_data([], [])
        for c in circs:
            ax.add_patch(c)
        return circs,

    def animate(k):
        for c in circs:
            heading = Yaw[k] / np.pi * 180.0
            t = mpl.transforms.Affine2D().rotate_deg_around(Xs[k], Ys[k], heading) + ax.transData
            c.set_transform(t)
            c.set_xy((Xs[k] - .3, Ys[k] - .13))
        x = np.copy(Xs[:k])
        y = np.copy(Ys[:k])
        sys.stdout.write("Making Animation: %d/%d \r" % (k, len(Xs)))
        sys.stdout.flush()
        line.set_data(x, y)
        line.set_alpha(.1)
        return circs,

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(Xs), interval=20, blit=True)
    #anim.save(save_as, fps=50, writer='ffmpeg', bitrate=2500)


if __name__ == "__main__":
    steering = np.clip(np.random.randn(250), -1, 1)
    throttle = np.clip(np.random.randn(250), -1, 0.75)
    x = np.array([0, 0, 0, 0, 5.0, 0.5, 0])
    state_history = []

    f = AutoRallyNet("alpha_nnet.npz")

    for i in range(250):
        state_history.append(x)
        x = f(x, steering[i], throttle[i])

    animate(np.asarray(state_history), "random_controls")
