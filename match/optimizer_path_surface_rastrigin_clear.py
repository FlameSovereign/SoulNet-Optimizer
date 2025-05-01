
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from torch.optim import Adam, SGD, RMSprop
from CollapseGrammarOptimizer_vGH1_0 import CollapseGrammarOptimizer_vGH1

class Optim2D(nn.Module):
    def __init__(self, init_x=3.0, init_y=3.0):
        super().__init__()
        self.x = nn.Parameter(torch.tensor(init_x))
        self.y = nn.Parameter(torch.tensor(init_y))

    def forward(self):
        A = 10
        return A * 2 + (self.x**2 - A * torch.cos(2 * np.pi * self.x)) + (self.y**2 - A * torch.cos(2 * np.pi * self.y))

def get_trajectory(optimizer_cls, name, steps=50):
    model = Optim2D()
    optimizer = optimizer_cls(model.parameters(), lr=3e-2)
    trajectory = []

    for _ in range(steps):
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()
        trajectory.append((model.x.item(), model.y.item(), loss.item()))

    return name, trajectory

def plot_surface_and_paths():
    X, Y = np.meshgrid(np.linspace(-5.12, 5.12, 200), np.linspace(-5.12, 5.12, 200))
    A = 10
    Z = A * 2 + (X**2 - A * np.cos(2 * np.pi * X)) + (Y**2 - A * np.cos(2 * np.pi * Y))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.inferno, alpha=0.6)

    color_map = {
        'CollapseGrammarGH': 'navy',
        'Adam': 'lime',
        'SGD': 'deepskyblue',
        'RMSprop': 'mediumorchid'
    }

    for name, opt in {
        'CollapseGrammarGH': lambda p, lr=3e-2: CollapseGrammarOptimizer_vGH1(p, lr=lr),
        'Adam': lambda p, lr=3e-2: Adam(p, lr=lr),
        'SGD': lambda p, lr=3e-2: SGD(p, lr=lr),
        'RMSprop': lambda p, lr=3e-2: RMSprop(p, lr=lr)
    }.items():
        label, traj = get_trajectory(opt, name)
        x_vals, y_vals, z_vals = zip(*traj)
        ax.plot(x_vals, y_vals, z_vals, label=label, linewidth=2, color=color_map[label])
        ax.scatter(x_vals, y_vals, z_vals, s=10, color=color_map[label])

        # start marker
        ax.scatter([x_vals[0]], [y_vals[0]], [z_vals[0]], c='red', s=50, marker='x')
        # end marker
        ax.scatter([x_vals[-1]], [y_vals[-1]], [z_vals[-1]], c='green', s=50, marker='o')

    ax.set_title("Optimizer Trajectories on Rastrigin Surface (Clear Colors)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig("optimizer_path_surface_rastrigin_clear.png")
    print("📈 Clarified trajectory surface saved to: optimizer_path_surface_rastrigin_clear.png")

if __name__ == "__main__":
    plot_surface_and_paths()
