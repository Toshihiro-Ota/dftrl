"""
The MIT License (MIT) Copyright (c) 2025. Toshihiro Ota
"""

import argparse
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
from torchdiffeq import odeint


parser = argparse.ArgumentParser()
parser.add_argument('--n_players', type=int, default=2,
                    help='Number of players (default: 2)')
parser.add_argument('--n_actions', type=int, default=2,
                    help='Number of actions (default: 2)')
parser.add_argument('--regularizer', type=str, default='entropic', choices=['entropic', 'euclidean'],
                    help='Type of regularizer function h (default: "entropic")')
parser.add_argument('--epsilon', type=float, default=0.1,
                    help='Dissipation parameter, assumed to be 0 or positive real number (default: 0.1)')
parser.add_argument('--initial_payoff', default=None, #choices=['rand', 'randn'],
                    help='Apply an adjustment to the initial state (default: None)')


def hreg(x: torch.Tensor, reg: str) -> torch.Tensor:
    """
    Regularizer function h.
    For reg='entropic', computes the sum of x*log(x).
    For reg='euclidean', computes 0.5 times the squared Euclidean norm of x.
    """
    if reg == 'entropic':
        return x @ torch.log(x)
    elif reg == 'euclidean':
        return (x**2).sum(dim=-1)/2  # equivalent to `torch.linalg.vector_norm(x, dim=-1)**2 / 2`
    else:
        raise NotImplementedError(f"Regularizer '{reg}' is not implemented.")

def hdual(y: torch.Tensor, reg: str) -> torch.Tensor:
    """
    Dual of the regularizer function.
    For reg='entropic', returns logsumexp over the last dimension.
    For reg='euclidean', computes a shifted squared norm difference.
    """
    if reg == 'entropic':
        return torch.logsumexp(y, dim=-1)
    elif reg == 'euclidean':  # faster convergence, slower computation, than the entropic
        c = ((y.sum(dim=-1, keepdim=True) - 1.0) / y.size(-1)) * torch.ones_like(y)  #? `y.mean(dim=-1, keepdim=True) - (1.0 / y.size(-1))` does not work correctly
        return ((y**2).sum(dim=-1) - (c**2).sum(dim=-1)) / 2  # equivalent to `(torch.linalg.vector_norm(y, dim=-1)**2 - torch.linalg.vector_norm(c, dim=-1)**2) / 2`
    else:
        raise NotImplementedError(f"Regularizer '{reg}' is not implemented.")

def nabla_h(y: torch.Tensor, h) -> torch.Tensor:
    """Compute the gradient of function h at y."""
    return torch.func.grad(h)(y)

def hess_h(y: torch.Tensor, h) -> torch.Tensor:
    """
    Compute the Hessian of function h at y and cast to float32.

    This is equivalent to `jacfwd(jacrev(h))(y)`, the best practice to compute hessian:
    https://pytorch.org/docs/stable/generated/torch.func.hessian.html
    """
    return torch.func.hessian(h)(y).to(torch.float)

def fenchel(y: torch.Tensor,
            xs: torch.Tensor,
            reg: str,
            hh=hreg,
            h=hdual) -> torch.Tensor:
    """
    Computes the Fenchel coupling: hreg(xs) + hdual(y) - <y, xs>

    xs is a constant, basically supposed to be the Nash equilibrium.
    """
    return hh(xs, reg) + h(y, reg) - y @ xs


## PolyMatZeroSumGame: for now only 2- & 3-player cases are implemented
class TwoPlayerZeroSumGame(nn.Module):
    def __init__(
            self,
            payoffmat: torch.Tensor,
            regularizer: str,
            epsilon: float,
            n_players=2,
            ) -> None:
        """
        Initialize a two-player matrix zero-sum game.

        Parameters
        ----------
        payoffmat : torch.Tensor
            Payoff matrix for player 1 vs. player 2.
        regularizer : str
            Regularizer type ('entropic' or 'euclidean').
        epsilon : float
            Perturbation / dissipation parameter. epsilon=0. reduces to the ordinary FTRL dynamics.
        """
        super().__init__()
        self.A12 = payoffmat
        self.A21 = -payoffmat.T
        self.eps = epsilon
        self.hess = partial(hess_h, h=partial(hdual, reg=regularizer))

        self.split_sizes = [payoffmat.size(0)] * (2*n_players)  # [n_actions] * (2*n_players)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, y1, y2 = torch.split(x, self.split_sizes, dim=-1)

        H_y1 = self.hess(y1)
        H_y2 = self.hess(y2)
        xx1 = self.A12 @ x2
        xx2 = self.A21 @ x1

        g1 = self.A12 @ H_y2 @ xx2
        g2 = self.A21 @ H_y1 @ xx1
        f1 = H_y1 @ g1
        f2 = H_y2 @ g2

        x1 = H_y1 @ xx1 + self.eps * f1
        x2 = H_y2 @ xx2 + self.eps * f2
        y1 = xx1 + self.eps * g1
        y2 = xx2 + self.eps * g2
        return torch.cat((x1, x2, y1, y2), dim=-1)

class ThreePlayerZeroSumGame(nn.Module):
    def __init__(
            self,
            payoffmat: torch.Tensor,
            regularizer: str,
            epsilon: float,
            n_players=3,
            ) -> None:
        """
        Initialize a three-player matrix zero-sum game.

        Parameters
        ----------
        payoffmat : torch.Tensor
            Payoff matrix for players i vs. players j.
        regularizer : str
            Regularizer type ('entropic' or 'euclidean').
        epsilon : float
            Perturbation / dissipation parameter. epsilon=0 reduces to the ordinary FTRL dynamics.
        """
        super().__init__()
        self.A12 = self.A23 = self.A31 = payoffmat  #! assumption
        self.A21 = self.A32 = self.A13 = -payoffmat.T
        self.eps = epsilon
        self.hess = partial(hess_h, h=partial(hdual, reg=regularizer))

        self.split_sizes = [payoffmat.size(0)] * (2*n_players)  # [n_actions] * (2*n_players)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3, y1, y2, y3 = torch.split(x, self.split_sizes, dim=-1)

        H_y1 = self.hess(y1)
        H_y2 = self.hess(y2)
        H_y3 = self.hess(y3)
        xx1 = self.A12 @ x2 + self.A13 @ x3
        xx2 = self.A21 @ x1 + self.A23 @ x3
        xx3 = self.A31 @ x1 + self.A32 @ x2

        g1 = self.A12 @ H_y2 @ xx2 + self.A13 @ H_y3 @ xx3
        g2 = self.A21 @ H_y1 @ xx1 + self.A23 @ H_y3 @ xx3
        g3 = self.A31 @ H_y1 @ xx1 + self.A32 @ H_y2 @ xx2
        f1 = H_y1 @ g1
        f2 = H_y2 @ g2
        f3 = H_y3 @ g3

        x1 = H_y1 @ xx1 + self.eps * f1
        x2 = H_y2 @ xx2 + self.eps * f2
        x3 = H_y3 @ xx3 + self.eps * f3
        y1 = xx1 + self.eps * g1
        y2 = xx2 + self.eps * g2
        y3 = xx3 + self.eps * g3
        return torch.cat((x1, x2, x3, y1, y2, y3), dim=-1)


def payoffmat_nasheq(n_actions: int) -> tuple:
    if n_actions==2:
        ## Matching pennies-like game
        a = 2.0; b = 1.0
        A = torch.tensor([
            [   a, -1.0],
            [-1.0,    b]
        ])
        xs = torch.tensor(
            [(b + 1)/(a + b + 2), (a + 1)/(a + b + 2)]
        )
    elif n_actions==3:
        ## Rock-Paper-Scissors-like game
        a = 1.0; b = 2.0; c = 3.0
        A = torch.tensor([
            [ 0.0,  -a,   b],
            [   a, 0.0,  -c],
            [  -b,   c, 0.0]
        ])
        xs = torch.tensor(
            [c/(a + b + c), b/(a + b + c), a/(a + b + c)]
        )
    else:
        raise NotImplementedError(f"n_actions={n_actions} case is not implemented.")
    return A, xs

def run_game(
        n_players: int,
        n_actions: int,
        payoffmat: torch.Tensor,
        regularizer: str,
        epsilon: float,
        initial_payoff,
        ) -> tuple:
    """
    Numerically solves the game dynamics based on the given parameters.
    Returns the state split per player ({x_i}, {y_i}).

    Parameters
    ----------
    n_players : int
        Number of players (2 or 3).
    n_actions : int
        Number of actions per player (2 or 3).
    payoffmat : torch.Tensor
        Payoff matrix, shape=(n_actions, n_actions).
    regularizer : str
        Type of regularizer ('entropic' or 'euclidean').
    epsilon : float
        Perturbation / dissipation parameter. epsilon=0. reduces to the ordinary FTRL dynamics.
    initial_payoff : Optional[Callable]
        Initial payoff adjustment function. If provided, its output is added to y0.

    Returns
    -------
    tuple of torch.Tensor
        A tuple of state tensors split per player
        (x1, x2, ..., y1, y2, ...).
    """
    assert epsilon >= 0, 'Dissipation parameter is assumed to be 0 or positive real number'
    if n_players==2:
        game = TwoPlayerZeroSumGame(payoffmat, regularizer, epsilon)
    elif n_players==3:
        game = ThreePlayerZeroSumGame(payoffmat, regularizer, epsilon)
    else:
        raise NotImplementedError(f"n_players={n_players} case is not implemented.")

    # initial condition
    y0 = torch.zeros(n_players, n_actions)
    if initial_payoff=='rand':
        y0 += torch.rand_like(y0) * 2
    elif initial_payoff=='randn':
        y0 += torch.randn_like(y0) * 2
    x0 = torch.stack([nabla_h(y0[i], h=partial(hdual, reg=regularizer)) for i in range(n_players)])
    z = torch.cat((x0, y0)).flatten()

    # define time steps
    Tmax = 100 #50
    dt = 0.01
    NdT = int(Tmax/dt)
    t = torch.linspace(0, Tmax, NdT)

    sol = odeint(lambda t, x: game(x), z, t)
    print(f'Total time steps (Tmax/dt) = {sol.shape[0]}, 2*n_players*n_actions: {sol.shape[1]}')  # sol.shape = (NdT, 2*n_players*n_actions)
    return torch.split(sol, game.split_sizes, dim=-1)  # out=(x1, x2, ..., xn, y1, y2, ..., yn)


cmap = plt.get_cmap("tab10")  # temporary
#* sol_split[i]==xsol[i], sol_split[n_players + i]==ysol[i]

def plot_x(sol_split: tuple, n_players: int, n_actions: int):
    plt.figure(figsize=(6, 4))
    for i in range(n_players):
        for a in range(n_actions):
            plt.plot(sol_split[i][:, a].detach().numpy(), linewidth=2,
                     color=cmap(a + i*n_actions), alpha=0.6, label=f'x{i+1}[{a+1}]')
    plt.xlabel('time step')
    plt.title('dynamics of strategies')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.show()

def plot_xdiff(sol_split: tuple, n_players: int, n_actions: int, regularizer: str):
    """
    For each player, plots the difference between xsol and nabla_h(ysol) (i.e., the update residual),
    computed efficiently using vectorized (vmap) operations.
    """
    plt.figure(figsize=(6, 4))
    for i in range(n_players):
        xdiff = sol_split[i] - torch.func.vmap(
            lambda y: nabla_h(y, h=partial(hdual, reg=regularizer))
            )(sol_split[n_players + i])
        for a in range(n_actions):
            plt.plot(xdiff[:, a].detach().numpy(), linewidth=0.8,
                     color=cmap(a + i*n_actions), alpha=0.6, label=f'x{i+1}diff[{a+1}]')
    plt.xlabel('time step')
    plt.title('xdiff = x - nabla_h(y)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.show()

def plot_x3d(sol_split: tuple, n_players: int = 3, n_actions: int = 2):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')

    d = torch.linspace(0, 1, 100)
    x1sol1 = sol_split[0][:, 0]
    x2sol1 = sol_split[1][:, 0]
    x3sol1 = sol_split[2][:, 0]
    ax.plot(d, d, d, linewidth=1.0, color='black', label="Nash equilibria")  # diagonal line from (0,0,0) to (1,1,1)
    ax.plot(x1sol1, x2sol1, x3sol1,
            color=cmap(0), linestyle='-', linewidth=1.2, marker='o', markevery=[0], label="strategies trajectory")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel("x1[1]")
    ax.set_ylabel("x2[1]")
    ax.set_zlabel("x3[1]")
    ax.set_title("3-player Matching pennies-like game")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.show()

def plot_xsimplex(sol_split: tuple, xs: torch.Tensor, n_players: int = 2, n_actions: int = 3):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')

    vertices = torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]  # to close the triangle
    ])
    ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='black', linewidth=1)
    ax.scatter(1.1, 0, 0, marker='$(1,0,0)$', color='black', s=2000)
    ax.scatter(0, 1.1, 0, marker='$(0,1,0)$', color='black', s=2000)
    ax.scatter(0, 0, 1.1, marker='$(0,0,1)$', color='black', s=2000)

    ax.scatter(xs[0], xs[1], xs[2], marker='*', color='black', s=70)
    ax.plot(sol_split[0][:, 0], sol_split[0][:, 1], sol_split[0][:, 2],
            color=cmap(0), linestyle='-', linewidth=2, marker='o', markevery=[0])

    ax.view_init(elev=35.264, azim=45)  # view from (1, 1, 1)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # Disable pane filling, set axis line width to 0, and remove pane edge color for each axis
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.line.set_linewidth(0.)
        axis.pane.set_edgecolor('none')
    plt.title("2-player Rock-Paper-Scissors-like game")
    plt.show()

def plot_fenchel(sol_split: tuple, xs: torch.Tensor, n_players: int, regularizer: str):
    fenchel_vals = sum(
        fenchel(sol_split[n_players + i], xs, reg=regularizer)
        for i in range(n_players)
        )
    plt.figure(figsize=(6, 4))
    plt.plot(fenchel_vals.detach().numpy(), linewidth=2, color=cmap(0), alpha=0.6, label='fenchel')
    plt.plot(torch.zeros_like(fenchel_vals).detach().numpy(), color='black', alpha=0.6, linestyle='dashed')
    plt.ylim(bottom=-0.1*max(fenchel_vals))
    plt.xlabel('time step')
    plt.title('Fenchel coupling: hreg(xs) + hdual(y) - <y, xs>')
    plt.legend()
    plt.show()


def main(args):
    #* (1/3): Define a game
    A, xs = payoffmat_nasheq(args.n_actions)

    #* (2/3): Run the game
    sol_split = run_game(
        args.n_players,
        args.n_actions,
        A,
        args.regularizer,
        args.epsilon,
        args.initial_payoff,  # if 'rand' or 'randn', apply a random adjustment to the initial state
        )
    print('\n trajectory of x1sol:', sol_split[0][:10], ' ...', sol_split[0][-10:], sep='\n')
    print('\n trajectory of x2sol:', sol_split[1][:10], ' ...', sol_split[1][-10:], sep='\n')
    if args.n_players==3: print('\n trajectory of x3sol:', sol_split[2][:10], ' ...', sol_split[2][-10:], sep='\n')

    #* (3/3): Plot the results
    plot_x(sol_split, args.n_players, args.n_actions)
    plot_xdiff(sol_split, args.n_players, args.n_actions, args.regularizer)
    if args.n_players==3 and args.n_actions==2:
        plot_x3d(sol_split)
    elif args.n_players==2 and args.n_actions==3:
        plot_xsimplex(sol_split, xs)
        plot_fenchel(sol_split, xs, args.n_players, args.regularizer)
    else:
        plot_fenchel(sol_split, xs, args.n_players, args.regularizer)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
