# Dissipation FTRL

This is the official Python implementation of the *dissipation FTRL* algorithm presented in the paper "The Hamiltonian of Poly-matrix Zero-sum Games," [arXiv:2505.12609](https://arxiv.org/abs/2505.12609).

## Abstract

Understanding a dynamical system fundamentally relies on establishing an appropriate Hamiltonian function and elucidating its symmetries. By formulating agents' strategies and cumulative payoffs as canonically conjugate variables, we identify the Hamiltonian function that generates the dynamics of poly-matrix zero-sum games. We reveal the symmetries of our Hamiltonian and derive the associated conserved quantities, showing how the conservation of probability and the invariance of the Fenchel coupling are intrinsically encoded within the system. Furthermore, we propose the *dissipation FTRL* (DFTRL) dynamics by introducing a perturbation that dissipates the Fenchel coupling, proving convergence to the Nash equilibrium and linking DFTRL to last-iterate convergent algorithms. Our results highlight the potential of Hamiltonian dynamics in uncovering the structural properties of learning dynamics in games, and pave the way for broader applications of Hamiltonian dynamics in game theory and machine learning.

## Usage

### Requirements

We use the [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq) library. Dependencies can be installed with the following:

```bash
pip install -r requirements.txt
```

### Example

To solve the DFTRL dynamics for a game, e.g. two-player Rock-Paper-Scissors, run the following:

```bash
python dftrl.py \
    --n_players 2 \
    --n_actions 3 \
    --regularizer 'entropic' \
    --epsilon 0.1
```

The argument `epsilon` corresponds to the perturbation coefficient $\alpha$ in the paper. `epsilon=0.` reduces the dynamics to the ordinary FTRL. An illustrative demonstration is provided in the [`demo_dftrl.ipynb`](./demo_dftrl.ipynb) notebook.

## Citation

If you use our code, or otherwise find our work useful, please cite the accompanying paper:

```bibtex
@article{ota2025hamiltonian,
    title   = {The {H}amiltonian of Poly-matrix Zero-sum Games},
    author  = {Ota, Toshihiro and Fujimoto, Yuma},
    journal = {arXiv preprint arXiv:2505.12609},
    year    = {2025}
}
```
