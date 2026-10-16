# te-qpinn

Trainable Embedding QPINN implementation for 2D nonlinear hyperelasticity. Built with JAX, Equinox, and PennyLane.

---

## Usage

Install dependencies using [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync
source .venv/bin/activate
```

Train unsymmetrized or symmetrized models using a TOML configuration:

```bash
# unsym QPINN
python qpinn_sweepable.py -c cfgs/template.toml -o rez/run_standard

# sym QPINN
python qpinn_sweepable_sym.py -c cfgs/template.toml -o rez/run_sym
```

Evaluate relative $L_2$ and $L_\infty$ error against the FEM reference ([`refs/test.npz`](refs/test.npz)):

```bash
# single
python err_metrics.py -f path/to/prediction.npz

# batch
python err_metrics.py path/to/results/*_prediction.npz
```

Visualize the deformed grid and training loss history:

```bash
python plotterer.py -p path/to/prediction.npz -l path/to/loss_history.npz --save path/to/save
```

Compute von Neumann entanglement entropy $S(\rho)$:

```bash
python ntangle_ntropy.py -c cfgs/template.toml -m path/to/model.eqx -f path/to/prediction.npz
```

---

SLURM array sweep scripts are in [`jobs/`](jobs/)