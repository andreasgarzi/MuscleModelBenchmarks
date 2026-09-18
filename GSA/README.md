# Global sensitivity analysis

This directory contains the variance-based global sensitivity analysis (GSA)
used for the muscle-model benchmarks. The GSA does not modify the model or the
benchmark definitions: every sampled parameter set is applied to a private
copy of an existing benchmark case.

Only the source code is distributed. Numerical results are generated locally
under `GSA/results/` and are not required to reproduce the analysis.

## Files

- `config.py` defines the analysis blocks, representative trials, parameter
  ranges and sampling distributions.
- `curves.py` builds correlated uncertainty modes for the length-dependent
  calcium functions `f1` and `f2` from the digitised experimental data.
- `engine.py` applies one parameter sample to the benchmark cases and returns
  force/calcium traces and scalar descriptors.
- `run_gsa.py` creates the Saltelli design, runs or resumes model evaluations,
  and manages the saved checkpoints.
- `analysis.py` calculates scalar and time-resolved Sobol indices, confidence
  intervals, convergence estimates and temporal-fingerprint correlations.
- `test_gsa.py` checks the Jansen estimator against the analytical Ishigami
  sensitivity indices.

## Installation

Create the project environment and install the repository requirements. The
GSA uses the same environment as the benchmark scripts; in particular it
requires `numpy`, `scipy`, `matplotlib` and `safepython`.

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run all commands below from the repository root. On Linux or macOS, replace
`venv\Scripts\python.exe` with the Python executable from the active virtual
environment.

## Analysis design

For a block containing `M` uncertain parameters, SAFEpython constructs two
independent `N x M` Latin-hypercube matrices, `A` and `B`. It then creates one
hybrid matrix `C_i` for each parameter. Consequently, a complete block requires

```text
N * (M + 2)
```

model evaluations per trial. The rows are saved in the order `A`, `B`,
`C_1`, ..., `C_M`; `analysis.py` relies on this order when calculating the
Jansen first-order and total-order Sobol indices.

Inputs are sampled independently in a unit hypercube and then transformed to
uniform or log-uniform physical ranges. The ranges and their rationale are
recorded both in `config.py` and in the `metadata.json` file written for each
run.

The blocks are organised by physiological mechanism:

| Block | Main purpose |
|---|---|
| `slow_calcium`, `fast_calcium` | Calcium-transient and MUAP parameters against direct calcium measurements |
| `slow_ea`, `fast_ea` | Excitation-activation parameters across twitch, unfused and fused tetani |
| `slow_length`, `fast_length` | Force-length and length-dependent calcium effects |
| `slow_fv`, `fast_fv` | Force-velocity parameters and dynamic contractions |
| `slow_mu`, `fast_mu_cat`, `fast_mu_rat` | Motor-unit excitation-activation, including sag where present |
| `slow_dynamic_joint`, `fast_dynamic_joint` | Joint excitation-activation and mechanical screening in dynamic trials |

List the exact trials and parameter count for every block with:

```powershell
venv\Scripts\python.exe -m GSA.run_gsa --list-blocks
```

## Running the GSA

A small run is useful for checking the installation and estimating execution
time, but it is not suitable for scientific interpretation:

```powershell
venv\Scripts\python.exe -m GSA.run_gsa `
  --block slow_calcium --n 16 --workers 4 --no-plots
```

For the analyses reported with the benchmarks, `N = 256` was used as the
initial screening resolution. Blocks whose ranking or confidence intervals do
not stabilise should be repeated or extended at `N = 512` or above.

```powershell
venv\Scripts\python.exe -m GSA.run_gsa `
  --block fast_fv --n 256 --workers 12
```

Several blocks may be requested in one command by repeating `--block`. Seeds
are incremented between blocks so that their designs remain independent.

```powershell
venv\Scripts\python.exe -m GSA.run_gsa `
  --block slow_ea --block fast_ea --n 256 --workers 12
```

Use `--all` only when the computational cost of every configured block is
acceptable.

## Checkpoints and resuming

Model outputs are written after completed evaluations rather than being kept
only in memory. If a run is interrupted, repeat the same block, `N`, seed and
analysis time step with `--resume`:

```powershell
venv\Scripts\python.exe -m GSA.run_gsa `
  --block fast_fv --n 256 --workers 12 --resume
```

The runner verifies the existing metadata before resuming. A row is marked as
complete only after all trial outputs for that row have been stored.

The default pooled mode is the fastest option. If a small number of unusually
stiff rows fail to complete, individual evaluations can be isolated and given
a timeout:

```powershell
venv\Scripts\python.exe -m GSA.run_gsa `
  --block slow_length --n 256 --workers 8 --resume `
  --evaluation-timeout 300
```

`--evaluation-method BDF` or `Radau` may be combined with the timeout mode,
but an alternative solver should be used only after checking its output
against the default LSODA solution. Any fallback method and affected rows are
recorded in the run metadata.

## Outputs and post-processing

Each run is saved as
`GSA/results/<block>_N<base-sample-size>_seed<seed>/` and contains:

- the unit and physical sampling designs;
- completion checkpoints and metadata;
- raw and normalised model traces for each trial;
- scalar model descriptors;
- first- and total-order Sobol indices;
- paired-bootstrap 95% confidence intervals;
- convergence arrays at nested sample sizes;
- time-resolved indices and temporal fingerprints;
- a JSON report, a short Markdown report and optional diagnostic figures.

The normalised force traces remove the direct maximum-force scale from the
time-resolved analysis, so that the resulting indices describe changes in
waveform shape. Sobol indices are reported only where the output has measurable
variance. Temporal fingerprints are signed median finite-difference responses;
high correlation between two influential fingerprints is treated as a screen
for possible practical confounding, not as proof of structural
non-identifiability.

Analysis products can be regenerated from stored model outputs without
rerunning the simulations:

```powershell
venv\Scripts\python.exe -m GSA.run_gsa `
  --analyse-only GSA/results/fast_fv_N256_seed260826
```

Add `--no-plots` to calculate numerical results without generating PNG files.

## Numerical check

The standalone estimator is tested against the Ishigami function, for which
the first- and total-order indices are known analytically:

```powershell
venv\Scripts\python.exe -m unittest GSA.test_gsa
```
