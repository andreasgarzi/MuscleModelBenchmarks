r"""Command-line runner for the variance-based sensitivity analysis.

SAFEpython generates two independent Latin-hypercube matrices, A and B, and M
hybrid matrices in which one parameter column is exchanged.  A block with N
base samples and M uncertain inputs therefore requires ``N * (M + 2)`` model
evaluations for each trial.  Outputs are written as memory-mapped NumPy arrays
as soon as evaluations finish, allowing long analyses to be resumed after an
interruption.

Examples (run from the repository root):

    venv\Scripts\python.exe -m GSA.run_gsa --list-blocks
    venv\Scripts\python.exe -m GSA.run_gsa --block fast_fv --n 256 --workers 12
    venv\Scripts\python.exe -m GSA.run_gsa --analyse-only GSA/results/fast_fv_N256_seed260826
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import scipy.stats as st
from safepython import VBSA
from safepython.sampling import AAT_sampling

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from GSA.analysis import analyse_run
from GSA.config import BLOCKS, GSABlock, get_block
from GSA.curves import CurveBasis, build_curve_basis
from GSA.engine import METRIC_NAMES, evaluate_sample, initialise_worker, trial_shapes


def _isolated_evaluation_entry(
    block: GSABlock,
    basis_dict: dict[str, Any] | None,
    analysis_dt: float,
    row: int,
    sample: np.ndarray,
    solver_method: str | None,
    send_connection: Any,
) -> None:
    """Evaluate one row in a disposable process and return it through a pipe.

    This slower execution path is used only when ``--evaluation-timeout`` is
    requested.  A separate process lets the parent terminate a genuinely
    stalled integration without losing completed rows.  The optional solver
    override is local to that child and does not alter ``benchmark_model.py``.
    """

    try:
        if solver_method is not None:
            import benchmark_model as model_module

            original_solve_ivp = model_module.solve_ivp

            def solve_ivp_with_method(*args: Any, **kwargs: Any) -> Any:
                kwargs["method"] = solver_method
                return original_solve_ivp(*args, **kwargs)

            model_module.solve_ivp = solve_ivp_with_method
        initialise_worker(block, basis_dict, analysis_dt)
        send_connection.send(("ok", evaluate_sample((row, sample))))
    except BaseException as exc:
        send_connection.send(
            ("error", row, repr(exc), "".join(traceback.format_exception(exc)))
        )
    finally:
        send_connection.close()


def _sample_design(block: GSABlock, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create the SAFE A/B/C design in unit and physical parameter spaces.

    SAFEpython 0.2 uses NumPy's legacy global random state, hence the explicit
    seed call.  Sampling is performed on independent U(0, 1) inputs; each
    :class:`~GSA.config.ParameterSpec` then applies its uniform or log-uniform
    transform.  Both designs are saved to make every run reproducible.
    """

    np.random.seed(seed)
    m = len(block.parameters)
    unit_ab = AAT_sampling("lhs", m, st.uniform, [0.0, 1.0], 2 * n)
    unit_a, unit_b, unit_c = VBSA.vbsa_resampling(unit_ab)
    unit_design = np.concatenate((unit_a, unit_b, unit_c), axis=0)
    physical = np.empty_like(unit_design)
    for j, spec in enumerate(block.parameters):
        physical[:, j] = spec.transform(unit_design[:, j])
    return unit_design, physical


def _metadata(
    block: GSABlock,
    n: int,
    seed: int,
    analysis_dt: float,
    curve_basis: CurveBasis | None,
) -> dict[str, Any]:
    """Build the self-contained description stored beside each run."""

    return {
        "schema_version": 1,
        "method": "SAFEpython LHS and Saltelli A/B/C resampling; Jansen Sobol estimators",
        "sampling": "maximin Latin hypercube in independent unit inputs",
        "block": block.name,
        "description": block.description,
        "N": n,
        "M": len(block.parameters),
        "evaluations_per_trial": n * (len(block.parameters) + 2),
        "seed": seed,
        "analysis_dt_s": analysis_dt,
        "trials": list(block.trials),
        "signal": block.signal,
        "metric_names": list(METRIC_NAMES),
        "parameter_names": [p.name for p in block.parameters],
        "parameters": [
            {
                "name": p.name,
                "low": p.low,
                "high": p.high,
                "distribution": "log-uniform" if p.scale == "log" else "uniform",
                "units": p.units,
                "rationale": p.rationale,
            }
            for p in block.parameters
        ],
        "curve_basis": curve_basis.as_dict() if curve_basis else None,
        "status": "running",
        "created_unix": time.time(),
    }


def _create_storage(
    run_dir: Path,
    block: GSABlock,
    n_rows: int,
    analysis_dt: float,
) -> tuple[np.memmap, dict[str, dict[str, np.memmap]]]:
    """Allocate checkpoint and output arrays for a new run.

    Arrays are opened with NumPy's ``.npy`` memory-map format.  This keeps RAM
    use independent of the full design size and still produces files that can
    be read by ordinary ``numpy.load`` calls.
    """

    completed = np.lib.format.open_memmap(
        run_dir / "completed.npy", mode="w+", dtype=bool, shape=(n_rows,)
    )
    completed[:] = False
    stores: dict[str, dict[str, np.memmap]] = {}
    shapes = trial_shapes(block, analysis_dt)
    trials_root = run_dir / "trials"
    trials_root.mkdir(exist_ok=True)
    for trial, info in shapes.items():
        path = trials_root / trial
        path.mkdir(exist_ok=True)
        np.save(path / "time.npy", info["time"])
        stores[trial] = {
            "signal": np.lib.format.open_memmap(
                path / "signal.npy", mode="w+", dtype=np.float64, shape=(n_rows, info["n_time"])
            ),
            "signal_normalized": np.lib.format.open_memmap(
                path / "signal_normalized.npy",
                mode="w+",
                dtype=np.float64,
                shape=(n_rows, info["n_time"]),
            ),
            "metrics": np.lib.format.open_memmap(
                path / "metrics.npy", mode="w+", dtype=np.float64, shape=(n_rows, len(METRIC_NAMES))
            ),
        }
        for array in stores[trial].values():
            array[:] = np.nan
    return completed, stores


def _open_storage(
    run_dir: Path, block: GSABlock
) -> tuple[np.memmap, dict[str, dict[str, np.memmap]]]:
    """Reopen an existing checkpoint in read/write mode."""

    completed = np.lib.format.open_memmap(run_dir / "completed.npy", mode="r+")
    stores: dict[str, dict[str, np.memmap]] = {}
    for trial in block.trials:
        path = run_dir / "trials" / trial
        stores[trial] = {
            "signal": np.lib.format.open_memmap(path / "signal.npy", mode="r+"),
            "signal_normalized": np.lib.format.open_memmap(path / "signal_normalized.npy", mode="r+"),
            "metrics": np.lib.format.open_memmap(path / "metrics.npy", mode="r+"),
        }
    return completed, stores


def _flush(completed: np.memmap, stores: dict[str, dict[str, np.memmap]]) -> None:
    """Force checkpoint state and model outputs from buffers to disk."""

    completed.flush()
    for trial_store in stores.values():
        for array in trial_store.values():
            array.flush()


def _check_resume(metadata: dict[str, Any], block: GSABlock, n: int, seed: int, analysis_dt: float) -> None:
    """Prevent an existing directory from being resumed with a new design."""

    expected = (block.name, n, seed, analysis_dt, [p.name for p in block.parameters])
    found = (
        metadata.get("block"),
        int(metadata.get("N", -1)),
        int(metadata.get("seed", -1)),
        float(metadata.get("analysis_dt_s", -1)),
        metadata.get("parameter_names"),
    )
    if expected != found:
        raise ValueError("The requested run does not match the existing checkpoint metadata.")


def run_block(
    block: GSABlock,
    n: int,
    seed: int,
    workers: int,
    max_tasks_per_child: int | None,
    evaluation_timeout_s: float | None,
    evaluation_method: str | None,
    analysis_dt: float,
    n_bootstrap: int,
    results_root: Path,
    resume: bool,
    make_plots: bool,
    run_analysis: bool,
    curve_bootstrap: int,
) -> Path:
    """Run, resume and optionally analyse one configured GSA block.

    The normal execution path uses a process pool and periodically recycles it
    to avoid the gradual solver slowdown observed in very long batches.  With
    a timeout, each row instead runs in a disposable process.  In both modes a
    row is marked complete only after every trial output has been copied to
    disk, so ``--resume`` never treats a partial row as valid.

    Returns
    -------
    pathlib.Path
        Directory containing metadata, samples, checkpoints and analysis
        products for the block.
    """

    run_dir = results_root / f"{block.name}_N{n}_seed{seed}"
    metadata_path = run_dir / "metadata.json"

    if run_dir.exists() and not resume:
        raise FileExistsError(f"{run_dir} already exists. Use --resume or choose another seed/N.")
    run_dir.mkdir(parents=True, exist_ok=True)

    if metadata_path.exists():
        # A resumed run must reuse the original curve basis and exact sample
        # matrices.  Rebuilding either from current files would invalidate the
        # already completed model rows.
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        _check_resume(metadata, block, n, seed, analysis_dt)
        curve_basis = CurveBasis.from_dict(metadata["curve_basis"]) if metadata.get("curve_basis") else None
        unit_design = np.load(run_dir / "unit_samples.npy")
        samples = np.load(run_dir / "samples.npy")
        completed, stores = _open_storage(run_dir, block)
    else:
        curve_basis = (
            build_curve_basis(REPO_ROOT, n_bootstrap=curve_bootstrap, seed=seed + 101)
            if block.includes_curve_uncertainty
            else None
        )
        unit_design, samples = _sample_design(block, n, seed)
        np.save(run_dir / "unit_samples.npy", unit_design)
        np.save(run_dir / "samples.npy", samples)
        metadata = _metadata(block, n, seed, analysis_dt, curve_basis)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        completed, stores = _create_storage(run_dir, block, len(samples), analysis_dt)

    pending = np.flatnonzero(~np.asarray(completed, dtype=bool))
    if evaluation_method is not None and len(pending):
        metadata.setdefault("solver_fallbacks", []).append(
            {
                "method": evaluation_method,
                "rows": [int(i) for i in pending],
                "recorded_unix": time.time(),
                "validation": (
                    "Use only after comparison against the nominal LSODA solution; "
                    "the runner records affected rows for reproducibility."
                ),
            }
        )
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"[{block.name}] N={n}, M={len(block.parameters)}, "
        f"evaluations={len(samples)} per trial, pending={len(pending)}, workers={workers}",
        flush=True,
    )
    if curve_basis is not None:
        print(
            f"[{block.name}] f1 variance captured={curve_basis.f1_explained.sum():.3%}; "
            f"f2 variance captured={curve_basis.f2_explained.sum():.3%}",
            flush=True,
        )

    failures: list[dict[str, Any]] = []
    start = time.perf_counter()
    last_update = start
    done_at_start = int(np.sum(completed))
    if len(pending):
        basis_dict = curve_basis.as_dict() if curve_basis else None

        def record_result(row: int, result: dict[str, dict[str, np.ndarray]]) -> None:
            """Copy one complete design row into its memory-mapped arrays."""

            for trial, outputs in result.items():
                stores[trial]["signal"][row] = outputs["signal"]
                stores[trial]["signal_normalized"][row] = outputs["signal_normalized"]
                stores[trial]["metrics"][row] = outputs["metrics"]
            completed[row] = True

        def report_progress(force: bool = False) -> None:
            """Flush checkpoints and print a rate-based ETA at most every 15 s."""

            nonlocal last_update
            now = time.perf_counter()
            if not force and now - last_update < 15 and not bool(np.all(completed)):
                return
            _flush(completed, stores)
            done = int(np.sum(completed))
            elapsed = now - start
            newly_done = done - done_at_start
            rate = newly_done / elapsed if elapsed > 0 else 0.0
            eta = (len(samples) - done) / rate if rate > 0 else np.nan
            print(
                f"[{block.name}] {done}/{len(samples)} complete "
                f"({100 * done / len(samples):.1f}%), "
                f"ETA {eta / 60:.1f} min, failures={len(failures)}",
                flush=True,
            )
            last_update = now

        if evaluation_timeout_s is not None:
            # Each active entry owns one process and the receiving end of its
            # pipe.  The parent polls these lightweight connections so several
            # timed evaluations can still run concurrently.
            context = mp.get_context("spawn")
            cursor = 0
            active: dict[int, tuple[Any, Any, float, int]] = {}
            while cursor < len(pending) or active:
                while cursor < len(pending) and len(active) < workers:
                    row = int(pending[cursor])
                    cursor += 1
                    receive_connection, send_connection = context.Pipe(duplex=False)
                    process = context.Process(
                        target=_isolated_evaluation_entry,
                        args=(
                            block,
                            basis_dict,
                            analysis_dt,
                            row,
                            np.asarray(samples[row]),
                            evaluation_method,
                            send_connection,
                        ),
                    )
                    process.start()
                    send_connection.close()
                    active[process.pid] = (
                        process,
                        receive_connection,
                        time.perf_counter(),
                        row,
                    )

                finished: list[int] = []
                now = time.perf_counter()
                for pid, (process, connection, started, row) in list(active.items()):
                    if connection.poll():
                        message = connection.recv()
                        process.join(timeout=5)
                        if message[0] == "ok":
                            result_row, result = message[1]
                            record_result(result_row, result)
                        else:
                            failures.append(
                                {"row": row, "error": message[2], "traceback": message[3]}
                            )
                        connection.close()
                        finished.append(pid)
                    elif not process.is_alive():
                        process.join(timeout=5)
                        connection.close()
                        failures.append(
                            {
                                "row": row,
                                "error": f"Worker exited with code {process.exitcode}",
                                "traceback": "",
                            }
                        )
                        finished.append(pid)
                    elif now - started > evaluation_timeout_s:
                        # No model output has been written for this row yet, so
                        # it remains pending and can be retried with --resume.
                        process.terminate()
                        process.join(timeout=5)
                        connection.close()
                        failures.append(
                            {
                                "row": row,
                                "error": f"Evaluation exceeded {evaluation_timeout_s:g} s",
                                "traceback": "",
                            }
                        )
                        finished.append(pid)
                for pid in finished:
                    del active[pid]
                report_progress()
                if not finished:
                    time.sleep(0.1)
        else:
            # Recreating the pool between batches releases solver and BLAS
            # state accumulated by long-lived worker processes.
            pool_size = (
                len(pending)
                if max_tasks_per_child is None
                else max(1, workers * max_tasks_per_child)
            )
            for batch_start in range(0, len(pending), pool_size):
                batch = pending[batch_start : batch_start + pool_size]
                with ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=initialise_worker,
                    initargs=(block, basis_dict, analysis_dt),
                ) as pool:
                    futures = {
                        pool.submit(evaluate_sample, (int(i), np.asarray(samples[i]))): int(i)
                        for i in batch
                    }
                    for future in as_completed(futures):
                        requested_index = futures[future]
                        try:
                            row, result = future.result()
                            record_result(row, result)
                        except Exception as exc:  # retain checkpoint and full diagnostic
                            failures.append(
                                {
                                    "row": requested_index,
                                    "error": repr(exc),
                                    "traceback": "".join(traceback.format_exception(exc)),
                                }
                            )
                        report_progress()

    _flush(completed, stores)
    if failures:
        (run_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    missing = np.flatnonzero(~np.asarray(completed, dtype=bool))
    if len(missing):
        metadata["status"] = "incomplete"
        metadata["completed_evaluations"] = int(np.sum(completed))
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        raise RuntimeError(
            f"{len(missing)} evaluations failed or remain incomplete. See {run_dir / 'failures.json'} and resume."
        )

    metadata["status"] = "complete"
    metadata["completed_evaluations"] = len(samples)
    metadata["completed_unix"] = time.time()
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if run_analysis:
        print(f"[{block.name}] Computing scalar and time-resolved indices...", flush=True)
        analyse_run(run_dir, n_bootstrap=n_bootstrap, make_plots=make_plots)
    print(f"[{block.name}] Results saved in {run_dir}", flush=True)
    return run_dir


def _parser() -> argparse.ArgumentParser:
    """Construct the public command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", action="append", choices=list(BLOCKS), help="GSA block; repeat as needed.")
    parser.add_argument("--all", action="store_true", help="Run every configured block.")
    parser.add_argument("--list-blocks", action="store_true", help="List blocks, trials and parameter counts.")
    parser.add_argument("--n", type=int, default=256, help="Saltelli base sample size (default: 256).")
    parser.add_argument("--seed", type=int, default=260826, help="Random seed.")
    parser.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 1)))
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=32,
        help=(
            "Recycle the complete worker pool after approximately this many tasks "
            "per worker to avoid long-run solver slowdown (default: 32; use 0 to disable)."
        ),
    )
    parser.add_argument(
        "--evaluation-timeout",
        type=float,
        default=0.0,
        help=(
            "Run each row in a disposable process with this timeout in seconds; "
            "0 uses the faster pooled mode (default: 0)."
        ),
    )
    parser.add_argument(
        "--evaluation-method",
        choices=("LSODA", "BDF", "Radau"),
        help=(
            "Override solve_ivp only in disposable-process mode; intended as a "
            "validated fallback for numerically stiff rows."
        ),
    )
    parser.add_argument("--analysis-dt", type=float, default=0.002, help="Time spacing saved for TVSA [s].")
    parser.add_argument("--bootstrap", type=int, default=500, help="Bootstrap resamples for scalar indices.")
    parser.add_argument("--curve-bootstrap", type=int, default=1000, help="Bootstrap fits used for f1/f2 PCA.")
    parser.add_argument("--results-root", type=Path, default=Path("GSA") / "results")
    parser.add_argument("--resume", action="store_true", help="Resume an existing compatible checkpoint.")
    parser.add_argument("--no-analysis", action="store_true", help="Only run and save model evaluations.")
    parser.add_argument("--no-plots", action="store_true", help="Compute indices but do not create PNG figures.")
    parser.add_argument("--analyse-only", type=Path, help="Recompute indices for an existing completed run.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse command-line arguments and dispatch requested blocks/analysis."""

    args = _parser().parse_args(argv)
    if args.list_blocks:
        for block in BLOCKS.values():
            print(f"{block.name:16s} M={len(block.parameters):2d} trials={len(block.trials):2d}  {block.description}")
        return
    if args.analyse_only:
        report = analyse_run(args.analyse_only, n_bootstrap=args.bootstrap, make_plots=not args.no_plots)
        print(f"Analysed {report['block']} in {args.analyse_only}")
        return
    names = list(BLOCKS) if args.all else (args.block or [])
    if not names:
        raise SystemExit("Select --block NAME, --all, --list-blocks, or --analyse-only RUN_DIR.")
    if args.n < 2:
        raise SystemExit("--n must be at least 2.")
    if args.workers < 1:
        raise SystemExit("--workers must be positive.")
    if args.max_tasks_per_child < 0:
        raise SystemExit("--max-tasks-per-child cannot be negative.")
    if args.evaluation_timeout < 0:
        raise SystemExit("--evaluation-timeout cannot be negative.")
    if args.evaluation_method and not args.evaluation_timeout:
        raise SystemExit("--evaluation-method requires --evaluation-timeout.")
    for offset, name in enumerate(names):
        run_block(
            get_block(name),
            n=args.n,
            seed=args.seed + offset,
            workers=args.workers,
            max_tasks_per_child=args.max_tasks_per_child or None,
            evaluation_timeout_s=args.evaluation_timeout or None,
            evaluation_method=args.evaluation_method,
            analysis_dt=args.analysis_dt,
            n_bootstrap=args.bootstrap,
            results_root=args.results_root,
            resume=args.resume,
            make_plots=not args.no_plots,
            run_analysis=not args.no_analysis,
            curve_bootstrap=args.curve_bootstrap,
        )


if __name__ == "__main__":
    main()
