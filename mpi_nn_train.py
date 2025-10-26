import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from mpi4py import MPI


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------


class BaseActivation:
    def forward(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def derivative(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SigmoidActivation(BaseActivation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        z_clip = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clip))

    def derivative(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        del z  # unused
        return a * (1.0 - a)


class ReLUActivation(BaseActivation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    def derivative(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        del a  # unused
        return (z > 0.0).astype(z.dtype)


class TanhActivation(BaseActivation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    def derivative(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        del z  # unused
        return 1.0 - a * a


def create_activation(name: str) -> BaseActivation:
    if name == "sigmoid":
        return SigmoidActivation()
    if name == "relu":
        return ReLUActivation()
    if name == "tanh":
        return TanhActivation()
    raise ValueError(f"Unsupported activation: {name}")


# ---------------------------------------------------------------------------
# Neural network core
# ---------------------------------------------------------------------------


class NeuralNetwork:
    def __init__(
        self,
        n_hidden: int,
        n_features: int,
        activation_key: str,
        seed: int,
        dtype: np.dtype,
    ) -> None:
        self.n_hidden = n_hidden
        self.n_features = n_features
        self.dtype = np.dtype(dtype)
        self.activation_key = activation_key
        self.activation = create_activation(activation_key)

        self.theta_size = n_hidden + 1 + n_hidden * n_features + n_hidden
        self.v_slice = slice(0, n_hidden)
        self.c_index = n_hidden
        self.W_slice = slice(n_hidden + 1, n_hidden + 1 + n_hidden * n_features)
        self.b_slice = slice(self.W_slice.stop, self.W_slice.stop + n_hidden)

        self._init_params(seed)

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def _init_params(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        fan_in = self.n_features
        fan_out = self.n_hidden

        init = "he" if self.activation_key == "relu" else "xavier"
        if init == "he":
            std_W = math.sqrt(2.0 / float(fan_in))
        else:
            std_W = math.sqrt(2.0 / float(fan_in + fan_out))

        self.W = (rng.standard_normal((self.n_hidden, self.n_features)) * std_W).astype(self.dtype)
        self.b = np.zeros(self.n_hidden, dtype=self.dtype)
        std_v = 1.0 / math.sqrt(max(1, self.n_hidden))
        self.v = (rng.standard_normal(self.n_hidden) * std_v).astype(self.dtype)
        self.c = self.dtype.type(0.0)

    def get_theta(self) -> np.ndarray:
        theta = np.empty(self.theta_size, dtype=self.dtype)
        theta[self.v_slice] = self.v
        theta[self.c_index] = self.c
        theta[self.W_slice] = self.W.reshape(-1)
        theta[self.b_slice] = self.b
        return theta

    def set_theta(self, theta: np.ndarray) -> None:
        self.v = theta[self.v_slice].astype(self.dtype, copy=False)
        self.c = self.dtype.type(theta[self.c_index])
        self.W = theta[self.W_slice].reshape((self.n_hidden, self.n_features)).astype(self.dtype, copy=False)
        self.b = theta[self.b_slice].astype(self.dtype, copy=False)

    def zero_like_theta(self) -> np.ndarray:
        return np.zeros(self.theta_size, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        Z = X @ self.W.T + self.b
        A = self.activation.forward(Z)
        return A @ self.v + self.c

    def gradient_for_batch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        loss_clip: float,
        out: np.ndarray,
    ) -> None:
        out.fill(0.0)
        if indices.size == 0:
            return

        Xb = X[indices]
        yb = y[indices]

        Z = Xb @ self.W.T + self.b
        A = self.activation.forward(Z)
        y_pred = A @ self.v + self.c
        errors = y_pred - yb
        if loss_clip and loss_clip > 0.0:
            max_abs_e = math.sqrt(loss_clip)
            np.clip(errors, -max_abs_e, max_abs_e, out=errors)

        grad_c = np.sum(errors, dtype=self.dtype)
        grad_v = A.T @ errors
        deriv = self.activation.derivative(Z, A)
        delta = (errors[:, None] * self.v[None, :]) * deriv
        grad_b = np.sum(delta, axis=0)
        grad_W = delta.T @ Xb

        out[self.v_slice] = grad_v
        out[self.c_index] = grad_c
        out[self.W_slice] = grad_W.reshape(-1)
        out[self.b_slice] = grad_b

    def sse_for_indices(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        block: int,
        loss_clip: float,
    ) -> float:
        if indices.size == 0:
            return 0.0

        total = 0.0
        for start in range(0, indices.size, block):
            end = min(indices.size, start + block)
            idx = indices[start:end]
            Xb = X[idx]
            yb = y[idx]
            preds = self.forward_batch(Xb)
            errors = preds - yb
            if loss_clip and loss_clip > 0.0:
                max_abs_e = math.sqrt(loss_clip)
                np.clip(errors, -max_abs_e, max_abs_e, out=errors)
            err64 = errors.astype(np.float64)
            total += float(np.sum(err64 * err64))
        return total


# ---------------------------------------------------------------------------
# Utility structures
# ---------------------------------------------------------------------------


def array_splits_lengths(total: int, size: int) -> List[int]:
    base = total // size
    rem = total % size
    return [base + (1 if r < rem else 0) for r in range(size)]


def split_indices(total: int, size: int) -> List[Tuple[int, int]]:
    lengths = array_splits_lengths(total, size)
    starts = np.cumsum([0] + lengths[:-1]).tolist()
    return list(zip(starts, [s + l for s, l in zip(starts, lengths)]))


def filter_local_batch(global_indices: np.ndarray, start: int, end: int) -> np.ndarray:
    mask = (global_indices >= start) & (global_indices < end)
    return global_indices[mask] - start


def load_preprocessed(data_dir: str):
    base = data_dir if data_dir and os.path.isdir(data_dir) else "."
    X_train = np.load(os.path.join(base, "X_train_scaled.npy"), mmap_mode="r")
    y_train = np.load(os.path.join(base, "y_train.npy"), mmap_mode="r")
    X_test = np.load(os.path.join(base, "X_test_scaled.npy"), mmap_mode="r")
    y_test = np.load(os.path.join(base, "y_test.npy"), mmap_mode="r")
    return X_train, y_train, X_test, y_test


@dataclass
class DatasetInfo:
    X_local: np.ndarray
    y_local: np.ndarray
    X_test_local: np.ndarray
    y_test_local: np.ndarray
    local_start: int
    local_end: int
    train_splits: List[Tuple[int, int]]
    test_splits: List[Tuple[int, int]]
    N: int
    N_test: int
    n_features: int
    y_mean: float
    y_std: float

    @property
    def local_count(self) -> int:
        return self.local_end - self.local_start


@dataclass
class LabelMeta:
    standardize: bool
    log1p: bool
    y_mean: float
    y_std: float


@dataclass
class TrainingResult:
    theta: np.ndarray
    history: List[Dict[str, float]]
    best_R: float
    best_iter: int
    comm_time: float
    compute_time: float


# ---------------------------------------------------------------------------
# Helper functions for training pipeline
# ---------------------------------------------------------------------------


def setup_mpi(dtype_str: str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    np_dtype = np.float32 if dtype_str == "float32" else np.float64
    mpi_dtype = MPI.FLOAT if np_dtype == np.float32 else MPI.DOUBLE
    return comm, rank, size, np_dtype, mpi_dtype


def load_and_distribute(
    data_dir: str,
    comm,
    rank: int,
    size: int,
    np_dtype,
    standardize_y: bool,
) -> Tuple[DatasetInfo, float]:
    # Rank 0 reads metadata only; all ranks will locally slice-load their partitions
    if rank == 0:
        X_train, y_train, X_test, y_test = load_preprocessed(data_dir)
        N, n_features = X_train.shape
        N_test = X_test.shape[0]
        if standardize_y:
            y_mean = float(np.mean(y_train))
            y_std = float(np.std(y_train) + 1e-12)
        else:
            y_mean = 0.0
            y_std = 1.0
        train_splits = split_indices(N, size)
        test_splits = split_indices(N_test, size)
        meta = {
            "N": N,
            "n_features": n_features,
            "N_test": N_test,
            "train_splits": train_splits,
            "test_splits": test_splits,
            "y_mean": y_mean,
            "y_std": y_std,
        }
        # free references early; ranks will re-open files locally
        del X_train, y_train, X_test, y_test
    else:
        meta = None

    t0 = time.perf_counter()
    meta = comm.bcast(meta, root=0)
    comm_time = time.perf_counter() - t0

    train_splits = meta["train_splits"]
    test_splits = meta["test_splits"]

    # Each rank locally loads its slices via memmap
    my_start, my_end = train_splits[rank]
    my_start_t, my_end_t = test_splits[rank]

    X_train_mm = np.load(os.path.join(data_dir if data_dir else ".", "X_train_scaled.npy"), mmap_mode="r")
    y_train_mm = np.load(os.path.join(data_dir if data_dir else ".", "y_train.npy"), mmap_mode="r")
    X_test_mm = np.load(os.path.join(data_dir if data_dir else ".", "X_test_scaled.npy"), mmap_mode="r")
    y_test_mm = np.load(os.path.join(data_dir if data_dir else ".", "y_test.npy"), mmap_mode="r")

    X_local = np.asarray(X_train_mm[my_start:my_end]).astype(np_dtype, copy=False)
    y_local = np.asarray(y_train_mm[my_start:my_end]).astype(np_dtype, copy=False)
    X_test_local = np.asarray(X_test_mm[my_start_t:my_end_t]).astype(np_dtype, copy=False)
    y_test_local = np.asarray(y_test_mm[my_start_t:my_end_t]).astype(np_dtype, copy=False)

    local_start, local_end = train_splits[rank]

    # Cast to desired compute dtype
    X_local = X_local.astype(np_dtype, copy=False)
    X_test_local = X_test_local.astype(np_dtype, copy=False)
    y_local = y_local.astype(np_dtype, copy=False)
    y_test_local = y_test_local.astype(np_dtype, copy=False)

    dataset = DatasetInfo(
        X_local=X_local,
        y_local=y_local,
        X_test_local=X_test_local,
        y_test_local=y_test_local,
        local_start=local_start,
        local_end=local_end,
        train_splits=train_splits,
        test_splits=test_splits,
        N=meta["N"],
        N_test=meta["N_test"],
        n_features=meta["n_features"],
        y_mean=float(meta["y_mean"]),
        y_std=float(meta["y_std"]),
    )
    return dataset, comm_time


def transform_labels(dataset: DatasetInfo, np_dtype, standardize_y: bool, log1p_y: bool) -> LabelMeta:
    y_local = dataset.y_local.astype(np_dtype, copy=False)
    y_test_local = dataset.y_test_local.astype(np_dtype, copy=False)

    if standardize_y:
        y_local = (y_local - dataset.y_mean) / dataset.y_std
        y_test_local = (y_test_local - dataset.y_mean) / dataset.y_std

    if log1p_y:
        y_local = np.log1p(np.maximum(y_local, -0.999999))
        y_test_local = np.log1p(np.maximum(y_test_local, -0.999999))

    dataset.y_local = y_local
    dataset.y_test_local = y_test_local

    return LabelMeta(standardize_y, log1p_y, dataset.y_mean, dataset.y_std)


def select_eval_indices(
    rng: np.random.Generator,
    n_samples: int,
    eval_frac: float,
    eval_max: int,
) -> np.ndarray:
    if n_samples == 0:
        return np.empty(0, dtype=np.int64)

    target = n_samples
    if eval_frac < 1.0:
        target = min(target, max(1, int(round(eval_frac * n_samples))))
    if eval_max and eval_max > 0:
        target = min(target, eval_max)

    if target >= n_samples:
        return np.arange(n_samples, dtype=np.int64)
    return rng.choice(n_samples, size=target, replace=False).astype(np.int64)


def inverse_transform_labels(values: np.ndarray, meta: LabelMeta) -> np.ndarray:
    out = values.astype(np.float64, copy=True)
    if meta.log1p:
        out = np.expm1(out)
    if meta.standardize:
        out = out * meta.y_std + meta.y_mean
    return out


def run_training_loop(
    model: NeuralNetwork,
    theta: np.ndarray,
    dataset: DatasetInfo,
    label_meta: LabelMeta,
    args,
    comm,
    rank: int,
    size: int,
    np_dtype,
    mpi_dtype,
) -> TrainingResult:
    X_local = dataset.X_local
    y_local = dataset.y_local
    rng = np.random.default_rng(args.seed)
    rng_eval = np.random.default_rng(args.seed + 123456)

    grad_buf = model.zero_like_theta()
    total_grad_buf = model.zero_like_theta()

    history: List[Dict[str, float]] = []
    best_R = math.inf
    best_iter = -1
    current_lr = args.lr
    comm_time = 0.0
    compute_time = 0.0

    N_global = dataset.N
    local_start = dataset.local_start
    local_end = dataset.local_end

    for it in range(1, args.iters + 1):
        batch_size = min(args.batch, N_global)
        if rank == 0:
            batch_indices = rng.choice(N_global, size=batch_size, replace=False).astype(np.int64)
        else:
            batch_indices = np.empty(batch_size, dtype=np.int64)

        t0 = time.perf_counter()
        comm.Bcast([batch_indices, MPI.LONG_LONG], root=0)
        comm_time += time.perf_counter() - t0

        local_batch = filter_local_batch(batch_indices, local_start, local_end)

        t0 = time.perf_counter()
        model.gradient_for_batch(X_local, y_local, local_batch, args.loss_clip, grad_buf)
        compute_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        comm.Allreduce([grad_buf, mpi_dtype], [total_grad_buf, mpi_dtype], op=MPI.SUM)
        comm_time += time.perf_counter() - t0

        grad_avg = total_grad_buf / float(batch_size)
        if args.weight_decay > 0.0:
            grad_avg[model.v_slice] += args.weight_decay * theta[model.v_slice]
            grad_avg[model.W_slice] += args.weight_decay * theta[model.W_slice]

        if args.grad_clip and args.grad_clip > 0.0:
            gnorm = float(np.linalg.norm(grad_avg))
            if gnorm > args.grad_clip:
                grad_avg *= args.grad_clip / (gnorm + 1e-12)

        theta -= current_lr * grad_avg
        model.set_theta(theta)

        do_eval = (it % args.eval_every == 0) or (it == 1)
        stop_training = False

        if do_eval:
            eval_idx = select_eval_indices(rng_eval, X_local.shape[0], args.eval_train_frac, args.eval_train_max_samples)

            t0 = time.perf_counter()
            local_sse = model.sse_for_indices(X_local, y_local, eval_idx, args.sse_block, args.loss_clip)
            compute_time += time.perf_counter() - t0

            total_sse = np.array([local_sse], dtype=np.float64)
            t0 = time.perf_counter()
            comm.Allreduce(MPI.IN_PLACE, total_sse, op=MPI.SUM)
            comm_time += time.perf_counter() - t0

            local_eval_count = np.array([eval_idx.size], dtype=np.float64)
            t0 = time.perf_counter()
            comm.Allreduce(MPI.IN_PLACE, local_eval_count, op=MPI.SUM)
            comm_time += time.perf_counter() - t0

            if label_meta.log1p:
                scale = 1.0
            else:
                scale = (label_meta.y_std * label_meta.y_std) if label_meta.standardize else 1.0
            R = (total_sse[0] * scale) / (2.0 * float(local_eval_count[0]))

            R_buf = np.array([R], dtype=np.float64)
            t0 = time.perf_counter()
            comm.Bcast(R_buf, root=0)
            comm_time += time.perf_counter() - t0
            R_latest = float(R_buf[0])

            if rank == 0:
                history.append({"iter": it, "R": R_latest})
                improved = (best_R - R_latest) > args.tol
                if improved:
                    best_R = R_latest
                    best_iter = it
                elif best_iter >= 0 and (it - best_iter) >= args.patience:
                    stop_training = True
            stop_buf = np.array([1 if stop_training else 0], dtype=np.int32)
            t0 = time.perf_counter()
            comm.Bcast(stop_buf, root=0)
            comm_time += time.perf_counter() - t0
            stop_training = bool(stop_buf[0])

        if stop_training:
            break

        if args.lr_decay_steps and args.lr_decay_steps > 0 and (it % args.lr_decay_steps == 0):
            current_lr *= args.lr_decay

    return TrainingResult(theta=theta, history=history, best_R=best_R, best_iter=best_iter, comm_time=comm_time, compute_time=compute_time)


def compute_rmse_local(model: NeuralNetwork, X: np.ndarray, y: np.ndarray, block: int, label_meta: LabelMeta) -> float:
    total_sse = 0.0
    n = X.shape[0]
    for start in range(0, n, block):
        end = min(n, start + block)
        Xb = X[start:end]
        yb = y[start:end]
        preds_trans = model.forward_batch(Xb)
        preds = inverse_transform_labels(preds_trans, label_meta)
        target = inverse_transform_labels(yb, label_meta)
        diff = preds - target
        diff64 = diff.astype(np.float64)
        total_sse += float(np.sum(diff64 * diff64))
    return total_sse


def compute_final_metrics(
    model: NeuralNetwork,
    dataset: DatasetInfo,
    label_meta: LabelMeta,
    args,
    comm,
    mpi_dtype,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    t_compute = 0.0
    t_comm = 0.0

    t0 = time.perf_counter()
    local_sse_train = compute_rmse_local(model, dataset.X_local, dataset.y_local, args.sse_block, label_meta)
    local_sse_test = compute_rmse_local(model, dataset.X_test_local, dataset.y_test_local, args.sse_block, label_meta)
    t_compute += time.perf_counter() - t0

    buf = np.array([local_sse_train, local_sse_test, float(dataset.local_count), float(dataset.X_test_local.shape[0])], dtype=np.float64)
    t0 = time.perf_counter()
    comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)
    t_comm += time.perf_counter() - t0

    total_sse_train, total_sse_test, total_N_train, total_N_test = buf
    rmse_train = math.sqrt(total_sse_train / total_N_train)
    rmse_test = math.sqrt(total_sse_test / total_N_test)

    metrics = {
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "total_sse_train": total_sse_train,
        "total_sse_test": total_sse_test,
        "N_train": int(total_N_train),
        "N_test": int(total_N_test),
    }
    return metrics, {"compute": t_compute, "comm": t_comm}


def quantiles(values: np.ndarray, probs: Sequence[float]) -> Dict[str, float]:
    return {f"p{int(p * 100)}": float(np.quantile(values, p)) for p in probs}


def collect_residual_stats(
    model: NeuralNetwork,
    dataset: DatasetInfo,
    label_meta: LabelMeta,
    seed: int,
    sample_size: int = 200_000,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    stats: Dict[str, float] = {}

    def sample_block(X: np.ndarray, y: np.ndarray, prefix: str) -> None:
        n = X.shape[0]
        if n == 0:
            return
        k = min(sample_size, n)
        idx = rng.choice(n, size=k, replace=False)
        preds_t = model.forward_batch(X[idx])
        preds = inverse_transform_labels(preds_t, label_meta)
        target = inverse_transform_labels(y[idx], label_meta)
        residuals = (preds - target).astype(np.float64)

        stats[f"{prefix}_y_mean"] = float(np.mean(target))
        stats[f"{prefix}_y_std"] = float(np.std(target))
        stats.update({f"{prefix}_{k}": v for k, v in quantiles(target, (0.5, 0.9, 0.95, 0.99)).items()})
        abs_res = np.abs(residuals)
        stats[f"{prefix}_abs_err_mean"] = float(np.mean(abs_res))
        stats.update({f"{prefix}_abs_err_{k}": v for k, v in quantiles(abs_res, (0.9, 0.95, 0.99)).items()})

    sample_block(dataset.X_local, dataset.y_local, "train")
    sample_block(dataset.X_test_local, dataset.y_test_local, "test")
    return stats


def save_results(
    args,
    size: int,
    training_result: TrainingResult,
    metrics: Dict[str, float],
    total_time: float,
    compute_time: float,
    comm_time: float,
    dataset: DatasetInfo,
    label_meta: LabelMeta,
    stats: Optional[Dict[str, float]],
) -> None:
    output = {
        "config": {
            "activation": args.activation,
            "n_hidden": args.hidden,
            "batch_size": args.batch,
            "lr": args.lr,
            "lr_decay": args.lr_decay,
            "lr_decay_steps": args.lr_decay_steps,
            "max_iters": args.iters,
            "eval_every": args.eval_every,
            "eval_train_frac": args.eval_train_frac,
            "eval_train_max_samples": args.eval_train_max_samples,
            "patience": args.patience,
            "tol": args.tol,
            "seed": args.seed,
            "processes": size,
            "standardize_y": args.standardize_y,
            "log1p_y": args.log1p_y,
            "weight_decay": args.weight_decay,
            "dtype": args.dtype,
            "grad_clip": args.grad_clip,
            "loss_clip": args.loss_clip,
        },
        "history": training_result.history,
        "metrics": {
            **metrics,
            "time_total_sec": total_time,
            "time_compute_sec": compute_time,
            "time_comm_sec": comm_time,
        },
        "data": {
            "N_train": dataset.N,
            "N_test": dataset.N_test,
            "n_features": dataset.n_features,
            "y_mean": label_meta.y_mean,
            "y_std": label_meta.y_std,
        },
    }

    if stats:
        output["residual_stats"] = stats

    with open(args.history_out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="One-hidden-layer NN with MPI SGM training")
    p.add_argument("--data-dir", default="data_preprocessed", help="Directory containing preprocessed .npy files")
    p.add_argument("--activation", default="relu", choices=["relu", "sigmoid", "tanh"])
    p.add_argument("--hidden", type=int, default=128, help="Hidden layer size")
    p.add_argument("--batch", type=int, default=256, help="Batch size M")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--iters", type=int, default=500, help="Max training iterations")
    p.add_argument("--eval-every", type=int, default=10, help="Evaluate R(theta) every k iters")
    p.add_argument("--patience", type=int, default=20, help="Early stopping patience in eval steps")
    p.add_argument("--tol", type=float, default=1e-6, help="Minimum R improvement to reset patience")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--history-out", default="training_history.json")
    p.add_argument("--standardize-y", action="store_true", help="Standardize y globally before training")
    p.add_argument("--log1p-y", action="store_true", help="Train on log1p(y) (after optional standardization) and invert for RMSE")
    p.add_argument("--weight-decay", type=float, default=0.0, help="L2 weight decay on W and v")
    p.add_argument("--lr-decay", type=float, default=1.0, help="Multiply lr by this factor at each lr-decay-steps")
    p.add_argument("--lr-decay-steps", type=int, default=0, help="Apply lr decay every this many iterations (0=off)")
    p.add_argument("--eval-train-frac", type=float, default=1.0, help="Fraction of local train samples to evaluate R(theta) on")
    p.add_argument("--eval-train-max-samples", type=int, default=0, help="Cap on number of local train samples used in evaluation (0=all)")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--sse-block", type=int, default=65536, help="Block size when computing SSE")
    p.add_argument("--grad-clip", type=float, default=0.0, help="Global L2 grad clip (0=off)")
    p.add_argument("--loss-clip", type=float, default=0.0, help="Clip squared error at this value in training (0=off)")
    p.add_argument("--stats-out", default=None, help="If set, embed residual/y stats in history JSON")
    return p.parse_args()


def main_train(args) -> None:
    comm, rank, size, np_dtype, mpi_dtype = setup_mpi(args.dtype)
    total_t0 = time.perf_counter()

    if rank == 0:
        print(f"[rank0] Starting. dtype={args.dtype}, activation={args.activation}, hidden={args.hidden}, batch={args.batch}", flush=True)
        print(f"[rank0] Data dir: {args.data_dir}", flush=True)

    dataset, comm_time = load_and_distribute(args.data_dir, comm, rank, size, np_dtype, args.standardize_y)

    if rank == 0:
        print(f"[rank0] Loaded meta: N={dataset.N}, N_test={dataset.N_test}, n_features={dataset.n_features}", flush=True)
        print(f"[rank0] My split: train[{dataset.local_start}:{dataset.local_end}] -> {dataset.local_count}", flush=True)

    label_meta = transform_labels(dataset, np_dtype, args.standardize_y, args.log1p_y)

    model = NeuralNetwork(args.hidden, dataset.n_features, args.activation, args.seed, np_dtype)

    if rank == 0:
        theta = model.get_theta()
    else:
        theta = np.empty(model.theta_size, dtype=np_dtype)

    t0 = time.perf_counter()
    comm.Bcast([theta, mpi_dtype], root=0)
    comm_time += time.perf_counter() - t0
    model.set_theta(theta)

    if rank == 0:
        print("[rank0] Enter training loop", flush=True)

    train_result = run_training_loop(model, theta, dataset, label_meta, args, comm, rank, size, np_dtype, mpi_dtype)

    metrics, metrics_times = compute_final_metrics(model, dataset, label_meta, args, comm, mpi_dtype)

    total_time = time.perf_counter() - total_t0
    compute_time = train_result.compute_time + metrics_times["compute"]
    comm_time += train_result.comm_time + metrics_times["comm"]

    stats = None
    if rank == 0 and args.stats_out:
        stats = collect_residual_stats(model, dataset, label_meta, args.seed)

    if rank == 0:
        save_results(
            args=args,
            size=size,
            training_result=train_result,
            metrics=metrics,
            total_time=total_time,
            compute_time=compute_time,
            comm_time=comm_time,
            dataset=dataset,
            label_meta=label_meta,
            stats=stats,
        )
        print(f"[rank0] Done. Total: {total_time:.3f}s, compute={compute_time:.3f}s, comm={comm_time:.3f}s", flush=True)


def train_mpi(args) -> None:
    main_train(args)


if __name__ == "__main__":
    main_train(parse_args())
