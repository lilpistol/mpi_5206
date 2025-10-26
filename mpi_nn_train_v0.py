import os
import time
import json
import math
import argparse
from typing import Tuple, List, Dict, Optional

import numpy as np
from mpi4py import MPI


# -----------------------------
# Activation functions
# -----------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    z_clip = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clip))


def d_sigmoid_from_a(a: np.ndarray) -> np.ndarray:
    # derivative using output a for numerical stability
    return a * (1.0 - a)


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def d_relu(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(z.dtype)


def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def d_tanh_from_a(a: np.ndarray) -> np.ndarray:
    return 1.0 - a * a


ACTIVATIONS = {
    "sigmoid": {
        "f": sigmoid,
        "df_from_a": d_sigmoid_from_a,
        "needs_z_for_grad": False,
    },
    "relu": {
        "f": relu,
        "df": d_relu,  # needs z
        "needs_z_for_grad": True,
    },
    "tanh": {
        "f": tanh,
        "df_from_a": d_tanh_from_a,
        "needs_z_for_grad": False,
    },
}


# -----------------------------
# Parameter packing / unpacking
# -----------------------------

def init_theta(
    n_hidden: int,
    n_features: int,
    activation: str,
    seed: int = 42,
    dtype: np.dtype = np.float64,
    init: str = "auto",
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fan_in = n_features
    fan_out = n_hidden

    if init == "auto":
        init = "he" if activation == "relu" else "xavier"

    if init == "he":
        std_W = math.sqrt(2.0 / float(fan_in))
    elif init == "xavier":
        std_W = math.sqrt(2.0 / float(fan_in + fan_out))
    else:
        std_W = 0.01

    W = (rng.standard_normal((n_hidden, n_features)) * std_W).astype(dtype)
    b = np.zeros(n_hidden, dtype=dtype)
    # Output layer weights: use 1/sqrt(n_hidden) for stability
    std_v = 1.0 / math.sqrt(max(1, n_hidden))
    v = (rng.standard_normal(n_hidden) * std_v).astype(dtype)
    # Cast scalar to desired dtype safely
    c = np.asarray(0.0, dtype=dtype).item()
    return pack_params(W, b, v, c)


def pack_params(W: np.ndarray, b: np.ndarray, v: np.ndarray, c: float) -> np.ndarray:
    # theta layout: [v (n), c (1), W (n*m), b (n)]
    n_hidden, n_features = W.shape
    theta = np.empty(n_hidden + 1 + n_hidden * n_features + n_hidden, dtype=W.dtype)
    offset = 0
    theta[offset:offset + n_hidden] = v
    offset += n_hidden
    theta[offset] = c
    offset += 1
    theta[offset:offset + n_hidden * n_features] = W.reshape(-1)
    offset += n_hidden * n_features
    theta[offset:offset + n_hidden] = b
    return theta


def unpack_params(theta: np.ndarray, n_hidden: int, n_features: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    # inverse of pack_params
    offset = 0
    v = theta[offset:offset + n_hidden]
    offset += n_hidden
    c = float(theta[offset])
    offset += 1
    W = theta[offset:offset + n_hidden * n_features].reshape((n_hidden, n_features))
    offset += n_hidden * n_features
    b = theta[offset:offset + n_hidden]
    return W, b, v, c


# -----------------------------
# Forward / Backward (single sample)
# -----------------------------

def forward_single(x: np.ndarray,
                   theta: np.ndarray,
                   n_hidden: int,
                   n_features: int,
                   act_key: str) -> Tuple[float, Dict[str, np.ndarray]]:
    W, b, v, c = unpack_params(theta, n_hidden, n_features)
    act = ACTIVATIONS[act_key]
    z = W @ x + b  # (n_hidden,)
    a = act["f"](z)
    y_pred = float(v @ a + c)
    cache = {"x": x, "z": z, "a": a}
    return y_pred, cache


def backward_single(y_pred: float,
                    y_true: float,
                    theta: np.ndarray,
                    cache: Dict[str, np.ndarray],
                    n_hidden: int,
                    n_features: int,
                    act_key: str) -> np.ndarray:
    W, b, v, c = unpack_params(theta, n_hidden, n_features)
    x = cache["x"]
    z = cache["z"]
    a = cache["a"]
    e = y_pred - y_true

    # Output layer grads
    grad_c = e
    grad_v = e * a  # (n_hidden,)

    # Hidden layer grads
    act = ACTIVATIONS[act_key]
    if act.get("needs_z_for_grad", False):
        da_dz = act["df"](z)
    else:
        # use derivative from a
        da_dz = act["df_from_a"](a)

    delta = e * v * da_dz  # (n_hidden,)
    grad_b = delta
    grad_W = delta[:, None] * x[None, :]  # (n_hidden, n_features)

    # pack in same order as theta
    grad = pack_params(grad_W, grad_b, grad_v, grad_c)
    return grad


# -----------------------------
# Utilities
# -----------------------------

def array_splits_lengths(total: int, size: int) -> List[int]:
    base = total // size
    rem = total % size
    return [base + (1 if r < rem else 0) for r in range(size)]


def split_indices(total: int, size: int) -> List[Tuple[int, int]]:
    lengths = array_splits_lengths(total, size)
    starts = np.cumsum([0] + lengths[:-1]).tolist()
    return list(zip(starts, [s + l for s, l in zip(starts, lengths)]))


def filter_local_batch(global_indices: np.ndarray, start: int, end: int) -> np.ndarray:
    # select indices j in [start, end)
    mask = (global_indices >= start) & (global_indices < end)
    return global_indices[mask] - start


def compute_local_gradient_sum(theta: np.ndarray,
                               X_local: np.ndarray,
                               y_local: np.ndarray,
                               local_batch_indices: np.ndarray,
                               n_hidden: int,
                               act_key: str,
                               loss_clip: float = 0.0) -> np.ndarray:
    # Vectorized over the local portion of the batch
    if local_batch_indices.size == 0:
        return np.zeros_like(theta)

    W, b, v, c = unpack_params(theta, n_hidden, X_local.shape[1])
    Xb = X_local[local_batch_indices]  # (B, m)
    yb = y_local[local_batch_indices]  # (B,)

    # Forward for batch
    Z = Xb @ W.T + b  # (B, n_hidden)
    act = ACTIVATIONS[act_key]
    if act_key == "relu":
        A = relu(Z)
        dA_dZ = d_relu(Z)
    elif act_key == "sigmoid":
        A = sigmoid(Z)
        dA_dZ = d_sigmoid_from_a(A)
    else:  # tanh
        A = tanh(Z)
        dA_dZ = d_tanh_from_a(A)

    y_pred = A @ v + c  # (B,)
    E = y_pred - yb      # (B,)
    if loss_clip and loss_clip > 0.0:
        max_abs_e = math.sqrt(loss_clip)
        E = np.clip(E, -max_abs_e, max_abs_e)

    # Gradients
    grad_c = np.sum(E, dtype=theta.dtype)
    grad_v = A.T @ E  # (n_hidden,)
    Delta = (E[:, None] * v[None, :] * dA_dZ)  # (B, n_hidden)
    grad_b = np.sum(Delta, axis=0)
    grad_W = Delta.T @ Xb  # (n_hidden, n_features)

    return pack_params(grad_W, grad_b, grad_v, grad_c)


def compute_sse_over_indices(theta: np.ndarray,
                             X: np.ndarray,
                             y: np.ndarray,
                             indices: np.ndarray,
                             n_hidden: int,
                             act_key: str,
                             block: int = 65536,
                             loss_clip: float = 0.0) -> float:
    # Vectorized SSE over a subset of rows, processed in blocks
    if indices.size == 0:
        return 0.0
    W, b, v, c = unpack_params(theta, n_hidden, X.shape[1])
    total = 0.0
    act = ACTIVATIONS[act_key]
    for start in range(0, indices.size, block):
        end = min(indices.size, start + block)
        idx = indices[start:end]
        Xb = X[idx]
        yb = y[idx]
        Z = Xb @ W.T + b
        if act_key == "relu":
            A = relu(Z)
        elif act_key == "sigmoid":
            A = sigmoid(Z)
        else:
            A = tanh(Z)
        y_pred = A @ v + c
        E = y_pred - yb
        if loss_clip and loss_clip > 0.0:
            max_abs_e = math.sqrt(loss_clip)
            E = np.clip(E, -max_abs_e, max_abs_e)
        E64 = E.astype(np.float64)
        total += float(np.sum(E64 * E64))
    return total


def load_preprocessed(data_dir: str):
    # Prefer data_dir if exists; else try current directory
    base = data_dir if data_dir and os.path.isdir(data_dir) else "."
    X_train_path = os.path.join(base, "X_train_scaled.npy")
    y_train_path = os.path.join(base, "y_train.npy")
    X_test_path = os.path.join(base, "X_test_scaled.npy")
    y_test_path = os.path.join(base, "y_test.npy")

    X_train = np.load(X_train_path, mmap_mode="r")
    y_train = np.load(y_train_path, mmap_mode="r")
    X_test = np.load(X_test_path, mmap_mode="r")
    y_test = np.load(y_test_path, mmap_mode="r")
    return X_train, y_train, X_test, y_test


# -----------------------------
# Training loop with MPI
# -----------------------------

def train_mpi(
    data_dir: str = "data_preprocessed",
    activation: str = "relu",
    n_hidden: int = 64,
    batch_size: int = 256,
    lr: float = 1e-3,
    max_iters: int = 500,
    eval_every: int = 10,
    patience: int = 20,
    tol: float = 1e-6,
    seed: int = 42,
    history_out: str = "training_history.json",
    # New features
    standardize_y: bool = False,
    log1p_y: bool = False,
    weight_decay: float = 0.0,
    lr_decay: float = 1.0,
    lr_decay_steps: int = 0,
    eval_train_frac: float = 1.0,
    eval_train_max_samples: int = 0,
    dtype: str = "float64",
    sse_block: int = 65536,
    grad_clip: float = 0.0,
    stats_out: Optional[str] = None,
    loss_clip: float = 0.0,
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if activation not in ACTIVATIONS:
        raise ValueError(f"Unsupported activation: {activation}. Choose from {list(ACTIVATIONS.keys())}")

    # Timers
    total_t0 = time.perf_counter()
    comm_time_total = 0.0
    compute_time_total = 0.0

    # Resolve dtype and MPI datatype
    np_dtype = np.float32 if dtype == "float32" else np.float64
    mpi_dtype = MPI.FLOAT if np_dtype == np.float32 else MPI.DOUBLE

    # 1) Rank 0 loads and scatters data
    if rank == 0:
        X_train, y_train, X_test, y_test = load_preprocessed(data_dir)
        N, n_features = X_train.shape
        N_test = X_test.shape[0]
        # y standardization parameters (global)
        if standardize_y:
            y_mean = float(np.mean(y_train))
            y_std = float(np.std(y_train) + 1e-12)
        else:
            y_mean = 0.0
            y_std = 1.0
        # optional log1p transform statistics (none needed; applied elementwise)
        # convert to numpy arrays in memory per split to scatter
        # Split train
        train_splits = split_indices(N, size)
        X_train_splits = [np.asarray(X_train[s:e]) for (s, e) in train_splits]
        y_train_splits = [np.asarray(y_train[s:e]) for (s, e) in train_splits]

        # Split test as well to avoid broadcasting massive arrays
        test_splits = split_indices(N_test, size)
        X_test_splits = [np.asarray(X_test[s:e]) for (s, e) in test_splits]
        y_test_splits = [np.asarray(y_test[s:e]) for (s, e) in test_splits]

        meta = {
            "N": N,
            "n_features": n_features,
            "N_test": N_test,
            "train_splits": train_splits,
            "test_splits": test_splits,
            "y_mean": y_mean,
            "y_std": y_std,
        }
    else:
        X_train_splits = y_train_splits = None
        X_test_splits = y_test_splits = None
        meta = None

    t0 = time.perf_counter()
    meta = comm.bcast(meta, root=0)
    comm_time_total += time.perf_counter() - t0

    N = meta["N"]
    n_features = meta["n_features"]
    N_test = meta["N_test"]
    train_splits = meta["train_splits"]
    test_splits = meta["test_splits"]
    y_mean = float(meta["y_mean"])  # for potential de-standardization
    y_std = float(meta["y_std"]) if standardize_y else 1.0

    # Receive my local data
    t0 = time.perf_counter()
    X_local = comm.scatter(X_train_splits, root=0)
    y_local = comm.scatter(y_train_splits, root=0)
    X_test_local = comm.scatter(X_test_splits, root=0)
    y_test_local = comm.scatter(y_test_splits, root=0)
    comm_time_total += time.perf_counter() - t0

    local_start, local_end = train_splits[rank]
    local_count = local_end - local_start

    # Cast to desired dtype
    X_local = X_local.astype(np_dtype, copy=False)
    X_test_local = X_test_local.astype(np_dtype, copy=False)
    y_local = y_local.astype(np_dtype, copy=False)
    y_test_local = y_test_local.astype(np_dtype, copy=False)

    # Apply label transforms (order: standardize then log1p if requested)
    if standardize_y:
        y_local = (y_local - y_mean) / y_std
        y_test_local = (y_test_local - y_mean) / y_std
    if log1p_y:
        # log1p in the current label space
        y_local = np.log1p(np.maximum(y_local, -0.999999))
        y_test_local = np.log1p(np.maximum(y_test_local, -0.999999))

    # 2) Initialize theta on rank 0 then broadcast
    if rank == 0:
        theta = init_theta(n_hidden, n_features, activation=activation, seed=seed, dtype=np_dtype)
    else:
        theta = np.empty(n_hidden + 1 + n_hidden * n_features + n_hidden, dtype=np_dtype)

    t0 = time.perf_counter()
    comm.Bcast([theta, mpi_dtype], root=0)
    comm_time_total += time.perf_counter() - t0

    # 3) Training loop
    rng = np.random.default_rng(seed)
    rng_eval = np.random.default_rng(seed + 123456)
    history = []  # list of dicts: {iter, R}
    best_R = math.inf
    best_iter = -1

    # convenience buffers
    grad_buf = np.zeros_like(theta)
    total_grad_buf = np.zeros_like(theta)

    for it in range(1, max_iters + 1):
        # 3.1) Sample batch indices on rank 0 and broadcast
        if rank == 0:
            # sample without replacement
            batch_indices = rng.choice(N, size=min(batch_size, N), replace=False).astype(np.int64)
        else:
            batch_indices = np.empty(min(batch_size, N), dtype=np.int64)

        t0 = time.perf_counter()
        comm.Bcast([batch_indices, MPI.LONG_LONG], root=0)
        comm_time_total += time.perf_counter() - t0

        # 3.2) Compute local gradient sum over my portion of batch
        t0 = time.perf_counter()
        local_batch = filter_local_batch(batch_indices, local_start, local_end)
        grad_buf[:] = compute_local_gradient_sum(theta, X_local, y_local, local_batch, n_hidden, activation, loss_clip=loss_clip)
        compute_time_total += time.perf_counter() - t0

        # 3.3) Allreduce to obtain global gradient sum
        t0 = time.perf_counter()
        comm.Allreduce([grad_buf, mpi_dtype], [total_grad_buf, mpi_dtype], op=MPI.SUM)
        comm_time_total += time.perf_counter() - t0

        # 3.4) Update theta identically on all processes
        grad_avg = total_grad_buf / float(batch_indices.size)
        # L2 weight decay on W and v only
        if weight_decay > 0.0:
            # slices: v [0:n_hidden], c [n_hidden], W [n_hidden+1 : n_hidden+1+n_hidden*n_features], b [last n_hidden]
            v_slice = slice(0, n_hidden)
            W_start = n_hidden + 1
            W_end = W_start + n_hidden * n_features
            W_slice = slice(W_start, W_end)
            grad_avg[v_slice] += weight_decay * theta[v_slice]
            grad_avg[W_slice] += weight_decay * theta[W_slice]
        # optional gradient clipping (global L2 norm)
        if grad_clip and grad_clip > 0.0:
            gnorm = float(np.linalg.norm(grad_avg))
            if gnorm > grad_clip:
                grad_avg *= (grad_clip / (gnorm + 1e-12))
        theta -= lr * grad_avg

        # 3.5) Evaluate R(theta) periodically for convergence/history
        do_eval = (it % eval_every == 0) or (it == 1)
        if do_eval:
            # Compute local SSE on (possibly a subset of) training set
            t0 = time.perf_counter()
            # choose subset indices
            nloc = X_local.shape[0]
            if eval_train_frac < 1.0 or (eval_train_max_samples and eval_train_max_samples > 0):
                target = nloc
                if eval_train_frac < 1.0:
                    target = min(target, max(1, int(round(eval_train_frac * nloc))))
                if eval_train_max_samples and eval_train_max_samples > 0:
                    target = min(target, eval_train_max_samples)
                if target < nloc:
                    idx = rng_eval.choice(nloc, size=target, replace=False)
                else:
                    idx = np.arange(nloc)
            else:
                idx = np.arange(nloc)

            local_sse = compute_sse_over_indices(theta, X_local, y_local, idx, n_hidden, activation, block=sse_block, loss_clip=loss_clip)
            compute_time_total += time.perf_counter() - t0

            # Aggregate SSE across processes
            t0 = time.perf_counter()
            total_sse = np.array([local_sse], dtype=np.float64)
            comm.Allreduce(MPI.IN_PLACE, total_sse, op=MPI.SUM)
            comm_time_total += time.perf_counter() - t0

            # Determine denominator based on whether we evaluated full set or subset
            # For monitoring we normalize by the evaluated sample count
            # Gather local counts used for evaluation
            t0 = time.perf_counter()
            local_eval_count = np.array([idx.size], dtype=np.float64)
            comm.Allreduce(MPI.IN_PLACE, local_eval_count, op=MPI.SUM)
            comm_time_total += time.perf_counter() - t0

            # Convert R back to original scale
            # If log1p_y: approximate using delta in original space ~ exp(y_hat)-1 - exp(y_true)-1
            # For history we keep label space used in training but report in original units when possible.
            if log1p_y:
                # For efficiency, we approximate scale via local transform on subset idx used, but here we lack predictions.
                # Fallback: cannot convert R exactly; report in transformed space for history.
                scale = 1.0
            else:
                scale = (y_std * y_std) if standardize_y else 1.0
            R = (total_sse[0] * scale) / (2.0 * float(local_eval_count[0]))
            if rank == 0:
                history.append({"iter": it, "R": R})
                print(f"Iter {it:6d}  R(theta)={R:.6f}")

            # Early stopping check on rank 0
            t0 = time.perf_counter()
            R_buf = np.array([R], dtype=np.float64)
            comm.Bcast(R_buf, root=0)
            comm_time_total += time.perf_counter() - t0

            # All ranks get latest R value
            R_latest = float(R_buf[0])
            if rank == 0:
                improved = (best_R - R_latest) > tol
                if improved:
                    best_R = R_latest
                    best_iter = it
                elif (it - best_iter) >= patience and best_iter >= 0:
                    print(f"Early stopping at iter {it} (best_iter={best_iter}, best_R={best_R:.6f})")
                    break

        # 3.6) Learning rate decay
        if lr_decay_steps and lr_decay_steps > 0 and (it % lr_decay_steps == 0):
            lr *= lr_decay

    # 4) Final metrics: RMSE on train and test (parallel)
    # Train SSE already computed above periodically; recompute now for final RMSE
    t0 = time.perf_counter()
    # SSE is computed in the training label space; convert to original later
    # For final RMSE, do NOT clip loss; report true RMSE
    local_sse_train = compute_sse_over_indices(theta, X_local, y_local, np.arange(X_local.shape[0]), n_hidden, activation, block=sse_block, loss_clip=0.0)
    local_sse_test = compute_sse_over_indices(theta, X_test_local, y_test_local, np.arange(X_test_local.shape[0]), n_hidden, activation, block=sse_block, loss_clip=0.0)
    compute_time_total += time.perf_counter() - t0

    t0 = time.perf_counter()
    buf = np.array([local_sse_train, local_sse_test, float(local_count), float(X_test_local.shape[0])], dtype=np.float64)
    comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)
    comm_time_total += time.perf_counter() - t0

    total_sse_train, total_sse_test, total_N_train, total_N_test = buf
    # Convert back to original scale
    if log1p_y:
        # Need exact recomputation in original space for RMSE: do a second pass computing predictions and residuals after inverse transform
        def rmse_original(theta, X, y_t, n_hidden, act_key):
            n = X.shape[0]
            n_features = X.shape[1]
            W, b, v, c = unpack_params(theta, n_hidden, n_features)
            sse = 0.0
            for start in range(0, n, sse_block):
                end = min(n, start + sse_block)
                Xb = X[start:end]
                yb = y_t[start:end]
                Z = Xb @ W.T + b
                if act_key == "relu":
                    A = relu(Z)
                elif act_key == "sigmoid":
                    A = sigmoid(Z)
                else:
                    A = tanh(Z)
                y_pred_t = A @ v + c
                # inverse transforms
                y_pred = np.expm1(y_pred_t)
                y_true = np.expm1(yb)
                if standardize_y:
                    y_pred = y_pred * y_std + y_mean
                    y_true = y_true * y_std + y_mean
                err = (y_pred - y_true).astype(np.float64)
                sse += float(np.sum(err * err))
            return math.sqrt(sse / float(n))

        rmse_train = rmse_original(theta, X_local, y_local, n_hidden, activation)
        rmse_test = rmse_original(theta, X_test_local, y_test_local, n_hidden, activation)
    else:
        if standardize_y:
            total_sse_train *= (y_std * y_std)
            total_sse_test *= (y_std * y_std)
        rmse_train = math.sqrt(total_sse_train / total_N_train)
        rmse_test = math.sqrt(total_sse_test / total_N_test)

    total_time = time.perf_counter() - total_t0

    if rank == 0:
        print("\nTraining complete.")
        print(f"Processes: {size}")
        print(f"Total time: {total_time:.3f}s  (compute: {compute_time_total:.3f}s, comm: {comm_time_total:.3f}s)")
        print(f"RMSE train: {rmse_train:.6f}")
        print(f"RMSE test : {rmse_test:.6f}")

        # Save history and summary
        out = {
            "config": {
                "activation": activation,
                "n_hidden": n_hidden,
                "batch_size": batch_size,
                "lr": lr,
                "lr_decay": lr_decay,
                "lr_decay_steps": lr_decay_steps,
                "max_iters": max_iters,
                "eval_every": eval_every,
                "eval_train_frac": eval_train_frac,
                "eval_train_max_samples": eval_train_max_samples,
                "patience": patience,
                "tol": tol,
                "seed": seed,
                "processes": size,
                "standardize_y": standardize_y,
                "log1p_y": log1p_y,
                "weight_decay": weight_decay,
                "dtype": dtype,
                "grad_clip": grad_clip,
                "loss_clip": loss_clip,
            },
            "history": history,
            "metrics": {
                "rmse_train": rmse_train,
                "rmse_test": rmse_test,
                "time_total_sec": total_time,
                "time_compute_sec": compute_time_total,
                "time_comm_sec": comm_time_total,
            },
            "data": {
                "N_train": int(total_N_train),
                "N_test": int(total_N_test),
                "n_features": n_features,
                "y_mean": y_mean,
                "y_std": y_std,
            },
        }
        # Optional residual/y stats sampling
        if stats_out:
            def sample_stats(theta, X, y_t, n_hidden, act_key, name, k=200000):
                n = X.shape[0]
                sel = np.random.default_rng(seed).choice(n, size=min(k, n), replace=False)
                W, b, v, c = unpack_params(theta, n_hidden, X.shape[1])
                Xb = X[sel]
                yb = y_t[sel]
                Z = Xb @ W.T + b
                if act_key == "relu":
                    A = relu(Z)
                elif act_key == "sigmoid":
                    A = sigmoid(Z)
                else:
                    A = tanh(Z)
                y_pred_t = A @ v + c
                # inverse transform
                if log1p_y:
                    y_pred = np.expm1(y_pred_t)
                    y_true = np.expm1(yb)
                else:
                    y_pred = y_pred_t
                    y_true = yb
                if standardize_y:
                    y_pred = y_pred * y_std + y_mean
                    y_true = y_true * y_std + y_mean
                err = (y_pred - y_true).astype(np.float64)
                def q(x, p):
                    return float(np.quantile(x, p))
                return {
                    f"{name}_y_mean": float(np.mean(y_true)),
                    f"{name}_y_std": float(np.std(y_true)),
                    f"{name}_y_p95": q(y_true, 0.95),
                    f"{name}_y_p99": q(y_true, 0.99),
                    f"{name}_abs_err_mean": float(np.mean(np.abs(err))),
                    f"{name}_abs_err_p95": q(np.abs(err), 0.95),
                    f"{name}_abs_err_p99": q(np.abs(err), 0.99),
                }

            stats = {}
            try:
                stats.update(sample_stats(theta, X_local, y_local, n_hidden, activation, name="train"))
                stats.update(sample_stats(theta, X_test_local, y_test_local, n_hidden, activation, name="test"))
                out["residual_stats"] = stats
            except Exception as e:
                print(f"Warning: stats sampling failed: {e}")
        try:
            with open(history_out, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            print(f"Saved training history to {history_out}")
        except Exception as e:
            print(f"Warning: could not save history to {history_out}: {e}")


def parse_args():
    p = argparse.ArgumentParser(description="One-hidden-layer NN with MPI SGM training")
    p.add_argument("--data-dir", default="data_preprocessed", help="Directory containing preprocessed .npy files")
    p.add_argument("--activation", default="relu", choices=list(ACTIVATIONS.keys()))
    p.add_argument("--hidden", type=int, default=128, help="Hidden layer size")
    p.add_argument("--batch", type=int, default=256, help="Batch size M")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--iters", type=int, default=500, help="Max training iterations")
    p.add_argument("--eval-every", type=int, default=10, help="Evaluate R(theta) every k iters")
    p.add_argument("--patience", type=int, default=20, help="Early stopping patience in eval steps")
    p.add_argument("--tol", type=float, default=1e-6, help="Minimum R improvement to reset patience")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--history-out", default="training_history.json")
    # New options
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


if __name__ == "__main__":
    args = parse_args()
    train_mpi(
        data_dir=args.data_dir,
        activation=args.activation,
        n_hidden=args.hidden,
        batch_size=args.batch,
        lr=args.lr,
        max_iters=args.iters,
        eval_every=args.eval_every,
        patience=args.patience,
        tol=args.tol,
        seed=args.seed,
        history_out=args.history_out,
        standardize_y=args.standardize_y,
        log1p_y=args.log1p_y,
        weight_decay=args.weight_decay,
        lr_decay=args.lr_decay,
        lr_decay_steps=args.lr_decay_steps,
        eval_train_frac=args.eval_train_frac,
        eval_train_max_samples=args.eval_train_max_samples,
        dtype=args.dtype,
        sse_block=args.sse_block,
        grad_clip=args.grad_clip,
        loss_clip=args.loss_clip,
        stats_out=args.stats_out,
    )
