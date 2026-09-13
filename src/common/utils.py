"""English documentation."""
import os
import sys
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42):
    """English documentation."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    for env_var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[env_var] = "1"
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    if x.size == 0 or y.size == 0 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def bray_curtis(x: np.ndarray, y: np.ndarray, eps: float = 1e-10) -> float:
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    return float(np.sum(np.abs(x - y)) / (np.sum(x + y) + eps))


def mse_rmse_mae(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return mse, rmse, mae


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-20:
        return 0.0
    return float(1 - ss_res / ss_tot)


def split_subjects(subject_ids, split="82", train_ratio=0.8, n_splits=5,
                   train_sids=None, test_sids=None, seed=42):
    """English documentation."""
    sids = sorted(set(subject_ids))
    if len(sids) < 2:
        raise ValueError("A subject-level split requires at least two subjects")
    rng = np.random.RandomState(seed)

    if split in ("82", "73"):
        ratio = float(split[0]) / 10.0  # "73" -> 0.7
        if split == "82":
            ratio = train_ratio if train_ratio != 0.8 else 0.8
        if not 0 < ratio < 1:
            raise ValueError("The training ratio must be between 0 and 1")
        n_train = min(len(sids) - 1, max(1, int(len(sids) * ratio)))
        perm = rng.permutation(sids)
        return [(list(perm[:n_train]), list(perm[n_train:]))]

    if split == "5fold":
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=min(n_splits, len(sids)), shuffle=True, random_state=seed)
        return [([sids[i] for i in tr], [sids[i] for i in te]) for tr, te in kf.split(sids)]

    if split == "loso":
        return [([s for s in sids if s != leave], [leave]) for leave in sids]

    if split == "custom":
        def _read_sids_from_xlsx(path, name):
            """English documentation."""
            if not path:
                raise ValueError("split=custom requires --train-sids and --test-sids xlsx files")
            df = pd.read_excel(path)
            if df.shape[1] == 0:
                raise ValueError(f"split=custom {name}  file is empty: {path}")
            cols = list(df.columns)
            id_col = next((c for c in cols
                           if str(c).lower() in ("subjectid", "sampleid", "subject_id", "patientid")),
                          cols[0])
            vals = [str(v).strip() for v in df[id_col].dropna().unique()]
            vals = [v for v in vals if v]
            if not vals:
                raise ValueError(f"split=custom {name}  file contains no SubjectID: {path}")
            return vals

        tr = _read_sids_from_xlsx(train_sids, "--train-sids")
        te = _read_sids_from_xlsx(test_sids, "--test-sids")
        lookup = {str(s): s for s in subject_ids}
        unknown = sorted((set(tr) | set(te)) - set(lookup))
        if unknown:
            raise ValueError(f"Custom split contains unknown SubjectID values: {unknown}")
        overlap = sorted(set(tr) & set(te))
        if overlap:
            raise ValueError(f"Custom training and test sets overlap: {overlap}")
        tr = [lookup[s] for s in tr]
        te = [lookup[s] for s in te]
        return [(tr, te)]

    raise ValueError(f"Unknown split={split}; supported: 82, 73, 5fold, loso, custom")


def label_encode_group(group_series):
    """English documentation."""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    encoded = le.fit_transform(group_series.astype(str))
    return encoded, list(le.classes_), le


def detect_group_column(df: pd.DataFrame):
    """English documentation."""
    for col in df.columns:
        if str(col).strip().lower() in ("group", "label", "class"):
            return col
    return None


def parse_time_col(time_values):
    """English documentation."""
    result = []
    for v in time_values:
        if isinstance(v, (int, float, np.integer, np.floating)):
            result.append(float(v))
            continue
        s = str(v).strip().lower()
        if s.startswith("t"):
            try:
                result.append(float(s[1:]))
                continue
            except ValueError:
                pass
        try:
            result.append(float(s))
        except Exception:
            result.append(np.nan)
    arr = np.array(result, float)
    if np.any(np.isnan(arr)):
        warnings.warn("Some Time values could not be parsed; using sequential indices")
        for i, v in enumerate(arr):
            if np.isnan(v):
                arr[i] = float(i)
    return arr



def try_save_figure(fig, out_path: str, dpi=150):
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    except Exception as e:
        warnings.warn(f"Failed to save image {out_path}: {e}")
