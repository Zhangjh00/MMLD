import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..common import (
    set_seed, ensure_dir,
    pearson_r, bray_curtis,
    split_subjects, build_history_from_abundances,
    MicrobiomeDataLoader, gLVModel, gLVModel_RK, build_glv_model,
    train_glv_model, load_glv_checkpoint,
)


def _load_or_train_fold_model(loader, train_sids, args, verbose=True):
    """English documentation."""
    device = getattr(args, "device", "cpu")
    glv_method = getattr(args, "glv_method", "euler")
    model_file = getattr(args, "model", None)
    if model_file and os.path.exists(model_file):
        try:
            ckpt = torch.load(model_file, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(model_file, map_location=device)
        checkpoint_scale = ckpt.get("SCALE", ckpt.get("scale"))
        if checkpoint_scale is not None:
            loader.rescale(float(checkpoint_scale))
        model = load_glv_checkpoint(ckpt, device=device)
        model.eval()
        return model

    loader.prepare_training_data(sid_subset=train_sids)
    if len(loader.training_pairs) == 0:
        raise RuntimeError(f"Training subject subset {train_sids} cannot produce training pairs; check the number of time points.")
    model = build_glv_model(glv_method, loader.N, loader.lag_min, loader.lag_max, loader.edge_info)
    train_glv_model(
        model, loader.training_pairs,
        epochs=getattr(args, "epochs", 10),
        lr=getattr(args, "lr", 0.01),
        device=device,
        verbose=verbose,
    )
    model.to(device)
    model.eval()
    return model


def _recent_state_queue(abundances, current_index, lag_max, device):
    """Return states at lags 1..lag_max relative to current_index."""
    recent = []
    for lag in range(1, lag_max + 1):
        index = current_index - lag
        state = abundances[index] if index >= 0 else abundances[0]
        recent.append(torch.tensor(state, dtype=torch.float32, device=device))
    return recent


# =========================================================
# =========================================================
def predict_sequence(model, loader, test_sid, initial_ratio=0.15, device="cpu"):
    """English documentation."""
    sample = loader.samples[test_sid]
    T = sample["T"]
    abundances = sample["abundances"]
    times = sample["times"]
    n_init = max(loader.lag_max + 1, int(np.ceil(T * initial_ratio)))
    if n_init > T:
        n_init = T

    recent_states = _recent_state_queue(
        abundances, n_init - 1, loader.lag_max, device
    )

    h = torch.tensor(abundances[n_init - 1, :], dtype=torch.float32, device=device)
    pred = []
    true = []

    for step in range(n_init, T):
        hist_list = []
        for lag in range(model.lag_min, model.lag_max + 1):
            hist_list.append(recent_states[lag - 1])
        history_tensor = torch.stack(hist_list)
        dt = torch.tensor(float(times[step] - times[step - 1]), dtype=torch.float32, device=device)
        h_next = model(h, history_tensor, dt)
        pred.append(h_next.detach().cpu().numpy().copy())
        true.append(abundances[step, :].copy())
        recent_states = [h.detach().clone()] + recent_states[: loader.lag_max - 1]
        h = h_next
    if not true:
        empty = np.zeros((0, loader.N), dtype=float)
        return times[n_init:n_init], empty, empty, n_init
    return times[n_init:], np.vstack(true), np.vstack(pred), n_init


def select_similar_training(loader, train_sids, test_sample, top_percent=0.01,
                            initial_ratio=0.15):
    """English documentation."""
    test_count = max(
        loader.lag_max + 1,
        int(np.ceil(test_sample["T"] * float(initial_ratio))),
    )
    test_count = min(test_count, test_sample["T"])
    test_profile = test_sample["abundances"][:test_count].mean(axis=0)
    distances = []
    for sid in train_sids:
        train_sample = loader.samples[sid]
        train_count = min(test_count, train_sample["T"])
        train_profile = train_sample["abundances"][:train_count].mean(axis=0)
        distances.append((sid, bray_curtis(test_profile, train_profile)))
    distances.sort(key=lambda x: x[1])
    n_keep = max(1, int(np.ceil(len(distances) * top_percent)))
    return [sid for sid, _ in distances[:n_keep]]


def predict_match(loader, train_sids, test_sid, initial_ratio=0.15,
                  top_percent=0.01, device="cpu", args=None,
                  use_similar=True,
                  _cache=None):
    """English documentation."""
    if use_similar:
        similar_sids = select_similar_training(
            loader, train_sids, loader.samples[test_sid], top_percent,
            initial_ratio=initial_ratio,
        )
    else:
        similar_sids = list(train_sids)
    if len(similar_sids) == 0:
        similar_sids = list(train_sids)

    cache_key = (float(loader.SCALE), tuple(sorted(similar_sids)))
    if _cache is not None and cache_key in _cache:
        model = _cache[cache_key]
    else:
        loader.prepare_training_data(sid_subset=similar_sids)
        if len(loader.training_pairs) == 0:
            raise RuntimeError(f"[match] Similar training subjects {similar_sids} cannot produce training pairs.")
        glv_method = getattr(args, "glv_method", "euler") if args is not None else "euler"
        model = build_glv_model(glv_method, loader.N, loader.lag_min, loader.lag_max, loader.edge_info)
        epochs = getattr(args, "epochs", 10) if args is not None else 10
        lr = getattr(args, "lr", 0.01) if args is not None else 0.01
        print(f"  [match sid={test_sid}] similar={len(similar_sids)} subjects → train a dedicated MMLD model")
        train_glv_model(model, loader.training_pairs, epochs=epochs, lr=lr, device=device, verbose=False)
        model.to(device)
        model.eval()
        if _cache is not None:
            _cache[cache_key] = model

    return predict_sequence(model, loader, test_sid, initial_ratio=initial_ratio, device=device)


def predict_step1(model, loader, test_sid, initial_ratio=0.15, device="cpu"):
    """English documentation."""
    sample = loader.samples[test_sid]
    T = sample["T"]
    abundances = sample["abundances"]
    times = sample["times"]
    n_init = max(loader.lag_max + 1, int(np.ceil(T * initial_ratio)))
    if n_init >= T:
        n_init = max(loader.lag_max + 1, T - 1)

    pred_list, true_list = [], []
    for t in range(n_init - 1, T - 1):
        history = build_history_from_abundances(abundances, t, model.lag_min, model.lag_max).to(device)
        h_cur = torch.tensor(abundances[t, :], dtype=torch.float32, device=device)
        dt = torch.tensor(float(times[t + 1] - times[t]), dtype=torch.float32, device=device)
        pred = model(h_cur, history, dt).detach().cpu().numpy()
        pred_list.append(pred)
        true_list.append(abundances[t + 1, :].copy())
    if not true_list:
        return times[-1:], abundances[-2:-1], abundances[-2:-1], n_init
    return times[n_init:], np.vstack(true_list), np.vstack(pred_list), n_init


def predict_stepn(model, loader, test_sid, initial_ratio=0.15, n_steps=3, device="cpu"):
    """English documentation."""
    sample = loader.samples[test_sid]
    T = sample["T"]
    abundances = sample["abundances"]
    times = sample["times"]
    n_init = max(loader.lag_max + 1, int(np.ceil(T * initial_ratio)))
    if n_init > T:
        n_init = T

    pred = []
    true = []

    t = n_init - 1
    while t < T - 1:
        recent_states = _recent_state_queue(
            abundances, t, loader.lag_max, device
        )
        h = torch.tensor(abundances[t, :], dtype=torch.float32, device=device)

        for s in range(n_steps):
            if t + s + 1 >= T:
                break
            hist_list = []
            for lag in range(model.lag_min, model.lag_max + 1):
                hist_list.append(recent_states[lag - 1])
            history_tensor = torch.stack(hist_list)
            dt = torch.tensor(float(times[t + s + 1] - times[t + s]), dtype=torch.float32, device=device)
            h_next = model(h, history_tensor, dt)
            pred.append(h_next.detach().cpu().numpy().copy())
            true.append(abundances[t + s + 1, :].copy())
            recent_states = [h.detach().clone()] + recent_states[: loader.lag_max - 1]
            h = h_next
        t += n_steps
    if not true:
        empty = np.zeros((0, loader.N), dtype=float)
        return times[n_init:n_init], empty, empty, n_init
    return times[n_init:n_init + len(true)], np.vstack(true), np.vstack(pred), n_init


def main_predict(args):
    set_seed(getattr(args, "seed", 42))
    out_dir = getattr(args, "output_dir", "outputs/01_prediction")
    ensure_dir(out_dir)

    loader = MicrobiomeDataLoader(
        args.abundance, args.graph,
        group_is_present=(None if not getattr(args, "no_group", False) else False),
    )
    loader.load_abundance()
    loader.load_graph()

    split = getattr(args, "split", "82")
    mode = str(getattr(args, "mode", "sequence"))
    m_step = re.fullmatch(r"step([1-9]\d*)", mode)
    if mode not in ("sequence", "match") and not m_step:
        raise ValueError(f"Unknown mode={mode}; supported: sequence, match, or stepN")

    subject_ids = list(loader.samples.keys())
    fold_list = split_subjects(
        subject_ids, split=split,
        train_ratio=getattr(args, "train_ratio", 0.8),
        n_splits=getattr(args, "n_splits", 5),
        train_sids=getattr(args, "train_sids", None),
        test_sids=getattr(args, "test_sids", None),
        seed=getattr(args, "seed", 42),
    )
    if split in ("73",) and not hasattr(args, "_train_ratio_set"):
        pass

    initial_ratio = float(getattr(args, "initial_ratio", 0.15))
    n_steps = int(m_step.group(1)) if m_step else 3
    top_percent = float(getattr(args, "top_percent", 0.01))
    use_similar = True
    device = getattr(args, "device", "cpu")

    match_model_cache = {}

    all_true_rows_wide = []
    all_long_rows = []
    fold_summaries = []
    subject_metric_rows = []
    per_time_r_rows = []
    similar_export_rows = []
    fold_scales = []

    for fold_idx, (train_sids, test_sids) in enumerate(fold_list, 1):
        print(f"\n===== Fold {fold_idx}/{len(fold_list)}  train={len(train_sids)} test={len(test_sids)}  mode={mode} split={split} =====")

        uses_checkpoint = (
            mode != "match"
            and getattr(args, "model", None)
            and os.path.exists(args.model)
        )
        if not uses_checkpoint:
            loader.rescale(loader.scaling_factor(train_sids))

        public_model = None
        if mode != "match":
            public_model = _load_or_train_fold_model(loader, train_sids, args)
        fold_scales.append({"Fold": fold_idx, "SCALE": float(loader.SCALE)})

        for test_sid in test_sids:
            if mode == "sequence":
                times, true_arr, pred_arr, _ = predict_sequence(
                    public_model, loader, test_sid, initial_ratio=initial_ratio, device=device)
            elif mode == "match":
                before = sorted(list(train_sids))
                if use_similar:
                    after = sorted(select_similar_training(
                        loader, train_sids, loader.samples[test_sid], top_percent,
                        initial_ratio=initial_ratio,
                    ))
                else:
                    after = before
                similar_export_rows.append({
                    "Fold": fold_idx, "Test_SubjectID": test_sid,
                    "Before_Count": len(before), "After_Count": len(after),
                    "Before_Subjects": "; ".join(before),
                    "After_Subjects": "; ".join(after),
                })
                times, true_arr, pred_arr, _ = predict_match(
                    loader, train_sids, test_sid,
                    initial_ratio=initial_ratio, top_percent=top_percent, device=device,
                    args=args, use_similar=use_similar, _cache=match_model_cache,
                )
            elif mode == "step1":
                times, true_arr, pred_arr, _ = predict_step1(
                    public_model, loader, test_sid, initial_ratio=initial_ratio, device=device)
            else:
                times, true_arr, pred_arr, _ = predict_stepn(
                    public_model, loader, test_sid,
                    initial_ratio=initial_ratio, n_steps=n_steps, device=device)

            if true_arr.size == 0 or pred_arr.size == 0:
                print(f"  Skipping SubjectID={test_sid}: no predicted time points")
                continue

            for ti, tval in enumerate(times):
                row = {"SubjectID": test_sid, "Time": float(tval)}
                for sp_idx, sp in enumerate(loader.species_names):
                    row[f"{sp}_true_abundance"] = float(true_arr[ti, sp_idx] * loader.SCALE)
                    row[f"{sp}_predicted_abundance"] = float(pred_arr[ti, sp_idx] * loader.SCALE)
                all_true_rows_wide.append(row)

            for ti, tval in enumerate(times):
                for sp_idx, sp in enumerate(loader.species_names):
                    all_long_rows.append({
                        "Fold": fold_idx, "SubjectID": test_sid,
                        "Time": float(tval), "Species": sp,
                        "true_abundance": float(true_arr[ti, sp_idx] * loader.SCALE),
                        "predicted_abundance": float(pred_arr[ti, sp_idx] * loader.SCALE),
                    })

            for ti, tval in enumerate(times):
                r = pearson_r(true_arr[ti, :], pred_arr[ti, :])
                per_time_r_rows.append({
                    "Fold": fold_idx, "SubjectID": test_sid,
                    "Time": float(tval), "Pearson_R": r,
                })

            true_subject_raw = true_arr * loader.SCALE
            pred_subject_raw = pred_arr * loader.SCALE
            subject_metric_rows.append({
                "Fold": fold_idx,
                "SubjectID": test_sid,
                "Pearson_R": pearson_r(true_arr.ravel(), pred_arr.ravel()),
                "RMSE": float(np.sqrt(np.mean((true_subject_raw - pred_subject_raw) ** 2))),
                "Predicted_Timepoints": int(len(times)),
                "NumberOfTaxa": int(loader.N),
            })

        fold_subject_metrics = pd.DataFrame([
            row for row in subject_metric_rows if row["Fold"] == fold_idx
        ])
        if fold_subject_metrics.empty:
            continue
        r = float(fold_subject_metrics["Pearson_R"].mean())
        rmse = float(fold_subject_metrics["RMSE"].mean())
        fold_summaries.append({
            "Fold": fold_idx,
            "Pearson_R": r,
            "RMSE": rmse,
        })
        print(
            f"  Fold {fold_idx}  subject-mean summary:  "
            f"Pearson_R={r:.4f}  RMSE={rmse:.4e}"
        )

    wide_df = pd.DataFrame(all_true_rows_wide)
    ordered_cols = ["SubjectID", "Time"]
    for sp in loader.species_names:
        ordered_cols += [f"{sp}_true_abundance", f"{sp}_predicted_abundance"]
    wide_df = wide_df[[c for c in ordered_cols if c in wide_df.columns]]
    wide_df.to_excel(os.path.join(out_dir, "prediction_abundance_wide.xlsx"), index=False)

    long_df = pd.DataFrame(all_long_rows)
    long_df.to_excel(os.path.join(out_dir, "prediction_abundance_long.xlsx"), index=False)

    if fold_summaries:
        fold_metrics = pd.DataFrame(fold_summaries)
        prediction_summary = pd.DataFrame([{
            "Pearson_R": float(fold_metrics["Pearson_R"].mean()),
            "RMSE": float(fold_metrics["RMSE"].mean()),
        }])
    else:
        prediction_summary = pd.DataFrame(columns=["Pearson_R", "RMSE"])
    prediction_summary.to_excel(
        os.path.join(out_dir, "prediction_summary.xlsx"), index=False
    )
    legacy_summary = Path(out_dir) / "cross_validation_summary.xlsx"
    if legacy_summary.exists():
        legacy_summary.unlink()

    pd.DataFrame(per_time_r_rows).to_excel(os.path.join(out_dir, "pearson_r_by_subject_time.xlsx"), index=False)
    pd.DataFrame(subject_metric_rows).to_excel(
        os.path.join(out_dir, "prediction_metrics_by_subject.xlsx"), index=False
    )

    if similar_export_rows:
        pd.DataFrame(similar_export_rows).to_excel(
            os.path.join(out_dir, "match_sample_selection.xlsx"), index=False)

    try:
        with open(os.path.join(out_dir, "predict_run_context.pkl"), "wb") as f:
            pickle.dump({
                "fold_list": fold_list, "species_names": loader.species_names,
                "fold_scales": fold_scales, "group_col": loader.group_col,
                "split": split, "mode": mode,
            }, f)
    except Exception:
        pass

    if len(fold_summaries) > 0:
        mean_row = prediction_summary.iloc[0]
        print("\n===== Mean across folds =====")
        for k in ["Pearson_R", "RMSE"]:
            print(f"  Avg {k}: {mean_row[k]:.6f}")

    print(f"\n[predict] Complete. Outputs: {out_dir}")
    return True
