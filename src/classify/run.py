"""English documentation."""
import os
import warnings

import numpy as np
import pandas as pd
import torch

from ..common import (
    set_seed, ensure_dir,
    split_subjects, build_history_from_abundances,
    MicrobiomeDataLoader, build_glv_model, train_glv_model,
)


def _is_valid_label(value):
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    if isinstance(value, str) and value.strip().lower() in ("", "nan", "none", "null", "na", "n/a"):
        return False
    return True


def _timepoint_labels(loader):
    """Return labels in the same time order as each subject's abundance rows."""
    labels = {}
    for sid, sample in loader.samples.items():
        groups = sample.get("groups", None)
        if not groups:
            labels[sid] = [None] * int(sample.get("T", 0))
            continue
        labels[sid] = [str(g) if _is_valid_label(g) else None for g in groups]
        uniq = sorted({label for label in labels[sid] if _is_valid_label(label)})
        if len(uniq) > 1:
            warnings.warn(
                f"SubjectID={sid} has time-varying labels={uniq}; classify uses each timepoint label"
            )
    return labels


def _expanded_split_rows(subject_ids, timepoint_labels):
    """Expand valid timepoint labels while retaining SubjectID as the split group."""
    labels, groups = [], []
    for sid in subject_ids:
        for label in timepoint_labels.get(sid, []):
            if _is_valid_label(label):
                labels.append(str(label))
                groups.append(sid)
    return np.asarray(labels, dtype=object), np.asarray(groups, dtype=object)


def _best_group_holdout(subject_ids, timepoint_labels, train_ratio, seed):
    """Select a subject-level holdout that best preserves timepoint label counts."""
    from sklearn.model_selection import GroupShuffleSplit

    labels, groups = _expanded_split_rows(subject_ids, timepoint_labels)
    if len(labels) == 0 or len(np.unique(groups)) < 2:
        raise ValueError("Classification requires at least two subjects with valid labels.")

    classes = np.unique(labels)
    global_distribution = np.asarray([(labels == cls).mean() for cls in classes])
    splitter = GroupShuffleSplit(
        n_splits=min(1000, max(100, len(np.unique(groups)) * 20)),
        train_size=train_ratio,
        random_state=seed,
    )
    best = None
    for train_idx, test_idx in splitter.split(labels, labels, groups):
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]
        train_distribution = np.asarray([(train_labels == cls).mean() for cls in classes])
        test_distribution = np.asarray([(test_labels == cls).mean() for cls in classes])
        missing_penalty = sum(
            int(not np.any(train_labels == cls)) + int(not np.any(test_labels == cls))
            for cls in classes
        )
        subject_ratio = len(np.unique(groups[train_idx])) / len(np.unique(groups))
        score = (
            missing_penalty * 10.0
            + np.abs(train_distribution - global_distribution).sum()
            + np.abs(test_distribution - global_distribution).sum()
            + abs(subject_ratio - train_ratio)
        )
        if best is None or score < best[0]:
            best = (score, sorted(set(groups[train_idx])), sorted(set(groups[test_idx])))
    return [(best[1], best[2])]


def _classification_splits(subject_ids, timepoint_labels, args):
    """Split by SubjectID while balancing all valid timepoint labels."""
    split = getattr(args, "split", "82")
    seed = getattr(args, "seed", 42)
    labeled_sids = [sid for sid in subject_ids if any(
        _is_valid_label(label) for label in timepoint_labels.get(sid, [])
    )]
    if len(labeled_sids) < 2:
        raise ValueError("Classification requires at least two labeled subjects.")

    if split == "custom":
        return split_subjects(
            subject_ids,
            split=split,
            train_ratio=getattr(args, "train_ratio", 0.8),
            n_splits=getattr(args, "n_splits", 5),
            train_sids=getattr(args, "train_sids", None),
            test_sids=getattr(args, "test_sids", None),
            seed=seed,
        )

    if split == "loso":
        return [([s for s in labeled_sids if s != leave], [leave]) for leave in labeled_sids]

    if split == "5fold":
        from sklearn.model_selection import StratifiedGroupKFold

        n_splits = min(int(getattr(args, "n_splits", 5)), len(labeled_sids))
        if n_splits < 2:
            raise ValueError("Classification cross-validation requires at least two labeled subjects.")
        labels, groups = _expanded_split_rows(labeled_sids, timepoint_labels)
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = []
        for train_idx, test_idx in cv.split(np.zeros(len(labels)), labels, groups):
            folds.append((sorted(set(groups[train_idx])), sorted(set(groups[test_idx]))))
        return folds

    if split in ("82", "73"):
        ratio = 0.7 if split == "73" else float(getattr(args, "train_ratio", 0.8))
        return _best_group_holdout(labeled_sids, timepoint_labels, ratio, seed)

    return split_subjects(
        labeled_sids,
        split=split,
        train_ratio=getattr(args, "train_ratio", 0.8),
        n_splits=getattr(args, "n_splits", 5),
        seed=seed,
    )


def _collect_train_real_abundance(loader, train_sids, timepoint_labels):
    """English documentation."""
    X_rows, y_rows = [], []
    for sid in train_sids:
        sample = loader.samples.get(sid)
        labels = timepoint_labels.get(sid, [])
        if sample is None:
            continue
        abundances = np.asarray(sample["abundances"], dtype=float)
        for t_idx in range(len(abundances)):
            label = labels[t_idx] if t_idx < len(labels) else None
            if not _is_valid_label(label):
                continue
            X_rows.append(abundances[t_idx].copy())
            y_rows.append(str(label))
    if not X_rows:
        return np.zeros((0, loader.N), dtype=float), np.array([], dtype=object)
    return np.asarray(X_rows, dtype=float), np.asarray(y_rows, dtype=object)


def _single_step_predict_test(model, loader, test_sids, device="cpu"):
    """English documentation."""
    model.to(device)
    model.eval()

    true_rows, pred_rows, meta_rows = [], [], []
    with torch.no_grad():
        for sid in test_sids:
            sample = loader.samples.get(sid)
            if sample is None:
                continue
            T = sample["T"]
            if T < loader.lag_max + 2:
                warnings.warn(f"SubjectID={sid} has too few time points for one-step gLV prediction; skipping")
                continue
            abundances = np.asarray(sample["abundances"], dtype=float)
            times = np.asarray(sample["times"], dtype=float)
            for t in range(loader.lag_max, T - 1):
                history = build_history_from_abundances(
                    abundances, t, loader.lag_min, loader.lag_max
                ).to(device)
                current = torch.tensor(abundances[t], dtype=torch.float32, device=device)
                dt_value = float(times[t + 1] - times[t])
                if not np.isfinite(dt_value) or dt_value <= 0:
                    dt_value = 1.0
                dt = torch.tensor(dt_value, dtype=torch.float32, device=device)
                pred = model(current, history, dt).detach().cpu().numpy()

                true_rows.append(abundances[t + 1].copy())
                pred_rows.append(pred.copy())
                meta_rows.append({
                    "SubjectID": sid,
                    "Time": float(times[t + 1]),
                    "Time_index": int(t + 1),
                })

    if not true_rows:
        empty = np.zeros((0, loader.N), dtype=float)
        return empty, empty, []
    return np.asarray(true_rows, dtype=float), np.asarray(pred_rows, dtype=float), meta_rows


def _dynamic_auc(y_test, y_prob, classes):
    from sklearn.metrics import roc_auc_score

    present_classes = np.unique(y_test)
    if len(present_classes) < 2:
        return float("nan")
    if len(present_classes) == 2:
        positive_class = int(present_classes[-1])
        binary_target = (y_test == positive_class).astype(int)
        return float(roc_auc_score(binary_target, y_prob[:, positive_class]))
    y_score = y_prob[:, present_classes]
    row_sums = y_score.sum(axis=1, keepdims=True)
    y_score = np.divide(y_score, row_sums, out=np.zeros_like(y_score), where=row_sums > 0)
    return float(roc_auc_score(
        y_test,
        y_score,
        multi_class="ovr",
        average="macro",
        labels=present_classes,
    ))


def _run_logreg_classification(
    X_train_raw,
    y_train_raw,
    X_test_pred,
    meta_rows,
    timepoint_labels,
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    if len(X_train_raw) == 0 or len(X_test_pred) == 0:
        return [], {"Accuracy": float("nan"), "AUC": float("nan"), "Train_Samples": 0, "Test_Samples": 0}

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    if len(le.classes_) < 2:
        warnings.warn("The training set has only one class; skipping this fold")
        return [], {"Accuracy": float("nan"), "AUC": float("nan"), "Train_Samples": len(y_train_raw), "Test_Samples": 0}

    y_test_labels = []
    for meta in meta_rows:
        sid_labels = timepoint_labels.get(meta["SubjectID"], [])
        time_index = int(meta["Time_index"])
        y_test_labels.append(sid_labels[time_index] if time_index < len(sid_labels) else None)
    y_test_labels = np.asarray(y_test_labels, dtype=object)

    known_labels = set(le.classes_)
    valid_mask = np.asarray([
        _is_valid_label(label) and str(label) in known_labels for label in y_test_labels
    ], dtype=bool)
    if not np.any(valid_mask):
        warnings.warn("The test set has no valid label represented in training; skipping this fold")
        return [], {"Accuracy": float("nan"), "AUC": float("nan"), "Train_Samples": len(y_train_raw), "Test_Samples": 0}

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_pred[valid_mask])
    y_test = le.transform(y_test_labels[valid_mask].astype(str))
    meta_valid = [meta_rows[i] for i in np.where(valid_mask)[0]]

    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    try:
        auc = _dynamic_auc(y_test, y_prob, le.classes_)
    except ValueError as e:
        warnings.warn(f"AUC could not be computed and is set to NaN: {e}")
        auc = float("nan")

    rows = []
    for i, meta in enumerate(meta_valid):
        rec = {
            "SubjectID": meta["SubjectID"],
            "Time": meta["Time"],
            "Time_index": meta["Time_index"],
            "True_Label": str(le.inverse_transform([y_test[i]])[0]),
            "Pred_Label": str(le.inverse_transform([y_pred[i]])[0]),
        }
        for ci, cls in enumerate(le.classes_):
            rec[f"Prob_{cls}"] = float(y_prob[i, ci])
        rows.append(rec)

    return rows, {
        "Accuracy": accuracy,
        "AUC": auc,
        "Train_Samples": int(len(X_train_raw)),
        "Test_Samples": int(len(X_test)),
        "Classes": ";".join(map(str, le.classes_)),
    }


def _save_roc_pdf(class_rows, out_path):
    """Save one-vs-rest ROC curves for all classes in one PDF."""
    if not class_rows:
        warnings.warn("No classification predictions were available for ROC curves")
        return False

    from sklearn.metrics import auc, roc_curve

    true_labels = np.asarray([row["True_Label"] for row in class_rows], dtype=object)
    probability_columns = sorted({
        key for row in class_rows for key in row if key.startswith("Prob_")
    })
    if len(probability_columns) < 2:
        warnings.warn("ROC curves require at least two predicted classes")
        return False

    classes = [key[len("Prob_"):] for key in probability_columns]
    probabilities = np.asarray([
        [float(row.get(key, 0.0)) for key in probability_columns]
        for row in class_rows
    ], dtype=float)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    plotted = 0
    for class_index, class_name in enumerate(classes):
        binary_true = (true_labels == class_name).astype(int)
        if binary_true.min() == binary_true.max():
            continue
        fpr, tpr, _ = roc_curve(binary_true, probabilities[:, class_index])
        curve_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{class_name} (AUC = {curve_auc:.3f})")
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        warnings.warn("ROC curves could not be computed because no class has both outcomes")
        return False

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest ROC curves")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return True


def _save_prediction_tables(fold_idx, loader, true_scaled, pred_scaled, meta_rows):
    rows_scaled, rows_raw = [], []
    for i, meta in enumerate(meta_rows):
        row_s = {
            "Fold": fold_idx,
            "SubjectID": meta["SubjectID"],
            "Time": meta["Time"],
            "Time_index": meta["Time_index"],
        }
        row_r = dict(row_s)
        for j, sp in enumerate(loader.species_names):
            row_s[f"{sp}_true"] = float(true_scaled[i, j])
            row_s[f"{sp}_pred"] = float(pred_scaled[i, j])
            row_r[f"{sp}_true"] = float(true_scaled[i, j] * loader.SCALE)
            row_r[f"{sp}_pred"] = float(pred_scaled[i, j] * loader.SCALE)
        rows_scaled.append(row_s)
        rows_raw.append(row_r)
    return rows_scaled, rows_raw


def main_classify(args):
    set_seed(getattr(args, "seed", 42))
    out_dir = getattr(args, "output_dir", "outputs/03_classify")
    ensure_dir(out_dir)

    graph_path = getattr(args, "graph", None)
    if not graph_path:
        raise ValueError("Classification requires a causal graph via --graph.")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Causal graph not found: {graph_path}")

    loader = MicrobiomeDataLoader(
        args.abundance, graph_path,
        group_is_present=(None if not getattr(args, "no_group", False) else False),
    )
    loader.load_abundance()
    if not loader.group_col:
        raise ValueError("Classification requires a Group column in the abundance table.")
    loader.load_graph()

    subject_ids = list(loader.samples.keys())
    timepoint_labels = _timepoint_labels(loader)
    split = getattr(args, "split", "82")
    fold_list = _classification_splits(subject_ids, timepoint_labels, args)
    glv_method = str(getattr(args, "glv_method", "euler")).strip().lower()
    epochs = int(getattr(args, "epochs", 10))
    lr = float(getattr(args, "lr", 0.01))
    device = "cpu"

    all_class_rows = []
    all_pred_scaled_rows = []
    all_pred_raw_rows = []
    fold_summary_rows = []

    for fold_idx, (train_sids, test_sids) in enumerate(fold_list, 1):
        print(f"===== Classification fold {fold_idx}/{len(fold_list)} train={len(train_sids)} test={len(test_sids)} =====")

        loader.rescale(loader.scaling_factor(train_sids))
        loader.prepare_training_data(sid_subset=train_sids)
        if len(loader.training_pairs) == 0:
            warnings.warn(f"Fold {fold_idx} has no usable gLV training pairs; skipping")
            continue

        model = build_glv_model(glv_method, loader.N, loader.lag_min, loader.lag_max, loader.edge_info)
        train_glv_model(
            model,
            loader.training_pairs,
            epochs=epochs,
            lr=lr,
            device=device,
            verbose=True,
        )

        true_scaled, pred_scaled, meta_rows = _single_step_predict_test(model, loader, test_sids, device=device)
        if len(meta_rows) == 0:
            warnings.warn(f"Fold {fold_idx} has no valid test predictions; skipping")
            continue

        X_train, y_train = _collect_train_real_abundance(loader, train_sids, timepoint_labels)
        class_rows, class_metrics = _run_logreg_classification(
            X_train,
            y_train,
            pred_scaled,
            meta_rows,
            timepoint_labels,
        )
        for row in class_rows:
            row["Fold"] = fold_idx
        all_class_rows.extend(class_rows)

        pred_scaled_rows, pred_raw_rows = _save_prediction_tables(
            fold_idx, loader, true_scaled, pred_scaled, meta_rows
        )
        all_pred_scaled_rows.extend(pred_scaled_rows)
        all_pred_raw_rows.extend(pred_raw_rows)

        fold_summary = {
            "Fold": fold_idx,
            "AUC": class_metrics.get("AUC", np.nan),
        }
        fold_summary_rows.append(fold_summary)

        fold_dir = os.path.join(out_dir, f"fold_{fold_idx}")
        ensure_dir(fold_dir)
        pd.DataFrame(class_rows).to_excel(os.path.join(fold_dir, "logreg_classification_results.xlsx"), index=False)
        pd.DataFrame(pred_scaled_rows).to_excel(os.path.join(fold_dir, "predictions_scaled.xlsx"), index=False)
        pd.DataFrame(pred_raw_rows).to_excel(os.path.join(fold_dir, "predictions_raw.xlsx"), index=False)
        pd.DataFrame([fold_summary]).to_excel(os.path.join(fold_dir, "metrics.xlsx"), index=False)
        torch.save({
            "state_dict": model.state_dict(),
            "N": loader.N,
            "lag_min": loader.lag_min,
            "lag_max": loader.lag_max,
            "edge_info": loader.edge_info,
            "species_names": loader.species_names,
            "SCALE": loader.SCALE,
            "model_type": glv_method,
        }, os.path.join(fold_dir, "MMLD_model.pt"))

        auc_value = fold_summary["AUC"]
        auc_msg = f"{auc_value:.4f}" if np.isfinite(auc_value) else "N/A"
        print(f"  Fold {fold_idx}: AUC={auc_msg}")

    result_df = pd.DataFrame(all_class_rows)
    result_path = os.path.join(out_dir, "classification_results.xlsx")
    result_df.to_excel(result_path, index=False)

    pred_scaled_df = pd.DataFrame(all_pred_scaled_rows)
    pred_scaled_df.to_excel(os.path.join(out_dir, "MMLD_one_step_abundance_scaled.xlsx"), index=False)

    pred_raw_df = pd.DataFrame(all_pred_raw_rows)
    pred_raw_df.to_excel(os.path.join(out_dir, "MMLD_one_step_abundance_raw.xlsx"), index=False)

    fold_metrics_df = pd.DataFrame(fold_summary_rows)
    if not fold_metrics_df.empty:
        summary_df = pd.DataFrame([{
            "AUC": float(fold_metrics_df["AUC"].mean(skipna=True)),
        }])
    else:
        summary_df = pd.DataFrame(columns=["AUC"])
    summary_path = os.path.join(out_dir, "classification_summary.xlsx")
    summary_df.to_excel(summary_path, index=False)

    roc_path = os.path.join(out_dir, "classification_roc_curves.pdf")
    _save_roc_pdf(all_class_rows, roc_path)

    if len(summary_df) > 0:
        mean_auc = float(summary_df.loc[0, "AUC"])
        print(f"[classify] Complete. Mean AUC={mean_auc:.4f}, outputs: {out_dir}")
    else:
        print(f"[classify] Complete, but no valid results were generated. Outputs: {out_dir}")
    return result_df
