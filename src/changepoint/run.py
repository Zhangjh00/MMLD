"""Adjacent-time change-point analysis based on held-out MMLD predictions."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mmld_changepoint_mpl_cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from ..common import (
    MicrobiomeDataLoader,
    build_glv_model,
    build_history_from_abundances,
    ensure_dir,
    set_seed,
    split_subjects,
    train_glv_model,
)


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _safe_pearson(observed, predicted):
    observed = np.asarray(observed, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    mask = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[mask], predicted[mask]
    if observed.size < 4 or np.std(observed) == 0 or np.std(predicted) == 0:
        return np.nan
    return float(np.corrcoef(observed, predicted)[0, 1])


def _resolve_sheet(path, requested=None):
    book = pd.ExcelFile(path)
    if requested is not None:
        if requested not in book.sheet_names:
            raise ValueError(
                f"Abundance sheet '{requested}' was not found. "
                f"Available sheets: {book.sheet_names}"
            )
        return requested
    if "absolute_abundance" in book.sheet_names:
        return "absolute_abundance"
    return book.sheet_names[0]


def _read_abundance(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [str(column).strip() for column in df.columns]
    id_col = next(
        (column for column in df.columns
         if column.lower() in ("subjectid", "sampleid", "subject_id", "patientid")),
        df.columns[0],
    )
    time_col = next(
        (column for column in df.columns
         if column.lower() in ("time", "timepoint", "time_point", "day")),
        None,
    )
    if time_col is None:
        raise ValueError("The abundance table must contain a Time column")
    df[time_col] = pd.to_numeric(df[time_col], errors="raise")
    return df, id_col, time_col


def _make_padded_pair(loader, sid, current_index):
    sample = loader.samples[sid]
    abundances = np.asarray(sample["abundances"], dtype=float)
    times = np.asarray(sample["times"], dtype=float)
    history = build_history_from_abundances(
        abundances, current_index, loader.lag_min, loader.lag_max
    )
    dt_value = float(times[current_index + 1] - times[current_index])
    if not np.isfinite(dt_value) or dt_value <= 0:
        dt_value = 1.0
    return {
        "sample_id": sid,
        "current": torch.tensor(abundances[current_index], dtype=torch.float32),
        "history": history,
        "target": torch.tensor(abundances[current_index + 1], dtype=torch.float32),
        "dt": torch.tensor(dt_value, dtype=torch.float32),
    }


def _training_pairs(loader, subject_ids):
    pairs = []
    for sid in subject_ids:
        sample = loader.samples.get(sid)
        if sample is None:
            continue
        for current_index in range(max(0, sample["T"] - 1)):
            pairs.append(_make_padded_pair(loader, sid, current_index))
    return pairs


def _filtered_graph(loader, graph_file, p_threshold, fold):
    book = pd.ExcelFile(graph_file)
    sheet = "edge_list" if "edge_list" in book.sheet_names else book.sheet_names[0]
    graph = pd.read_excel(graph_file, sheet_name=sheet)
    graph.columns = [str(column).strip() for column in graph.columns]
    required = {"source", "target"}
    if not required.issubset(graph.columns):
        raise ValueError(f"The causal graph is missing columns: {sorted(required - set(graph.columns))}")
    if "lag" not in graph.columns:
        graph["lag"] = 1
    if "p_value" not in graph.columns:
        graph["p_value"] = 0.05
    if "test_statistic" not in graph.columns:
        graph["test_statistic"] = 0.0
    graph["lag"] = pd.to_numeric(graph["lag"], errors="coerce").abs()
    graph["p_value"] = pd.to_numeric(graph["p_value"], errors="coerce")
    graph["test_statistic"] = pd.to_numeric(graph["test_statistic"], errors="coerce")
    graph = graph.dropna(subset=["lag", "p_value", "test_statistic"]).copy()
    graph["lag"] = graph["lag"].astype(int)
    graph = graph[
        (graph["lag"] >= 1)
        & (graph["p_value"] <= p_threshold)
        & (graph["source"] != graph["target"])
        & graph["source"].isin(loader.species_to_idx)
        & graph["target"].isin(loader.species_to_idx)
    ].copy()
    graph.insert(0, "LagMode", "full_lag")
    graph.insert(0, "Fold", fold)
    return graph.reset_index(drop=True)


def _heldout_metrics(args, sheet):
    probe = MicrobiomeDataLoader(
        args.abundance, args.graph,
        group_is_present=(False if args.no_group else None),
        sheet_name=sheet,
    )
    probe.load_abundance()
    fold_list = split_subjects(
        list(probe.samples),
        split=getattr(args, "split", "82"),
        train_ratio=getattr(args, "train_ratio", 0.8),
        n_splits=args.n_splits,
        train_sids=getattr(args, "train_sids", None),
        test_sids=getattr(args, "test_sids", None),
        seed=args.seed,
    )
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    used_graphs = []
    for fold, (train_ids, test_ids) in enumerate(fold_list, start=1):
        set_seed(args.seed + fold)
        loader = MicrobiomeDataLoader(
            args.abundance, args.graph,
            group_is_present=(False if args.no_group else None),
            sheet_name=sheet,
        )
        loader.load_abundance()
        loader.rescale(loader.scaling_factor(train_ids))
        loader.load_graph(p_threshold=args.p_threshold)
        if not loader.edge_info:
            raise ValueError(f"Fold {fold} has no usable causal edges")
        used_graphs.append(_filtered_graph(loader, args.graph, args.p_threshold, fold))

        pairs = _training_pairs(loader, train_ids)
        if not pairs:
            raise RuntimeError(f"Fold {fold} has no usable training transitions")
        model = build_glv_model(
            args.glv_method, loader.N, loader.lag_min, loader.lag_max, loader.edge_info
        )
        train_glv_model(
            model, pairs, epochs=args.epochs, lr=args.lr,
            device=device, verbose=True,
        )
        model.to(device).eval()
        with torch.no_grad():
            for sid in test_ids:
                sample = loader.samples[sid]
                for current_index in range(max(0, sample["T"] - 1)):
                    pair = _make_padded_pair(loader, sid, current_index)
                    predicted = model(
                        pair["current"].to(device),
                        pair["history"].to(device),
                        pair["dt"].to(device),
                    ).detach().cpu().numpy()
                    observed = pair["target"].numpy()
                    rows.append({
                        "SubjectID": sid,
                        "Time": float(sample["times"][current_index + 1]),
                        "Pearson_R": _safe_pearson(observed, predicted),
                        "RMSE": float(np.sqrt(np.mean((observed - predicted) ** 2))),
                        "NumberOfTaxa": int(len(observed)),
                        "Fold": fold,
                        "Scale": float(loader.SCALE),
                        "PredictionMode": "held-out rolling one-step",
                        "LagMode": "full_lag",
                    })
    graph_used = pd.concat(used_graphs, ignore_index=True) if used_graphs else pd.DataFrame()
    return pd.DataFrame(rows), graph_used


def _metrics_from_prediction(path, fallback_n_taxa):
    pred = pd.read_excel(path)
    pred.columns = [str(column).strip() for column in pred.columns]
    id_col = next(
        (column for column in pred.columns
         if column.lower() in ("subjectid", "sampleid", "subject_id", "patientid")),
        None,
    )
    if id_col is None or "Time" not in pred.columns:
        raise ValueError("The prediction file must contain SubjectID/SampleID and Time")
    if id_col != "SubjectID":
        pred = pred.rename(columns={id_col: "SubjectID"})

    if {"Species", "true_abundance", "predicted_abundance"}.issubset(pred.columns):
        rows = []
        for (sid, time), group in pred.groupby(["SubjectID", "Time"], sort=False):
            observed = pd.to_numeric(group["true_abundance"], errors="coerce").to_numpy()
            predicted = pd.to_numeric(group["predicted_abundance"], errors="coerce").to_numpy()
            mask = np.isfinite(observed) & np.isfinite(predicted)
            rows.append({
                "SubjectID": sid,
                "Time": time,
                "Pearson_R": _safe_pearson(observed[mask], predicted[mask]),
                "RMSE": float(np.sqrt(np.mean((observed[mask] - predicted[mask]) ** 2))),
                "NumberOfTaxa": int(mask.sum()),
                "PredictionMode": "external true/predicted abundance",
                "LagMode": "full_lag",
            })
        return pd.DataFrame(rows)

    suffix_pairs = [
        ("_true_abundance", "_predicted_abundance"),
        ("_true", "_pred"),
    ]
    for true_suffix, predicted_suffix in suffix_pairs:
        true_columns = {
            column[:-len(true_suffix)]: column
            for column in pred.columns if column.endswith(true_suffix)
        }
        predicted_columns = {
            column[:-len(predicted_suffix)]: column
            for column in pred.columns if column.endswith(predicted_suffix)
        }
        taxa = sorted(set(true_columns) & set(predicted_columns))
        if taxa:
            rows = []
            for _, record in pred.iterrows():
                observed = np.asarray([record[true_columns[taxon]] for taxon in taxa], dtype=float)
                predicted = np.asarray([record[predicted_columns[taxon]] for taxon in taxa], dtype=float)
                mask = np.isfinite(observed) & np.isfinite(predicted)
                rows.append({
                    "SubjectID": record["SubjectID"],
                    "Time": record["Time"],
                    "Pearson_R": _safe_pearson(observed[mask], predicted[mask]),
                    "RMSE": float(np.sqrt(np.mean((observed[mask] - predicted[mask]) ** 2))),
                    "NumberOfTaxa": int(mask.sum()),
                    "PredictionMode": "external true/predicted abundance",
                    "LagMode": "full_lag",
                })
            return pd.DataFrame(rows)

    if "Pearson_R" not in pred.columns:
        raise ValueError(
            "The prediction file must contain Pearson_R or paired true/predicted abundance columns"
        )
    if "RMSE" not in pred.columns:
        pred["RMSE"] = np.nan
    if "NumberOfTaxa" not in pred.columns:
        pred["NumberOfTaxa"] = fallback_n_taxa
    if "PredictionMode" not in pred.columns:
        pred["PredictionMode"] = "external metrics"
    if "LagMode" not in pred.columns:
        pred["LagMode"] = "full_lag"
    return pred[[
        "SubjectID", "Time", "Pearson_R", "RMSE", "NumberOfTaxa",
        "PredictionMode", "LagMode",
    ]].copy()


def _fisher_independent_test(previous_r, current_r, previous_n, current_n):
    if not np.isfinite(previous_r) or not np.isfinite(current_r):
        return np.nan, np.nan
    if previous_n <= 3 or current_n <= 3:
        return np.nan, np.nan
    previous_r = float(np.clip(previous_r, -0.999999, 0.999999))
    current_r = float(np.clip(current_r, -0.999999, 0.999999))
    difference = np.arctanh(current_r) - np.arctanh(previous_r)
    standard_error = np.sqrt(1.0 / (previous_n - 3) + 1.0 / (current_n - 3))
    statistic = difference / standard_error
    return float(statistic), float(2.0 * stats.norm.sf(abs(statistic)))


def _subject_adjacent_tests(metrics, alpha):
    rows = []
    for sid, subject in metrics.groupby("SubjectID", sort=False):
        subject = subject.sort_values("Time").reset_index(drop=True)
        for index in range(1, len(subject)):
            previous, current = subject.iloc[index - 1], subject.iloc[index]
            statistic, p_value = _fisher_independent_test(
                previous["Pearson_R"], current["Pearson_R"],
                int(previous["NumberOfTaxa"]), int(current["NumberOfTaxa"]),
            )
            delta = current["Pearson_R"] - previous["Pearson_R"]
            rows.append({
                "SubjectID": sid,
                "PreviousTime": previous["Time"],
                "CurrentTime": current["Time"],
                "PreviousR": previous["Pearson_R"],
                "CurrentR": current["Pearson_R"],
                "DeltaR": delta,
                "Direction": "R_increase" if delta > 0 else "R_decrease",
                "FisherZStatistic": statistic,
                "PValue": p_value,
                "Significant": bool(np.isfinite(p_value) and p_value < alpha),
            })
    return pd.DataFrame(rows)


def _group_adjacent_tests(metrics, alpha):
    r_table = metrics.pivot_table(
        index="SubjectID", columns="Time", values="Pearson_R", aggfunc="mean"
    ).sort_index(axis=1)
    times = r_table.columns.to_numpy(dtype=float)
    rows = []
    for index in range(1, len(times)):
        previous_time, current_time = times[index - 1], times[index]
        pair = r_table[[previous_time, current_time]].dropna()
        if len(pair) >= 2:
            previous_z = np.arctanh(np.clip(pair[previous_time], -0.999999, 0.999999))
            current_z = np.arctanh(np.clip(pair[current_time], -0.999999, 0.999999))
            statistic, p_value = stats.ttest_rel(current_z, previous_z)
        else:
            statistic, p_value = np.nan, np.nan
        delta = pair[current_time].mean() - pair[previous_time].mean() if len(pair) else np.nan
        rows.append({
            "PreviousTime": previous_time,
            "CurrentTime": current_time,
            "PreviousMeanR": pair[previous_time].mean() if len(pair) else np.nan,
            "CurrentMeanR": pair[current_time].mean() if len(pair) else np.nan,
            "DeltaMeanR": delta,
            "NumberOfPairedSubjects": len(pair),
            "PairedFisherZStatistic": statistic,
            "PValue": p_value,
            "Significant": bool(np.isfinite(p_value) and p_value < alpha),
        })
    return pd.DataFrame(rows)


def _format_time(value):
    return f"{float(value):g}"


def _format_p(value):
    if not np.isfinite(value):
        return "P=NA"
    return f"P={value:.2e}" if value < 0.001 else f"P={value:.3f}"


def _annotate_significant(axis, tests, group_curve=False, positions=None):
    if tests.empty or "Significant" not in tests.columns:
        return
    significant = tests.loc[tests["Significant"]].reset_index(drop=True)
    if significant.empty:
        return
    y_columns = (
        ("PreviousMeanR", "CurrentMeanR") if group_curve
        else ("PreviousR", "CurrentR")
    )
    values = significant[list(y_columns)].to_numpy(dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    bottom, top = axis.get_ylim()
    span = max(top - bottom, 0.2)
    base = max(top, float(values.max()) + 0.04 * span)
    step, height = 0.10 * span, 0.025 * span
    axis.set_ylim(bottom, base + step * len(significant) + 0.08 * span)
    for level, (_, row) in enumerate(significant.iterrows()):
        previous_time, current_time = float(row["PreviousTime"]), float(row["CurrentTime"])
        x1 = positions.get(previous_time) if positions is not None else previous_time
        x2 = positions.get(current_time) if positions is not None else current_time
        if x1 is None or x2 is None:
            continue
        y = base + level * step
        axis.plot(
            [x1, x1, x2, x2], [y - height, y, y, y - height],
            color="#C73659", linewidth=1.0, clip_on=False,
        )
        axis.text(
            (x1 + x2) / 2.0, y + 0.012 * span, _format_p(float(row["PValue"])),
            ha="center", va="bottom", fontsize=7.5,
            color="#C73659", fontweight="bold",
        )


def _plot_r_rmse(metrics, tests, observed_times, output, title, group_curve=False):
    if group_curve:
        curve = metrics.groupby("Time", as_index=False).agg(
            Pearson_R=("Pearson_R", "mean"),
            R_SE=("Pearson_R", lambda x: x.std(ddof=1) / np.sqrt(x.notna().sum())
                  if x.notna().sum() > 1 else np.nan),
            RMSE=("RMSE", "mean"),
            RMSE_SE=("RMSE", lambda x: x.std(ddof=1) / np.sqrt(x.notna().sum())
                     if x.notna().sum() > 1 else np.nan),
        )
    else:
        curve = metrics.sort_values("Time").copy()
        curve["R_SE"] = np.nan
        curve["RMSE_SE"] = np.nan
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.5), sharex=True)
    axes[0].errorbar(
        curve["Time"], curve["Pearson_R"], yerr=curve["R_SE"],
        fmt="o-", color="#176B87", lw=1.8, ms=4, capsize=2,
    )
    axes[0].axhline(0, color="black", lw=0.7)
    axes[0].set_ylabel("MMLD Pearson R")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.2)
    axes[1].errorbar(
        curve["Time"], curve["RMSE"], yerr=curve["RMSE_SE"],
        fmt="^-", color="#6F4E7C", lw=1.7, ms=4, capsize=2,
    )
    axes[1].set_ylabel("MMLD RMSE\n(scaled abundance)")
    axes[1].set_xlabel("Real sampling time")
    axes[1].grid(alpha=0.2)
    _annotate_significant(axes[0], tests, group_curve=group_curve)
    times = np.sort(np.unique(np.asarray(observed_times, dtype=float)))
    axes[1].set_xticks(times)
    axes[1].set_xticklabels([_format_time(time) for time in times], rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_group_boxplot(metrics, tests, output, title):
    times = np.sort(metrics["Time"].dropna().unique().astype(float))
    distributions = [
        metrics.loc[metrics["Time"] == time, "Pearson_R"].dropna().to_numpy(dtype=float)
        for time in times
    ]
    positions = np.arange(len(times), dtype=float)
    position_map = {float(time): float(position) for time, position in zip(times, positions)}
    fig, axis = plt.subplots(figsize=(max(7.0, 0.55 * len(times)), 4.2))
    axis.boxplot(
        distributions, positions=positions, widths=0.55, patch_artist=True,
        showfliers=False,
        medianprops={"color": "#7A1F3D", "linewidth": 1.3},
        boxprops={"facecolor": "#BFD7E5", "edgecolor": "#176B87", "linewidth": 1.0},
        whiskerprops={"color": "#176B87", "linewidth": 0.9},
        capprops={"color": "#176B87", "linewidth": 0.9},
    )
    medians = np.asarray([
        np.nanmedian(values) if values.size else np.nan for values in distributions
    ])
    valid = np.isfinite(medians)
    axis.plot(
        positions[valid], medians[valid], color="#7A1F3D", linewidth=1.35,
        marker="o", markersize=3.6, markerfacecolor="white",
        markeredgewidth=0.9, label="Median R", zorder=4,
    )
    rng = np.random.default_rng(42)
    for position, values in zip(positions, distributions):
        if values.size:
            axis.scatter(
                position + rng.normal(0.0, 0.045, size=values.size), values,
                s=14, color="#176B87", alpha=0.65,
                edgecolors="white", linewidths=0.35, zorder=3,
            )
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set_xticks(positions)
    axis.set_xticklabels([_format_time(time) for time in times], rotation=45, ha="right")
    axis.set_xlabel("Real sampling time")
    axis.set_ylabel("MMLD Pearson R")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.18)
    axis.legend(loc="best", fontsize=8)
    _annotate_significant(axis, tests, group_curve=True, positions=position_map)
    fig.tight_layout()
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main_changepoint(args):
    out_dir = Path(args.output_dir or "outputs/05_changepoint")
    ensure_dir(str(out_dir))
    sheet = _resolve_sheet(args.abundance, args.sheet)
    abundance, id_col, time_col = _read_abundance(args.abundance, sheet)
    species_count = len([
        column for column in abundance.columns
        if column not in {id_col, time_col}
        and column.strip().lower() not in {"group", "label", "class"}
    ])

    if args.prediction_file:
        metrics = _metrics_from_prediction(args.prediction_file, species_count)
        graph_used = pd.DataFrame()
        prediction_mode = "external"
    else:
        if not args.graph or not Path(args.graph).is_file():
            raise FileNotFoundError(
                "Provide a causal graph via --graph when --prediction-file is not used"
            )
        metrics, graph_used = _heldout_metrics(args, sheet)
        prediction_mode = "held-out rolling one-step"
    if metrics.empty:
        raise RuntimeError("No valid prediction time points were generated")
    metrics["Time"] = pd.to_numeric(metrics["Time"], errors="raise")
    subject_tests = _subject_adjacent_tests(metrics, args.alpha)
    group_tests = _group_adjacent_tests(metrics, args.alpha)

    for sid, subject_metrics in metrics.groupby("SubjectID", sort=False):
        subject_times = abundance.loc[
            abundance[id_col].astype(str) == str(sid), time_col
        ].to_numpy(dtype=float)
        tests = subject_tests[
            subject_tests["SubjectID"].astype(str) == str(sid)
        ] if not subject_tests.empty else subject_tests
        _plot_r_rmse(
            subject_metrics, tests, subject_times,
            out_dir / f"subject_{sid}_adjacent_R_change.pdf",
            f"Subject {sid}: adjacent-time changes (full_lag)",
        )
    _plot_r_rmse(
        metrics, group_tests, abundance[time_col].unique(),
        out_dir / "mean_adjacent_R_change.pdf",
        "Cross-subject prediction changes (full_lag)", group_curve=True,
    )
    _plot_group_boxplot(
        metrics, group_tests, out_dir / "mean_R_boxplot.pdf",
        "Cross-subject Pearson R distributions (full_lag)",
    )

    config = {
        "abundance_file": str(args.abundance),
        "sheet": sheet,
        "graph_file": str(args.graph or ""),
        "prediction_file": str(args.prediction_file or ""),
        "prediction_mode": prediction_mode,
        "prediction_split": getattr(args, "split", "82"),
        "lag_mode": "full_lag",
        "alpha": args.alpha,
        "test": "adjacent Pearson R difference by Fisher r-to-z",
        "prediction_epochs": args.epochs,
        "prediction_learning_rate": args.lr,
        "prediction_folds": len(metrics["Fold"].dropna().unique()) if "Fold" in metrics else 0,
        "prediction_p_threshold": args.p_threshold,
        "prediction_seed": args.seed,
        "device": args.device,
    }
    workbook = out_dir / "adjacent_R_changepoint_results.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="per_timepoint_R_RMSE", index=False)
        subject_tests.to_excel(writer, sheet_name="subject_adjacent_R_tests", index=False)
        group_tests.to_excel(writer, sheet_name="group_adjacent_R_tests", index=False)
        graph_used.to_excel(writer, sheet_name="graph_used", index=False)
        pd.DataFrame([config]).to_excel(writer, sheet_name="configuration", index=False)
    (out_dir / "run_configuration.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[changepoint] Complete. Outputs: {out_dir}")
    return True
