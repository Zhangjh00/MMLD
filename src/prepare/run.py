"""English documentation."""
import os
import re
import shutil
import tempfile
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch


from ..common import (
    ensure_dir,
    MicrobiomeDataLoader, gLVModel, gLVModel_RK, build_glv_model,
)


# ======================================================
# ======================================================
def _build_cond_ind_test(ci_test_name="ParCorr"):
    """English documentation."""
    try:
        if ci_test_name == "ParCorr":
            from tigramite.independence_tests.parcorr import ParCorr
            return ParCorr(significance="analytic")
        elif ci_test_name == "RobustParCorr":
            from tigramite.independence_tests.robust_parcorr import RobustParCorr
            return RobustParCorr(significance="analytic")
        elif ci_test_name == "ParCorrWLS":
            from tigramite.independence_tests.parcorr_wls import ParCorrWLS
            return ParCorrWLS(significance="analytic")
        elif ci_test_name == "GPDC":
            from tigramite.independence_tests.gpdc import GPDC
            return GPDC(significance="analytic")
        elif ci_test_name == "CMIknn":
            from tigramite.independence_tests.cmiknn import CMIknn
            return CMIknn(significance="analytic")
        raise ValueError(f"Unknown CI test: {ci_test_name}")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("tigramite is not installed. Install it with pip or conda-forge.") from e


def _time_to_order(x):
    """English documentation."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    match = re.search(r"-?\d+(?:\.\d+)?", str(x))
    if match:
        return float(match.group())
    return np.nan


# ======================================================
# ======================================================
def run_pcmci_from_excel(
    input_excel, output_excel,
    alpha=0.05, tau_min=1, tau_max=3,
    ci_test_name="ParCorr", verbosity=1,
):
    """English documentation."""
    try:
        import tigramite.data_processing as pp
        from tigramite.pcmci import PCMCI
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("tigramite is not installed.") from e

    input_excel = Path(input_excel); output_excel = Path(output_excel)
    output_excel.parent.mkdir(parents=True, exist_ok=True)
    if tau_min < 1:
        raise ValueError("tau_min must be at least 1 to avoid same-time information.")
    if tau_max < tau_min:
        raise ValueError("tau_max must be greater than or equal to tau_min.")

    cond_ind_test = _build_cond_ind_test(ci_test_name)
    absolute = pd.read_excel(input_excel)
    absolute.columns = [str(c).strip() for c in absolute.columns]
    absolute = absolute.rename(columns={absolute.columns[0]: "SubjectID", absolute.columns[1]: "Time"})

    from ..common.utils import detect_group_column
    group_col = detect_group_column(absolute)

    species_cols = [c for c in absolute.columns if c not in ["SubjectID", "Time"] +
                    ([group_col] if group_col else [])]
    if len(species_cols) == 0:
        raise ValueError("No species columns were found.")
    absolute["Time_order"] = absolute["Time"].apply(_time_to_order)
    if absolute["Time_order"].isna().any():
        raise ValueError(f"Time contains unsortable values: {absolute.loc[absolute['Time_order'].isna(),'Time'].unique()[:10]}")
    for col in species_cols:
        absolute[col] = pd.to_numeric(absolute[col], errors="coerce")
    df = absolute.sort_values(["SubjectID", "Time_order"]).reset_index(drop=True)

    print(f"[PCMCI] Subjects={df['SubjectID'].nunique()}，species={len(species_cols)}")

    data_dict = {}
    skipped_subjects = []
    for subject_id, sub_df in df.groupby("SubjectID"):
        sub_df = sub_df.sort_values("Time_order")
        arr = sub_df[species_cols].fillna(0).to_numpy(dtype=float)
        if arr.shape[0] <= tau_max:
            skipped_subjects.append((subject_id, arr.shape[0]))
            print(f"  Skipping {subject_id}: T={arr.shape[0]} <= tau_max={tau_max}")
            continue
        data_dict[subject_id] = arr
    if len(data_dict) == 0:
        raise ValueError("No usable time series. Check SubjectID, Time, and tau_max.")

    dataframe = pp.DataFrame(data=data_dict, var_names=species_cols, analysis_mode="multiple")
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=verbosity)

    try:
        results = pcmci.run_pcmci(tau_min=tau_min, tau_max=tau_max,
                                   pc_alpha=alpha, alpha_level=alpha)
    except TypeError:
        results = pcmci.run_pcmci(tau_min=tau_min, tau_max=tau_max, pc_alpha=alpha)

    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]

    try:
        graph = pcmci.get_graph_from_pmatrix(p_matrix=p_matrix, alpha_level=alpha,
                                              tau_min=tau_min, tau_max=tau_max)
    except AttributeError:
        graph = np.full(p_matrix.shape, "", dtype=object)
        for lag in range(tau_min, tau_max + 1):
            sig = p_matrix[:, :, lag] < alpha
            graph[:, :, lag][sig] = "-->"

    rows_edges = []
    for i, source in enumerate(species_cols):
        for j, target in enumerate(species_cols):
            for lag in range(tau_min, tau_max + 1):
                if str(graph[i, j, lag]).strip() != "":
                    rows_edges.append({
                        "source": source,
                        "target": target,
                        "lag": lag,
                        "meaning": f"{source}(t-{lag}) -> {target}(t)",
                        "graph_mark": str(graph[i, j, lag]),
                        "p_value": float(p_matrix[i, j, lag]),
                        "test_statistic": float(val_matrix[i, j, lag]),
                    })
    edge_list = pd.DataFrame(rows_edges, columns=[
        "source", "target", "lag", "meaning", "graph_mark", "p_value", "test_statistic"])

    summary = pd.DataFrame([{
        "subjects": len(data_dict),
        "nodes": len(species_cols),
        "edges": len(edge_list),
        "tau_min": tau_min, "tau_max": tau_max,
        "alpha": alpha,
        "ci_test": ci_test_name,
    }])

    with pd.ExcelWriter(output_excel) as writer:
        edge_list.to_excel(writer, sheet_name="edge_list", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
        if skipped_subjects:
            pd.DataFrame(skipped_subjects, columns=["SubjectID", "T"]).to_excel(
                writer, sheet_name="skipped_subjects", index=False)
    print(f"[PCMCI] Complete; edges={len(edge_list)}; output: {output_excel}")
    return output_excel


# ======================================================
# ======================================================
def run_lingam(abundance_file, out_path, p_default=0.05, lag_default=1):
    try:
        import lingam
    except ImportError:
        raise RuntimeError("lingam is not installed. Install it with pip.")
    loader = MicrobiomeDataLoader(abundance_file)
    loader.load_abundance()
    stack = []
    for sid, sample in loader.samples.items():
        stack.append(sample["abundances"][-1])
    X = np.log(np.vstack(stack) + 1e-8)
    model = lingam.DirectLiNGAM()
    model.fit(X)
    adj = model.adjacency_matrix_
    species = loader.species_names
    rows = []
    for j in range(len(species)):
        for i in range(len(species)):
            w = float(adj[j, i])
            if abs(w) > 1e-10:
                rows.append({
                    "source": species[i], "target": species[j],
                    "lag": lag_default, "p_value": p_default, "test_statistic": w,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["source", "target", "lag", "meaning", "graph_mark", "p_value", "test_statistic"])
    else:
        df["meaning"] = df.apply(lambda r: f"{r['source']}(t-{int(r['lag'])}) -> {r['target']}(t)", axis=1)
        df["graph_mark"] = "-->"
        df = df[["source", "target", "lag", "meaning", "graph_mark", "p_value", "test_statistic"]]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path) as writer:
        df.to_excel(writer, sheet_name="edge_list", index=False)
    return out_path


def _normalize_graph_output(df_source_target, lag_default=1, p_default=0.05, graph_mark_default="-->"):
    """English documentation."""
    if df_source_target is None or len(df_source_target) == 0:
        return pd.DataFrame(columns=["source", "target", "lag", "meaning", "graph_mark", "p_value", "test_statistic"])
    out = pd.DataFrame({
        "source": df_source_target["source"],
        "target": df_source_target["target"],
        "lag": df_source_target["lag"] if "lag" in df_source_target.columns else lag_default,
        "p_value": df_source_target["p_value"] if "p_value" in df_source_target.columns else p_default,
        "test_statistic": df_source_target["test_statistic"] if "test_statistic" in df_source_target.columns else 0.0,
    })
    out["lag"] = pd.to_numeric(out["lag"], errors="coerce").fillna(lag_default).astype(int)
    out.loc[out["lag"] < 1, "lag"] = 1
    out["p_value"] = pd.to_numeric(out["p_value"], errors="coerce").fillna(p_default)
    out["test_statistic"] = pd.to_numeric(out["test_statistic"], errors="coerce").fillna(0.0)
    if "meaning" in df_source_target.columns:
        out["meaning"] = df_source_target["meaning"].fillna("")
    else:
        out["meaning"] = ""
    if "graph_mark" in df_source_target.columns:
        out["graph_mark"] = df_source_target["graph_mark"].fillna(graph_mark_default)
    else:
        out["graph_mark"] = graph_mark_default
    need_fill = out["meaning"].astype(str).str.len() == 0
    out.loc[need_fill, "meaning"] = out.loc[need_fill].apply(
        lambda r: f"{r['source']}(t-{int(r['lag'])}) -> {r['target']}(t)", axis=1
    )
    out["graph_mark"] = out["graph_mark"].astype(str).replace({"nan": graph_mark_default})
    return out[["source", "target", "lag", "meaning", "graph_mark", "p_value", "test_statistic"]]


def _build_regression_graph(loader, max_lag=1, mode="granger", alpha=0.05):
    """English documentation."""
    from sklearn.linear_model import LassoCV
    import statsmodels.api as sm

    if loader.N is None or not loader.samples:
        loader.load_abundance()

    species = loader.species_names
    if not species:
        raise ValueError("No usable species columns were found.")

    max_lag = max(1, int(max_lag))
    rows = []
    samples = [(sid, sample) for sid, sample in loader.samples.items() if sample["T"] > max_lag]
    if not samples:
        raise ValueError("No time series is long enough for lagged-regression causal discovery.")

    for tgt_idx, tgt_name in enumerate(species):
        X_rows = []
        y_rows = []
        for _, sample in samples:
            abund = np.asarray(sample["abundances"], dtype=float)
            T = sample["T"]
            for t in range(max_lag, T):
                features = []
                for lag in range(1, max_lag + 1):
                    features.extend(abund[t - lag, :].tolist())
                X_rows.append(features)
                y_rows.append(float(abund[t, tgt_idx]))
        X = np.asarray(X_rows, dtype=float)
        y = np.asarray(y_rows, dtype=float)
        if X.size == 0 or y.size == 0:
            continue

        feature_names = []
        for lag in range(1, max_lag + 1):
            for src_name in species:
                feature_names.append((src_name, lag))

        if mode == "lasso_dbn":
            scaler_mean = X.mean(axis=0)
            scaler_std = X.std(axis=0)
            scaler_std[scaler_std == 0] = 1.0
            Xs = (X - scaler_mean) / scaler_std
            model = LassoCV(cv=min(5, max(2, len(y) // 5)), random_state=42, n_alphas=50, max_iter=10000)
            model.fit(Xs, y)
            coefs = model.coef_ / scaler_std
            for idx, coef in enumerate(coefs):
                if abs(coef) <= 1e-12:
                    continue
                src_name, lag = feature_names[idx]
                rows.append({
                    "source": src_name,
                    "target": tgt_name,
                    "lag": int(lag),
                    "meaning": f"{src_name}(t-{int(lag)}) -> {tgt_name}(t)",
                    "graph_mark": "-->",
                    "p_value": 0.05,
                    "test_statistic": float(coef),
                })
            continue

        Xs = sm.add_constant(X, has_constant="add")
        fit = sm.OLS(y, Xs).fit()
        for idx, (src_name, lag) in enumerate(feature_names, start=1):
            coef = float(fit.params[idx])
            pval = float(fit.pvalues[idx])
            if not np.isfinite(coef) or not np.isfinite(pval):
                continue
            if pval < alpha and abs(coef) > 1e-12:
                rows.append({
                    "source": src_name,
                    "target": tgt_name,
                    "lag": int(lag),
                    "meaning": f"{src_name}(t-{int(lag)}) -> {tgt_name}(t)",
                    "graph_mark": "-->",
                    "p_value": pval,
                    "test_statistic": coef,
                })

    if not rows:
        warnings.warn("Lagged-regression causal discovery produced no significant edges.")
    return _normalize_graph_output(pd.DataFrame(rows))


def _build_neural_graph(loader, method="cmlp", max_lag=2, epochs=10, lr=0.01):
    """English documentation."""
    try:
        from hybridmetrics import NeuralGrangerCausality
    except ImportError:
        raise ImportError(
            "cMLP and cLSTM require the hybridmetrics package."
        )

    if loader.N is None or not loader.samples:
        loader.load_abundance()

    species = loader.species_names
    if not species:
        raise ValueError("No usable species columns were found.")

    method = str(method or "cmlp").strip().lower()
    if method not in ("cmlp", "clstm"):
        raise ValueError(f"Unknown neural Granger method: {method}")

    max_lag = max(1, int(max_lag))
    blocks = []
    for _, sample in loader.samples.items():
        if sample["T"] <= max_lag:
            continue
        blocks.append(np.asarray(sample["abundances"], dtype=float))
    if not blocks:
        raise ValueError("No time series is long enough for cMLP or cLSTM causal discovery.")
    data = np.vstack(blocks)  # (sum_T, k)

    try:
        model = NeuralGrangerCausality(
            method=method,
            lag=max_lag,
            hidden=32,
            num_layers=1,
            lam=0.0,
            lam_ridge=1e-4,
            lr=lr,
            epochs=epochs,
        )
    except TypeError:
        model = NeuralGrangerCausality(method=method, lag=max_lag)
    model.fit(data, verbose=False)

    # causality_pairs: cause / effect / strength
    pairs = model.causality_pairs(threshold=1e-3)
    rows = []
    for _, r in pairs.iterrows():
        rows.append({
            "source": str(r["cause"]),
            "target": str(r["effect"]),
            "lag": max_lag,
            "meaning": f"{r['cause']}(t-{max_lag}) -> {r['effect']}(t)",
            "graph_mark": "-->",
            "p_value": 0.05,
            "test_statistic": float(r["strength"]),
        })
    if not rows:
        warnings.warn(f"{method} causal discovery produced no significant edges; consider increasing --epochs.")
    return _normalize_graph_output(pd.DataFrame(rows))


def _build_pc_graph(loader, alpha=0.05):
    """English documentation."""
    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz
    except ImportError as e:
        raise ImportError("PC requires the causal-learn package.") from e

    if loader.N is None or not loader.samples:
        loader.load_abundance()

    species = loader.species_names
    if not species:
        raise ValueError("No usable species columns were found.")

    data = np.vstack([np.asarray(sample["abundances"][-1], dtype=float) for sample in loader.samples.values()])
    if data.shape[0] < 3:
        raise ValueError("PC requires at least three cross-sectional samples.")

    try:
        cg = pc(
            data,
            alpha=float(alpha),
            indep_test=fisherz,
            stable=True,
            uc_rule=0,
            uc_priority=2,
            verbose=False,
            show_progress=False,
            node_names=species,
        )
    except TypeError:
        cg = pc(data, alpha=float(alpha), indep_test=fisherz)
    graph = cg.G.graph

    rows = []
    for i, source in enumerate(species):
        for j, target in enumerate(species):
            if i == j:
                continue
            if not (graph[i, j] == -1 and graph[j, i] == 1):
                continue
            x = data[:, i].astype(float)
            y = data[:, j].astype(float)
            x_std = float(np.std(x))
            y_std = float(np.std(y))
            if x_std == 0.0 or y_std == 0.0:
                coef = 0.0
            else:
                xz = (x - np.mean(x)) / x_std
                yz = (y - np.mean(y)) / y_std
                coef = float(np.corrcoef(xz, yz)[0, 1])
                if not np.isfinite(coef):
                    coef = 0.0
            rows.append({
                "source": source,
                "target": target,
                "lag": 1,
                "meaning": f"{source}(t-1) -> {target}(t)",
                "graph_mark": "-->",
                "p_value": 0.05,
                "test_statistic": coef,
            })

    if not rows:
        warnings.warn("PC causal discovery produced no significant directed edges.")
    return _normalize_graph_output(pd.DataFrame(rows))


def build_causal_graph(args):
    """English documentation."""
    out_dir = args.output_dir or "outputs/00_prepare"
    ensure_dir(out_dir)
    graph_arg = getattr(args, "graph", None)
    graph_path = Path(graph_arg) if graph_arg else None
    if graph_path and graph_path.exists():
        xls = pd.ExcelFile(graph_path)
        sheet = "edge_list" if "edge_list" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(graph_path, sheet_name=sheet)
        return _normalize_graph_output(df)
    alg = getattr(args, "alg", getattr(args, "causal_method", "pcmci")).lower()
    import tempfile
    _tmpdir = tempfile.mkdtemp(prefix="mmld_causal_")
    tmp_out = str(graph_path) if graph_path and not graph_path.exists() else os.path.join(_tmpdir, "raw_causal_graph.xlsx")
    if alg == "pcmci":
        run_pcmci_from_excel(
            input_excel=args.abundance,
            output_excel=tmp_out,
            alpha=getattr(args, "alpha", 0.05),
            tau_min=getattr(args, "tau_min", 1),
            tau_max=getattr(args, "tau_max", 2),
            ci_test_name=getattr(args, "ci_test", "ParCorr"),
            verbosity=1,
        )
    elif alg == "lingam":
        run_lingam(args.abundance, tmp_out)
    elif alg in ("granger", "lasso_dbn", "cmlp", "clstm"):
        loader = MicrobiomeDataLoader(args.abundance)
        loader.load_abundance()
        max_lag = int(getattr(args, "tau_max", 2))
        if alg in ("cmlp", "clstm"):
            df = _build_neural_graph(
                loader,
                method=alg,
                max_lag=max_lag,
                epochs=int(getattr(args, "epochs", 10)),
                lr=float(getattr(args, "lr", 0.01)),
            )
        else:
            df = _build_regression_graph(
                loader,
                max_lag=max_lag,
                mode="lasso_dbn" if alg == "lasso_dbn" else "granger",
                alpha=float(getattr(args, "alpha", 0.05)),
            )
        Path(tmp_out).parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(tmp_out) as writer:
            df.to_excel(writer, sheet_name="edge_list", index=False)
    elif alg == "pc":
        loader = MicrobiomeDataLoader(args.abundance)
        loader.load_abundance()
        df = _build_pc_graph(loader, alpha=float(getattr(args, "alpha", 0.05)))
        Path(tmp_out).parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(tmp_out) as writer:
            df.to_excel(writer, sheet_name="edge_list", index=False)
    else:
        raise NotImplementedError(f"Algorithm not implemented: {alg}. Supply another algorithm as a seven-column causal graph xlsx via --graph.")
    raw_df = pd.read_excel(tmp_out, sheet_name="edge_list")
    return _normalize_graph_output(raw_df)


# ======================================================
# ======================================================
def train_mmld_weighted(model, train_pairs, epochs=10, lr=0.01, device="cpu", verbose=True):
    """English documentation."""
    import torch.nn.functional as F
    if not train_pairs:
        warnings.warn("No usable gLV training pairs; model parameters remain initialized.")
        return []
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    import random
    history_log = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_mse = 0.0; count = 0
        random.shuffle(train_pairs)
        for pair in train_pairs:
            current = pair["current"].to(device)
            hist = pair["history"].to(device)
            target = pair["target"].to(device)
            dt = pair["dt"].to(device) if hasattr(pair["dt"], "to") else torch.tensor(pair["dt"], dtype=torch.float32, device=device)
            optimizer.zero_grad()
            pred = model(current, hist, dt)
            mse = torch.mean((pred - target) ** 2)
            self_interaction_reg = 0.001 * torch.mean(F.softplus(model.self_interaction_raw) ** 2)
            A_edge = model.get_A_edge()
            if A_edge.numel() > 0:
                edge_reg = 0.001 * torch.mean(A_edge ** 2)
            else:
                edge_reg = torch.zeros((), device=device)
            loss = mse + self_interaction_reg + edge_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_mse += float(mse.item())
            count += 1
        avg = total_mse / max(count, 1)
        history_log.append(avg)
    return history_log


def _is_cross_sectional_loader(loader):
    return all(sample["T"] <= 1 for sample in loader.samples.values())


def _build_steady_state_training_pairs(loader, n_steps_per_subject=5):
    """English documentation."""
    pairs = []
    lag_min, lag_max = loader.lag_min, loader.lag_max
    t_pseudo = max(lag_max + 2, n_steps_per_subject + lag_max + 1)
    for sid, sample in loader.samples.items():
        h0 = np.asarray(sample["abundances"][-1], dtype=np.float32)
        pseudo_abund = np.stack([h0] * t_pseudo)
        for t in range(lag_max, t_pseudo - 1):
            history = []
            for lag in range(lag_min, lag_max + 1):
                history.append(pseudo_abund[t - lag])
            pairs.append({
                "sample_id": sid,
                "time_index": int(t),
                "current": torch.tensor(pseudo_abund[t], dtype=torch.float32),
                "history": torch.tensor(np.stack(history), dtype=torch.float32),
                "target": torch.tensor(pseudo_abund[t + 1], dtype=torch.float32),
                "dt": torch.tensor(1.0, dtype=torch.float32),
            })
    return pairs


def _glv_parameter_tables(model, species_names):
    with torch.no_grad():
        growth = model.intrinsic_growth_rate.detach().cpu().numpy()
        self_raw = model.self_interaction_raw.detach().cpu().numpy()
        self_interaction = model.get_self_interaction().detach().cpu().numpy()
    growth_df = pd.DataFrame({
        "species": species_names,
        "intrinsic_growth_rate": growth,
    })
    self_df = pd.DataFrame({
        "species": species_names,
        "self_interaction": self_interaction,
        "self_interaction_raw": self_raw,
    })
    return growth_df, self_df


def mmld_weight(loader: MicrobiomeDataLoader, epochs=10, lr=0.01, device="cpu"):
    loader.prepare_training_data()
    if not loader.training_pairs and _is_cross_sectional_loader(loader):
        print("[prepare cross_section] Training gLV parameters with steady-state fixed-point pairs.")
        loader.training_pairs = _build_steady_state_training_pairs(loader, n_steps_per_subject=5)
    model = build_glv_model(getattr(loader, "model_type", "euler"), loader.N, loader.lag_min, loader.lag_max, loader.edge_info)
    train_mmld_weighted(model, loader.training_pairs, epochs=epochs, lr=lr, device=device)
    with torch.no_grad():
        A_edge = model.get_A_edge().cpu().numpy()
    growth_df, self_df = _glv_parameter_tables(model, loader.species_names)
    rows = []
    for edge, aij in zip(loader.edge_info, A_edge):
        src = loader.species_names[edge["source_idx"]]
        tgt = loader.species_names[edge["target_idx"]]
        rows.append({
            "source": src,
            "target": tgt,
            "lag": edge["lag"],
            "meaning": f"{src}(t-{edge['lag']}) -> {tgt}(t)",
            "graph_mark": "-->",
            "p_value": edge.get("p_value", 0.05),
            "test_statistic": float(aij),
        })
    if rows:
        weighted_df = pd.DataFrame(rows)[["source", "target", "lag", "meaning", "graph_mark", "p_value", "test_statistic"]]
    else:
        weighted_df = pd.DataFrame(columns=["source", "target", "lag", "meaning", "graph_mark", "p_value", "test_statistic"])
    return model, weighted_df, growth_df, self_df


# ======================================================
# ======================================================
def _abbrev_name(name):
    """English documentation."""
    parts = str(name).split('-')
    if len(parts) >= 2:
        return f"{parts[0][0]}. {parts[1]}"
    return name


def plot_causal_network(graph_df, out_pdf, driver_species=None, top_edges=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
        import matplotlib.lines as mlines
    except ImportError:
        warnings.warn("matplotlib or networkx is not installed; skipping the plot")
        return
    df = graph_df.copy()
    if df.empty:
        warnings.warn("The causal graph has no edges; skipping the network plot.")
        return
    if top_edges and len(df) > top_edges:
        df = df.reindex(df["test_statistic"].abs().sort_values(ascending=False).index).head(top_edges)

    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(_abbrev_name(row["source"]), _abbrev_name(row["target"]),
                   weight=abs(float(row["test_statistic"])) * 10,
                   sign=(float(row["test_statistic"]) > 0))

    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'sans-serif']
    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)

    pos = nx.circular_layout(G)
    node_size = 4000

    color_pos = '#ca6661'
    color_neg = '#4ca7bf'

    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='#FDFDFD',
                           edgecolors='#D1D5DB', linewidths=2.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold',
                            font_color='#2C3E50', ax=ax)

    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d['sign']]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if not d['sign']]
    pos_weights = [G[u][v]['weight'] for u, v in pos_edges]
    neg_weights = [G[u][v]['weight'] for u, v in neg_edges]

    if pos_edges:
        nx.draw_networkx_edges(G, pos, edgelist=pos_edges, edge_color=color_pos,
                               width=pos_weights, arrowsize=45, arrowstyle='-|>',
                               connectionstyle='arc3,rad=0.15', ax=ax,
                               alpha=0.9, node_size=node_size)
    if neg_edges:
        nx.draw_networkx_edges(G, pos, edgelist=neg_edges, edge_color=color_neg,
                               width=neg_weights, arrowsize=45, arrowstyle='-|>',
                               connectionstyle='arc3,rad=0.15', ax=ax,
                               alpha=0.9, node_size=node_size)

    if driver_species:
        drs = set(_abbrev_name(s) for s in driver_species)
        node_list = [n for n in G.nodes() if n in drs]
        if node_list:
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=node_list,
                                   node_size=node_size + 600,
                                   node_color='none', edgecolors='#d94848',
                                   linewidths=3.0)

    pos_line = mlines.Line2D([], [], color=color_pos, linewidth=4,
                             marker='>', markersize=12, label='Positive Causality')
    neg_line = mlines.Line2D([], [], color=color_neg, linewidth=4,
                             marker='>', markersize=12, label='Negative Causality')
    legend_handles = [pos_line, neg_line]
    if driver_species:
        legend_handles.append(
            mlines.Line2D([], [], marker='o', color='#d94848', label='Driver Species',
                          linestyle='None', markerfacecolor='none',
                          markersize=12, markeredgewidth=2.5)
        )
    ax.legend(handles=legend_handles, loc='upper left', frameon=True,
              fontsize=12, borderpad=1, shadow=True, bbox_to_anchor=(0.0, 1.05))

    ax.set_title('Causal Microbial Interaction Network', fontsize=22,
                 fontweight='bold', pad=30, color='#1A1A1A')
    ax.axis('off')
    fig.tight_layout()
    ensure_dir(os.path.dirname(out_pdf) or ".")
    fig.savefig(out_pdf, format='pdf', bbox_inches='tight')
    plt.close(fig)


def _copy_prepare_inputs(args, out_dir):
    """Copy user-supplied prepare inputs into the prepare output directory."""
    abundance = Path(getattr(args, "abundance", "") or "")
    if not abundance.is_file():
        raise FileNotFoundError(f"Abundance table not found: {abundance}")

    copied = {}
    destinations = [(abundance, Path(out_dir) / "input_abundance.xlsx", "abundance")]
    graph_arg = getattr(args, "graph", None)
    if graph_arg:
        graph = Path(graph_arg)
        if graph.is_file():
            destinations.append((graph, Path(out_dir) / "input_causal_graph.xlsx", "graph"))

    # MPC uses separate initial- and target-state workbooks. Explicit paths take
    # precedence; otherwise discover the conventional filenames next to the
    # abundance workbook so the example directory works with the minimal
    # prepare command.
    for attr, default_name, output_name, key in (
        ("mpc_initial", "initial.xlsx", "input_mpc_initial.xlsx", "mpc_initial"),
        ("mpc_target", "target.xlsx", "input_mpc_target.xlsx", "mpc_target"),
    ):
        explicit_path = getattr(args, attr, None)
        source = Path(explicit_path) if explicit_path else abundance.parent / default_name
        if explicit_path and not source.is_file():
            raise FileNotFoundError(f"MPC input table not found: {source}")
        if source.is_file():
            destinations.append((source, Path(out_dir) / output_name, key))

    for source, destination, key in destinations:
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        copied[key] = str(destination)
        print(f"  Copied input: {destination}")
    return copied


# ======================================================
# ======================================================
def main_prepare(args):
    out_dir = getattr(args, "output_dir", None) or "outputs/00_prepare"
    ensure_dir(out_dir)
    glv_method = str(getattr(args, "glv_method", "euler")).strip().lower()
    if glv_method not in ("euler", "rk4"):
        raise ValueError("Unknown gLV solver: %s (supported: euler or rk4)" % glv_method)

    print("[prepare] Step 0: archive input files")
    copied_inputs = _copy_prepare_inputs(args, out_dir)

    print("[prepare] Step 1: build or load causal graph")
    args.output_dir = out_dir
    raw_graph_df = build_causal_graph(args)

    print("[prepare] Step 2: load and preprocess data")
    # The normalized raw graph is an internal intermediate file only. Keeping it
    # in a temporary directory avoids exposing a second, easily confused graph
    # workbook in the prepare outputs.
    with tempfile.TemporaryDirectory(prefix="mmld_prepare_") as temporary_dir:
        internal_graph_path = os.path.join(temporary_dir, "causal_graph.xlsx")
        with pd.ExcelWriter(internal_graph_path) as writer:
            raw_graph_df.to_excel(writer, sheet_name="edge_list", index=False)
        loader = MicrobiomeDataLoader(
            args.abundance, internal_graph_path,
            group_is_present=(None if not getattr(args, "no_group", False) else False),
        )
        loader.load_abundance()
        loader.load_graph()
    loader.model_type = glv_method
    data_type = "cross_section" if _is_cross_sectional_loader(loader) else "time_series"
    print(f"[prepare] data_type={data_type}")

    print("[prepare] Step 3: train MMLD")
    model, weighted_df, growth_df, self_df = mmld_weight(
        loader,
        epochs=getattr(args, "epochs", 10),
        lr=getattr(args, "lr", 0.01),
        device="cpu",
    )
    weighted_path = os.path.join(out_dir, "MMLD_processed_causal_graph.xlsx")
    with pd.ExcelWriter(weighted_path) as writer:
        weighted_df.to_excel(writer, sheet_name="edge_list", index=False)
        growth_df.to_excel(writer, sheet_name="growth_rate", index=False)
        self_df.to_excel(writer, sheet_name="self_interaction", index=False)
    print(f"  Saved: {weighted_path}")

    model_path = os.path.join(out_dir, "MMLD_model.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "N": loader.N, "lag_min": loader.lag_min, "lag_max": loader.lag_max,
        "edge_info": loader.edge_info, "species_names": loader.species_names,
        "SCALE": loader.SCALE,
        "data_type": data_type,
        "model_type": glv_method,
    }, model_path)
    print(f"  Saved: {model_path}")

    cache_path = os.path.join(out_dir, "preprocessed_data.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump({
            "loader_samples": loader.samples, "species_names": loader.species_names,
            "N": loader.N, "SCALE": loader.SCALE,
            "lag_min": loader.lag_min, "lag_max": loader.lag_max,
            "num_lags": loader.num_lags, "edge_info": loader.edge_info,
            "group_col": loader.group_col, "group_classes": loader.group_classes,
            "data_type": data_type,
        }, f)
    print(f"  Saved: {cache_path}")

    print("[prepare] Step 4: plot original causal network")
    pdf_path = os.path.join(out_dir, "causal_network.pdf")
    plot_causal_network(raw_graph_df, pdf_path)
    print(f"  Saved: {pdf_path}")

    print(f"[prepare] Complete. Outputs: {out_dir}")
    return {
        "out_dir": out_dir,
        "loader": loader,
        "model": model,
        "weighted_graph": weighted_df,
        "growth_rate": growth_df,
        "self_interaction": self_df,
        "data_type": data_type,
        "copied_inputs": copied_inputs,
    }
