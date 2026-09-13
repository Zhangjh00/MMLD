"""English documentation."""
import os
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.integrate import solve_ivp
from scipy.linalg import expm, solve_discrete_are

from ..common import set_seed, ensure_dir, MicrobiomeDataLoader


TAU_DEFAULT = 1.0
SELF_LOOP_DELTA = 1.0
Q_WEIGHT_DEFAULT = 2e4
R_WEIGHT_DEFAULT = 1.5e-1
REWIRING_PROBABILITY_DEFAULT = 0.0
EPSILON_COEFFICIENT_DEFAULT = 0.1
EPS = 1e-10
DEVICE = "cpu"


def safe_torch_load(path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def f_holling_ii(x, theta):
    return x / (1.0 + theta * x)


def system_dynamics(t, x, A, growth_rate, theta):
    x = np.maximum(x, 0.0)
    f_val = f_holling_ii(x, theta)
    dx = x * (A @ f_val + growth_rate)
    dx[x <= 0] = 0.0
    return dx


def get_final_state(x0, tau, A, growth_rate, theta):
    x0 = np.maximum(x0, 0.0).copy()
    x0[x0 <= 0] = 0
    sol = solve_ivp(system_dynamics, [0, tau], x0, args=(A, growth_rate, theta),
                    method="RK45", rtol=1e-7, atol=1e-7)
    return np.maximum(sol.y[:, -1], 0.0)


def get_final_state_no_control(x0, tau, A, growth_rate, theta):
    x0 = np.maximum(x0, 0.0).copy()
    x0[x0 <= 0] = 0

    def _dyn(t, x):
        x2 = np.maximum(x, 0.0)
        f_val = f_holling_ii(x2, theta)
        dx = x2 * (A @ f_val + growth_rate)
        dx[x2 <= 0] = 0.0
        return dx
    sol = solve_ivp(_dyn, [0, tau], x0, method="RK45", rtol=1e-7, atol=1e-7)
    return np.maximum(sol.y[:, -1], 0.0)


def rewire_network(A, p):
    N = A.shape[0]
    A = A.copy()
    row, col = np.where(np.abs(A) > 0)
    if p > 0.0 and len(row) >= 2:
        n_rewire = int(len(row) * p)
        for _ in range(n_rewire):
            idx1, idx2 = np.random.choice(len(row), 2, replace=False)
            A[row[idx1], col[idx1]], A[row[idx2], col[idx2]] = \
                A[row[idx2], col[idx2]], A[row[idx1], col[idx1]]
    return A


def compute_lqr_gain(A_discrete, B_discrete, Q, R):
    P = solve_discrete_are(A_discrete, B_discrete, Q, R)
    K = np.linalg.inv(R + B_discrete.T @ P @ B_discrete) @ B_discrete.T @ P @ A_discrete
    return K


def build_B_matrix(N, driver_indices):
    m = len(driver_indices)
    B = np.zeros((N, m))
    for i, idx in enumerate(driver_indices):
        B[idx, i] = np.random.rand()
    return B


def pearsonr_safe(x, y):
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    if len(x) < 2:
        return float("nan")
    if np.std(x) < EPS or np.std(y) < EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def calc_metrics(x, y):
    x, y = np.asarray(x), np.asarray(y)
    err = x - y
    rmse = float(math.sqrt(np.mean(err ** 2)))
    pr = pearsonr_safe(x, y)
    return {"RMSE": rmse, "Pearson_R": pr}


def direction_text(delta, tol=1e-8):
    if delta > tol:
        return "increase"
    if delta < -tol:
        return "decrease"
    return "unchanged"


def GetControllerSuccess(x0, xd, tau, MaxNumberOfSteps, epsilon,
                         A, growth_rate, theta, RewiringProbability, driver_indices,
                         q_weight=Q_WEIGHT_DEFAULT, r_weight=R_WEIGHT_DEFAULT):
    N = len(x0)
    x0 = np.maximum(x0, 0.0).copy()
    x0[x0 <= 0] = 0
    initial_error = np.linalg.norm(x0 - xd, np.inf)
    if initial_error <= epsilon:
        return -1, [x0.copy()], []
    ctr_success = 0
    trajectory = [x0.copy()]
    controls = []

    Atemp = A.copy() + SELF_LOOP_DELTA * np.eye(N)
    Atemp_rewired = rewire_network(Atemp, RewiringProbability)
    A_rewired = Atemp_rewired - SELF_LOOP_DELTA * np.eye(N)

    m = len(driver_indices)
    B = build_B_matrix(N, driver_indices)
    Q = q_weight * np.eye(N)
    Rmat = r_weight * np.eye(m)

    A_discrete = expm(A_rewired * tau)
    B_discrete = A_discrete @ B

    try:
        K = compute_lqr_gain(A_discrete, B_discrete, Q, Rmat)
    except np.linalg.LinAlgError:
        K = np.zeros((m, N))

    xprev = x0.copy()
    for k in range(MaxNumberOfSteps):
        xnext = get_final_state(xprev - B @ K @ (xprev - xd), tau, A, growth_rate, theta)
        control_input = -K @ (xprev - xd)
        controls.append(control_input.copy())
        trajectory.append(xnext.copy())
        xprev = xnext.copy()
    return ctr_success, trajectory, controls


def extract_model_parameters(checkpoint):
    N = int(checkpoint["N"])
    edge_info = list(checkpoint["edge_info"])
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint.get("model_state_dict")
    if state_dict is None:
        raise ValueError("The model checkpoint contains neither state_dict nor model_state_dict")
    growth_rate_key = "intrinsic_growth_rate" if "intrinsic_growth_rate" in state_dict else "r"
    growth_rate = state_dict[growth_rate_key].cpu().numpy().flatten()
    _A_raw_tensor = state_dict["_A_raw"].detach().cpu()
    signs = np.array([edge.get("sign", 1.0) for edge in edge_info])
    A_edge = F.softplus(_A_raw_tensor).numpy().flatten() * signs
    A = np.zeros((N, N), dtype=np.float64)
    for i, edge in enumerate(edge_info):
        src, tgt = int(edge["source_idx"]), int(edge["target_idx"])
        A[tgt, src] = A_edge[i]
    A = A - SELF_LOOP_DELTA * np.eye(N)
    alpha_key = "self_interaction_raw" if "self_interaction_raw" in state_dict else "_alpha_raw"
    if alpha_key in state_dict:
        alpha_raw = state_dict[alpha_key].cpu().numpy().flatten()
        self_interaction = -np.log1p(np.exp(alpha_raw))
        theta = np.abs(self_interaction) * 0.1
    else:
        theta = np.full(N, 0.01)
    train_scale = checkpoint.get("SCALE") or checkpoint.get("scale")
    if train_scale is not None:
        train_scale = float(train_scale)
    return A, growth_rate, theta, train_scale


def read_abundance_with_group(path, species_names=None):
    df = pd.read_excel(path)
    cols = list(df.columns)
    if cols[0] != "SubjectID" or cols[1] != "Time":
        df.columns = ["SubjectID", "Time"] + cols[2:]
    df["SubjectID"] = df["SubjectID"].astype(str)
    return df


def _detect_label_column(df, species_names):
    """English documentation."""
    from ..common.utils import detect_group_column
    col = detect_group_column(df)
    return col


def compute_global_scale(current_df, target_df, species_names):
    all_sums = []
    for df in [current_df, target_df]:
        vals = df[species_names].values.astype(np.float64)
        all_sums.extend(vals.sum(axis=1).tolist())
    all_sums = np.array([s for s in all_sums if s > EPS])
    if len(all_sums) == 0:
        return 1.0
    median_total = np.median(all_sums)
    scale = median_total if median_total > 0 else 1.0
    return float(scale)


def build_target_map(target_df, species_names, global_scale, default_last_n=5):
    """English documentation."""
    out = {}
    for sid, sub in target_df.groupby("SubjectID"):
        sub = sub.sort_values("Time").reset_index(drop=True)
        mat = sub[species_names].values.astype(np.float64)
        if len(mat) == 0:
            continue
        mean_vec = mat[-1]
        out[str(sid)] = np.maximum(mean_vec, 0.0) / max(global_scale, EPS)
    return out


def _best_pearson_row(df):
    """English documentation."""
    if df.empty:
        return None
    valid = df[np.isfinite(pd.to_numeric(df["Pearson_R_controlled"], errors="coerce"))]
    if valid.empty:
        return df.iloc[0]
    return valid.loc[valid["Pearson_R_controlled"].idxmax()]


def parse_driver_species(driver_species_arg, species_names):
    if isinstance(driver_species_arg, str):
        names = [s.strip() for s in driver_species_arg.split(",") if s.strip()]
    else:
        names = list(driver_species_arg)
    missing = [n for n in names if n not in species_names]
    if missing:
        raise ValueError(f"Driver species are absent from the model species list: {missing}\nSpecies count={len(species_names)}")
    indices = [species_names.index(n) for n in names]
    return names, indices


def get_all_timepoint_initials(current_df, species_names, global_scale, group_col=None):
    initials = []
    group_vals = {}
    for sid, sub in current_df.groupby("SubjectID"):
        sub = sub.sort_values("Time").reset_index(drop=True)
        times = sub["Time"].values
        abund = sub[species_names].values.astype(np.float64)
        if group_col and group_col in sub.columns:
            gvals = sub[group_col].tolist()
        else:
            gvals = [None] * len(sub)
        diffs = np.array([], dtype=float)
        try:
            diffs = np.diff(np.asarray(times).astype(float))
            diffs = diffs[np.isfinite(diffs)]
        except Exception:
            pass
        dt = max(float(np.mean(diffs)) if len(diffs) > 0 else 1.0, 1.0)
        for t_idx in range(len(sub)):
            x0 = np.maximum(abund[t_idx] / global_scale, 0.0)
            t_val = None
            try:
                t_val = float(times[t_idx])
            except Exception:
                t_val = float(t_idx)
            initials.append({
                "SubjectID": str(sid), "time_pos": t_idx, "dt": dt,
                "x0": x0, "current_time": t_val, "time_raw": times[t_idx],
                "group": gvals[t_idx] if t_idx < len(gvals) else None,
            })
            group_vals[(str(sid), t_idx)] = gvals[t_idx] if t_idx < len(gvals) else None
    return initials, group_vals


def _plot_sample_curves(sid_dir, species_names, orig, controlled, target, times_raw, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ncols = 3
    nrows = int(math.ceil(len(species_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 2.5 * nrows), squeeze=False)
    x = np.arange(len(times_raw))
    for k, sp in enumerate(species_names):
        ax = axes[k // ncols][k % ncols]
        ax.plot(x, orig[:, k], "o-", color="#444", label="Original", lw=1, markersize=3)
        ax.plot(x, controlled[:, k], "s-", color="#b23b3b", label="Controlled", lw=1, markersize=3)
        ax.axhline(target[k], color="#2f7d32", ls="--", label="Target", lw=1)
        ax.set_xticks(x[:: max(1, len(x) // 6)])
        ax.set_xticklabels([f"{times_raw[i]:g}" if hasattr(times_raw[i], "__float__") else str(times_raw[i])
                            for i in ax.get_xticks().astype(int)], rotation=30, fontsize=6)
        ax.set_title(sp, fontsize=8)
    for a in axes.ravel():
        if not a.has_data():
            a.set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8, ncol=3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf_path = os.path.join(sid_dir, "species_abundance_before_after_intervention.pdf")
    ensure_dir(sid_dir)
    fig.savefig(pdf_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run_mpc(args):
    set_seed(getattr(args, "seed", 42))
    out_dir = getattr(args, "output_dir", "outputs/02_driver_mpc")
    ensure_dir(out_dir)

    if not getattr(args, "model", None) or not os.path.exists(args.model):
        raise FileNotFoundError("Provide a trained model via --model. Run MMLD prepare first if needed.")
    checkpoint = safe_torch_load(args.model)
    species_names = list(checkpoint["species_names"])
    N = len(species_names)
    A_base, growth_rate_base, theta_base, train_scale = extract_model_parameters(checkpoint)

    initial_file = getattr(args, "initial", None) or getattr(args, "abundance", None)
    if not initial_file or not os.path.exists(initial_file):
        raise FileNotFoundError("Provide an initial-state abundance table via --initial.")
    target_file = getattr(args, "target", None)
    if not target_file or not os.path.exists(target_file):
        raise FileNotFoundError("Provide a target-state abundance table via --target.")

    initial_df = read_abundance_with_group(initial_file, species_names)
    target_df = read_abundance_with_group(target_file, species_names)
    for col in species_names:
        if col not in initial_df.columns:
            raise ValueError(f"Initial-state data is missing species column: {col}")
        if col not in target_df.columns:
            raise ValueError(f"Target-state data is missing species column: {col}")

    group_col = _detect_label_column(initial_df, species_names)
    target_group_col = _detect_label_column(target_df, species_names)
    if group_col is None:
        raise ValueError("The initial-state abundance table must contain a Group/label/class column.")
    if target_group_col is None:
        raise ValueError("The target-state abundance table must contain a Group/label/class column.")

    global_scale = train_scale if train_scale is not None else compute_global_scale(initial_df, target_df, species_names)
    if train_scale is None:
        warnings.warn("The checkpoint has no SCALE value; it was recalculated.")

    target_map = build_target_map(target_df, species_names, global_scale)
    if not target_map:
        raise ValueError("The target-state table has no usable SubjectID or abundance records.")
    initials, group_map = get_all_timepoint_initials(initial_df, species_names, global_scale, group_col=group_col)

    if not getattr(args, "driver_species", None):
        raise ValueError("Provide comma-separated driver species via --driver-species.")
    driver_names, driver_indices = parse_driver_species(args.driver_species, species_names)

    tau = float(getattr(args, "tau", TAU_DEFAULT))
    steps = int(getattr(args, "steps", 1))
    q_weight = float(getattr(args, "q_weight", Q_WEIGHT_DEFAULT))
    r_weight = float(getattr(args, "r_weight", R_WEIGHT_DEFAULT))
    rew_p = float(getattr(args, "rewiring_probability", REWIRING_PROBABILITY_DEFAULT))
    eps_coeff = float(getattr(args, "epsilon_coefficient", EPSILON_COEFFICIENT_DEFAULT))
    if steps < 1:
        raise ValueError("--steps must be an integer greater than or equal to 1.")
    if tau <= 0:
        raise ValueError("--tau must be greater than 0.")
    if q_weight <= 0 or r_weight <= 0:
        raise ValueError("--q-weight and --r-weight must be greater than 0.")

    print(f"[MPC] Species={N}, driver_species={len(driver_indices)}, max_steps={steps}，tau={tau}")

    all_results = {}          # (sid, time_pos, step_k) -> state
    no_ctrl_results = {}      # (sid, time_pos, step_k) -> state
    pearson_table_rows = []

    for info in initials:
        sid = info["SubjectID"]
        tp = info["time_pos"]
        x0 = info["x0"]
        target_vec = target_map.get(sid)
        if target_vec is None:
            mean_t = np.mean(np.stack(list(target_map.values())), axis=0)
            target_map[sid] = target_vec = mean_t

        # no-control
        x_nc = x0.copy()
        nc_states = []
        for _ in range(steps):
            x_nc = get_final_state_no_control(x_nc, tau, A_base, growth_rate_base, theta_base)
            nc_states.append(x_nc.copy())

        epsilon = eps_coeff * np.linalg.norm(x0 - target_vec, np.inf)
        _, trajectory, controls = GetControllerSuccess(
            x0, target_vec, tau, steps, epsilon,
            A_base, growth_rate_base, theta_base, rew_p, driver_indices,
            q_weight=q_weight, r_weight=r_weight,
        )

        for k in range(1, steps + 1):
            ctrl_state = trajectory[k] if k < len(trajectory) else trajectory[-1]
            nc_state = nc_states[k - 1]
            pear_r_ctrl = pearsonr_safe(ctrl_state, target_vec)
            pear_r_nc = pearsonr_safe(nc_state, target_vec)
            all_results[(sid, tp, k)] = ctrl_state
            no_ctrl_results[(sid, tp, k)] = nc_state
            pearson_table_rows.append({
                "SubjectID": sid, "Start_TimePos": tp, "Start_Time_raw": info["time_raw"],
                "Step": k, "Time_after_control": info["current_time"] + k * tau,
                "Pearson_R_controlled": pear_r_ctrl, "Pearson_R_no_control": pear_r_nc,
                "Metrics_controlled": calc_metrics(ctrl_state, target_vec),
                "Metrics_no_control": calc_metrics(nc_state, target_vec),
                "Group": group_map.get((sid, tp)),
            })

    pear_df = pd.DataFrame(pearson_table_rows)
    per_sid_best_rows = []
    for sid, sub in pear_df.groupby("SubjectID"):
        if steps == 1:
            best_tp_row = _best_pearson_row(sub)
            if best_tp_row is not None:
                per_sid_best_rows.append(best_tp_row.to_dict())
        else:
            sub = sub.copy()
            sub["__best_step_in_tp"] = False
            tp_keys = []
            for (sid2, tp), s2 in sub.groupby(["SubjectID", "Start_TimePos"]):
                best = _best_pearson_row(s2)
                if best is not None:
                    sub.at[best.name, "__best_step_in_tp"] = True
                    tp_keys.append(best.name)
            tp_best = sub.loc[tp_keys]
            final = _best_pearson_row(tp_best)
            if final is not None:
                per_sid_best_rows.append(final.to_dict())

    per_sid_best_df = pd.DataFrame(per_sid_best_rows)

    wide_rows = []
    best_start_by_sid = {}
    if not per_sid_best_df.empty:
        best_start_by_sid = {
            str(row["SubjectID"]): int(row["Start_TimePos"])
            for _, row in per_sid_best_df.iterrows()
        }
    for info in initials:
        sid = info["SubjectID"]
        tp = info["time_pos"]
        selected_tp = best_start_by_sid.get(str(sid))
        if selected_tp == tp:
            if steps == 1:
                best_step = 1
            else:
                sel = per_sid_best_df[per_sid_best_df["SubjectID"] == sid]
                best_step = int(sel.iloc[0]["Step"]) if not sel.empty else None
        else:
            best_step = None
        row = {"SubjectID": sid, "Time": info["time_raw"]}
        x0 = info["x0"]
        if best_step is not None and (sid, tp, best_step) in all_results:
            ctrl = all_results[(sid, tp, best_step)]
        else:
            ctrl = x0.copy()
        target_vec = target_map.get(sid, np.zeros(N))
        for j, sp in enumerate(species_names):
            row[f"{sp}_original_abundance"] = float(x0[j]) * global_scale
            row[f"{sp}_controlled_abundance"] = float(ctrl[j]) * global_scale
            row[f"{sp}_target_abundance"] = float(target_vec[j]) * global_scale
        if group_col:
            row[group_col] = info["group"]
        wide_rows.append(row)

    wide_df = pd.DataFrame(wide_rows)
    col_order = ["SubjectID", "Time"]
    for sp in species_names:
        col_order.extend([f"{sp}_original_abundance", f"{sp}_controlled_abundance", f"{sp}_target_abundance"])
    if group_col:
        col_order.append(group_col)
    wide_df = wide_df[[c for c in col_order if c in wide_df.columns]]

    wide_path = os.path.join(out_dir, "MPC_abundance_results.xlsx")
    with pd.ExcelWriter(wide_path, engine="openpyxl") as writer:
        wide_df.to_excel(writer, sheet_name="abundance_wide", index=False)
    print(f"[MPC] Saved: {wide_path}")

    metric_detail_rows = []
    for r in pearson_table_rows:
        m = r["Metrics_controlled"].copy()
        m_nc = r["Metrics_no_control"]
        row_out = {
            "SubjectID": r["SubjectID"],
            "Start_Time_raw": r["Start_Time_raw"],
            "Step": r["Step"], "Time_after_control": r["Time_after_control"],
            "Pearson_R": r["Pearson_R_controlled"],
            **{f"Ctrl_{k}": m[k] for k in ["RMSE"]},
            **{f"NoCtrl_{k}": m_nc[k] for k in ["RMSE"]},
            "Pearson_R_no_control": r["Pearson_R_no_control"],
            "Group": r.get("Group"),
        }
        metric_detail_rows.append(row_out)
    metric_detail_df = pd.DataFrame(metric_detail_rows)
    metric_path = os.path.join(out_dir, "MPC_metrics.xlsx")
    with pd.ExcelWriter(metric_path, engine="openpyxl") as writer:
        metric_detail_df.to_excel(writer, sheet_name="subject_start_step", index=False)
        grp_cols = ["Step"] if steps > 1 else ["Start_Time_raw"]
        agg_cols = [c for c in metric_detail_df.columns if c.startswith("Ctrl_") or c.startswith("NoCtrl_") or c in ("Pearson_R", "Pearson_R_no_control")]
        try:
            metric_detail_df.groupby(grp_cols, as_index=False)[agg_cols].mean().to_excel(
                writer, sheet_name="group_means", index=False)
        except Exception:
            pass
    print(f"[MPC] Saved: {metric_path}")

    best_summary_path = os.path.join(out_dir, "MPC_best_intervention_by_subject.xlsx")
    with pd.ExcelWriter(best_summary_path, engine="openpyxl") as writer:
        best_export_drop = ["Start_TimePos", "Metrics_controlled", "Metrics_no_control"]
        per_sid_best_df.drop(columns=best_export_drop, errors="ignore").to_excel(
            writer, sheet_name="best_start", index=False)
        if steps > 1:
            tmp_rows = []
            for (sid, tp), s2 in pear_df.groupby(["SubjectID", "Start_TimePos"]):
                best = _best_pearson_row(s2)
                if best is not None:
                    tmp_rows.append(best.to_dict())
            tpd = pd.DataFrame(tmp_rows).drop(
                columns=best_export_drop, errors="ignore")
            tpd.to_excel(writer, sheet_name="best_step_by_time", index=False)
    print(f"[MPC] Saved: {best_summary_path}")

    if getattr(args, "no_plot", False) is False:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            pass
        else:
            sid_best_row = {row["SubjectID"]: row for _, row in per_sid_best_df.iterrows()}
            for sid, sub in initial_df.groupby("SubjectID"):
                sub = sub.sort_values("Time").reset_index(drop=True)
                abund = sub[species_names].values.astype(np.float64)
                times_raw = sub["Time"].values.tolist()
                T = len(sub)
                orig = abund.copy()
                ctrl = abund.copy()
                target_vec = target_map.get(sid, np.zeros(N)) * global_scale
                br = sid_best_row.get(sid)
                if br is not None:
                    tp_sel = int(br["Start_TimePos"])
                    step_sel = int(br["Step"])
                    key = (sid, tp_sel, step_sel)
                    if key in all_results:
                        ctrl_end = all_results[key] * global_scale
                        end_tp = min(T - 1, tp_sel + step_sel)
                        start_arr = abund[tp_sel].copy()
                        for t in range(tp_sel, min(end_tp + 1, T)):
                            alpha = (t - tp_sel) / max(step_sel, 1)
                            alpha = max(0.0, min(1.0, alpha))
                            ctrl[t] = start_arr * (1 - alpha) + ctrl_end * alpha
                sid_dir = os.path.join(out_dir, "subjects", str(sid))
                ensure_dir(sid_dir)
                _plot_sample_curves(sid_dir, species_names, orig, ctrl, target_vec, times_raw,
                                    title=f"Subject {sid}  before and after intervention")
                data_rows = []
                for t_idx in range(T):
                    row = {"Time": times_raw[t_idx]}
                    for j, sp in enumerate(species_names):
                        row[f"{sp}_original"] = float(orig[t_idx, j])
                        row[f"{sp}_controlled"] = float(ctrl[t_idx, j])
                        row[f"{sp}_target"] = float(target_vec[j])
                    data_rows.append(row)
                pd.DataFrame(data_rows).to_excel(
                    os.path.join(sid_dir, "plot_data.xlsx"), index=False)
    print(f"[MPC] Complete. Outputs: {out_dir}")


def main_mpc(args):
    run_mpc(args)
