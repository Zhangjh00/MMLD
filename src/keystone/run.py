"""English documentation."""
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from scipy.stats import f_oneway, ttest_ind
except Exception:
    f_oneway, ttest_ind = None, None


from ..common import (
    set_seed, ensure_dir,
    MicrobiomeDataLoader, train_glv_model, build_glv_model,
    bray_curtis,
)


# =========================================================
# =========================================================
class KnockoutExperiment:
    def __init__(self, model, loader, eps=1e-10, strict_knockout=False):
        self.model = model
        self.loader = loader
        self.eps = eps
        # Match the reference implementations: cross-sectional knockout is
        # maintained at zero at every simulated step, whereas longitudinal
        # knockout removes the species from the initial current/history states
        # only.
        self.strict_knockout = bool(strict_knockout)

    def get_last_state(self, sample_id):
        sample = self.loader.samples[sample_id]
        abundances = sample["abundances"]
        times = sample["times"]
        T = sample["T"]
        last_idx = T - 1
        h0 = torch.tensor(abundances[last_idx], dtype=torch.float32)
        history_init = np.zeros((self.loader.lag_max, self.loader.N))
        for lag in range(1, self.loader.lag_max + 1):
            idx = last_idx - lag
            history_init[lag - 1, :] = abundances[idx, :] if idx >= 0 else abundances[0, :]
        history_init = torch.tensor(history_init, dtype=torch.float32)
        avg_dt = float(np.mean(np.diff(times))) if T > 1 else 1.0
        return h0, history_init, avg_dt

    def _simulate_n_steps(self, h0, history_init, avg_dt, n_steps=100, knock_idx=None):
        self.model.eval()
        full_recent = [history_init[i].clone() for i in range(self.loader.lag_max)]
        h = h0.clone()
        if knock_idx is not None:
            h[knock_idx] = 0.0
        trajectory = [h.numpy().copy()]
        with torch.no_grad():
            for _ in range(n_steps):
                dt = torch.tensor(avg_dt, dtype=torch.float32)
                history_list = [
                    full_recent[lag - 1]
                    for lag in range(self.loader.lag_min, self.loader.lag_max + 1)
                ]
                history_tensor = torch.stack(history_list)
                h_next = self.model(h, history_tensor, dt)
                if knock_idx is not None:
                    h_next[knock_idx] = 0.0
                full_recent = [h.clone()] + full_recent[:-1]
                h = h_next
                trajectory.append(h_next.numpy().copy())
        return trajectory[-1], np.array(trajectory)

    def _normalize(self, x):
        x = np.asarray(x, dtype=float)
        x = np.maximum(x, self.eps)
        return x / (x.sum() + self.eps)

    def _subcomposition(self, wt, ko, knock_idx):
        mask = [i for i in range(len(wt)) if i != knock_idx]
        wt_sub = self._normalize(wt[mask])
        ko_sub = self._normalize(ko[mask])
        return wt_sub, ko_sub

    def subcomp_bc(self, wt, ko, knock_idx):
        wt_sub, ko_sub = self._subcomposition(wt, ko, knock_idx)
        return float(np.sum(np.abs(wt_sub - ko_sub)) / (np.sum(wt_sub + ko_sub) + self.eps))

    def _relative_abundance_matrix(self, states):
        states = np.asarray(states, dtype=float)
        states = np.maximum(states, self.eps)
        return states / (states.sum(axis=1, keepdims=True) + self.eps)

    def _clr_transform(self, states, pseudocount=5.0):
        states = np.asarray(states, dtype=float)
        raw_scale_states = np.maximum(states * self.loader.SCALE, 0.0)
        log_values = np.log(raw_scale_states + pseudocount)
        return log_values - log_values.mean(axis=1, keepdims=True)

    def _require_scipy(self):
        if f_oneway is None or ttest_ind is None:
            raise ImportError("The three-metric ANOVA analysis requires scipy.")

    def _secondary_impact_analysis(self, wt_finals, ko_finals, thresholds=None, min_threshold_hits=7):
        if thresholds is None:
            thresholds = np.arange(0.001, 0.0071, 0.0005)
        thresholds = np.asarray(thresholds, dtype=float)
        if min_threshold_hits <= 0 or min_threshold_hits > len(thresholds):
            raise ValueError("min_threshold_hits must be between 1 and len(thresholds)")
        wt_relative = self._relative_abundance_matrix(wt_finals)
        all_ko_states = [
            self._relative_abundance_matrix(ko_finals[idx])
            for idx in range(self.loader.N) if len(ko_finals[idx]) > 0
        ]
        overall_relative = np.vstack([wt_relative] + all_ko_states)
        rows = []
        for knock_idx in range(self.loader.N):
            if len(ko_finals[knock_idx]) == 0:
                continue
            ko_relative = self._relative_abundance_matrix(ko_finals[knock_idx])
            for target_idx in range(self.loader.N):
                if target_idx == knock_idx:
                    continue
                invasion_hits = 0
                extinction_hits = 0
                for threshold in thresholds:
                    full_occurrence = float(np.mean(wt_relative[:, target_idx] > threshold))
                    ko_occurrence = float(np.mean(ko_relative[:, target_idx] > threshold))
                    overall_occurrence = float(np.mean(overall_relative[:, target_idx] > threshold))

                    if 0.2 < overall_occurrence < 0.8:
                        ko_mean = ko_relative[:, target_idx].mean()
                        ko_std = ko_relative[:, target_idx].std(ddof=0)
                        overall_mean = overall_relative[:, target_idx].mean()
                        overall_std = overall_relative[:, target_idx].std(ddof=0)
                        is_extinct = bool(ko_mean + ko_std < overall_mean - overall_std)
                        is_invading = bool(ko_mean - ko_std > overall_mean + overall_std)
                    else:
                        occurrence_difference = ko_occurrence - full_occurrence
                        is_extinct = bool(occurrence_difference < -0.5)
                        is_invading = bool(occurrence_difference > 0.5)
                    extinction_hits += int(is_extinct)
                    invasion_hits += int(is_invading)
                rows.append({
                    "ko_species": self.loader.species_names[knock_idx],
                    "ko_idx": knock_idx,
                    "target_species": self.loader.species_names[target_idx],
                    "target_idx": target_idx,
                    "invasion_threshold_hits": invasion_hits,
                    "extinction_threshold_hits": extinction_hits,
                    "secondary_invasion": bool(invasion_hits >= min_threshold_hits),
                    "secondary_extinction": bool(extinction_hits >= min_threshold_hits),
                })
        details = pd.DataFrame(rows)
        summary = {}
        if details.empty:
            return details, summary
        details["secondary_impact"] = details["secondary_invasion"] | details["secondary_extinction"]
        for knock_idx in range(self.loader.N):
            subset = details[details["ko_idx"] == knock_idx]
            summary[knock_idx] = {
                "secondary_impact_count": int(subset["secondary_impact"].sum()),
                "secondary_invasion_count": int(subset["secondary_invasion"].sum()),
                "secondary_extinction_count": int(subset["secondary_extinction"].sum()),
            }
        return details, summary

    def _clr_anova_analysis(self, wt_finals, ko_finals, alpha=0.05, pseudocount=5.0):
        self._require_scipy()
        condition_states = {"Full": np.asarray(wt_finals, dtype=float)}
        for idx in range(self.loader.N):
            if len(ko_finals[idx]) > 0:
                condition_states[idx] = np.asarray(ko_finals[idx], dtype=float)
        clr_conditions = {
            cond: self._clr_transform(states, pseudocount=pseudocount)
            for cond, states in condition_states.items()
        }
        rows = []
        for target_idx in range(self.loader.N):
            groups = []
            for condition, clr_values in clr_conditions.items():
                if condition == target_idx:
                    continue
                if len(clr_values) >= 2:
                    groups.append(clr_values[:, target_idx])
            if len(groups) < 2:
                continue
            anova_result = f_oneway(*groups)
            anova_pvalue = float(anova_result.pvalue)
            anova_significant = bool(anova_pvalue < alpha)
            if "Full" not in clr_conditions:
                continue
            full_values = clr_conditions["Full"][:, target_idx]
            for knock_idx in range(self.loader.N):
                if knock_idx == target_idx or knock_idx not in clr_conditions:
                    continue
                ko_values = clr_conditions[knock_idx][:, target_idx]
                if len(full_values) < 2 or len(ko_values) < 2:
                    continue
                full_pvalue = float(ttest_ind(
                    ko_values, full_values, equal_var=True, nan_policy="omit").pvalue)
                other_significant = 0
                for other_idx in range(self.loader.N):
                    if other_idx in (knock_idx, target_idx):
                        continue
                    if other_idx not in clr_conditions:
                        continue
                    other_values = clr_conditions[other_idx][:, target_idx]
                    if len(other_values) < 2:
                        continue
                    o_p = float(ttest_ind(ko_values, other_values, equal_var=True,
                                          nan_policy="omit").pvalue)
                    other_significant += int(o_p < alpha)
                rows.append({
                    "ko_species": self.loader.species_names[knock_idx],
                    "ko_idx": knock_idx,
                    "target_species": self.loader.species_names[target_idx],
                    "target_idx": target_idx,
                    "anova_statistic": float(anova_result.statistic),
                    "anova_pvalue": anova_pvalue,
                    "anova_significant": anova_significant,
                    "full_mean_clr": float(np.mean(full_values)),
                    "eko_mean_clr": float(np.mean(ko_values)),
                    "eko_vs_full_pvalue": full_pvalue,
                    "other_communities_significant": int(other_significant),
                    "anova_impact": bool(
                        anova_significant and full_pvalue < alpha and other_significant >= 3
                    ),
                })
        details = pd.DataFrame(rows)
        summary = {}
        if details.empty:
            return details, summary
        for knock_idx in range(self.loader.N):
            subset = details[details["ko_idx"] == knock_idx]
            summary[knock_idx] = {
                "anova_impact_count": int(subset["anova_impact"].sum()),
            }
        return details, summary

    def run_knockout_all_samples(self, n_steps=100,
                                  secondary_thresholds=None, secondary_min_hits=7,
                                  anova_alpha=0.05, clr_pseudocount=5.0):
        metrics_accum = {idx: {"bc": [], "abundance": []} for idx in range(self.loader.N)}
        wt_finals = []
        ko_finals = {idx: [] for idx in range(self.loader.N)}
        for sample_id in list(self.loader.samples.keys()):
            h0, history_init, avg_dt = self.get_last_state(sample_id)
            wt_final, _ = self._simulate_n_steps(h0, history_init, avg_dt, n_steps=n_steps)
            if np.isnan(wt_final).any() or float(wt_final.sum()) <= 0:
                continue
            wt_finals.append(wt_final)
            for idx in range(self.loader.N):
                h0_ko = h0.clone()
                h0_ko[idx] = 0.0
                history_ko = history_init.clone()
                history_ko[:, idx] = 0.0
                ko_kwargs = {"knock_idx": idx} if self.strict_knockout else {}
                ko_final, _ = self._simulate_n_steps(
                    h0_ko, history_ko, avg_dt, n_steps=n_steps,
                    **ko_kwargs)
                if np.isnan(ko_final).any():
                    metrics_accum[idx]["bc"].append(1.0)
                else:
                    metrics_accum[idx]["bc"].append(
                        self.subcomp_bc(wt_final, ko_final, idx))
                    ko_finals[idx].append(ko_final)
                metrics_accum[idx]["abundance"].append(float(h0[idx].item()))
        if not wt_finals:
            raise RuntimeError("Keystone knockout produced no valid wild-type final state. Check the data format and lag_max.")

        sec_details, sec_summary = self._secondary_impact_analysis(
            wt_finals, ko_finals, thresholds=secondary_thresholds,
            min_threshold_hits=secondary_min_hits,
        )
        anova_details, anova_summary = ({}, {})
        try:
            anova_details, anova_summary = self._clr_anova_analysis(
                wt_finals, ko_finals, alpha=anova_alpha, pseudocount=clr_pseudocount)
        except Exception as e:
            warnings.warn(f"Skipping CLR-ANOVA: {e}")

        results = []
        for idx in range(self.loader.N):
            valid_count = len(metrics_accum[idx]["bc"])
            if valid_count == 0:
                continue
            results.append({
                "species": self.loader.species_names[idx],
                "idx": int(idx),
                "abundance_scaled_mean": float(np.mean(metrics_accum[idx]["abundance"])),
                "bc_impact": float(np.mean(metrics_accum[idx]["bc"])),
                "valid_samples": int(valid_count),
                **sec_summary.get(idx, {}),
                **anova_summary.get(idx, {}),
            })
        results.sort(key=lambda x: x["bc_impact"], reverse=True)
        self.knockout_summary = pd.DataFrame(results)
        self.secondary_impact_details = sec_details
        self.clr_anova_details = anova_details
        return results


# =========================================================
# =========================================================
def _build_steady_state_pairs(loader):
    """English documentation."""
    pairs = []
    lag_min, lag_max = loader.lag_min, loader.lag_max
    N = loader.N
    for sid, sample in loader.samples.items():
        h = torch.tensor(sample["abundances"][0], dtype=torch.float32)
        history = torch.stack([h.clone() for _ in range(lag_max - lag_min + 1)])
        pairs.append({
            "current": h,
            "history": history,
            "dt": torch.tensor(1.0, dtype=torch.float32),
            "target": h.clone(),
        })
    return pairs


def _build_steady_state_extended_pairs(loader, n_steps_per_subject=5):
    """English documentation."""
    from ..common.model import build_history_from_abundances
    pairs = []
    lag_min, lag_max = loader.lag_min, loader.lag_max
    for sid, sample in loader.samples.items():
        h0_arr = sample["abundances"][0]
        T_pseudo = max(lag_max + 1, n_steps_per_subject + lag_max + 1)
        pseudo_abund = np.stack([h0_arr] * T_pseudo)
        pseudo_times = np.arange(1, T_pseudo + 1, dtype=float)
        for t in range(lag_max, T_pseudo - 1):
            hist = build_history_from_abundances(pseudo_abund, t, lag_min, lag_max)
            cur = torch.tensor(pseudo_abund[t], dtype=torch.float32)
            tar = torch.tensor(pseudo_abund[t + 1], dtype=torch.float32)
            dt = torch.tensor(float(pseudo_times[t + 1] - pseudo_times[t]), dtype=torch.float32)
            pairs.append({"current": cur, "history": hist, "target": tar, "dt": dt})
    return pairs


# =========================================================
# =========================================================
def main_keystone(args):
    set_seed(getattr(args, "seed", 42))
    out_dir = getattr(args, "output_dir", "outputs/04_keystone")
    ensure_dir(out_dir)
    device = getattr(args, "device", "cpu")
    glv_method = str(getattr(args, "glv_method", "euler")).strip().lower()

    data_type = getattr(args, "data_type", "auto").lower()

    loader = MicrobiomeDataLoader(
        args.abundance, args.graph,
        group_is_present=(None if not getattr(args, "no_group", False) else False),
    )
    loader.load_abundance()
    loader.load_graph()

    def _is_cross_sectional(load):
        return all(v["T"] <= 1 for v in load.samples.values())
    if data_type == "auto":
        data_type = "cross_section" if _is_cross_sectional(loader) else "time_series"
    print(f"[keystone] data_type={data_type}")

    if data_type == "cross_section":
        print("[keystone cross_section] Using steady-state fixed-point pairs, with each T=1 subject treated as a fixed point.")

    max_epochs = int(getattr(args, "max_epochs", 10))
    lr = float(getattr(args, "lr", 0.01))

    if data_type == "cross_section":
        train_pairs = _build_steady_state_extended_pairs(loader, n_steps_per_subject=5)
        if len(train_pairs) == 0:
            raise RuntimeError("Failed to build cross-sectional training pairs. Verify that each subject has T=1.")
        loader.training_pairs = train_pairs
    else:
        loader.prepare_training_data()

    model = build_glv_model(glv_method, loader.N, loader.lag_min, loader.lag_max, loader.edge_info)

    final_ranks = []
    dataset_name = os.path.basename(args.abundance)

    train_glv_model(
        model,
        loader.training_pairs,
        epochs=max_epochs,
        lr=lr,
        device=device,
        verbose=True,
    )

    exp = KnockoutExperiment(
        model,
        loader,
        strict_knockout=(data_type == "cross_section"),
    )
    results_summary_list = exp.run_knockout_all_samples(
        n_steps=100,
        secondary_thresholds=None,
        secondary_min_hits=7,
        anova_alpha=float(getattr(args, "anova_alpha", 0.05)),
        clr_pseudocount=5.0,
    )

    # Rank knockout metrics only once, after the final training epoch.
    bc_arr = np.zeros(loader.N, dtype=float) + np.nan
    ext_cnt = np.zeros(loader.N, dtype=float) + np.nan
    inv_cnt = np.zeros(loader.N, dtype=float) + np.nan
    anova_cnt = np.zeros(loader.N, dtype=float) + np.nan
    for result in results_summary_list:
        bc_arr[result["idx"]] = result["bc_impact"]
        ext_cnt[result["idx"]] = result.get("secondary_extinction_count", 0)
        inv_cnt[result["idx"]] = result.get("secondary_invasion_count", 0)
        anova_cnt[result["idx"]] = result.get("anova_impact_count", 0)

    def _rank_desc_ignore_nan(values):
        mask_ok = np.isfinite(values)
        ranks = np.full(len(values), np.nan)
        order = np.argsort(-values[mask_ok], kind="stable")
        local_ranks = np.arange(1, int(np.sum(mask_ok)) + 1, dtype=int)
        orig_idx = np.where(mask_ok)[0][order]
        ranks[orig_idx] = local_ranks
        return ranks

    # The final ranking is based exclusively on the mean subcomposition
    # Bray–Curtis impact. Secondary-effect and ANOVA values are retained as
    # descriptive columns, but are not combined into another ranking.
    rk_bc = _rank_desc_ignore_nan(bc_arr)
    sec_combined = np.nansum(np.stack([ext_cnt, inv_cnt]), axis=0)

    for idx, sp in enumerate(loader.species_names):
        final_ranks.append({
            "dataset": dataset_name,
            "species": sp,
            "species_idx": int(idx),
            "rank_BrayCurtis": rk_bc[idx] if np.isfinite(rk_bc[idx]) else None,
            "BrayCurtis_mean": None if not np.isfinite(bc_arr[idx]) else float(bc_arr[idx]),
            "Secondary_Impact_Total": None if not np.isfinite(sec_combined[idx]) else float(sec_combined[idx]),
            "Secondary_Extinction": None if not np.isfinite(ext_cnt[idx]) else float(ext_cnt[idx]),
            "Secondary_Invasion": None if not np.isfinite(inv_cnt[idx]) else float(inv_cnt[idx]),
            "ANOVA_Impact_Count": None if not np.isfinite(anova_cnt[idx]) else float(anova_cnt[idx]),
        })

    final_df = pd.DataFrame(final_ranks).sort_values("rank_BrayCurtis", na_position="last")
    ranking_path = os.path.join(out_dir, "keystone_knockout_ranking.xlsx")
    with pd.ExcelWriter(ranking_path, engine="openpyxl") as writer:
        final_df.to_excel(writer, sheet_name="final_ranking", index=False)
    print(f"[keystone] Saved: {ranking_path}")

    model_path = os.path.join(out_dir, "MMLD_model.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "N": loader.N, "lag_min": loader.lag_min, "lag_max": loader.lag_max,
        "edge_info": loader.edge_info, "species_names": loader.species_names,
        "SCALE": loader.SCALE, "data_type": data_type,
        "model_type": glv_method,
    }, model_path)
    print(f"[keystone] Saved: {model_path}")
    print(f"[keystone] Complete. Outputs: {out_dir}")
    return True
