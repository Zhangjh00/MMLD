"""English documentation."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .utils import detect_group_column, parse_time_col


class MicrobiomeDataLoader:
    """English documentation."""

    def __init__(self, abundance_file: str, pcmci_file: str = None,
                 group_is_present: bool = None, sheet_name=0):
        self.abundance_file = abundance_file
        self.pcmci_file = pcmci_file
        self.group_is_present = group_is_present
        self.sheet_name = sheet_name

        self.species_names = None
        self.N = None
        self.samples = {}
        self.training_pairs = []
        self.lag_min = 1
        self.lag_max = 1
        self.num_lags = 1
        self.species_to_idx = {}
        self.edge_info = []
        self.SCALE = 1.0
        self.group_col = None
        self.group_classes = None

    def load_abundance(self):
        df = pd.read_excel(self.abundance_file, sheet_name=self.sheet_name)
        self.samples = {}
        self.group_classes = None

        cols = list(df.columns)
        id_col = None
        for c in cols:
            if str(c).lower() in ("subjectid", "sampleid", "subject_id", "patientid"):
                id_col = c
                break
        if id_col is None:
            id_col = cols[0]
            warnings.warn(f"SubjectID column not found; using the first column '{id_col}' as SubjectID")

        time_col = None
        for c in cols:
            if c != id_col and str(c).lower() in ("time", "timepoint", "time_point", "day"):
                time_col = c
                break
        if time_col is None:
            time_col = cols[1]
            warnings.warn(f"Time column not found; using the second column '{time_col}' as Time")

        df = df.rename(columns={id_col: "SubjectID", time_col: "Time"})

        df["Time"] = parse_time_col(df["Time"].values)

        group_col = detect_group_column(df)
        if self.group_is_present is True and group_col is None:
            warnings.warn("A Group column was requested but not detected; using the last column")
            group_col = df.columns[-1]
        if self.group_is_present is False:
            group_col = None
        self.group_col = group_col

        exclude = {"SubjectID", "Time"}
        if group_col:
            exclude.add(group_col)
        species_cols = [c for c in df.columns if c not in exclude]

        for col in species_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        non_zero_species = [sp for sp in species_cols if df[sp].sum() > 0]
        self.species_names = non_zero_species
        self.N = len(non_zero_species)
        self.species_to_idx = {name: i for i, name in enumerate(self.species_names)}

        all_totals = []
        for sid in df["SubjectID"].unique():
            raw = df[df["SubjectID"] == sid].sort_values("Time")[non_zero_species].values
            all_totals.extend(raw.sum(axis=1))
        valid_totals = np.asarray(all_totals, dtype=float)
        valid_totals = valid_totals[np.isfinite(valid_totals) & (valid_totals > 0)]
        median_total = float(np.median(valid_totals)) if valid_totals.size else 0.0
        self.SCALE = median_total if median_total > 0 else 1.0
        print(f"[loader] Species N={self.N}  Subjects={df['SubjectID'].nunique()}  SCALE={self.SCALE:.1e}  Group={group_col}")

        for sid in df["SubjectID"].unique():
            sample_df = df[df["SubjectID"] == sid].sort_values("Time")
            times = sample_df["Time"].values.astype(float)
            raw = sample_df[non_zero_species].values
            abundances = np.maximum(raw.astype(float), 1e-10) / self.SCALE
            entry = {"times": times, "abundances": abundances, "T": len(times)}
            if group_col:
                entry["groups"] = sample_df[group_col].astype(str).tolist()
                if self.group_classes is None:
                    self.group_classes = sorted(set(entry["groups"]))
                else:
                    self.group_classes = sorted(set(self.group_classes) | set(entry["groups"]))
            self.samples[sid] = entry
        return self.samples

    def rescale(self, new_scale: float):
        """Rescale loaded abundance arrays without changing their raw values."""
        new_scale = float(new_scale)
        if not np.isfinite(new_scale) or new_scale <= 0:
            raise ValueError("The abundance scaling factor must be a positive finite number")
        old_scale = float(self.SCALE)
        if np.isclose(old_scale, new_scale):
            self.SCALE = new_scale
            return
        factor = old_scale / new_scale
        for sample in self.samples.values():
            sample["abundances"] = sample["abundances"] * factor
        self.SCALE = new_scale

    def scaling_factor(self, sid_subset=None):
        """Return the median sample-total abundance for the selected subjects."""
        sids = list(self.samples) if sid_subset is None else list(sid_subset)
        totals = []
        for sid in sids:
            sample = self.samples.get(sid)
            if sample is None:
                continue
            raw = np.asarray(sample["abundances"], dtype=float) * float(self.SCALE)
            totals.extend(raw.sum(axis=1).tolist())
        totals = np.asarray(totals, dtype=float)
        totals = totals[np.isfinite(totals) & (totals > 0)]
        return float(np.median(totals)) if totals.size else 1.0

    def load_graph(self, p_threshold: float = 0.05, remove_self_edges: bool = True,
                   graph_file: str = None):
        gfile = graph_file or self.pcmci_file
        if gfile is None:
            raise ValueError("No causal graph file was provided")

        xls = pd.ExcelFile(gfile)
        sheet = "edge_list" if "edge_list" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(gfile, sheet_name=sheet)

        required = {"source", "target"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Causal graph is missing required columns: {missing}")

        if "lag" not in df.columns:
            df["lag"] = 1
        if "p_value" not in df.columns:
            df["p_value"] = 0.05
        if "test_statistic" not in df.columns:
            df["test_statistic"] = 0.0

        df["lag"] = pd.to_numeric(df["lag"], errors="coerce").fillna(1).astype(int)
        df.loc[df["lag"] < 1, "lag"] = 1

        df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce").fillna(0.05)
        df["test_statistic"] = pd.to_numeric(df["test_statistic"], errors="coerce").fillna(0.0)

        self.edge_info = []

        df_sig = df[df["p_value"] <= p_threshold].copy()
        if remove_self_edges:
            df_sig = df_sig[df_sig["source"] != df_sig["target"]]

        if len(df_sig) == 0:
            warnings.warn("No significant edges passed the threshold; using lag_min=1 and lag_max=1")
            self.lag_min = 1
            self.lag_max = 1
            self.num_lags = 1
            return self.edge_info

        self.lag_min = int(df_sig["lag"].min())
        self.lag_max = int(df_sig["lag"].max())
        self.num_lags = self.lag_max - self.lag_min + 1

        for _, row in df_sig.iterrows():
            source, target, lag = row["source"], row["target"], int(row["lag"])
            if source not in self.species_to_idx or target not in self.species_to_idx:
                warnings.warn(f"Skipping edge {source}->{target}: species is absent from the abundance table")
                continue
            sign = 1.0 if row["test_statistic"] >= 0 else -1.0
            self.edge_info.append({
                "source_idx": self.species_to_idx[source],
                "target_idx": self.species_to_idx[target],
                "lag": lag,
                "sign": sign,
                "lag_idx": lag - self.lag_min,
                "p_value": float(row["p_value"]),
                "raw_stat": float(row["test_statistic"]),
            })
        print(f"[loader] Lag range {self.lag_min}-{self.lag_max}; valid edges {len(self.edge_info)}")
        return self.edge_info

    def prepare_training_data(self, sid_subset=None):
        self.training_pairs = []
        sids = list(self.samples.keys()) if sid_subset is None else list(sid_subset)
        for sid in sids:
            if sid not in self.samples:
                continue
            sample = self.samples[sid]
            times = sample["times"]
            abundances = sample["abundances"]
            T = sample["T"]
            for t in range(self.lag_max, T - 1):
                history = np.zeros((self.num_lags, self.N), dtype=np.float32)
                for k, lag in enumerate(range(self.lag_min, self.lag_max + 1)):
                    idx = t - lag
                    history[k, :] = abundances[idx, :] if idx >= 0 else abundances[0, :]
                self.training_pairs.append({
                    "sample_id": sid,
                    "time_index": t,
                    "current": torch.tensor(abundances[t, :], dtype=torch.float32),
                    "history": torch.tensor(history, dtype=torch.float32),
                    "target": torch.tensor(abundances[t + 1, :], dtype=torch.float32),
                    "dt": float(times[t + 1] - times[t]),
                })
        print(f"[loader] Built training pairs: {len(self.training_pairs)}")
        return self.training_pairs

    def raw_graph_df(self, graph_file: str = None):
        gfile = graph_file or self.pcmci_file
        return pd.read_excel(gfile)

    def get_groups_for_sample(self, sid):
        s = self.samples.get(sid)
        if s is None:
            return None
        return s.get("groups", None)
