import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from ..common import ensure_dir, detect_group_column


def _generate_julia_code(N, rows, cols, vals, idx_to_strain, module_path, max_sets=10000):
    strain_map_lines = []
    for k, v in idx_to_strain.items():
        strain_map_lines.append(f"    {k} => \"{v}\",")
    strain_map_block = "\n".join(strain_map_lines)
    rows_str = "[" + ", ".join(str(int(r)) for r in rows) + "]"
    cols_str = "[" + ", ".join(str(int(c)) for c in cols) + "]"
    vals_str = "[" + ", ".join(str(float(v)) for v in vals) + "]"

    script = f"""using SparseArrays
include(raw\"{module_path}\")
using .DriverSpeciesModule

N = {N}
strain_mapping = Dict(
{strain_map_block}
)
A = sparse(Int64{rows_str}, Int64{cols_str}, Float64{vals_str}, {N}, {N})

driver_species = DriverSpecies(A)
all_driver_sets = AllDriverSpecies(A; max_sets={int(max_sets)})
println("DRIVER_SPECIES:")
for idx in sort(driver_species)
    println("$(idx),$(strain_mapping[idx])")
end
println("COUNT:", length(driver_species))
println("ALL_DRIVER_SETS:")
for (set_id, ds) in enumerate(all_driver_sets)
    println("SET_BEGIN,$set_id,$(length(ds))")
    for idx in sort(ds)
        println("$set_id,$idx,$(strain_mapping[idx])")
    end
    println("SET_END,$set_id")
end
println("SET_COUNT:", length(all_driver_sets))
"""
    return script


def _parse_julia_output(stdout_lines):
    driver_species = []
    all_driver_sets = []
    parsing_single = False
    current_set = None

    for raw in stdout_lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("DRIVER_SPECIES:"):
            parsing_single = True
            continue
        if line.startswith("COUNT:"):
            parsing_single = False
            continue
        if line.startswith("ALL_DRIVER_SETS:"):
            parsing_single = False
            continue
        if line.startswith("SET_COUNT:"):
            continue
        if line.startswith("SET_BEGIN"):
            parts = line.split(",")
            current_set = {"set_id": int(parts[1]), "set_size": int(parts[2]), "members": []}
            continue
        if line.startswith("SET_END"):
            if current_set is not None:
                all_driver_sets.append(current_set)
                current_set = None
            continue
        if current_set is not None and "," in line:
            set_id, idx, name = line.split(",", 2)
            current_set["members"].append((int(idx), name))
        elif parsing_single and "," in line:
            idx, name = line.split(",", 1)
            driver_species.append((int(idx), name))
    return driver_species, all_driver_sets


def _read_driver_input(graph_path, abundance_path):
    graph_book = pd.ExcelFile(graph_path)
    if "edge_list" in graph_book.sheet_names:
        graph_sheet = "edge_list"
    elif "Network_Edges" in graph_book.sheet_names:
        graph_sheet = "Network_Edges"
    else:
        graph_sheet = graph_book.sheet_names[0]
    graph_df = pd.read_excel(graph_path, sheet_name=graph_sheet)
    abundance_df = pd.read_excel(abundance_path)
    required_graph_columns = {"source", "target", "test_statistic"}
    if not required_graph_columns.issubset(graph_df.columns):
        raise ValueError(
            "The causal graph is missing required columns: "
            f"{sorted(required_graph_columns - set(graph_df.columns))}"
        )

    group_col = detect_group_column(abundance_df)
    excluded_columns = {
        "SubjectID", "Time", "Time_order", "SampleID", "subject", "time",
    }
    if group_col is not None:
        excluded_columns.add(group_col)
    species_names = sorted(
        column for column in abundance_df.columns
        if column not in excluded_columns and not str(column).startswith("Unnamed")
    )
    strain_to_idx = {name: index + 1 for index, name in enumerate(species_names)}
    rows, cols, vals = [], [], []
    for _, row in graph_df.iterrows():
        src = row["source"]
        tgt = row["target"]
        if src == tgt:
            continue
        if src not in strain_to_idx or tgt not in strain_to_idx:
            continue
        rows.append(strain_to_idx[src])
        cols.append(strain_to_idx[tgt])
        vals.append(float(row["test_statistic"]))
    N = len(species_names)
    idx_to_strain = {v: k for k, v in strain_to_idx.items()}
    return N, rows, cols, vals, idx_to_strain


def main_driver(args):
    out_dir = getattr(args, "output_dir", "outputs/02_driver_mpc")
    ensure_dir(out_dir)

    if not getattr(args, "graph", None) or not os.path.exists(args.graph):
        raise FileNotFoundError("Provide a causal graph xlsx via --graph.")
    if not getattr(args, "abundance", None) or not os.path.exists(args.abundance):
        raise FileNotFoundError("Provide an abundance xlsx via --abundance.")

    N, rows, cols, vals, idx_to_strain = _read_driver_input(args.graph, args.abundance)
    if N == 0:
        raise ValueError("No species columns were found in the abundance table.")
    max_sets = int(getattr(args, "max_sets", 10000))

    repo_root = Path(__file__).resolve().parents[2]
    pinned_julia_project = repo_root / "julia_env"
    julia_project = pinned_julia_project
    julia_bin = shutil.which("julia")
    if julia_bin is None:
        julia_bin = str(Path.home() / ".juliaup" / "bin" / "julia")
    if not Path(julia_bin).is_file() or not os.access(julia_bin, os.X_OK):
        raise RuntimeError(
            "Julia was not found. Install Julia 1.12.6 and run "
            "scripts/install_fresh_conda_env.sh before MMLD driver."
        )
    if not (julia_project / "Project.toml").is_file() or not (julia_project / "Manifest.toml").is_file():
        raise RuntimeError(f"Pinned Julia environment is incomplete: {julia_project}")

    version_check = subprocess.run(
        [julia_bin, "--version"], capture_output=True, text=True, timeout=30,
    )
    if version_check.returncode != 0:
        raise RuntimeError(
            "MMLD driver could not run the installed Julia executable. "
            f"Detected: {version_check.stdout.strip() or version_check.stderr.strip()}"
        )

    print(f"[driver] {version_check.stdout.strip()}; using the exact mFDIC algorithm.")
    module_path = (julia_project / "DriverSpeciesModule.jl").as_posix()
    if not os.path.isfile(module_path):
        raise RuntimeError(f"Julia driver module is missing: {module_path}")
    julia_code = _generate_julia_code(
        N, rows, cols, vals, idx_to_strain, module_path, max_sets=max_sets,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".jl", delete=False) as f:
        f.write(julia_code)
        temp_path = f.name
    try:
        proc = subprocess.run(
            [julia_bin, f"--project={julia_project}", temp_path],
            capture_output=True, text=True, timeout=3600,
            cwd=str(repo_root),
        )
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass
    if proc.returncode != 0:
        raise RuntimeError(
            "Julia driver execution failed. Run the fresh installer to instantiate "
            f"the pinned packages. Details: {proc.stderr[:1200]}"
        )
    one, sets = _parse_julia_output(proc.stdout.splitlines())

    one_df = pd.DataFrame({
        "index": [idx for idx, _ in one],
        "species": [name for _, name in one],
    })
    summary_rows = []
    long_rows = []
    for item in sets:
        ids = sorted([idx for idx, _ in item["members"]])
        names = [name for _, name in sorted(item["members"], key=lambda x: x[0])]
        summary_rows.append({
            "set_id": item["set_id"],
            "set_size": item["set_size"],
            "species_indices": "; ".join(map(str, ids)),
            "species_names": "; ".join(names),
        })
        for idx, name in sorted(item["members"], key=lambda x: x[0]):
            long_rows.append({
                "set_id": item["set_id"],
                "set_size": item["set_size"],
                "index": idx,
                "species": name,
            })
    summary_df = pd.DataFrame(summary_rows)
    long_df = pd.DataFrame(long_rows)
    out_path = os.path.join(out_dir, "minimum_driver_species_sets.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        one_df.to_excel(writer, sheet_name="one_solution", index=False)
        summary_df.to_excel(writer, sheet_name="all_sets_summary", index=False)
        long_df.to_excel(writer, sheet_name="all_sets_long", index=False)
    print(f"[driver] Saved: {out_path} (one_solution, all_sets_summary, all_sets_long; {len(summary_df)} sets)")
    if one:
        print(f"[driver] One driver-species solution: {[n for _, n in one]}")
    return out_path
