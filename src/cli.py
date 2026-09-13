#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""English documentation."""
import argparse
import os
import sys
from pathlib import Path


def _add_common_args(p: argparse.ArgumentParser, with_graph=True, with_model=False,
                     need_abundance=True, graph_required=False, model_required=False):
    if need_abundance:
        p.add_argument("-a", "--abundance", type=str, required=True,
                       help="Abundance workbook")
    if with_graph:
        p.add_argument("-g", "--graph", type=str,
                       required=graph_required,
                       help="Causal-graph workbook")
    if with_model:
        p.add_argument("-m", "--model", type=str,
                       required=model_required,
                       help="MMLD model checkpoint")
    p.add_argument("-o", "--output-dir", type=str, default=None, help="Output directory")
    p.add_argument("--no-group", action="store_true",
                   help="Declare that the data has no Group/label/class column")
    p.add_argument("--glv-method", type=str, default="euler", choices=["euler", "rk4"], help="gLV solver: Euler or RK4")
    p.add_argument("--seed", type=int, default=42, help="Random seed")


def _mode_type(s: str) -> str:
    """English documentation."""
    import re
    if s in ("sequence", "match"):
        return s
    if re.fullmatch(r"step[1-9]\d*", s):
        return s
    raise argparse.ArgumentTypeError(
        f"Invalid mode='{s}'; supported values: sequence / match / stepN (for example, step1 or step4)")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="MMLD",
        description="Microbial community longitudinal analysis toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ----- prepare -----
    p1 = sub.add_parser("prepare", help="Build a causal graph and train a MMLD model")
    _add_common_args(p1, with_graph=True, with_model=False, need_abundance=True)
    p1.add_argument("--alg", type=str, default="pcmci",
                    choices=["pcmci", "granger", "cmlp", "clstm", "lasso_dbn", "lingam", "pc"],
                    help="Causal discovery method; an existing --graph file is loaded directly")
    p1.add_argument("--tau-min", type=int, default=1,
                    help="Minimum lag used for PCMCI")
    p1.add_argument("--tau-max", type=int, default=2,
                    help="Maximum lag used for PCMCI")
    p1.add_argument("--alpha", type=float, default=0.05,
                    help="Shared alpha level for conditional independence tests and edge filtering")
    p1.add_argument("--ci-test", type=str, default="ParCorr",
                    choices=["ParCorr", "RobustParCorr", "GPDC", "CMIknn"],
                    help="Conditional-independence test used by PCMCI")
    p1.add_argument("--epochs", type=int, default=10, help="Number of MMLD training epochs")
    p1.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    p1.add_argument(
        "--mpc-initial", type=str, default=None,
        help="Optional MPC initial-state workbook to archive with prepare outputs",
    )
    p1.add_argument(
        "--mpc-target", type=str, default=None,
        help="Optional MPC target-state workbook to archive with prepare outputs",
    )
    p1.set_defaults(func=cmd_prepare)

    # ----- predict -----
    p2 = sub.add_parser("predict", help="Predict future abundance")
    _add_common_args(
        p2, with_graph=True, with_model=True, need_abundance=True,
        graph_required=True,
    )
    p2.add_argument("--split", type=str, default="82",
                    choices=["82", "73", "5fold", "loso", "custom"],
                    help="Subject-level split: 82, 73, 5fold, loso, or custom")
    p2.add_argument("--train-ratio", type=float, default=0.8, help="Training ratio for split=82")
    p2.add_argument("--n-splits", type=int, default=5, help="Number of folds for split=5fold")
    p2.add_argument("--train-sids", type=str, default=None,
                    help="Training SubjectID xlsx path for split=custom")
    p2.add_argument("--test-sids", type=str, default=None,
                    help="Test SubjectID xlsx path for split=custom")
    p2.add_argument("--mode", type=_mode_type, default="sequence",
                    help="Prediction mode: sequence, match, or stepN (for example, step1 or step4)")
    p2.add_argument("--initial-ratio", type=float, default=0.15, help="Fraction of real test time points used for initialization")
    p2.add_argument("--top-percent", type=float, default=0.01, help="Top percent of similar training subjects for mode=match")
    p2.add_argument("--epochs", type=int, default=10, help="MMLD training epochs when no model is supplied")
    p2.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    p2.set_defaults(func=cmd_predict)

    # ----- driver -----
    p3 = sub.add_parser("driver", help="Identify minimum driver species sets")
    _add_common_args(
        p3, with_graph=True, with_model=False, need_abundance=True,
        graph_required=True,
    )
    p3.add_argument("--max-sets", type=int, default=10000, help="Maximum number of driver sets to enumerate")
    p3.set_defaults(func=cmd_driver)

    # ----- mpc -----
    p4 = sub.add_parser("mpc", help="MPC intervention with LQR")
    _add_common_args(
        p4, with_graph=True, with_model=True, need_abundance=False,
        graph_required=True, model_required=True,
    )
    p4.add_argument("--initial", type=str, required=True, help="Initial-state abundance xlsx with a final Group/label/class column")
    p4.add_argument("-t", "--target", type=str, required=True, help="Target-state abundance xlsx with a final Group/label/class column")
    p4.add_argument("--driver-species", type=str, required=True, help="Comma-separated driver species names")
    p4.add_argument("--steps", type=int, default=1, help="Intervention steps: 1 for one step or N for multiple steps")
    p4.add_argument("--q-weight", type=float, default=2e4, help="LQR Q weight")
    p4.add_argument("--r-weight", type=float, default=1.5e-1, help="LQR R weight")
    p4.add_argument("--tau", type=float, default=1.0, help="Intervention duration tau")
    p4.set_defaults(func=cmd_mpc)

    # ----- classify -----
    p5 = sub.add_parser("classify", help="Classify MMLD one-step abundance predictions")
    p5.add_argument("-a", "--abundance", type=str, required=True,
                    help="Prepared input_abundance.xlsx")
    p5.add_argument("-g", "--graph", type=str, required=True,
                    help="Prepared MMLD_processed_causal_graph.xlsx")
    p5.add_argument("-o", "--output-dir", type=str, default=None, help="Output directory")
    p5.add_argument("--no-group", action="store_true",
                    help="Declare that the data has no Group/label/class column")
    p5.add_argument("--split", type=str, default="82",
                    choices=["82", "73", "5fold", "loso", "custom"],
                    help="Subject-level split: 82, 73, 5fold, loso, or custom")
    p5.add_argument("--train-ratio", type=float, default=0.8,
                    help="Training subject ratio for split=82; ignored by other splits")
    p5.add_argument("--n-splits", type=int, default=5,
                    help="Number of folds for split=5fold")
    p5.add_argument("--train-sids", type=str, default=None,
                    help="Xlsx file containing SubjectID values for the custom training split")
    p5.add_argument("--test-sids", type=str, default=None,
                    help="Xlsx file containing SubjectID values for the custom test split")
    p5.add_argument("--epochs", type=int, default=10, help="MMLD training epochs in each split")
    p5.add_argument("--lr", type=float, default=0.01, help="MMLD learning rate in each fold")
    p5.add_argument("--glv-method", type=str, default="euler", choices=["euler", "rk4"], help="gLV solver: Euler or RK4")
    p5.add_argument("--seed", type=int, default=42, help="Random seed")
    p5.set_defaults(func=cmd_classify)

    # ----- keystone -----
    p6 = sub.add_parser("keystone", help="Rank keystone species")
    _add_common_args(
        p6, with_graph=True, with_model=False, need_abundance=True,
        graph_required=True,
    )
    p6.add_argument("--max-epochs", type=int, default=10, help="Maximum number of training epochs")
    p6.add_argument("--lr", type=float, default=0.01,
                    help="Learning rate used to train the MMLD model")
    p6.add_argument("--presence-threshold", type=float, default=1e-5, help="Extinction threshold")
    p6.add_argument("--anova-alpha", type=float, default=0.05, help="ANOVA significance level")
    p6.set_defaults(func=cmd_keystone)

    # ----- changepoint -----
    p7 = sub.add_parser(
        "changepoint",
        help="Detect adjacent-time changes in MMLD prediction fidelity",
    )
    _add_common_args(
        p7, with_graph=True, with_model=False, need_abundance=True,
        graph_required=True,
    )
    p7.add_argument(
        "--sheet", type=str, default=None,
        help="Abundance worksheet; defaults to absolute_abundance when available",
    )
    p7.add_argument(
        "--prediction-file", type=str, default=None,
        help=("Optional xlsx containing per-timepoint metrics or paired observed/predicted "
              "abundances; when omitted, MMLD is trained using the selected subject split"),
    )
    p7.add_argument(
        "--alpha", type=float, default=0.01,
        help=("Change-point significance threshold (significance threshold for "
              "adjacent-time Pearson-R comparisons; default: 0.01)"),
    )
    p7.add_argument(
        "--epochs", type=int, default=10,
        help="MMLD training epochs in each split",
    )
    p7.add_argument(
        "--split", type=str, default="82",
        choices=["82", "73", "5fold", "loso", "custom"],
        help="Subject-level split: 82, 73, 5fold, loso, or custom",
    )
    p7.add_argument("--train-ratio", type=float, default=0.8,
                    help="Training subject ratio for split=82")
    p7.add_argument("--train-sids", type=str, default=None,
                    help="Training SubjectID xlsx path for split=custom")
    p7.add_argument("--test-sids", type=str, default=None,
                    help="Test SubjectID xlsx path for split=custom")
    p7.add_argument("--lr", type=float, default=0.01, help="MMLD learning rate")
    p7.add_argument(
        "--n-splits", type=int, default=5,
        help="Number of subject-level cross-validation folds",
    )
    p7.add_argument(
        "--p-threshold", type=float, default=0.01,
        help="Maximum causal-edge P value retained from the supplied graph",
    )
    p7.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
        help="Computation device",
    )
    p7.set_defaults(func=cmd_changepoint)

    # ----- all -----
    p_all = sub.add_parser("all", help="Run the complete workflow")
    _add_common_args(p_all, with_graph=True, with_model=False, need_abundance=True)
    p_all.add_argument("--alg", type=str, default="pcmci",
                       choices=["pcmci", "granger", "cmlp", "clstm", "lasso_dbn", "lingam", "pc"],
                       help="Causal discovery method used during prepare; ignored when --graph is an existing file")
    p_all.add_argument("--tau-min", type=int, default=1,
                       help="Minimum lag used by longitudinal causal discovery")
    p_all.add_argument("--tau-max", type=int, default=2,
                       help="Maximum lag used by longitudinal causal discovery")
    p_all.add_argument("--alpha", type=float, default=0.05,
                       help="Shared alpha level for conditional independence tests and edge filtering")
    p_all.add_argument("--ci-test", type=str, default="ParCorr",
                       choices=["ParCorr", "RobustParCorr", "GPDC", "CMIknn"],
                       help="Conditional-independence test used by PCMCI")
    # predict
    p_all.add_argument("--split", type=str, default="82",
                       choices=["82", "73", "5fold", "loso", "custom"],
                       help="Subject-level split used by prediction and classification")
    p_all.add_argument("--train-ratio", type=float, default=0.8,
                       help="Training subject ratio for split=82; ignored by other splits")
    p_all.add_argument("--n-splits", type=int, default=5,
                       help="Number of folds for split=5fold")
    p_all.add_argument("--train-sids", type=str, default=None,
                       help="Training SubjectID xlsx path for split=custom")
    p_all.add_argument("--test-sids", type=str, default=None,
                       help="Test SubjectID xlsx path for split=custom")
    p_all.add_argument("--mode", type=_mode_type, default="sequence",
                       help="Prediction mode: sequence, match, or stepN (for example, step1 or step4)")
    p_all.add_argument("--initial-ratio", type=float, default=0.15,
                       help="Fraction of each test subject's real prefix used to initialize prediction")
    p_all.add_argument("--top-percent", type=float, default=0.01,
                       help="Top fraction of similar training subjects used by prediction mode=match")
    p_all.add_argument("--epochs", type=int, default=10,
                       help="MMLD training epochs used by prediction and classification")
    p_all.add_argument("--lr", type=float, default=0.01,
                       help="Learning rate used by MMLD training")
    # driver
    p_all.add_argument("--max-sets", type=int, default=10000,
                       help="Maximum number of minimum driver-species sets to enumerate")
    # mpc
    p_all.add_argument("--mpc-initial", type=str, default=None, help="Optional MPC initial state")
    p_all.add_argument("--mpc-target", type=str, default=None, help="Optional MPC target state")
    p_all.add_argument("--driver-species", type=str, default=None, help="Optional comma-separated driver species")
    p_all.add_argument("--mpc-steps", type=int, default=1,
                       help="Number of MPC intervention steps; used only when MPC inputs are provided")
    p_all.add_argument("--q-weight", type=float, default=2e4,
                       help="MPC state-error weight in the LQR objective")
    p_all.add_argument("--r-weight", type=float, default=1.5e-1,
                       help="MPC intervention-cost weight in the LQR objective")
    p_all.add_argument("--tau", type=float, default=1.0,
                       help="MPC intervention duration for each control step")
    # keystone
    p_all.add_argument("--max-epochs", type=int, default=10,
                       help="Maximum MMLD training epochs used by keystone analysis")
    p_all.add_argument("--anova-alpha", type=float, default=0.05,
                       help="ANOVA significance level used by keystone analysis")
    p_all.set_defaults(func=cmd_all)

    return parser


def _apply_default_outdir(args, subdir):
    if not args.output_dir:
        base = getattr(args, "_root_outdir", "outputs")
        args.output_dir = os.path.join(base, subdir)
    return args.output_dir


def _infer_prepare_subdir(abundance_file):
    """English documentation."""
    input_stem = Path(abundance_file).stem if abundance_file else "dataset"
    safe_stem = "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in input_stem
    ).strip("_") or "dataset"
    try:
        import pandas as pd
        df = pd.read_excel(abundance_file)
        cols = list(df.columns)
        id_col = next((c for c in cols
                       if str(c).lower() in ("subjectid", "sampleid", "subject_id", "patientid")),
                      cols[0])
        counts = df.groupby(id_col).size()
        kind = "cross_section" if len(counts) > 0 and counts.max() <= 1 else "time_series"
        return f"00_prepare_{kind}_{safe_stem}"
    except Exception:
        return f"00_prepare_{safe_stem}"


def cmd_prepare(args):
    args.output_dir = args.output_dir or os.path.join(
        "outputs", _infer_prepare_subdir(getattr(args, "abundance", None))
    )
    args._root_outdir = os.path.dirname(args.output_dir) or "outputs"
    from src.prepare import main_prepare
    return main_prepare(args)


def cmd_predict(args):
    _apply_default_outdir(args, "01_prediction")
    from src.predict import main_predict
    return main_predict(args)


def cmd_driver(args):
    _apply_default_outdir(args, "02_driver_mpc")
    from src.driver import main_driver
    return main_driver(args)


def cmd_mpc(args):
    _apply_default_outdir(args, "02_driver_mpc")
    from src.mpc import main_mpc
    return main_mpc(args)


def cmd_classify(args):
    _apply_default_outdir(args, "03_classify")
    from src.classify import main_classify
    return main_classify(args)


def cmd_keystone(args):
    _apply_default_outdir(args, "04_keystone")
    from src.keystone import main_keystone
    return main_keystone(args)


def cmd_changepoint(args):
    _apply_default_outdir(args, "05_changepoint")
    from src.changepoint import main_changepoint
    return main_changepoint(args)


def cmd_all(args):
    root_out = args.output_dir or "outputs"
    os.makedirs(root_out, exist_ok=True)
    args._root_outdir = root_out

    # 1) prepare
    print("\n" + "=" * 60)
    print("[ALL] 1/6 → prepare")
    print("=" * 60)
    from src.prepare import main_prepare
    args.output_dir = os.path.join(root_out, _infer_prepare_subdir(args.abundance))
    prep = main_prepare(args)
    prepared_abundance = os.path.join(args.output_dir, "input_abundance.xlsx")
    weighted_graph = os.path.join(args.output_dir, "MMLD_processed_causal_graph.xlsx")
    model_pt = os.path.join(args.output_dir, "MMLD_model.pt")
    copied_inputs = prep.get("copied_inputs", {})

    def _run(name, fn, setters, required=None):
        if required and not all(getattr(args, r, None) for r in required):
            print(f"[ALL] Skipping {name} (missing arguments: {required})")
            return
        from argparse import Namespace
        nargs = Namespace(**vars(args))
        for k, v in setters.items():
            setattr(nargs, k, v)
        print("\n" + "=" * 60)
        print(f"[ALL] → {name}")
        print("=" * 60)
        try:
            fn(nargs)
        except Exception as e:
            print(f"[ALL] ⚠ {name} failed: {e}")

    # 2) predict
    _run("predict", cmd_predict, {
        "abundance": prepared_abundance,
        "graph": weighted_graph,
        "model": None,
        "output_dir": os.path.join(root_out, "01_prediction"),
        "epochs": getattr(args, "epochs", 10),
    })

    # 3) driver
    _run("driver", cmd_driver, {
        "abundance": prepared_abundance,
        "graph": weighted_graph,
        "output_dir": os.path.join(root_out, "02_driver_mpc"),
    })

    args.initial = copied_inputs.get("mpc_initial") or getattr(args, "mpc_initial", None)
    args.target = copied_inputs.get("mpc_target") or getattr(args, "mpc_target", None)
    args.steps = getattr(args, "mpc_steps", 1)
    if getattr(args, "driver_species", None):
        _run("mpc", cmd_mpc, {
            "graph": weighted_graph,
            "model": model_pt,
            "initial": args.initial,
            "target": args.target,
            "driver_species": args.driver_species,
            "steps": args.steps,
            "output_dir": os.path.join(root_out, "02_driver_mpc"),
        }, required=["initial", "target", "driver_species"])
    else:
        print("[ALL] Skipping mpc (missing --driver-species, --mpc-initial, or --mpc-target)")

    from src.common.utils import detect_group_column
    try:
        import pandas as pd
        has_group = detect_group_column(pd.read_excel(args.abundance, nrows=5)) is not None
    except Exception:
        has_group = False
    if has_group:
        _run("classify", cmd_classify, {
            "abundance": prepared_abundance,
            "graph": weighted_graph,
            "output_dir": os.path.join(root_out, "03_classify"),
            "epochs": getattr(args, "epochs", 10),
        })
    else:
        print("[ALL] Skipping classify: the abundance table has no Group/label/class column")

    # 6) keystone
    _run("keystone", cmd_keystone, {
        "abundance": prepared_abundance,
        "graph": weighted_graph,
        "output_dir": os.path.join(root_out, "04_keystone"),
    })

    print("\n[ALL] Workflow complete. Outputs: ", root_out)
    return True


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    result = args.func(args)
    return 1 if result is False else 0


if __name__ == "__main__":
    sys.exit(main())
