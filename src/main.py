import numpy as np  
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed


from preprocessing.cleaning import prepare_datasets
from lsh.minhashing import build_minhash_for_brands 
from evaluation.lsh_eval import tune_lsh_parameters
from lsh.lsh import run_lsh_for_all_brands  
from msm.msm import generate_small_brand_candidate_pairs
from evaluation.msm_eval import tune_msm_params, run_msm_and_evaluate
from evaluation.curves import eval_full_model_for_lsh_configs

def run_single_bootstrap(raw_data,
                         seed,
                         num_perm=128,
                         lsh_min_PC=0.9):
    
    random.seed(seed)
    np.random.seed(seed)
    # --- Prepare datasets (this already does a bootstrap split internally) ---
    datasets = prepare_datasets(raw_data, seed=seed)
    cleaned_data = datasets["cleaned_all"]
    data_train = datasets["data_train"]
    data_test = datasets["data_test"]
    brand_blocks_train = datasets["brand_blocks_train"]
    brand_blocks_test = datasets["brand_blocks_test"]
    known_brands = datasets["known_brands"]

    print(f"[Bootstrap seed={seed}] Data cleaned and split.")
    print("  Train clusters:", len(data_train))
    print("  Test  clusters:", len(data_test))

    # --- LSH on TRAIN: build signatures and tune (b, r) ---
    brand_signatures_train, small_brand_offers_train, _ = build_minhash_for_brands(
        brand_blocks_train, num_perm=num_perm, seed=seed
    )
    print("[Bootstrap] Minhash on TRAIN done.")

    best_lsh_params, best_lsh_metrics = tune_lsh_parameters(
        brand_signatures=brand_signatures_train,
        data=raw_data,
        num_perm=num_perm,
        min_PC=lsh_min_PC,
    )
    b, r = best_lsh_params
    print(f"[Bootstrap] Best LSH params: b={b}, r={r}")
    print(f"  LSH PC (recall) = {best_lsh_metrics['PC']}")
    print(f"  LSH PQ (precision) = {best_lsh_metrics['PQ']}")
    print(f"  LSH FC (fraction comparisons) = {best_lsh_metrics['FC']}")

    brand_lsh_candidates_train = run_lsh_for_all_brands(
        brand_signatures_train, b, r
    )

    small_brand_candidates_train, _ = generate_small_brand_candidate_pairs(
        small_brand_offers_train
    )
    brand_candidates_train = {
        **brand_lsh_candidates_train,
        **small_brand_candidates_train,
    }

    train_clusters = sorted(set(data_train.keys()))

    # --- MSM tuning on TRAIN ---
    best_msm_params, best_msm_metrics = tune_msm_params(
    brand_candidates=brand_candidates_train,
    data=cleaned_data,
    train_cluster_ids=train_clusters,
    n_trials=30, 
    timeout=3600,       # cap in seconds
    gamma_range=(0.3, 0.7),
    epsilon_range=(0.1, 0.4),
    mu_range=(0.5, 0.9),
    alpha_range=(0.4, 0.8),
    beta_range=(0.0,0.2),
    eta_range=(0.3,0.7),
    delta_range=(0.3,0.7),
    seed=seed
)

    print(f"[Bootstrap] Best MSM params: {best_msm_params}")
    print(f"[Bootstrap] Train F1: {best_msm_metrics['F1']}")

    # --- Run LSH + MSM on TEST with tuned params ---
    brand_signatures_test, small_brand_offers_test, _ = build_minhash_for_brands(
        brand_blocks_test, num_perm=num_perm, seed=seed
    )
    brand_lsh_candidates_test = run_lsh_for_all_brands(
        brand_signatures_test, b=b, r=r
    )
    small_brand_candidates_test, _ = generate_small_brand_candidate_pairs(
        small_brand_offers_test
    )
    brand_candidates_test = {
        **brand_lsh_candidates_test,
        **small_brand_candidates_test,
    }

    test_clusters = set(data_test.keys())

    test_metrics = run_msm_and_evaluate(
        brand_candidates=brand_candidates_test,
        data=cleaned_data,
        gamma=best_msm_params["gamma"],
        epsilon=best_msm_params["epsilon"],
        mu=best_msm_params["mu"],
        alpha=best_msm_params["alpha"],
        beta=best_msm_params["beta"],
        delta=best_msm_params["delta"],
        eta=best_msm_params["eta"],
        allowed_cluster_ids=test_clusters,
    )

    print("[Bootstrap] TEST metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    curve_points = eval_full_model_for_lsh_configs(
    brand_signatures_test=brand_signatures_test,
    small_brand_offers_test=small_brand_offers_test,
    cleaned_data=cleaned_data,
    test_clusters=test_clusters,
    msm_params=best_msm_params,
    num_perm=num_perm,
    max_delta=2,  # or 1/2/3 depending on how dense you want FC grid
    )

    return {
        "lsh_params": {"b": b, "r": r},
        "lsh_metrics": best_lsh_metrics,
        "msm_params": best_msm_params,
        "train_metrics": best_msm_metrics,
        "test_metrics": test_metrics,
        "curve_points": curve_points,
    }


def main(args):
    with open(args.path) as f:
        raw_data = json.load(f)

    #for testing only through main guard
    if args.max_clusters is not None:
        raw_data = dict(list(raw_data.items())[:args.max_clusters])


    num_perm = 240
    all_results = []

    # ---- SEQUENTIAL VERSION (fallback / n_jobs_bootstrap=1) ----
    if args.n_jobs_bootstrap == 1:
        for i in range(args.bootstraps):
            seed = args.seed + i
            print(f"\n========== BOOTSTRAP {i+1}/{args.bootstraps} (seed={seed}) ==========\n")
            res = run_single_bootstrap(
                raw_data,
                seed,
                num_perm,
                args.lsh_min_PC,
            )
            all_results.append(res)

    # ---- PARALLEL VERSION ----
    else:
        seeds = [args.seed + i for i in range(args.bootstraps)]

        with ProcessPoolExecutor(max_workers=args.n_jobs_bootstrap) as ex:
            futures = {
                ex.submit(
                    run_single_bootstrap,
                    raw_data,      
                    s,             
                    num_perm,      
                    args.lsh_min_PC,  
                ): s
                for s in seeds
            }

            done_count = 0
            for fut in as_completed(futures):
                seed_done = futures[fut]
                done_count += 1
                print(
                    f"\n[Main] Bootstrap with seed={seed_done} finished "
                    f"({done_count}/{args.bootstraps})"
                )
                all_results.append(fut.result())

    # ---- Aggregate TEST metrics over bootstraps ----
    if not all_results:
        print("No bootstrap results collected.")
        return

    test_metrics_list = [r["test_metrics"] for r in all_results]
    metric_keys = list(test_metrics_list[0].keys())
    avg_test_metrics = {
        k: float(np.mean([m[k] for m in test_metrics_list]))
        for k in metric_keys
    }

    print("\n========== AVERAGED TEST METRICS OVER BOOTSTRAPS ==========")
    for k, v in avg_test_metrics.items():
        print(f"{k}: {v}")



    # ---- collect all curve points from all bootstraps ----
    all_curve_rows = []
    for res in all_results:
        all_curve_rows.extend(res["curve_points"])

    curve_df = pd.DataFrame(all_curve_rows)

    # ---- bin FC into regular intervals, e.g. width 0.05 ----
    bin_width = 0.05
    bin_edges = np.arange(0.0, 1.0 + bin_width, bin_width)  # [0.0, 0.05, ..., 1.0]

    curve_df["FC_bin"] = pd.cut(
        curve_df["FC"],
        bins=bin_edges,
        include_lowest=True,
        labels=False,      
    )

    # ---- aggregate per bin ----
    agg = (
        curve_df
        .groupby("FC_bin")
        .agg(
            FC=("FC", "mean"),     
            F1=("F1", "mean"),
            PQ=("PQ", "mean"),
            PC=("PC", "mean"),
            F1_star=("F1*", "mean"),
            count=("FC", "size"),  # how many points in the bin
        )   
        .reset_index(drop=True)
    )

    # optional: drop bins with very few points (to reduce noise)
    min_points_per_bin = 2
    agg = agg[agg["count"] >= min_points_per_bin]

    # make sure rows are sorted by FC
    agg = agg.sort_values("FC").reset_index(drop=True)


    def plot_metric_vs_FC(df, metric, out_dir="graphs"):
        os.makedirs(out_dir, exist_ok=True)
        df = df.sort_values("FC")  # just to be safe

        x = df["FC"]
        y = df[metric]

        plt.figure()
        plt.plot(x, y, marker="o")
        plt.xlabel("Fraction of comparisons (FC)")
        plt.ylabel(metric)
        plt.title(f"{metric} vs FC")
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{metric}_vs_FC.png"))
        plt.close()


    # note: use the column name "F1_star" here, not "F1*"
    for metric in ["F1", "PQ", "PC", "F1_star"]:
        plot_metric_vs_FC(agg, metric)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/TVs-all-merged.json")
    parser.add_argument("--bootstraps", type=int, default=7,
                        help="Number of bootstrap repetitions.")
    parser.add_argument("--seed", type=int, default=123,
                        help="Base random seed.")
    parser.add_argument("--lsh_min_PC", type=float, default=0.8,
                        help="Minimum allowed ratio of true duplicates retained after LSH.")
    parser.add_argument("--max_clusters", type=int, default=None,
                        help="If set, use only the first N clusters for quick runs.")
    parser.add_argument("--n_jobs_bootstrap", type=int, default=7,
                        help="Number of processes to use for parallel bootstraps.")
    args = parser.parse_args()
    start = time.perf_counter()
    main(args)
    end = time.perf_counter()

    elapsed = end - start
    print(f"\n===== TOTAL WALL-CLOCK RUNTIME: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds) =====")

# python src/main.py --path data/TVs-all-merged.json --bootstraps 3 --max_clusters 400 --n_jobs_bootstrap 3