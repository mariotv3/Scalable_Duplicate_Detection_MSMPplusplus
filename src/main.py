import numpy as np  
import json
import argparse

from preprocessing.cleaning import prepare_datasets
from lsh.minhashing import build_minhash_for_brands 
from evaluation.lsh_eval import tune_lsh_parameters
from lsh.lsh import run_lsh_for_all_brands  
from msm.msm import generate_small_brand_candidate_pairs
from evaluation.msm_eval import tune_msm_params, run_msm_and_evaluate


def run_single_bootstrap(raw_data,
                         seed,
                         num_perm=128,
                         lsh_max_FC=0.85,
                         gamma_grid=None,
                         epsilon_grid=None,
                         mu_grid=None,
                         alpha_grid=None):

    if gamma_grid is None:
        gamma_grid = [0.4, 0.5]
    if epsilon_grid is None:
        epsilon_grid = [0.2, 0.3]
    if mu_grid is None:
        mu_grid = [0.7, 0.8]
    if alpha_grid is None:
        alpha_grid = [0.5, 0.6]

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
        max_FC=lsh_max_FC,
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

    train_clusters = set(data_train.keys())

    # --- MSM tuning on TRAIN ---
    best_msm_params, best_msm_metrics = tune_msm_params(
    brand_candidates=brand_candidates_train,
    data=cleaned_data,
    brands=known_brands,
    train_cluster_ids=train_clusters,
    n_trials=30, 
    timeout=300,       # cap in seconds
    gamma_range=(0.3, 0.7),
    epsilon_range=(0.1, 0.4),
    mu_range=(0.5, 0.9),
    alpha_range=(0.4, 0.8),
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
        brands=known_brands,
        allowed_cluster_ids=test_clusters,
    )

    print("[Bootstrap] TEST metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    return {
        "lsh_params": {"b": b, "r": r},
        "lsh_metrics": best_lsh_metrics,
        "msm_params": best_msm_params,
        "train_metrics": best_msm_metrics,
        "test_metrics": test_metrics,
    }

def main(args):
    with open(args.path) as f:
        raw_data = json.load(f)

    # Test: restrict to first N clusters for quick testing
    if args.max_clusters is not None:
        raw_data = dict(list(raw_data.items())[:args.max_clusters])

    num_perm = 128

    gamma_grid = [0.45, 0.5]
    epsilon_grid = [0.25, 0.3]
    mu_grid = [0.65, 0.7]
    alpha_grid = [0.6, 0.65]

    all_results = []

    for i in range(args.bootstraps):
        seed = args.seed + i  # different seed per bootstrap
        print(f"\n========== BOOTSTRAP {i+1}/{args.bootstraps} (seed={seed}) ==========\n")

        res = run_single_bootstrap(
            raw_data=raw_data,
            seed=seed,
            num_perm=num_perm,
            lsh_max_FC=args.lsh_max_FC,
            gamma_grid=gamma_grid,
            epsilon_grid=epsilon_grid,
            mu_grid=mu_grid,
            alpha_grid=alpha_grid,
        )
        all_results.append(res)

    # ---- Aggregate TEST metrics over bootstraps ----
    test_metrics_list = [r["test_metrics"] for r in all_results]

   
    metric_keys = list(test_metrics_list[0].keys())
    avg_test_metrics = {
        k: float(np.mean([m[k] for m in test_metrics_list]))
        for k in metric_keys
    }

    print("\n========== AVERAGED TEST METRICS OVER BOOTSTRAPS ==========")
    for k, v in avg_test_metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/TVs-all-merged.json")
    parser.add_argument("--bootstraps", type=int, default=5,
                        help="Number of bootstrap repetitions.")
    parser.add_argument("--seed", type=int, default=123,
                        help="Base random seed.")
    parser.add_argument("--lsh_max_FC", type=float, default=0.85,
                        help="Maximum allowed fraction of comparisons for LSH tuning.")
    parser.add_argument("--max_clusters", type=int, default=None,
                        help="If set, use only the first N clusters for quick runs.")
    args = parser.parse_args()
    main(args)

# python src/main.py --path data/TVs-all-merged.json --bootstraps 3 --max_clusters 400