#!/usr/bin/env python3
import json
import argparse
import numpy as np
from itertools import combinations
import random
import matplotlib.pyplot as plt
import os

from preprocessing.cleaning import prepare_datasets
from lsh.minhashing import extract_shingles_for_offer
from lsh.lsh import lsh_for_brand_block, generate_lsh_configs

# -------------------------------------------------------
# MinHash Debugger
# -------------------------------------------------------
def debug_minhash_for_brand(brand, brand_blocks, brand_signatures, max_pairs=200, seed = 123):
    items = brand_blocks[brand]                     # [(offer_id, offer_dict), ...]
    offer_ids, sigs = brand_signatures[brand]       # sigs shape (num_perm, N)

    # --- True shingles ---
    shingle_sets = {oid: extract_shingles_for_offer(offer)
                    for oid, offer in items}

    # Map offer_id → column index in signature matrix
    col_index = {oid: j for j, oid in enumerate(offer_ids)}

    # Limit number of tested pairs
    all_pairs = list(combinations(offer_ids, 2))
    if len(all_pairs) > max_pairs:
        rng = random.Random(seed)
        all_pairs = random.sample(all_pairs, max_pairs)

    true_jacc = []
    est_jacc = []

    for oid1, oid2 in all_pairs:
        S1 = shingle_sets[oid1]
        S2 = shingle_sets[oid2]

        # True Jaccard
        if S1 or S2:
            jt = len(S1 & S2) / len(S1 | S2)
        else:
            jt = 0.0

        # MinHash Jaccard estimate
        v1 = sigs[:, col_index[oid1]]
        v2 = sigs[:, col_index[oid2]]
        je = np.mean(v1 == v2)

        true_jacc.append(jt)
        est_jacc.append(je)

    true_jacc = np.array(true_jacc)
    est_jacc = np.array(est_jacc)
    corr = np.corrcoef(true_jacc, est_jacc)[0, 1] if len(true_jacc) > 1 else np.nan

    print(f"\n=== MinHash Debug for brand '{brand}' ===")
    print("Pairs tested:", len(true_jacc))
    print("Mean true Jaccard:", true_jacc.mean())
    print("Mean est  Jaccard:", est_jacc.mean())
    print("Correlation:", corr)
    print("First 10 pairs:")
    for i in range(min(10, len(true_jacc))):
        print(f"  true={true_jacc[i]:.3f}, est={est_jacc[i]:.3f}")


# -------------------------------------------------------
# LSH Debugger
# -------------------------------------------------------
def debug_lsh_for_brand(brand, brand_signatures, num_perm, max_delta=2):
    offer_ids, sigs_full = brand_signatures[brand]
    N = len(offer_ids)
    total_pairs = N * (N - 1) // 2

    print(f"\n=== LSH candidate counts for brand '{brand}' ===")
    print(f"N = {N}, total possible = {total_pairs}")

    configs = generate_lsh_configs(num_perm=num_perm, max_delta=max_delta)

    for k_eff, b, r in configs:
        if k_eff > sigs_full.shape[0]:
            continue
        sigs = sigs_full[:k_eff, :]
        pairs, _ = lsh_for_brand_block(offer_ids, sigs, b, r)
        FC = len(pairs) / total_pairs if total_pairs else 0.0

        print(f"k_eff={k_eff:3d}, b={b:3d}, r={r:3d}  ->  "
              f"{len(pairs):4d} pairs,   FC={FC:.4f}")


def jaccard(a, b):
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def offer_cluster_id(offer_id: str) -> str:
    """Extract the cluster/model ID from 'clusterID#index'."""
    cid, _ = offer_id.split("#", 1)
    return cid


def plot_jaccard_histograms(
    brand_items,
    shingle_sets_by_oid,
    outfile: str = "jaccard_hist.png",
    max_pairs: int | None = 5000,
    seed = 123
):
    """
    brand_items: list of (offer_id, offer_dict) for ONE brand
    shingle_sets_by_oid: dict {offer_id: set_of_shingles}
    outfile: path to save the PNG
    max_pairs: optional cap on number of pairs to sample for speed
    """

    dup_vals = []
    nondup_vals = []

    all_pairs = list(combinations(brand_items, 2))
    if max_pairs is not None and len(all_pairs) > max_pairs:
        rng = random.Random(seed)
        all_pairs = rng.sample(all_pairs, max_pairs)

    for (oid1, _), (oid2, _) in all_pairs:
        s1 = shingle_sets_by_oid[oid1]
        s2 = shingle_sets_by_oid[oid2]
        j = jaccard(s1, s2)

        if offer_cluster_id(oid1) == offer_cluster_id(oid2):
            dup_vals.append(j)
        else:
            nondup_vals.append(j)

    print(f"#dup pairs used: {len(dup_vals)}")
    print(f"#non-dup pairs used: {len(nondup_vals)}")

    # ---- Plot histograms ----
    plt.figure(figsize=(8, 5))
    bins = 20

    plt.hist(dup_vals,     bins=bins, alpha=0.5, label="duplicates",     density=True)
    plt.hist(nondup_vals, bins=bins, alpha=0.5, label="non-duplicates", density=True)
    

    plt.xlabel("True Jaccard similarity")
    plt.ylabel("Count")
    plt.title("Jaccard similarity distributions (dup vs non-dup)")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    plt.savefig(outfile)
    plt.close()
    print(f"Saved histogram to {outfile}")

# -------------------------------------------------------
# Main guard
# -------------------------------------------------------
if __name__ == "__main__":
    random.seed(123)
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/TVs-all-merged.json")
    parser.add_argument("--brand", required=False,
                        help="Brand name to debug (default: first brand found).")
    parser.add_argument("--max_pairs", type=int, default=200)
    parser.add_argument("--num_perm", type=int, default=128)
    args = parser.parse_args()

    # Load raw dataset
    with open(args.path) as f:
        raw_data = json.load(f)

    # Optionally restrict to first 400 clusters for speed
    raw_data = dict(list(raw_data.items())[:400])

    # Preprocess normally
    datasets = prepare_datasets(raw_data, seed=123)
    brand_blocks = datasets["brand_blocks_train"]
    cleaned_data = datasets["cleaned_all"]  # not used below, but fine to keep

    # MinHash signatures
    from lsh.minhashing import build_minhash_for_brands
    brand_signatures, _, _ = build_minhash_for_brands(
        brand_blocks,
        num_perm=args.num_perm,
        seed=123,
    )

    # -----------------------------
    # Pick a brand to debug
    # -----------------------------
    if args.brand:
        brand = args.brand
        if brand not in brand_signatures:
            raise ValueError(f"Brand '{brand}' not found in brand_signatures.")
    else:
        brand = next(iter(brand_signatures.keys()))
        print(f"No brand specified – using '{brand}'")

    # Items and shingles for that brand
    brand_items = brand_blocks[brand]  # list of (offer_id, offer_dict)
    shingle_sets_by_oid = {
        oid: extract_shingles_for_offer(offer)
        for oid, offer in brand_items
    }

    # -----------------------------
    # Run MinHash and LSH debuggers
    # -----------------------------
    debug_minhash_for_brand(
        brand=brand,
        brand_blocks=brand_blocks,
        brand_signatures=brand_signatures,
        max_pairs=args.max_pairs,
    )

    debug_lsh_for_brand(
        brand=brand,
        brand_signatures=brand_signatures,
        num_perm=args.num_perm,
        max_delta=2,
    )

    # -----------------------------
    # Plot Jaccard histograms
    # -----------------------------
    plot_jaccard_histograms(
        brand_items=brand_items,
        shingle_sets_by_oid=shingle_sets_by_oid,
        outfile=f"{brand}_jaccard_hist.png",
        max_pairs=5000,
    )

    print(f"Done. Histogram saved to {brand}_jaccard_hist.png")

    # python src/debugging_minhash.py --brand samsung