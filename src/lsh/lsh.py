import numpy as np
from collections import defaultdict

def build_lsh_index(signatures: np.ndarray, b: int, r: int):
    k, N = signatures.shape
    assert k == b * r, f"k must equal b * r, got k={k}, b*r={b*r}"

    band_hash_tables = [defaultdict(list) for _ in range(b)]

    for doc_idx in range(N):
        for band_idx in range(b):
            start = band_idx * r
            end = start + r
            band_slice = signatures[start:end, doc_idx]
            band_bytes = band_slice.tobytes()
            band_hash_tables[band_idx][band_bytes].append(doc_idx)

    return band_hash_tables


def generate_all_candidate_pairs(band_hash_tables):
    seen_pairs = set()

    for band_table in band_hash_tables:
        for bucket_docs in band_table.values():
            if len(bucket_docs) < 2:
                continue
            bucket_docs = sorted(bucket_docs)
            for i_idx in range(len(bucket_docs)):
                for j_idx in range(i_idx + 1, len(bucket_docs)):
                    pair = (bucket_docs[i_idx], bucket_docs[j_idx])
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        yield pair


def lsh_for_brand_block(offer_ids, signatures, b, r):
    k, N = signatures.shape
    assert k == b * r, f"k must equal b * r, got k={k}, b*r={b*r}"

    band_hash_tables = build_lsh_index(signatures, b=b, r=r)
    idx_pairs = list(generate_all_candidate_pairs(band_hash_tables))

    pairs_offer_ids = [(offer_ids[i], offer_ids[j]) for (i, j) in idx_pairs]

    doc_to_cand_indices = {i: set() for i in range(N)}
    for i, j in idx_pairs:
        doc_to_cand_indices[i].add(j)
        doc_to_cand_indices[j].add(i)

    doc_to_cand_offers = {
        offer_ids[i]: {offer_ids[j] for j in cand_idxs}
        for i, cand_idxs in doc_to_cand_indices.items()
    }

    return pairs_offer_ids, doc_to_cand_offers


def factor_pairs(n: int):
    pairs = []
    for b in range(1, int(n**0.5) + 1):
        if n % b == 0:
            r = n // b
            pairs.append((b, r))
            if b != r:
                pairs.append((r, b))
    return sorted(pairs)


def run_lsh_for_all_brands(brand_signatures, b, r):
    brand_lsh_candidates = {}

    for brand, (offer_ids, signatures) in brand_signatures.items():
        k, N = signatures.shape
        if N < 2:
            continue
        if k != b * r:
            raise ValueError(
                f"LSH config mismatch for brand '{brand}': k={k}, b*r={b*r}"
            )

        pairs_offer_ids, _ = lsh_for_brand_block(offer_ids, signatures, b, r)
        brand_lsh_candidates[brand] = pairs_offer_ids

    return brand_lsh_candidates


def generate_lsh_configs(num_perm: int, max_delta: int = 2):
    configs = []

    for k_eff in range(max(1, num_perm - max_delta), num_perm + max_delta + 1):
        for b, r in factor_pairs(k_eff):
            configs.append((k_eff, b, r))

    configs = sorted(set(configs))
    return configs

def run_lsh_for_all_brands_with_keff(brand_signatures, k_eff, b, r):
    brand_lsh_candidates = {}
    brand_lsh_doc_cands = {}

    for brand, (offer_ids, sigs) in brand_signatures.items():
        k, n = sigs.shape
        if n < 2:
            continue

        if k_eff > k:
            raise ValueError(f"k_eff={k_eff} > k={k} for brand {brand}")

        sigs_eff = sigs[:k_eff, :]

        pairs, doc_to_cand = lsh_for_brand_block(offer_ids, sigs_eff, b, r)
        brand_lsh_candidates[brand] = pairs
        brand_lsh_doc_cands[brand] = doc_to_cand

    return brand_lsh_candidates, brand_lsh_doc_cands