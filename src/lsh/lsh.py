import numpy as np
from collections import defaultdict


def build_lsh_index(signatures: np.ndarray, b: int, r: int):
    k, n = signatures.shape
    if k != b * r:
        raise ValueError(f"k={k} but b*r={b*r}")
    tables = [defaultdict(list) for _ in range(b)]
    for j in range(n):
        for band in range(b):
            s = band * r
            e = s + r
            key = signatures[s:e, j].tobytes()
            tables[band][key].append(j)
    return tables


def generate_all_candidate_pairs(band_hash_tables):
    seen = set()
    for table in band_hash_tables:
        for idxs in table.values():
            if len(idxs) < 2:
                continue
            idxs = sorted(idxs)
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    pair = (idxs[i], idxs[j])
                    if pair not in seen:
                        seen.add(pair)
                        yield pair


def lsh_for_brand_block(offer_ids, signatures, b, r):
    k, n = signatures.shape
    if k != b * r:
        raise ValueError(f"k={k} but b*r={b*r}")
    tables = build_lsh_index(signatures, b=b, r=r)
    idx_pairs = list(generate_all_candidate_pairs(tables))
    pairs_offer_ids = [(offer_ids[i], offer_ids[j]) for (i, j) in idx_pairs]
    doc_to_cand = {oid: set() for oid in offer_ids}
    for i, j in idx_pairs:
        oi, oj = offer_ids[i], offer_ids[j]
        doc_to_cand[oi].add(oj)
        doc_to_cand[oj].add(oi)
    return pairs_offer_ids, doc_to_cand


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
        k, n = signatures.shape
        if n < 2:
            # Nothing to compare for this brand
            continue

        if k != b * r:
            raise ValueError(f"LSH config mismatch for brand '{brand}': k={k} but b*r={b*r}")

        pairs_offer_ids, _ = lsh_for_brand_block(offer_ids, signatures, b, r)
        brand_lsh_candidates[brand] = pairs_offer_ids

    return brand_lsh_candidates