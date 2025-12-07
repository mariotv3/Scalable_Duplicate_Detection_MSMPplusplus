import re
from typing import Dict, List, Tuple, Iterable, Set

import numpy as np

NUMERIC_PREFIX = re.compile(r"^\d+(?:\.\d+)?")
PRIME_32 = np.uint32(4294967291)


def augment_with_numeric_parts(tokens: Iterable[str]) -> Set[str]:
    tokens = set(tokens)
    out = set(tokens)
    for t in tokens:
        m = NUMERIC_PREFIX.match(t)
        if m:
            out.add(m.group(0))
    return out


def extract_shingles_for_offer(offer: dict) -> Set[str]:
    mw = offer.get("modelWords_title", set()) or set()
    mw = augment_with_numeric_parts(mw)
    feat = set(offer.get("featuresMap", []))
    return mw | feat


def shingles_to_ints_local(shingles: Iterable[str],
                           shingle_to_id: Dict[str, int]) -> Set[int]:
    out = set()
    for s in shingles:
        if s not in shingle_to_id:
            shingle_to_id[s] = len(shingle_to_id)
        out.add(shingle_to_id[s])
    return out


def compute_minhash_signature(shingle_ids: Iterable[int],
                              A: np.ndarray,
                              B: np.ndarray,
                              P: np.int32) -> np.ndarray:
    num_perm = A.shape[0]
    sig = np.full(num_perm, P, dtype=np.int32)
    shingle_ids = list(shingle_ids)
    if not shingle_ids:
        return sig
    for x in shingle_ids:
        x = np.uint32(x)
        hx = (A + B * x) % P
        sig = np.minimum(sig, hx)
    return sig.astype(np.uint32)


def process_brand_block(
    items: List[Tuple[str, dict]],
    num_perm: int,
    A: np.ndarray,
    B: np.ndarray,
    P: np.uint32,
) -> Tuple[List[str], np.ndarray]:
    shingle_to_id: Dict[str, int] = {}
    product_shingle_sets: Dict[str, Set[int]] = {}

    for offer_id, offer in items:
        shingles = extract_shingles_for_offer(offer)
        product_shingle_sets[offer_id] = shingles_to_ints_local(
            shingles, shingle_to_id
        )

    offer_ids = list(product_shingle_sets.keys())
    n = len(offer_ids)
    sigs = np.empty((num_perm, n), dtype=np.uint32)

    for j, oid in enumerate(offer_ids):
        sigs[:, j] = compute_minhash_signature(
            product_shingle_sets[oid], A, B, P
        )

    return offer_ids, sigs


def build_minhash_for_brands(
    brand_blocks: Dict[str, List[Tuple[str, dict]]],
    num_perm: int = 128,
    seed: int | None = None,
    min_offers_for_lsh: int = 4,
):
    rng = np.random.default_rng(seed)
    A = rng.integers(1, PRIME_32, size=num_perm, dtype=np.uint32)
    B = rng.integers(0, PRIME_32, size=num_perm, dtype=np.uint32)

    brand_signatures: Dict[str, Tuple[List[str], np.ndarray]] = {}
    small_brand_offers: Dict[str, List[Tuple[str, dict]]] = {}

    for brand, items in brand_blocks.items():
        n = len(items)
        if n < min_offers_for_lsh:
            small_brand_offers[brand] = items
            continue
        offer_ids, sigs = process_brand_block(items, num_perm, A, B, PRIME_32)
        brand_signatures[brand] = (offer_ids, sigs)

    params = {"NUM_PERM": num_perm, "P": PRIME_32, "A": A, "B": B}
    return brand_signatures, small_brand_offers, params