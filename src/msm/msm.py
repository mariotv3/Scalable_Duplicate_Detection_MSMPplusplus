import re
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from ordered_set import OrderedSet

INCH_REGEX = re.compile(
    r'(?<!\d)(\d{2,3}(?:\.\d+)?)\s*(?:inches|inch|in\b|["”])',
    re.IGNORECASE
)

STRONG_MW = re.compile(r"(?i)^(?=.*[a-z])(?=.*\d)(?!.*(inch|hz|p)$)[a-z0-9-]{6,}$")



def generate_small_brand_candidate_pairs(small_brand_offers):
    small_brand_candidates = {}
    all_pairs_flat = []

    for brand, items in small_brand_offers.items():
        offer_ids = [oid for oid, _ in items]

        pairs = []
        if len(offer_ids) >= 2:
            for i, j in combinations(range(len(offer_ids)), 2):
                pair = (offer_ids[i], offer_ids[j])
                pairs.append(pair)
                all_pairs_flat.append(pair)

        small_brand_candidates[brand] = pairs

    return small_brand_candidates, all_pairs_flat


def cosineSim(s1, s2):
    if not s1 and not s2:
        return 0.0

    t1 = s1.lower().split()
    t2 = s2.lower().split()

    vocab = set(t1) | set(t2)
    if not vocab:
        return 0.0

    v1 = np.array([t1.count(tok) for tok in vocab], dtype=float)
    v2 = np.array([t2.count(tok) for tok in vocab], dtype=float)

    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0

    return float(np.dot(v1, v2) / denom)


def q_gram_similarity(string_1, string_2, q=3):
    s1 = f" {str(string_1).lower()} "
    s2 = f" {str(string_2).lower()} "

    q1 = {s1[i:i + q] for i in range(len(s1) - q + 1)}
    q2 = {s2[i:i + q] for i in range(len(s2) - q + 1)}

    if not q1 and not q2:
        return 0.0

    return len(q1 & q2) / len(q1 | q2)


def levenshtein(a, b):
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        prev = dp[:]
        dp[0] = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            dp[j] = min(prev[j] + 1,
                        dp[j - 1] + 1,
                        prev[j - 1] + cost)
    return dp[-1]


def norm_lv(a, b):
    if not a and not b:
        return 0.0
    d = levenshtein(a, b)
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    return 1 - d / max_len


def split_numeric(token):
    token = token.lower()
    m = re.match(r'(\d+)(\D*)', token)
    if m:
        return m.group(2), m.group(1)
    m = re.match(r'(\D+)(\d+)', token)
    if m:
        return m.group(1), m.group(2)
    return token, ""


def avg_lv_sim(model_words_1, model_words_2, mw=False):
    if not model_words_1 or not model_words_2:
        return 0.0

    sims = []
    for w1 in model_words_1:
        for w2 in model_words_2:
            if mw:
                w1_non, _ = split_numeric(w1)
                w2_non, _ = split_numeric(w2)
                sims.append(norm_lv(w1_non, w2_non))
            else:
                sims.append(norm_lv(w1, w2))

    return sum(sims) / len(sims)


def get_original_offer(offer_id, data):
    cid, idx_str = offer_id.split("#", 1)
    return data[cid][int(idx_str)]


def same_shop(product_1, product_2, debug=False):
    if debug:
        print("shop1:", product_1.get("shop"))
        print("shop2:", product_2.get("shop"))
    return product_1.get("shop") == product_2.get("shop")


def same_resolution(product_1, product_2, debug=False):
    product_1_reso = "NA"
    product_2_reso = "NA"

    regex = r'(?<!\S)\d{3,}\s*[x]\s*\d{3,}(?!\S)'

    fm1 = product_1.get("featuresMap") or {}
    fm2 = product_2.get("featuresMap") or {}

    for key, val in fm1.items():
        if "resolution" in key.lower():
            temp = re.search(regex, str(val))
            if temp is not None:
                product_1_reso = temp.group(0)
                break

    for key, val in fm2.items():
        if "resolution" in key.lower():
            temp = re.search(regex, str(val))
            if temp is not None:
                product_2_reso = temp.group(0)
                break

    if debug:
        print("reso1:", product_1_reso)
        print("reso2:", product_2_reso)

    return (product_1_reso == product_2_reso or
            product_1_reso == "NA" or
            product_2_reso == "NA")

def extract_size_inches(product: dict) -> float | None:
    # Prefer title
    title = str(product.get("title") or "")
    m = INCH_REGEX.search(title)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    fm = product.get("featuresMap") or {}
    for k, v in fm.items():
        ks = str(k).lower()
        if any(tok in ks for tok in ["size", "screen", "diagonal"]):
            m = INCH_REGEX.search(str(v))
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
    return "NA"

def same_size(product1, product2):
    s1 = extract_size_inches(product1)
    s2 = extract_size_inches(product2)

    return (s1==s2 or
            s1 == "NA" or
            s2 == "NA")

def extract_model_words(features, keys):
    key_words_regex = r'^\d+\.\d+|\b\d+:\d+\b|(?<!\S)\d{3,}\s*[x]\s*\d{3,}(?!\S)'

    model_words = OrderedSet()
    for key in keys:
        if key in features:
            matches = re.findall(key_words_regex, str(features.get(key, "")))
            model_words.update(matches)

    return model_words

def strong_model_word(title_1, title_2):
    title_regex = r'([a-zA-Z0-9]*((\d*\.)?\d+[^0-9, ]+)[a-zA-Z0-9]*)'
    model_words_1 = OrderedSet()
    model_words_2 = OrderedSet()

    model_words_1.update(x[0] for x in re.findall(title_regex, title_1))
    model_words_2.update(x[0] for x in re.findall(title_regex, title_2))

    strong_1 = {w.lower() for w in model_words_1 if STRONG_MW.match(w)}
    strong_2 = {w.lower() for w in model_words_2 if STRONG_MW.match(w)}

    if strong_1 and strong_2 and not (strong_1 & strong_2):
        return False

    if strong_1 & strong_2:
        return True
    
    return False

def title_comp(title_1, title_2, alpha, beta, delta, eta):
    title_regex = r'([a-zA-Z0-9]*((\d*\.)?\d+[^0-9, ]+)[a-zA-Z0-9]*)' 

    if strong_model_word(title_1, title_2):
        return 1.0
    
    name_cosine_sim = cosineSim(title_1, title_2)

    model_words_1 = OrderedSet()
    model_words_2 = OrderedSet()

    model_words_1.update(x[0] for x in re.findall(title_regex, title_1))
    model_words_2.update(x[0] for x in re.findall(title_regex, title_2))

    strong_1 = {w.lower() for w in model_words_1 if STRONG_MW.match(w)}
    strong_2 = {w.lower() for w in model_words_2 if STRONG_MW.match(w)}

    if strong_1 and strong_2 and not (strong_1 & strong_2):
        return -1.0

    if name_cosine_sim > alpha:
        return 1.0
    

    similar_model_words = False

    for word_1 in model_words_1:
        non_numeric_1, numeric_1 = split_numeric(word_1)

        for word_2 in model_words_2:
            non_numeric_2, numeric_2 = split_numeric(word_2)

            approx_sim = norm_lv(non_numeric_1, non_numeric_2)

            if approx_sim > eta and numeric_1 != numeric_2:
                return -1
            elif approx_sim > eta and numeric_1 == numeric_2:
                similar_model_words = True

    final_name_sim = beta * name_cosine_sim + (1 - beta) * avg_lv_sim(
        model_words_1=model_words_1,
        model_words_2=model_words_2,
        mw=False
    )

    if similar_model_words:
        final_name_sim = delta * avg_lv_sim(
            model_words_1=model_words_1,
            model_words_2=model_words_2,
            mw=True
        ) + (1 - delta) * final_name_sim

    return final_name_sim


def msm_pair_dissimilarity(product_1,
                           product_2,
                           beta,
                           delta,
                           eta,
                           gamma,
                           mu,
                           alpha):
    
    if not strong_model_word(product_1.get("title", ""), product_2.get("title", ""),):
        if same_shop(product_1, product_2, debug=False):
            return 1.0

        if not same_resolution(product_1, product_2, debug=False):
            return 1.0
        
        if not same_size(product_1, product_2):
            return 1.0

    sim = 0.0
    mean_sim = 0.0
    m = 0
    w = 0

    features_1 = product_1.get("featuresMap") or {}
    features_2 = product_2.get("featuresMap") or {}

    no_match_keys_1 = list(features_1.keys())
    no_match_keys_2 = list(features_2.keys())

    for key_1 in list(features_1.keys()):
        for key_2 in list(no_match_keys_2):
            key_sim = q_gram_similarity(key_1, key_2, q=3)
            if key_sim > gamma:
                value_sim = q_gram_similarity(
                    str(features_1.get(key_1, "")),
                    str(features_2.get(key_2, "")),
                    q=3
                )
                sim += key_sim * value_sim
                m += 1
                w += key_sim

                if key_1 in no_match_keys_1:
                    no_match_keys_1.remove(key_1)
                if key_2 in no_match_keys_2:
                    no_match_keys_2.remove(key_2)
                break

    if w > 0:
        mean_sim = sim / w

    model_words_1 = extract_model_words(features_1, no_match_keys_1)
    model_words_2 = extract_model_words(features_2, no_match_keys_2)

    union_len = len(model_words_1.union(model_words_2))
    mw_percentage = 0.0 if union_len == 0 else len(
        model_words_1.intersection(model_words_2)
    ) / union_len
    
    title_sim = title_comp(
        product_1.get("title", ""),
        product_2.get("title", ""),
        alpha=alpha,
        beta=beta,
        delta=delta,
        eta=eta
    )

    if title_sim == -1:
        theta_1 = m / max(1, min(len(features_1), len(features_2)))
        theta_2 = 1 - theta_1
        h_sim = theta_1 * mean_sim + theta_2 * mw_percentage
    else:
        theta_1 = (1 - mu) * m / max(1, min(len(features_1), len(features_2)))
        theta_2 = 1 - mu - theta_1
        h_sim = theta_1 * mean_sim + theta_2 * mw_percentage + mu * title_sim

    return 1.0 - h_sim


def msm_clustering_for_brand(
    candidate_pairs,
    data,
    gamma,
    epsilon,
    mu,
    alpha,
    eta,
    beta,
    delta
):
    if not candidate_pairs:
        return [], pd.DataFrame()

    offer_ids = sorted({oid for pair in candidate_pairs for oid in pair})
    n = len(offer_ids)

    dissimilarity = pd.DataFrame(
        np.ones((n, n), dtype=float),
        index=offer_ids,
        columns=offer_ids
    )
    np.fill_diagonal(dissimilarity.values, 0.0)

    for oid1, oid2 in candidate_pairs:
        p1 = get_original_offer(oid1, data)
        p2 = get_original_offer(oid2, data)

        d = msm_pair_dissimilarity(
            p1, p2, beta=beta, delta = delta, eta=eta, gamma=gamma, mu=mu, alpha=alpha
        )
        if d > 0.4:
            d = 1
        dissimilarity.loc[oid1, oid2] = d
        dissimilarity.loc[oid2, oid1] = d

    if n == 1:
        return [[offer_ids[0]]], dissimilarity

    clustering = AgglomerativeClustering(
        metric="precomputed",
        linkage="complete",
        distance_threshold=epsilon,
        n_clusters=None
    )
    clustering.fit(dissimilarity.values)

    labels = clustering.labels_
    clusters_dict = defaultdict(list)
    for oid, lbl in zip(offer_ids, labels):
        clusters_dict[lbl].append(oid)

    clusters = list(clusters_dict.values())
    return clusters, dissimilarity


def msm_for_all_brands(
    brand_candidates,
    beta,
    delta,
    eta,
    data,
    gamma,
    epsilon,
    mu,
    alpha
):
    clusters_by_brand = {}
    dissim_by_brand = {}

    for brand, pairs in brand_candidates.items():
        clusters, dissim = msm_clustering_for_brand(
            candidate_pairs=pairs,
            data=data,
            gamma=gamma,
            epsilon=epsilon,
            mu=mu,
            alpha=alpha,
            beta=beta,
            delta=delta,
            eta=eta
        )
        clean_clusters = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue

            if has_forbidden_pair(cluster, dissim):
                for oid in cluster:
                    clean_clusters.append([oid])
            else:
                clean_clusters.append(cluster)
        clusters_by_brand[brand] = clusters
        dissim_by_brand[brand] = dissim

    return clusters_by_brand, dissim_by_brand

def has_forbidden_pair(cluster, dissim, cutoff=1.0):
    for i in range(len(cluster)):
        for j in range(i + 1, len(cluster)):
            if dissim.loc[cluster[i], cluster[j]] >= cutoff:
                return True
    return False

def clusters_to_pairs_by_brand(clusters_by_brand):
    brand_pairs = {}
    all_pairs = set()

    for brand, clusters in clusters_by_brand.items():
        pairs = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            cluster = sorted(cluster)
            for i, j in combinations(cluster, 2):
                pair = (i, j)
                pairs.append(pair)
                all_pairs.add(pair)
        brand_pairs[brand] = pairs

    return brand_pairs, all_pairs