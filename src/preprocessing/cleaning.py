import re
import copy
from collections import OrderedDict, defaultdict, Counter
import numpy as np

TRAILING_SUFFIXES = {
    "tv", "electronics", "inc", "inc.", "corp", "corporation", "company", "co"
}

def normalize_brand(raw):
    if not raw or not isinstance(raw, str):
        return "unknown"
    if raw.strip().lower() == "pansonic":
        return "panasonic"
    s = raw.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    words = s.split()
    while words and words[-1] in TRAILING_SUFFIXES:
        words.pop()
    if not words:
        return "unknown"
    return " ".join(words)

def extract_brand(features_map):
    if not isinstance(features_map, dict):
        return "unknown"
    for key in sorted(features_map.keys(), key=str):
        value = features_map[key]
        if isinstance(key, str) and re.search(r"brand", key, re.IGNORECASE):
            if isinstance(value, str):
                v = value.strip()
                if v:
                    return normalize_brand(v)
    return "unknown"

def extract_titles(data):
    cluster_to_titles = {}
    for cluster_id, offers in data.items():
        titles = []
        for offer in offers:
            title = offer.get("title")
            if isinstance(title, str):
                titles.append(title.strip())
        cluster_to_titles[cluster_id] = titles
    return cluster_to_titles

def propagate_brands_in_cluster(cluster_to_brands):
    updated = {}
    known_brands = set()
    for cid, brands in cluster_to_brands.items():
        non_unknown = [b for b in brands if b != "unknown"]
        if non_unknown:
            counts = Counter(non_unknown)
            max_count = max(counts.values())
            candidates = [b for b, c in counts.items() if c == max_count]
            most_common_brand = sorted(candidates)[0]
            updated[cid] = [most_common_brand] * len(brands)
            known_brands.add(most_common_brand)
        else:
            updated[cid] = list(brands)
    return updated, known_brands

def build_brand_pattern(brand_set):
    if not brand_set:
        return None
    brands_sorted = sorted(brand_set, key=len, reverse=True)
    escaped = [re.escape(b) for b in brands_sorted]
    pattern = r"\b(?:%s)\b" % "|".join(escaped)
    return re.compile(pattern, flags=re.IGNORECASE)

def guess_brand_from_title(title, brand_pattern):
    if not brand_pattern or not isinstance(title, str):
        return "unknown"
    m = brand_pattern.search(title)
    if m:
        return m.group(0).strip()
    return "unknown"

def fill_unknown_brands_from_titles(cluster_to_brands, cluster_to_titles, known_brands):
    brand_pattern = build_brand_pattern(known_brands)
    updated = {}
    for cid, brands in cluster_to_brands.items():
        if any(b != "unknown" for b in brands):
            updated[cid] = brands
            continue
        titles = cluster_to_titles.get(cid, [])
        inferred_brand = "unknown"
        for title in titles:
            inferred_brand = guess_brand_from_title(title, brand_pattern)
            inferred_brand = normalize_brand(inferred_brand)
            if inferred_brand != "unknown":
                break
        if inferred_brand != "unknown":
            updated[cid] = [inferred_brand] * len(brands)
            known_brands.add(inferred_brand)
        else:
            updated[cid] = brands
    return updated, known_brands

def annotate_offers_with_brand(data):
    cluster_to_brands = {}
    for cluster_id, offers in data.items():
        brands = []
        for offer in offers:
            features_map = offer.get("featuresMap", {})
            brand = extract_brand(features_map)
            brands.append(brand)
        cluster_to_brands[cluster_id] = brands
    cluster_to_brands, known_brands = propagate_brands_in_cluster(cluster_to_brands)
    cluster_to_titles = extract_titles(data)
    cluster_to_brands, known_brands = fill_unknown_brands_from_titles(
        cluster_to_brands, cluster_to_titles, known_brands
    )
    for cluster_id, offers in data.items():
        brands = cluster_to_brands[cluster_id]
        for offer, brand in zip(offers, brands):
            offer["brand"] = brand
    return data

CLEAN_MAP_MW = {
    "inch ": [
        "inches", "inch", "\"", "”", "'", "-inch"
    ],
    "hz ": [
        "hertz", "hz", "hz."
    ],
    "lb ": [
        "pounds", "pound", "lbs.", "lbs", "lb.", "lb"
    ],
    "kg ": [
        "kg.", "kg", "kgs", "kilograms", "kilogram", "kg)", "kg )", "Kg", "Kg.", "Kgs"
    ],
    "w ": [
        "watts", "watt", "w.", "w"
    ],
    "cd/m2 ": [
        "cd/mÂ²", "cd/m²", "cd/m2", "cd / m2", "cd / mÂ²", "cd / m²"
    ],
    "deg ": [
        "°", "º", "&#176;", "degrees", "degree"
    ],
}

CLEAN_MAP_TITLE_EXTRA = {
    " ": [
        "and", "or", "-", ",", "/", "&",
        "refurbished", "diagonal", "diag.",
        "best buy", "thenerds.net", "newegg.com",
    ]
}

CLEAN_MAP_TITLE = {**CLEAN_MAP_MW, **CLEAN_MAP_TITLE_EXTRA}

def build_regex_rules(clean_map):
    rules = []
    for replacement, permutations in clean_map.items():
        for perm in sorted(permutations, key=len, reverse=True):
            if perm.isalpha():
                pattern_str = r"\b" + re.escape(perm) + r"\b"
            else:
                pattern_str = re.escape(perm)
            pattern = re.compile(pattern_str, re.IGNORECASE)
            rules.append((pattern, replacement))
    return rules

MW_RULES = build_regex_rules(CLEAN_MAP_MW)
TITLE_RULES = build_regex_rules(CLEAN_MAP_TITLE)

def clean_string(string, rules):
    if not string:
        return ""
    s = string
    for pattern, replacement in rules:
        s = pattern.sub(replacement, s)
    s = re.sub(r"[^0-9a-zA-Z]+$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_data_optimized(data):
    cleaned_data_duplicates = OrderedDict()
    cleaned_data_non_duplicates = OrderedDict()
    for name, product in data.items():
        cleaned_product = []
        for product_per_shop in product:
            entry = dict(product_per_shop)
            title = (entry.get("title") or "").lower()
            entry["title"] = clean_string(title, TITLE_RULES)
            features_map = entry.get("featuresMap") or {}
            cleaned_features = {}
            for feature, value in features_map.items():
                v = (value or "").lower()
                cleaned_features[feature] = clean_string(v, MW_RULES)
            entry["featuresMap"] = cleaned_features
            cleaned_product.append(entry)
        if len(cleaned_product) > 1:
            cleaned_data_duplicates[name] = cleaned_product
        else:
            cleaned_data_non_duplicates[name] = cleaned_product
    return cleaned_data_duplicates, cleaned_data_non_duplicates

NUMERIC_VALUE_PATTERN = re.compile(
    r"\d+(?:\.\d+)?[a-zA-Z]+|\d+(?:\.\d+)?"
)

def extract_numeric_tokens_from_text(text: str):
    if not text:
        return []
    numeric_tokens = []
    for match in NUMERIC_VALUE_PATTERN.finditer(text):
        token = match.group(0)
        num_match = re.match(r"\d+(?:\.\d+)?", token)
        if num_match:
            numeric_tokens.append(num_match.group(0))
    return numeric_tokens

def extract_numeric_model_words_from_offer(offer: dict):
    features_map = offer.get("featuresMap") or {}
    numeric_set = set()
    for value in features_map.values():
        tokens = extract_numeric_tokens_from_text(value)
        numeric_set.update(tokens)
    return numeric_set

def keep_only_numeric_features(cleaned_data):
    def transform_dict(data_dict):
        for _, offers in data_dict.items():
            for offer in offers:
                nums = sorted(extract_numeric_model_words_from_offer(offer))
                offer["featuresMap"] = nums
    transform_dict(cleaned_data)
    return cleaned_data

def bootstrap_split_clusters(data, rng=None):
    if rng is None:
        rng = np.random.default_rng(123)
    cluster_ids = sorted(data.keys())
    N = len(cluster_ids)
    sample_indices = rng.integers(low=0, high=N, size=N, endpoint=False)
    bootstrap_ids = [cluster_ids[i] for i in sample_indices]
    train_ids = sorted(set(bootstrap_ids))
    test_ids = sorted(set(cluster_ids) - set(train_ids))
    assert set(train_ids).isdisjoint(set(test_ids))
    train_clusters = {cid: data[cid] for cid in train_ids}
    test_clusters = {cid: data[cid] for cid in test_ids}
    return train_clusters, test_clusters

def keep_only_features_brand_and_title(data):
    for _ , offers in data.items():
        for offer in offers:
            title = offer.get("title", "")
            features_map = offer.get("featuresMap", {})
            brand = offer.get("brand", "unknown")
            offer.clear()
            offer["title"] = title
            offer["featuresMap"] = features_map
            offer["brand"] = brand
    return data
# r"\b(?=\w*[a-zA-Z])(?=\w*\d)[a-zA-Z0-9-]+\b"
MODEL_WORD_PATTERN = re.compile(r"\b(?=\w*\d)[a-zA-Z0-9-]+\b")

def extract_title_model_words(title: str):
    if not isinstance(title, str):
        return set()
    tokens = MODEL_WORD_PATTERN.findall(title)
    return set(tokens)

def add_model_words_from_title(cleaned_data):
    for cluster_id, offers in cleaned_data.items():
        for offer in offers:
            title = offer.get("title", "")
            mw_set = extract_title_model_words(title)
            offer["modelWords_title"] = mw_set
    return cleaned_data

def group_offers_by_brand(data):
    brand_blocks = defaultdict(list)
    known_brands = set()
    for cluster_id, offers in data.items():
        for i, offer in enumerate(offers):
            brand = offer.get("brand", "unknown")
            if brand and brand != "unknown":
                known_brands.add(brand)
            offer_id = f"{cluster_id}#{i}"
            brand_blocks[brand].append((offer_id, offer))
    return brand_blocks, known_brands

def prepare_datasets(raw_data, seed=123):
    data = annotate_offers_with_brand(copy.deepcopy(raw_data))
    cleaned_dups, cleaned_single = clean_data_optimized(data)
    cleaned_all = {**cleaned_dups, **cleaned_single}
    rng = np.random.default_rng(seed)
    dups_train, dups_test = bootstrap_split_clusters(cleaned_dups, rng=rng)
    singles_train, singles_test = bootstrap_split_clusters(cleaned_single, rng=rng)
    data_train = {**dups_train, **singles_train}
    data_test = {**dups_test, **singles_test}
    data_train = keep_only_numeric_features(copy.deepcopy(data_train))
    data_test = keep_only_numeric_features(copy.deepcopy(data_test))
    data_train = keep_only_features_brand_and_title(data_train)
    data_test = keep_only_features_brand_and_title(data_test)
    data_train = add_model_words_from_title(data_train)
    data_test = add_model_words_from_title(data_test)
    brand_blocks_train, known_train = group_offers_by_brand(data_train)
    brand_blocks_test, known_test = group_offers_by_brand(data_test)
    known_brands = known_train | known_test
    return {
        "cleaned_all": cleaned_all,
        "data_train": data_train,
        "data_test": data_test,
        "brand_blocks_train": brand_blocks_train,
        "brand_blocks_test": brand_blocks_test,
        "known_brands": known_brands,
    }

# For testing
def run(data):
    data = annotate_offers_with_brand(copy.deepcopy(data))

    cleaned_dups, cleaned_single = clean_data_optimized(data)
    rng = np.random.default_rng(123)
    dups_train, dups_test = bootstrap_split_clusters(cleaned_dups, rng=rng)
    singles_train, singles_test = bootstrap_split_clusters(cleaned_single, rng=rng)

    train = {**dups_train, **singles_train}
    test  = {**dups_test,  **singles_test}

    brands_train, _ = group_offers_by_brand(train)
    brands_test, _  = group_offers_by_brand(test)

    return train, test, set(brands_train.keys()), set(brands_test.keys())

# Main guard
if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    with open(args.path) as f:
        data = json.load(f)

    train, test, brands_train, brands_test = run(data)
    print("Train clusters:", len(train))
    print("Test clusters:", len(test))