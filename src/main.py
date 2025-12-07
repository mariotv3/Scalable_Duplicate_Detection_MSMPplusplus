
import argparse
import json
from preprocessing.cleaning import prepare_datasets


def main(args):
    with open(args.path) as f:
        raw_data = json.load(f)

    datasets = prepare_datasets(raw_data, seed=42)

    train = datasets["data_train"]
    test = datasets["data_test"]
    brand_blocks_train = datasets["brand_blocks_train"]
    brand_blocks_test = datasets["brand_blocks_test"]
    known_brands = datasets["known_brands"]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/TVs-all-merged.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)