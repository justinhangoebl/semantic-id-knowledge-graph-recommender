import argparse
import csv
import json
import os

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert lfm_semids.pt to CSV with item_id and semantic_ids columns."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/ml1m_item_rqvae_semids.pt",
        help="Path to the .pt file containing semantic IDs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/ml1m_item_rqvae_semids.csv",
        help="Path to write the CSV output",
    )
    return parser.parse_args()


def to_list(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return list(value)


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    data = torch.load(args.input, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        semids = data["semantic_ids"]
        item_ids = data.get("item_ids", None)
    else:
        semids = data
        item_ids = None

    rows = to_list(semids)
    if item_ids is None:
        item_ids = list(range(1, len(rows) + 1))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["item_id", "semantic_ids"])
        for item_id, row in zip(item_ids, rows):
            writer.writerow([item_id, json.dumps(row)])

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
