import pandas as pd
import pickle

def fix_semanticids_item_ids(
    semanticids_path="./dataset/ml1m/ml1m.semanticids",
    link_path="./dataset/ml1m/ml1m.link",
    output_path="./dataset/ml1m/ml1m.semanticids_linked",
):
    """Rewrite semanticids to use entity IDs from link file.

    - item_id is mapped to entity_id using the link file.
    - rows without a link entry are skipped and counted.
    """
    df = pd.read_csv(semanticids_path, sep=",")
    links = pd.read_csv(link_path, sep="\t", header=0)
    link_map = dict(zip(links["item_id:token"], links["entity_id:token"]))

    rows = []
    skipped = 0
    for _, row in df.iterrows():
        item_id = int(row["item_id"])
        entity_id = link_map.get(item_id)
        if entity_id is None:
            skipped += 1
            continue
        rows.append({
            "item_id": int(entity_id),
            "semantic_ids": row["semantic_ids"],
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    print(
        f"Wrote {len(out_df)} rows to {output_path} (skipped {skipped} missing links)."
    )
    

def read_pd(build_tokenized=False):
    path = "./saved/KGGLM - ml1m - dataloaders/ml1m-for-KGGLM-finetune"
    with open(path, "rb") as f:
        obj = pickle.load(f)

    print("container type:", type(obj))
    print("num entries:", len(obj))

    for idx, (loader, payload) in enumerate(obj):
        print(f"\n== entry {idx} ==")
        print("loader type:", type(loader))
        print("payload dtype:", payload.dtype, "shape:", payload.shape)

        dataset = getattr(loader, "dataset", None)
        if dataset is not None:
            print("dataset type:", type(dataset))
            if hasattr(dataset, "__len__"):
                try:
                    print("dataset len:", len(dataset))
                except Exception as exc:
                    print("dataset len: <error>", repr(exc))
                    # Path-language datasets expose __len__ only after tokenization.
                    if hasattr(dataset, "_tokenized_dataset"):
                        is_ready = getattr(dataset, "_tokenized_dataset") is not None
                        print("tokenized dataset ready:", is_ready)

                    if hasattr(dataset, "inter_num"):
                        print("fallback inter_num:", getattr(dataset, "inter_num"))

                    if build_tokenized:
                        try:
                            if hasattr(dataset, "generate_user_path_dataset"):
                                dataset.generate_user_path_dataset()
                            if hasattr(dataset, "tokenize_path_dataset"):
                                dataset.tokenize_path_dataset()
                            print("dataset len (after build):", len(dataset))
                        except Exception as build_exc:
                            print("dataset len (after build): <error>", repr(build_exc))
            try:
                sample = dataset[0]
                print("dataset[0] type:", type(sample))
                print("dataset[0] preview:", sample)
            except Exception as exc:
                print("dataset[0]: <error>", repr(exc))

        try:
            it = iter(loader)
            batch = next(it)
            print("batch type:", type(batch))
            print("batch preview:", batch)
        except Exception as exc:
            print("batch: <error>", repr(exc))

def main():
    fix_semanticids_item_ids()

if __name__ == "__main__":
    main()

