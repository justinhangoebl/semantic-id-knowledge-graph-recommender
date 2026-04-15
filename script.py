import pandas as pd
import pickle
from tqdm import tqdm 


def test_sem_id_files():
    semantic_ids = (
        pd.read_csv("./dataset/ml1m/ml1m.semanticids", sep=",", index_col=0)['semantic_ids']
        .str.strip("\"[]\"")
        .str.split(", ")
        .apply(lambda x: [int(i) for i in x])
    ).to_dict()

    print("Semantic IDs loaded and processed.")
    
    return semantic_ids
    
import pickle
from tqdm import tqdm

def convert_KGGLM_to_SPRIG_paths():
    print("Converting KGGLM paths to SPRIG paths...")
    
    # Load your source paths
    try:
        with open("./paths/KGGLM-ml-1m.pkl", "rb") as f:
            kgglm_paths = pickle.load(f)
        print(f"KGGLM paths loaded. Found {len(kgglm_paths)} paths.")
    except FileNotFoundError:
        print("Error: Source pkl file not found.")
        return

    # Assuming test_sem_id_files() returns a dict or mapping if needed later
    # For now, we use the explicit mapping provided: I109 -> [SEM12, SEM54, SEM64]
    target_mapping = test_sem_id_files()  # Load semantic IDs if needed for mapping
    
    sprig_paths = []
    errors = 0
    
    for path in tqdm(kgglm_paths, desc=f"Converting paths {errors}"):
        new_path = ""
        for token in path.split(" "):
            # Check if token needs conversion
            if token.startswith("I"):
                sem = target_mapping.get(int(token[1:]), [])
                if sem:
                    new_path += " " + " ".join(f"SEM{sem_id}" for sem_id in sem)
                else:
                    new_path += " " + token  # If no mapping, keep original
                    errors += 1
            else:
                new_path += " " + token
        
        sprig_paths.append(new_path[1:])

    # Preview the conversion (using your example data logic)
    print("\nSample Conversion:")
    print(f"Original: {kgglm_paths[32] if kgglm_paths else 'N/A'}")
    print(f"Converted: {sprig_paths[32] if sprig_paths else 'N/A'}")

    print(f"Conversion completed with {errors} unmapped tokens.")
    # Save the result
    with open("./paths/SPRIG-ml-1m.pkl", "wb") as f:
        pickle.dump(sprig_paths, f)
    print("Conversion complete and saved.")

# Note: Ensure test_sem_id_files is defined in your environment
import os

def look_pkl():
    cache_file = (
        f"./paths/SPRIG - ml-1m_sprig_pretrain_raw_(5, 5).pkl"
    )

    if os.path.exists(cache_file):
        print(
            "Loading cached SPRIG pretrain raw paths from %s …", cache_file
        )
        with open(cache_file, "rb") as fh:
            raw_paths = pickle.load(fh)  # list of numpy path arrays
    for x in raw_paths:
        print(x)
        break

def transform_spaces():
    smeantic_ids = test_sem_id_files()
    links = pd.read_csv("./dataset/ml1m/ml1m.link", sep="\t", header=0)
    lsit = []
    for item_id, sem_ids in smeantic_ids.items():
        print(f"Processing item_id: {item_id} with semantic IDs: {sem_ids}")
        
        # Find the corresponding entity_id for this item_id
        entity_id = links.loc[links['item_id:token'] == item_id, 'entity_id:token'].values
        if len(entity_id) == 0:
            print(f"Warning: No entity_id found for item_id {item_id}. Skipping.")
            continue
        entity_id = entity_id[0]  # Assuming one-to-one mapping
        print(f"Found entity_id: {entity_id} for item_id: {item_id}")
        # Create the new format string
        lsit.append({
            "item_id": entity_id,
            "semantic_ids": sem_ids
        })
        
    df = pd.DataFrame(lsit)
    df.to_csv("./dataset/ml1m/ml1m.semanticids", index=False)
    

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
            
            
            
            
            
            
            
            

def convert_KGGLM_to_SPRIG_paths():
    print("Converting KGGLM paths to SPRIG paths...")
    
    # Load your source paths
    try:
        with open("./paths/OLD/KGGLM - pre - ml-1m small official_pretrain_paths_(5, 5).pkl", "rb") as f:
            kgglm_paths = pickle.load(f)
        print(f"KGGLM paths loaded. Found {len(kgglm_paths)} paths.")
    except FileNotFoundError:
        print("Error: Source pkl file not found.")
        return

    # Assuming test_sem_id_files() returns a dict or mapping if needed later
    # For now, we use the explicit mapping provided: I109 -> [SEM12, SEM54, SEM64]
    target_mapping = test_sem_id_files()  # Load semantic IDs if needed for mapping
    
    sprig_paths = []
    errors = 0
    
    for path in tqdm(kgglm_paths, desc=f"Converting paths {errors}"):
        new_path = ""
        for token in path.split(" "):
            # Check if token needs conversion
            if token.startswith("I"):
                sem = target_mapping.get(int(token[1:]), [])
                if sem:
                    new_path += " " + " ".join(f"SEM{sem_id}" for sem_id in sem)
                else:
                    new_path += " " + token  # If no mapping, keep original
                    errors += 1
            else:
                new_path += " " + token
        
        sprig_paths.append(new_path[1:])

    # Preview the conversion (using your example data logic)
    print("\nSample Conversion:")
    print(f"Original: {kgglm_paths[32] if kgglm_paths else 'N/A'}")
    print(f"Converted: {sprig_paths[32] if sprig_paths else 'N/A'}")

    print(f"Conversion completed with {errors} unmapped tokens.")
    # Save the result
    with open("./paths/SPRIG-ml-1m.pkl", "wb") as f:
        pickle.dump(sprig_paths, f)
    print("Conversion complete and saved.")

def main():
    # Set build_tokenized=True only if you want to force path generation/tokenization.
    convert_KGGLM_to_SPRIG_paths()
if __name__ == "__main__":
    main()

