import pandas as pd
import pickle
from tqdm import tqdm 


def test_sem_id_files():
    semantic_ids = (
        pd.read_csv("./dataset/ml-1m/ml-1m.semanticids", sep=",", index_col=0)['semantic_ids']
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

def main():
    convert_KGGLM_to_SPRIG_paths()

if __name__ == "__main__":
    main()

