import pickle
import warnings
import os

# Path to your saved dataloader
dataloader_path = "/home/justin-hangoebl/master-thesis/unica/hopwise/saved/KGGLM - ml-1m - dataloaders/ml-1m-for-KGGLM-pretrain"

# Output pickle path
output_pickle = "ml1m_all_paths.pkl"

# Load the dataloaders
with open(dataloader_path, "rb") as f:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataloaders = pickle.load(f)

all_paths = []

# --- 1. Train split (global pretraining paths) ---
train_loader = dataloaders[0][0]
train_dataset = train_loader.dataset

train_paths = train_dataset.path_dataset.strip().split("\n")
print(f"Train paths: {len(train_paths)}")
all_paths.extend(train_paths)

# --- 2. Validation split (dynamic paths) ---
valid_loader = dataloaders[1][0]

valid_count = 0
for batch in valid_loader:
    interaction, path_tuple, head_ids, tail_ids = batch
    # path_tuple is usually a list/tuple of paths for this interaction
    # Convert each to string if needed
    if isinstance(path_tuple, (list, tuple)):
        for p in path_tuple:
            if isinstance(p, str):
                all_paths.append(p)
            else:
                # convert token ids to string if stored as numbers
                all_paths.append(" ".join(map(str, p)))
    else:
        all_paths.append(str(path_tuple))
    valid_count += 1
print(f"Validation interactions processed: {valid_count}")

# --- 3. Test split (dynamic paths) ---
test_loader = dataloaders[2][0]

test_count = 0
for batch in test_loader:
    interaction, path_tuple, head_ids, tail_ids = batch
    if isinstance(path_tuple, (list, tuple)):
        for p in path_tuple:
            if isinstance(p, str):
                all_paths.append(p)
            else:
                all_paths.append(" ".join(map(str, p)))
    else:
        all_paths.append(str(path_tuple))
    test_count += 1
print(f"Test interactions processed: {test_count}")

# --- Save all paths to a pickle file ---
with open(output_pickle, "wb") as f:
    pickle.dump(all_paths, f)

print(f"Saved all paths ({len(all_paths)}) to {output_pickle}")