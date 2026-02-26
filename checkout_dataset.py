import torch
import sys
import pickle

path = "saved/ml-1m-small-KGGLMDataset.pth"

with open(path, "rb") as f:
    dataset = pickle.load(f)

print("=== Dataset Type ===")
print(type(dataset))

print("\n=== inter_feat (interactions) - first 5 rows ===")
print(dataset.inter_feat.head())

print("\n=== kg_feat (knowledge graph) - first 5 rows ===")
if dataset.kg_feat is not None:
    print(dataset.kg_feat.head())
else:
    print("None")

print("\n=== Tokenizer vocab (first 20 tokens) ===")
vocab = dataset.tokenizer.get_vocab()
for token, idx in sorted(vocab.items(), key=lambda x: x[1])[:20]:
    print(f"  {idx}: {token}")

print("\n=== item2entity (first 5) ===")
for k, v in list(dataset.item2entity.items())[:5]:
    print(f"  item {k} -> entity {v}")

print("\n=== Path dataset ===")
if dataset._path_dataset is not None:
    print(f"  Type: {type(dataset._path_dataset)}")
    print(f"  First 3 entries:")
    for i, entry in enumerate(dataset._path_dataset[:3]):
        print(f"    {entry}")
else:
    print("  None (paths not yet generated)")

print("\n=== Tokenized dataset ===")
if dataset._tokenized_dataset is not None:
    print(f"  Type: {type(dataset._tokenized_dataset)}")
    print(f"  First entry: {dataset._tokenized_dataset[0]}")
else:
    print("  None (not yet tokenized)")

print("\n=== Key config values ===")
print(f"  path_hop_length: {dataset.path_hop_length}")
print(f"  max_paths_per_user: {dataset.max_paths_per_user}")
print(f"  context_length: {dataset.context_length}")
print(f"  n_users: {dataset.user_num}")
print(f"  n_items: {dataset.item_num}")