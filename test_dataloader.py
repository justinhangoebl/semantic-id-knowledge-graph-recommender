"""
Script to inspect the ml-100k-for-KGGLM-pretrain dataloader
"""

import pickle
import os
import warnings

# Path to the saved dataloader
dataloader_path = "/home/justin-hangoebl/master-thesis/unica/hopwise/saved/KGGLM - ml-100k - dataloaders/ml-100k-for-KGGLM-pretrain"

print(f"Loading dataloader from: {dataloader_path}")
print(f"File exists: {os.path.exists(dataloader_path)}")


def _guess_split_name(idx, total):
    """Best-effort name for each saved dataloader split.

    For standard recommendation tasks we usually have
    (train, valid, test) -> total == 3.

    For KG + interaction evaluation we may have
    (train, valid_inter, valid_kg, test_inter, test_kg) -> total == 5.
    """

    if total == 3:
        return ["train", "valid", "test"][idx]
    if total == 5:
        return ["train", "valid_inter", "valid_kg", "test_inter", "test_kg"][idx]
    return None

if os.path.exists(dataloader_path):
    print(f"File size: {os.path.getsize(dataloader_path) / (1024*1024):.2f} MB")
    
    with open(dataloader_path, "rb") as f:
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            dataloaders_data = pickle.load(f)
    
    num_dataloaders = len(dataloaders_data)
    print(f"\nNumber of dataloaders: {num_dataloaders}")

    if num_dataloaders == 3:
        print("Split pattern guess: [0]=train, [1]=valid, [2]=test")
    elif num_dataloaders == 5:
        print(
            "Split pattern guess: [0]=train, [1]=valid_inter, [2]=valid_kg, "
            "[3]=test_inter, [4]=test_kg"
        )
    
    for i, dataloader_saved_data in enumerate(dataloaders_data):
        print(f"\n{'='*60}")
        split_name = _guess_split_name(i, num_dataloaders)
        if split_name is not None:
            print(f"Dataloader {i+1} (split: {split_name}):")
        else:
            print(f"Dataloader {i+1}:")
        print(f"{'='*60}")
        
        print(f"Type of saved data: {type(dataloader_saved_data)}")
        print(f"Length of saved data: {len(dataloader_saved_data)}")
        
        # First element is the dataloader itself
        dataloader = dataloader_saved_data[0]
        print(f"\nDataloader type: {type(dataloader).__name__}")
        print(f"Dataloader class: {dataloader.__class__}")
        
        # Check if it has common dataloader attributes
        if hasattr(dataloader, 'dataset'):
            dataset = dataloader.dataset
            print(f"Dataset type: {type(dataset).__name__}")
            try:
                print(f"Dataset length: {len(dataset)}")
            except ValueError as e:
                print(f"Dataset length: Cannot determine - {e}")

            # Try to expose split information if available on the dataset itself
            if hasattr(dataset, 'config') and 'eval_args' in dataset.config:
                split_args = dataset.config['eval_args'].get('split', None)
                print(f"Eval split args (from dataset.config): {split_args}")

            # Check for KGGLMDataset / KnowledgePathDataset specific attributes
            if hasattr(dataset, 'inter_feat'):
                print(f"Inter feat length: {len(dataset.inter_feat)}")
            if hasattr(dataset, 'kg_feat'):
                print(f"KG feat present: Yes")
            if hasattr(dataset, '_tokenized_path_dataset'):
                print(f"Tokenized path dataset present: {dataset._tokenized_path_dataset is not None}")
            if hasattr(dataset, 'field2token_id'):
                print(f"Field2token_id keys: {list(dataset.field2token_id.keys())[:5]}...")

            # Try to inspect pretraining paths if this is a path dataset
            if hasattr(dataset, '_path_dataset'):
                print("\n--- Inspecting pretraining paths (path_dataset) ---")
                path_string = None
                try:
                    # Preferred: use already-generated path dataset
                    path_string = dataset.path_dataset
                except Exception as e:
                    print(f"path_dataset not yet generated: {e}")
                    # As a fallback, try to (re)generate the path dataset
                    if hasattr(dataset, 'generate_user_path_dataset'):
                        try:
                            print("Generating path dataset via generate_user_path_dataset() ...")
                            dataset.generate_user_path_dataset()
                            path_string = dataset.path_dataset
                        except Exception as ee:
                            print(f"Failed to generate path dataset: {ee}")

                if isinstance(path_string, str):
                    # Each line corresponds to one path, already formatted as tokens
                    paths = [p for p in path_string.strip().split("\n") if p]
                    print(f"Total number of paths (lines in path_dataset): {len(paths)}")
                    sample_n = min(10, len(paths))
                    print(f"\nSample of {sample_n} pretraining paths:")
                    for j, p in enumerate(paths[:sample_n]):
                        print(f"  Path {j+1}: {p}")
                else:
                    if path_string is not None:
                        print(f"path_dataset type: {type(path_string)} (not a string; custom handling may be needed)")
        
        if hasattr(dataloader, 'config'):
            print(f"\nConfig keys: {list(dataloader.config.final_config_dict.keys())[:10]}...")
            print(f"Model: {dataloader.config['model']}")
            print(f"Dataset: {dataloader.config['dataset']}")
            print(f"Train stage: {dataloader.config['train_stage']}")
        
        if hasattr(dataloader, 'batch_size'):
            print(f"Batch size: {dataloader.batch_size}")
        
        if hasattr(dataloader, 'sampler'):
            print(f"Sampler type: {type(dataloader.sampler).__name__}")
            # For many dataloaders, sampler.phase encodes whether this is
            # train/valid/test split.
            phase = getattr(dataloader.sampler, 'phase', None)
            if phase is not None:
                print(f"Sampler phase: {phase}")
        
        # Check for generator state (second element)
        if len(dataloader_saved_data) > 1:
            print(f"\nGenerator state included: Yes")
            if len(dataloader_saved_data) == 3:
                print("This is a KnowledgeBasedDataLoader (has both general and kg generator states)")
        
        # Try to inspect the first batch
        try:
            print(f"\n{'='*40}")
            print("Inspecting first batch:")
            print(f"{'='*40}")
            
            # Restore generator if needed
            if hasattr(dataloader, 'generator') and dataloader.generator is None:
                import torch
                if len(dataloader_saved_data) >= 2:
                    generator = torch.Generator()
                    generator.set_state(dataloader_saved_data[1])
                    dataloader.generator = generator
                    if hasattr(dataloader, 'sampler'):
                        dataloader.sampler.generator = generator
            
            for batch in dataloader:
                print(f"Batch type: {type(batch)}")
                if isinstance(batch, dict):
                    print(f"Batch keys: {batch.keys()}")
                    for key, value in batch.items():
                        if hasattr(value, 'shape'):
                            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                        else:
                            print(f"  {key}: {type(value)}")
                elif isinstance(batch, (list, tuple)):
                    print(f"Batch length: {len(batch)}")
                    for j, item in enumerate(batch):
                        if hasattr(item, 'shape'):
                            print(f"  Item {j}: shape={item.shape}, dtype={item.dtype}")
                        else:
                            print(f"  Item {j}: {type(item)}")
                else:
                    print(f"Batch content: {batch}")
                
                # Only show first batch
                break
                
        except Exception as e:
            print(f"Error inspecting batch: {e}")
            import traceback
            traceback.print_exc()

else:
    print("Dataloader file not found!")
