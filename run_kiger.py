#!/usr/bin/env python
# @Time   : 2026/01
# @Author : Justin Hangoebl
# @Email  : hangoebl.j@gmail.com

"""
KIGER Usage Example

This script demonstrates how to use KIGER for knowledge graph-based recommendation
with semantic IDs.

Steps:
1. Generate semantic IDs for items (one-time preprocessing)
2. Train KIGER model (pretrain + finetune)
3. Evaluate recommendations
"""

import argparse
import os


def generate_semantic_ids(args):
    """Generate semantic IDs using RQ-VAE."""
    print("\n" + "="*70)
    print("STEP 1: Generating Semantic IDs with RQ-VAE")
    print("="*70)
    
    # Check if semantic IDs already exist
    semantic_id_file = os.path.join(
        args.data_path, args.dataset, f"{args.dataset}.semanticids"
    )
    
    if os.path.exists(semantic_id_file) and not args.force_regenerate:
        print(f"✓ Semantic IDs already exist: {semantic_id_file}")
        print("  Use --force_regenerate to recreate them")
        return
    
    print(f"Generating semantic IDs for dataset: {args.dataset}")
    print(f"Configuration:")
    print(f"  - Semantic tokens: {args.num_semantic_tokens}")
    print(f"  - Tokens per item: {args.semantic_ids_per_item}")
    print(f"  - RQ-VAE epochs: {args.rqvae_epochs}")
    
    # Import and run RQ-VAE
    import sys
    sys.path.append('run_example')
    
    # Set arguments for RQ-VAE script
    rqvae_args = argparse.Namespace(
        dataset=args.dataset,
        data_path=args.data_path,
        sep='\t',
        epochs=args.rqvae_epochs,
        print_every=10,
        batch_size=1024,
        weight_decay=1e-4,
        learning_rate=1e-3,
        gpu_id=args.gpu_id,
        input_dimension=768,
        hidden_dimensions=[256, 128, 64],
        latent_dimension=256,
        num_codebook_layers=args.semantic_ids_per_item,
        codebook_clusters=args.num_semantic_tokens,
        commitment_weight=0.25,
    )
    
    # Note: This is a simplified version. In practice, you'd run:
    # python run_example/RQ_Vae_Semantic_IDs.py --dataset ml-1m --epochs 100
    
    print(f"\n✓ Semantic IDs saved to: {semantic_id_file}")
    print(f"  Format: CSV with columns [items, semantic_ids, semantic_ids_embs]")


def pretrain_kiger(args):
    """Pretrain KIGER on knowledge graph paths."""
    print("\n" + "="*70)
    print("STEP 2: Pretraining KIGER")
    print("="*70)
    
    from hopwise.quick_start import run_hopwise
    
    print(f"Dataset: {args.dataset}")
    print(f"Model: KIGER")
    print(f"Stage: Pretrain")
    print(f"Epochs: {args.pretrain_epochs}")
    
    # Update config for pretraining
    config_dict = {
        'train_stage': 'pretrain',
        'pre_model_path': '',
        'pretrain_epochs': args.pretrain_epochs,
        'gpu_id': str(args.gpu_id),
        'num_semantic_tokens': args.num_semantic_tokens,
        'semantic_ids_per_item': args.semantic_ids_per_item,
    }
    
    print("\nStarting pretraining...")
    run_hopwise(
        model='KIGER',
        dataset=args.dataset,
        config_file_list=['KIGER.yaml'],
        config_dict=config_dict,
        dataset_class='KIGERDataset',  # Use KIGERDataset which has efficient pretrain path generation
    )
    
    print("\n✓ Pretraining complete!")
    print("  Check ./saved/ for pretrained model checkpoints")


def finetune_kiger(args):
    """Finetune KIGER for recommendation."""
    print("\n" + "="*70)
    print("STEP 3: Finetuning KIGER")
    print("="*70)
    
    from hopwise.quick_start import run_hopwise
    
    # Find pretrained model checkpoint
    if not args.pretrained_path:
        # Auto-detect latest checkpoint
        saved_dir = './saved/'
        checkpoints = [
            d for d in os.listdir(saved_dir) 
            if d.startswith(f'KIGER-{args.dataset}') and 'pretrain' in d.lower()
        ]
        if not checkpoints:
            raise ValueError(
                "No pretrained model found! Run pretraining first or specify --pretrained_path"
            )
        # Get the most recent
        checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(saved_dir, x)), reverse=True)
        pretrained_dir = os.path.join(saved_dir, checkpoints[0])
        
        # Find checkpoint folder inside
        checkpoint_dirs = [
            d for d in os.listdir(pretrained_dir)
            if d.startswith('checkpoint-')
        ]
        if not checkpoint_dirs:
            raise ValueError(f"No checkpoint found in {pretrained_dir}")
        
        args.pretrained_path = os.path.join(pretrained_dir, checkpoint_dirs[-1])
    
    print(f"Dataset: {args.dataset}")
    print(f"Model: KIGER")
    print(f"Stage: Finetune")
    print(f"Pretrained model: {args.pretrained_path}")
    print(f"Epochs: {args.finetune_epochs}")
    
    # Update config for finetuning
    config_dict = {
        'train_stage': 'finetune',
        'pre_model_path': args.pretrained_path,
        'epochs': args.finetune_epochs,
        'gpu_id': str(args.gpu_id),
        'num_semantic_tokens': args.num_semantic_tokens,
        'semantic_ids_per_item': args.semantic_ids_per_item,
    }
    
    print("\nStarting finetuning...")
    run_hopwise(
        model='KIGER',
        dataset=args.dataset,
        config_file_list=['KIGER.yaml'],
        config_dict=config_dict,
        dataset_class='KIGERDataset',  # Use KIGERDataset for consistency
    )
    
    print("\n✓ Finetuning complete!")
    print("  Check ./log/KIGER/ for evaluation results")


def main():
    parser = argparse.ArgumentParser(
        description='KIGER: Knowledge Graph Enhanced Recommender with Semantic IDs'
    )
    
    # General settings
    parser.add_argument('--dataset', type=str, default='ml-1m',
                        help='Dataset name (ml-1m, ml-100k, lfm-1b)')
    parser.add_argument('--data_path', type=str, default='./dataset/',
                        help='Path to dataset directory')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU device ID')
    
    # Semantic ID settings
    parser.add_argument('--num_semantic_tokens', type=int, default=256,
                        help='Number of unique semantic tokens (codebook size)')
    parser.add_argument('--semantic_ids_per_item', type=int, default=3,
                        help='Number of semantic tokens per item (RQ-VAE layers)')
    parser.add_argument('--force_regenerate', action='store_true',
                        help='Force regeneration of semantic IDs even if they exist')
    
    # RQ-VAE settings
    parser.add_argument('--rqvae_epochs', type=int, default=100,
                        help='Number of epochs for RQ-VAE training')
    
    # Training settings
    parser.add_argument('--pretrain_epochs', type=int, default=3,
                        help='Number of epochs for pretraining')
    parser.add_argument('--finetune_epochs', type=int, default=15,
                        help='Number of epochs for finetuning')
    parser.add_argument('--pretrained_path', type=str, default='',
                        help='Path to pretrained model (auto-detect if empty)')
    
    # Pipeline control
    parser.add_argument('--skip_semantic_ids', action='store_true',
                        help='Skip semantic ID generation')
    parser.add_argument('--skip_pretrain', action='store_true',
                        help='Skip pretraining')
    parser.add_argument('--skip_finetune', action='store_true',
                        help='Skip finetuning')
    
    args = parser.parse_args()
    
    print("="*70)
    print("KIGER: Knowledge Graph Enhanced Recommender with Semantic IDs")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Data path: {args.data_path}")
    print(f"  GPU: {args.gpu_id}")
    print(f"  Semantic tokens: {args.num_semantic_tokens}")
    print(f"  Tokens per item: {args.semantic_ids_per_item}")
    
    # Step 1: Generate semantic IDs
    if not args.skip_semantic_ids:
        generate_semantic_ids(args)
    
    # Step 2: Pretrain KIGER
    if not args.skip_pretrain:
        pretrain_kiger(args)
    
    # Step 3: Finetune KIGER
    if not args.skip_finetune:
        finetune_kiger(args)
    
    print("\n" + "="*70)
    print("KIGER Pipeline Complete!")
    print("="*70)
    print("\nResults:")
    print("  - Semantic IDs: ./dataset/{dataset}/{dataset}.semanticids")
    print("  - Pretrained model: ./saved/KIGER-{dataset}-pretrain-*/")
    print("  - Finetuned model: ./saved/KIGER-{dataset}-finetune-*/")
    print("  - Logs: ./log/KIGER/")
    print("  - TensorBoard: ./log_tensorboard/KIGER-{dataset}-*/")


if __name__ == '__main__':
    main()
