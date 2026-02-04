"""
Training Script for Chess Mirror Bot

Trains the policy network on your chess games using PyTorch.
Supports MPS (Apple Silicon), CUDA, and CPU.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.network import ChessPolicyNet, ChessMirrorModel
from data.process_pgn import MoveEncoder  # Needed for pickle to load the encoder


class ChessDataset(Dataset):
    """PyTorch dataset for chess training examples."""
    
    def __init__(self, npz_path: str):
        """Load dataset from numpy compressed file."""
        data = np.load(npz_path)
        self.boards = torch.from_numpy(data['boards']).float()
        self.moves = torch.from_numpy(data['moves']).long()
        
        print(f"Loaded {len(self)} examples from {npz_path}")
    
    def __len__(self) -> int:
        return len(self.moves)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.boards[idx], self.moves[idx]


def train_model(
    dataset_path: str,
    move_encoder_path: str,
    output_dir: str,
    model_name: str = 'model',
    epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    num_filters: int = 128,
    num_residual_blocks: int = 6,
    patience: int = 10,
    device: str = 'auto'
):
    """
    Train a chess policy model.
    
    Args:
        dataset_path: Path to the .npz dataset file
        move_encoder_path: Path to the move_encoder.pkl
        output_dir: Directory to save checkpoints
        model_name: Name prefix for saved models
        epochs: Maximum training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        weight_decay: L2 regularization strength
        num_filters: Number of CNN filters
        num_residual_blocks: Number of residual blocks
        patience: Early stopping patience
        device: 'auto', 'mps', 'cuda', or 'cpu'
    """
    
    # Setup device
    if device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using Apple Silicon MPS acceleration")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print("Using CUDA GPU acceleration")
        else:
            device = torch.device('cpu')
            print("Using CPU (training will be slower)")
    else:
        device = torch.device(device)
    
    # Load move encoder to get vocabulary size
    with open(move_encoder_path, 'rb') as f:
        move_encoder = pickle.load(f)
    num_moves = move_encoder.vocab_size
    print(f"Move vocabulary size: {num_moves}")
    
    # Load dataset
    dataset = ChessDataset(dataset_path)
    
    if len(dataset) == 0:
        print("Error: Dataset is empty!")
        return
    
    # Split into train/val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # MPS doesn't play well with multiprocessing
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Create model
    model = ChessPolicyNet(
        num_moves=num_moves,
        num_filters=num_filters,
        num_residual_blocks=num_residual_blocks
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.NLLLoss()  # Since model outputs log_softmax
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_val_acc = 0.0
    epochs_without_improvement = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\n{'='*60}")
    print(f"Starting training: {model_name}")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (boards, targets) in enumerate(train_loader):
            boards = boards.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(boards)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * boards.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(targets).sum().item()
            train_total += targets.size(0)
            
            # Progress update
            if (batch_idx + 1) % 50 == 0:
                print(f"\rEpoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}", end='')
        
        train_loss /= train_total
        train_acc = 100.0 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for boards, targets in val_loader:
                boards = boards.to(device)
                targets = targets.to(device)
                
                outputs = model(boards)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * boards.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(targets).sum().item()
                val_total += targets.size(0)
        
        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\rEpoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'num_moves': num_moves,
                'num_filters': num_filters,
                'num_residual_blocks': num_residual_blocks,
                'epoch': epoch,
                'val_acc': val_acc,
                'train_acc': train_acc,
                'history': history
            }
            
            torch.save(checkpoint, output_path / f'{model_name}_best.pt')
            print(f"  → Saved new best model (val_acc: {val_acc:.2f}%)")
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'num_moves': num_moves,
        'num_filters': num_filters,
        'num_residual_blocks': num_residual_blocks,
        'epoch': epoch,
        'val_acc': val_acc,
        'train_acc': train_acc,
        'history': history
    }
    torch.save(checkpoint, output_path / f'{model_name}_final.pt')
    
    # Save training history
    with open(output_path / f'{model_name}_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Models saved to: {output_path}")
    print(f"{'='*60}")
    
    return model, history


def train_all_models(data_dir: str = './processed', output_dir: str = './checkpoints', **kwargs):
    """
    Train all three models: white, black, and combined.
    """
    data_path = Path(data_dir)
    move_encoder_path = data_path / 'move_encoder.pkl'
    
    if not move_encoder_path.exists():
        print(f"Error: Move encoder not found at {move_encoder_path}")
        print("Please run process_pgn.py first to process your PGN files.")
        return
    
    models_to_train = [
        ('white_dataset.npz', 'white_hybrid_dataset.npz', 'roy_white'),
        ('black_dataset.npz', 'black_hybrid_dataset.npz', 'roy_black'),
        ('combined_dataset.npz', 'combined_hybrid_dataset.npz', 'roy_combined')
    ]
    
    for dataset_file, hybrid_file, model_name in models_to_train:
        # Prefer hybrid dataset if it exists
        hybrid_path = data_path / hybrid_file
        regular_path = data_path / dataset_file
        
        if hybrid_path.exists():
            dataset_path = hybrid_path
            print(f"Using hybrid dataset: {hybrid_file}")
        elif regular_path.exists():
            dataset_path = regular_path
        else:
            print(f"Warning: No dataset found for {model_name}, skipping...")
            continue
        
        # Check if dataset has data
        data = np.load(dataset_path)
        if len(data['moves']) == 0:
            print(f"Warning: Dataset {dataset_path.name} is empty, skipping...")
            continue
        
        print(f"\n{'#'*60}")
        print(f"Training model: {model_name}")
        print(f"Dataset: {dataset_file}")
        print(f"{'#'*60}\n")
        
        train_model(
            dataset_path=str(dataset_path),
            move_encoder_path=str(move_encoder_path),
            output_dir=output_dir,
            model_name=model_name,
            **kwargs
        )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train chess mirror models')
    parser.add_argument('--data-dir', type=str, default='./processed',
                        help='Directory containing processed datasets')
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'white', 'black', 'combined'],
                        help='Which model(s) to train')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    args = parser.parse_args()
    
    if args.model == 'all':
        train_all_models(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience
        )
    else:
        # Check for hybrid dataset first, fall back to regular
        data_path = Path(args.data_dir)
        
        dataset_map = {
            'white': ('white_hybrid_dataset.npz', 'white_dataset.npz'),
            'black': ('black_hybrid_dataset.npz', 'black_dataset.npz'),
            'combined': ('combined_hybrid_dataset.npz', 'combined_dataset.npz')
        }
        
        hybrid_file, regular_file = dataset_map[args.model]
        
        if (data_path / hybrid_file).exists():
            dataset_file = hybrid_file
            print(f"Using hybrid dataset: {hybrid_file}")
        else:
            dataset_file = regular_file
            print(f"Using regular dataset: {regular_file}")
        
        train_model(
            dataset_path=str(data_path / dataset_file),
            move_encoder_path=str(data_path / 'move_encoder.pkl'),
            output_dir=args.output_dir,
            model_name=f'roy_{args.model}',
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience
        )
