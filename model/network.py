"""
Chess Mirror Neural Network Model

A CNN-based policy network that learns to predict moves from board positions.
Architecture inspired by AlphaGo/AlphaZero but scaled down for our smaller dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers.
    Uses batch normalization and skip connections.
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual  # Skip connection
        x = F.relu(x)
        return x


class ChessPolicyNet(nn.Module):
    """
    Policy network that predicts move probabilities from board positions.
    
    Architecture:
    1. Input convolution: 14 planes → 256 channels
    2. Residual tower: 10 residual blocks
    3. Policy head: outputs logits for all possible moves
    
    This architecture is sized for ~65k training examples.
    """
    
    def __init__(self, 
                 input_channels: int = 14,
                 num_filters: int = 128,
                 num_residual_blocks: int = 6,
                 num_moves: int = 4672):  # Will be set from move_encoder
        super().__init__()
        
        self.num_moves = num_moves
        
        # Initial convolution
        self.input_conv = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_filters)
        
        # Residual tower
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_residual_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, num_moves)
    
    def forward(self, x: torch.Tensor, legal_move_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Board tensor of shape (batch, 14, 8, 8)
            legal_move_mask: Optional boolean mask of shape (batch, num_moves)
                            True for legal moves, False for illegal
        
        Returns:
            Log probabilities over moves (batch, num_moves)
        """
        # Input convolution
        x = F.relu(self.input_bn(self.input_conv(x)))
        
        # Residual tower
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head
        x = F.relu(self.policy_bn(self.policy_conv(x)))
        x = x.view(x.size(0), -1)  # Flatten
        logits = self.policy_fc(x)
        
        # Mask illegal moves if provided
        if legal_move_mask is not None:
            # Set illegal moves to very negative value before softmax
            logits = logits.masked_fill(~legal_move_mask, float('-inf'))
        
        # Return log probabilities
        return F.log_softmax(logits, dim=1)
    
    def predict_move(self, x: torch.Tensor, legal_move_mask: torch.Tensor, 
                     temperature: float = 1.0) -> torch.Tensor:
        """
        Predict a move (for inference).
        
        Args:
            x: Board tensor
            legal_move_mask: Boolean mask of legal moves
            temperature: Sampling temperature (lower = more deterministic)
        
        Returns:
            Selected move index
        """
        with torch.no_grad():
            log_probs = self.forward(x, legal_move_mask)
            
            if temperature == 0:
                # Greedy selection
                return log_probs.argmax(dim=1)
            else:
                # Sample from distribution
                probs = (log_probs / temperature).exp()
                probs = probs / probs.sum(dim=1, keepdim=True)  # Renormalize
                return torch.multinomial(probs, 1).squeeze(1)


class ChessMirrorModel:
    """
    Wrapper class that handles model loading, saving, and inference.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        # Auto-detect device
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.model: Optional[ChessPolicyNet] = None
        
        if model_path:
            self.load(model_path)
    
    def create_model(self, num_moves: int, **kwargs):
        """Create a new model with the specified vocabulary size."""
        self.model = ChessPolicyNet(num_moves=num_moves, **kwargs).to(self.device)
        return self.model
    
    def load(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Recreate model architecture
        self.model = ChessPolicyNet(
            num_moves=checkpoint['num_moves'],
            num_filters=checkpoint.get('num_filters', 128),
            num_residual_blocks=checkpoint.get('num_residual_blocks', 6)
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model from {path}")
        return self.model
    
    def save(self, path: str, **extra_info):
        """Save model checkpoint."""
        # Get actual config from model
        num_filters = self.model.input_conv.out_channels
        num_residual_blocks = len(self.model.residual_blocks)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'num_moves': self.model.num_moves,
            'num_filters': num_filters,
            'num_residual_blocks': num_residual_blocks,
            **extra_info
        }
        torch.save(checkpoint, path)
        print(f"Saved model to {path}")


if __name__ == '__main__':
    # Quick test
    print("Testing model architecture...")
    
    model = ChessPolicyNet(num_moves=4672)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch = torch.randn(4, 14, 8, 8)
    output = model(batch)
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test with legal move mask
    mask = torch.zeros(4, 4672, dtype=torch.bool)
    mask[:, :20] = True  # Pretend first 20 moves are legal
    output_masked = model(batch, mask)
    print(f"Masked output shape: {output_masked.shape}")
    print(f"Non-inf values per sample: {(output_masked > float('-inf')).sum(dim=1)}")
    
    print("\nModel test passed!")
