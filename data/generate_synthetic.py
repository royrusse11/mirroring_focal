"""
Synthetic Data Generator for Chess Mirror Bot

Generates additional training data for middlegame/endgame positions
while preserving your authentic opening moves.

Strategy:
- Moves 1-12: Keep YOUR moves only (your opening repertoire)
- Moves 13+: Add synthetic moves from Stockfish at limited strength
  to simulate human-like play and fill coverage gaps
"""

import chess
import chess.engine
import chess.pgn
import numpy as np
import pickle
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.process_pgn import BoardEncoder, MoveEncoder, TrainingExample


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    opening_cutoff: int = 12  # Moves 1-12 are "opening"
    stockfish_depth: int = 8  # Lower depth = more human-like mistakes
    stockfish_time: float = 0.05  # Time limit per move
    num_variations_per_position: int = 3  # How many alternate games to generate
    randomize_top_n: int = 3  # Pick randomly from top N moves (adds variety)
    min_move_for_synthetic: int = 10  # Don't generate synthetic before this move
    

class SyntheticDataGenerator:
    """Generates synthetic training data using Stockfish."""
    
    def __init__(self, 
                 stockfish_path: str = '/opt/homebrew/bin/stockfish',
                 config: Optional[SyntheticConfig] = None):
        self.stockfish_path = stockfish_path
        self.config = config or SyntheticConfig()
        self.board_encoder = BoardEncoder()
        self._engine = None
        
    @property
    def engine(self) -> chess.engine.SimpleEngine:
        """Lazy load Stockfish."""
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            # Configure for more human-like play
            self._engine.configure({
                "Skill Level": 10,  # Roughly 1500 ELO
                "UCI_LimitStrength": True,
                "UCI_Elo": 1400  # Target around your level
            })
        return self._engine
    
    def close(self):
        """Close Stockfish engine."""
        if self._engine:
            self._engine.quit()
            self._engine = None
    
    def get_human_like_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Get a human-like move using Stockfish with randomization.
        
        Instead of always picking the best move, we:
        1. Get top N moves from Stockfish
        2. Weight them by score
        3. Randomly select (favoring better moves but not always best)
        """
        try:
            # Get multiple moves with scores
            result = self.engine.analyse(
                board, 
                chess.engine.Limit(
                    depth=self.config.stockfish_depth,
                    time=self.config.stockfish_time
                ),
                multipv=self.config.randomize_top_n
            )
            
            if not result:
                return None
            
            # Handle single result vs multiple
            if isinstance(result, dict):
                result = [result]
            
            moves_with_scores = []
            for info in result:
                if 'pv' in info and info['pv']:
                    move = info['pv'][0]
                    score = info.get('score')
                    if score:
                        # Convert to centipawns
                        if score.is_mate():
                            cp = 10000 if score.relative.mate() > 0 else -10000
                        else:
                            cp = score.relative.score()
                        moves_with_scores.append((move, cp))
            
            if not moves_with_scores:
                return None
            
            # If only one move, return it
            if len(moves_with_scores) == 1:
                return moves_with_scores[0][0]
            
            # Weight selection towards better moves but with randomness
            # Convert scores to weights (higher score = higher weight)
            min_score = min(s for _, s in moves_with_scores)
            weights = []
            for move, score in moves_with_scores:
                # Shift scores to be positive and add base weight
                weight = (score - min_score + 100) ** 1.5  # Exponential favors better moves
                weights.append(weight)
            
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]
            
            # Random selection based on weights
            selected = random.choices(moves_with_scores, weights=weights, k=1)[0]
            return selected[0]
            
        except Exception as e:
            print(f"Error getting move: {e}")
            return None
    
    def generate_synthetic_game(self, 
                                 starting_board: chess.Board,
                                 perspective_color: chess.Color,
                                 num_moves: int = 30) -> List[TrainingExample]:
        """
        Generate a synthetic game continuation from a position.
        
        Returns training examples for moves made by perspective_color.
        """
        examples = []
        board = starting_board.copy()
        
        for _ in range(num_moves):
            if board.is_game_over():
                break
            
            move = self.get_human_like_move(board)
            if move is None:
                break
            
            # If it's our turn, record this as a training example
            if board.turn == perspective_color:
                try:
                    board_tensor = self.board_encoder.encode(board, perspective_color)
                    
                    example = TrainingExample(
                        board_tensor=board_tensor,
                        move_index=-1,  # Will be set later with move encoder
                        move_uci=move.uci(),
                        game_id="synthetic",
                        move_number=board.fullmove_number,
                        color='white' if perspective_color == chess.WHITE else 'black'
                    )
                    examples.append(example)
                except Exception as e:
                    print(f"Error creating example: {e}")
            
            board.push(move)
        
        return examples
    
    def extract_middlegame_positions(self, pgn_dir: Path, username: str) -> List[Tuple[chess.Board, chess.Color]]:
        """
        Extract middlegame positions from your actual games.
        These will be starting points for synthetic continuations.
        """
        positions = []
        username_lower = username.lower()
        
        pgn_files = list(pgn_dir.glob('*.pgn'))
        print(f"Scanning {len(pgn_files)} PGN files for middlegame positions...")
        
        for pgn_file in tqdm(pgn_files, desc="Extracting positions"):
            with open(pgn_file, 'r', encoding='utf-8', errors='replace') as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    
                    # Determine user's color
                    white_player = game.headers.get('White', '').lower()
                    black_player = game.headers.get('Black', '').lower()
                    
                    if username_lower in white_player:
                        user_color = chess.WHITE
                    elif username_lower in black_player:
                        user_color = chess.BLACK
                    else:
                        continue
                    
                    # Replay game and extract middlegame positions
                    board = game.board()
                    move_num = 0
                    
                    for move in game.mainline_moves():
                        board.push(move)
                        move_num += 1
                        
                        # Extract positions after the opening but not too late
                        if self.config.min_move_for_synthetic <= move_num <= 40:
                            # Sample some positions (not all, to avoid too much data)
                            if random.random() < 0.3:  # 30% chance to use this position
                                positions.append((board.copy(), user_color))
        
        print(f"Extracted {len(positions)} middlegame positions")
        return positions


def generate_hybrid_dataset(
    pgn_dir: str,
    username: str,
    original_data_dir: str,
    output_dir: str,
    stockfish_path: str = '/opt/homebrew/bin/stockfish',
    num_synthetic_games_per_position: int = 2,
    max_synthetic_positions: int = 2000
):
    """
    Generate a hybrid dataset:
    - Opening moves (1-12): Original data only
    - Middlegame/Endgame: Original + Synthetic
    """
    
    pgn_path = Path(pgn_dir)
    original_path = Path(original_data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load move encoder
    with open(original_path / 'move_encoder.pkl', 'rb') as f:
        move_encoder = pickle.load(f)
    
    # Load original data with metadata
    print("\nLoading original datasets...")
    
    datasets = {}
    for name in ['white', 'black', 'combined']:
        data = np.load(original_path / f'{name}_dataset.npz')
        with open(original_path / f'{name}_meta.pkl', 'rb') as f:
            meta = pickle.load(f)
        datasets[name] = {
            'boards': data['boards'],
            'moves': data['moves'],
            'meta': meta  # (game_id, move_number, color, move_uci)
        }
        print(f"  {name}: {len(data['moves'])} examples")
    
    # Separate opening moves from middlegame/endgame
    config = SyntheticConfig()
    
    def split_by_phase(boards, moves, meta):
        """Split data into opening vs middlegame/endgame."""
        opening_mask = np.array([m[1] <= config.opening_cutoff for m in meta])
        
        opening = {
            'boards': boards[opening_mask],
            'moves': moves[opening_mask],
            'meta': [m for m, is_opening in zip(meta, opening_mask) if is_opening]
        }
        midend = {
            'boards': boards[~opening_mask],
            'moves': moves[~opening_mask],
            'meta': [m for m, is_opening in zip(meta, opening_mask) if not is_opening]
        }
        return opening, midend
    
    print(f"\nSplitting data (opening = moves 1-{config.opening_cutoff})...")
    split_data = {}
    for name, data in datasets.items():
        opening, midend = split_by_phase(data['boards'], data['moves'], data['meta'])
        split_data[name] = {'opening': opening, 'midend': midend}
        print(f"  {name}: {len(opening['moves'])} opening, {len(midend['moves'])} middlegame/endgame")
    
    # Generate synthetic data
    print(f"\nGenerating synthetic middlegame/endgame data...")
    generator = SyntheticDataGenerator(stockfish_path=stockfish_path, config=config)
    
    try:
        # Extract positions to start synthetic games from
        positions = generator.extract_middlegame_positions(pgn_path, username)
        
        # Limit positions if needed
        if len(positions) > max_synthetic_positions:
            positions = random.sample(positions, max_synthetic_positions)
        
        print(f"\nGenerating synthetic continuations from {len(positions)} positions...")
        
        synthetic_examples = []
        for board, color in tqdm(positions, desc="Generating synthetic games"):
            for _ in range(num_synthetic_games_per_position):
                examples = generator.generate_synthetic_game(board, color, num_moves=20)
                synthetic_examples.extend(examples)
        
        print(f"Generated {len(synthetic_examples)} synthetic training examples")
        
        # Convert synthetic examples to arrays
        if synthetic_examples:
            synthetic_boards = np.stack([ex.board_tensor for ex in synthetic_examples])
            synthetic_moves = []
            valid_indices = []
            
            for i, ex in enumerate(synthetic_examples):
                try:
                    move_idx = move_encoder.move_to_idx[ex.move_uci]
                    synthetic_moves.append(move_idx)
                    valid_indices.append(i)
                except KeyError:
                    pass  # Move not in vocabulary
            
            synthetic_boards = synthetic_boards[valid_indices]
            synthetic_moves = np.array(synthetic_moves)
            synthetic_meta = [(ex.game_id, ex.move_number, ex.color, ex.move_uci) 
                            for i, ex in enumerate(synthetic_examples) if i in valid_indices]
            
            print(f"Valid synthetic examples: {len(synthetic_moves)}")
        else:
            synthetic_boards = np.array([])
            synthetic_moves = np.array([])
            synthetic_meta = []
    
    finally:
        generator.close()
    
    # Combine datasets
    print("\nCreating hybrid datasets...")
    
    for name in ['white', 'black', 'combined']:
        opening = split_data[name]['opening']
        midend = split_data[name]['midend']
        
        # Filter synthetic by color if needed
        if name == 'white':
            syn_mask = [m[2] == 'white' for m in synthetic_meta]
        elif name == 'black':
            syn_mask = [m[2] == 'black' for m in synthetic_meta]
        else:
            syn_mask = [True] * len(synthetic_meta)
        
        if synthetic_moves.size > 0 and any(syn_mask):
            syn_mask = np.array(syn_mask)
            filtered_syn_boards = synthetic_boards[syn_mask]
            filtered_syn_moves = synthetic_moves[syn_mask]
            filtered_syn_meta = [m for m, keep in zip(synthetic_meta, syn_mask) if keep]
            
            # Combine: opening (original only) + middlegame (original + synthetic)
            combined_boards = np.concatenate([
                opening['boards'],
                midend['boards'],
                filtered_syn_boards
            ])
            combined_moves = np.concatenate([
                opening['moves'],
                midend['moves'],
                filtered_syn_moves
            ])
            combined_meta = opening['meta'] + midend['meta'] + filtered_syn_meta
        else:
            combined_boards = np.concatenate([opening['boards'], midend['boards']])
            combined_moves = np.concatenate([opening['moves'], midend['moves']])
            combined_meta = opening['meta'] + midend['meta']
        
        # Save hybrid dataset
        np.savez_compressed(
            output_path / f'{name}_hybrid_dataset.npz',
            boards=combined_boards,
            moves=combined_moves
        )
        with open(output_path / f'{name}_hybrid_meta.pkl', 'wb') as f:
            pickle.dump(combined_meta, f)
        
        orig_count = len(opening['moves']) + len(midend['moves'])
        syn_count = len(combined_moves) - orig_count
        print(f"  {name}_hybrid: {len(combined_moves)} total ({orig_count} original + {syn_count} synthetic)")
    
    # Copy move encoder to output
    import shutil
    shutil.copy(original_path / 'move_encoder.pkl', output_path / 'move_encoder.pkl')
    
    print(f"\nHybrid datasets saved to {output_path}")
    print("\nTo train on hybrid data:")
    print(f"  python training/train.py --data-dir {output_path} --model combined")
    print("\nNote: The dataset files are named *_hybrid_dataset.npz")
    print("You may need to rename them or modify train.py to use the hybrid versions.")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate hybrid training dataset')
    parser.add_argument('--pgn-dir', type=str, default='./pgn',
                        help='Directory containing PGN files')
    parser.add_argument('--username', type=str, default='roy_russe11',
                        help='Your chess.com username')
    parser.add_argument('--original-data', type=str, default='./processed',
                        help='Directory with original processed data')
    parser.add_argument('--output-dir', type=str, default='./processed_hybrid',
                        help='Output directory for hybrid dataset')
    parser.add_argument('--stockfish', type=str, default='/opt/homebrew/bin/stockfish',
                        help='Path to Stockfish executable')
    parser.add_argument('--max-positions', type=int, default=2000,
                        help='Maximum middlegame positions to use for synthetic generation')
    parser.add_argument('--variations', type=int, default=2,
                        help='Synthetic game variations per position')
    
    args = parser.parse_args()
    
    generate_hybrid_dataset(
        pgn_dir=args.pgn_dir,
        username=args.username,
        original_data_dir=args.original_data,
        output_dir=args.output_dir,
        stockfish_path=args.stockfish,
        max_synthetic_positions=args.max_positions,
        num_synthetic_games_per_position=args.variations
    )
