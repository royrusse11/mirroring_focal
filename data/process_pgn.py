"""
PGN Parser and Data Processor for Chess Mirror Bot

This script:
1. Parses all PGN files from your chess.com exports
2. Extracts YOUR moves only (filtering by username)
3. Creates training examples: (board_state, your_move)
4. Saves separate datasets for White, Black, and Combined models
"""

import chess
import chess.pgn
import numpy as np
import pickle
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm
import re


@dataclass
class TrainingExample:
    """A single training example: board state → move"""
    board_tensor: np.ndarray  # 8x8x14 encoded board
    move_index: int           # Index in our move vocabulary
    move_uci: str            # UCI string for debugging
    game_id: str             # Which game this came from
    move_number: int         # Move number in game
    color: str               # 'white' or 'black'


class BoardEncoder:
    """
    Encodes a chess board into a tensor representation.
    
    We use 14 planes of 8x8:
    - Planes 0-5: Your pieces (P, N, B, R, Q, K)
    - Planes 6-11: Opponent pieces (p, n, b, r, q, k)
    - Plane 12: Current player's turn (all 1s if your turn, all 0s otherwise)
    - Plane 13: Castling rights + en passant encoded
    """
    
    PIECE_TO_PLANE = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }
    
    def encode(self, board: chess.Board, perspective_color: chess.Color) -> np.ndarray:
        """
        Encode board from perspective of the given color.
        This means 'your pieces' are always in planes 0-5.
        """
        tensor = np.zeros((14, 8, 8), dtype=np.float32)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            
            # Get row/col (flip if playing as black so your pieces are at bottom)
            row = chess.square_rank(square)
            col = chess.square_file(square)
            
            if perspective_color == chess.BLACK:
                row = 7 - row
                col = 7 - col
            
            plane = self.PIECE_TO_PLANE[piece.piece_type]
            
            # Your pieces in planes 0-5, opponent in 6-11
            if piece.color == perspective_color:
                tensor[plane, row, col] = 1.0
            else:
                tensor[plane + 6, row, col] = 1.0
        
        # Plane 12: whose turn (1 if it's our turn)
        if board.turn == perspective_color:
            tensor[12, :, :] = 1.0
        
        # Plane 13: castling rights encoded in corners, en passant in relevant square
        # Kingside castling: corner (0,7) for us, (7,7) for opponent
        # Queenside castling: corner (0,0) for us, (7,0) for opponent
        if perspective_color == chess.WHITE:
            if board.has_kingside_castling_rights(chess.WHITE):
                tensor[13, 0, 7] = 1.0
            if board.has_queenside_castling_rights(chess.WHITE):
                tensor[13, 0, 0] = 1.0
            if board.has_kingside_castling_rights(chess.BLACK):
                tensor[13, 7, 7] = 1.0
            if board.has_queenside_castling_rights(chess.BLACK):
                tensor[13, 7, 0] = 1.0
        else:
            # Flip for black's perspective
            if board.has_kingside_castling_rights(chess.BLACK):
                tensor[13, 0, 7] = 1.0
            if board.has_queenside_castling_rights(chess.BLACK):
                tensor[13, 0, 0] = 1.0
            if board.has_kingside_castling_rights(chess.WHITE):
                tensor[13, 7, 7] = 1.0
            if board.has_queenside_castling_rights(chess.WHITE):
                tensor[13, 7, 0] = 1.0
        
        # En passant square
        if board.ep_square is not None:
            ep_row = chess.square_rank(board.ep_square)
            ep_col = chess.square_file(board.ep_square)
            if perspective_color == chess.BLACK:
                ep_row = 7 - ep_row
                ep_col = 7 - ep_col
            tensor[13, ep_row, ep_col] = 0.5  # Use 0.5 to distinguish from castling
        
        return tensor


class MoveEncoder:
    """
    Encodes chess moves into indices and back.
    
    We use a simple approach: enumerate all possible moves in UCI format.
    There are ~1968 possible moves in chess (64 * 64 - duplicates + promotions).
    We'll build the vocabulary from the actual moves in the dataset.
    """
    
    def __init__(self):
        self.move_to_idx: Dict[str, int] = {}
        self.idx_to_move: Dict[int, str] = {}
        self._build_vocabulary()
    
    def _build_vocabulary(self):
        """Pre-build all possible UCI moves"""
        idx = 0
        
        # Regular moves: all square-to-square combinations
        for from_sq in chess.SQUARES:
            for to_sq in chess.SQUARES:
                if from_sq != to_sq:
                    move_uci = chess.square_name(from_sq) + chess.square_name(to_sq)
                    self.move_to_idx[move_uci] = idx
                    self.idx_to_move[idx] = move_uci
                    idx += 1
        
        # Pawn promotions
        for from_file in range(8):
            for to_file in range(8):
                if abs(from_file - to_file) <= 1:  # Can only promote straight or diagonal
                    # White promotion (rank 7 to 8)
                    from_sq = chess.square_name(chess.square(from_file, 6))
                    to_sq = chess.square_name(chess.square(to_file, 7))
                    for promo in ['q', 'r', 'b', 'n']:
                        move_uci = from_sq + to_sq + promo
                        if move_uci not in self.move_to_idx:
                            self.move_to_idx[move_uci] = idx
                            self.idx_to_move[idx] = move_uci
                            idx += 1
                    
                    # Black promotion (rank 2 to 1)
                    from_sq = chess.square_name(chess.square(from_file, 1))
                    to_sq = chess.square_name(chess.square(to_file, 0))
                    for promo in ['q', 'r', 'b', 'n']:
                        move_uci = from_sq + to_sq + promo
                        if move_uci not in self.move_to_idx:
                            self.move_to_idx[move_uci] = idx
                            self.idx_to_move[idx] = move_uci
                            idx += 1
        
        self.vocab_size = idx
        print(f"Move vocabulary size: {self.vocab_size}")
    
    def encode(self, move: chess.Move) -> int:
        """Convert a move to its index"""
        return self.move_to_idx[move.uci()]
    
    def decode(self, idx: int) -> str:
        """Convert an index back to UCI string"""
        return self.idx_to_move[idx]


def parse_pgn_file(filepath: Path, username: str, board_encoder: BoardEncoder, 
                   move_encoder: MoveEncoder) -> Tuple[List[TrainingExample], List[TrainingExample]]:
    """
    Parse a PGN file and extract training examples for the given username.
    
    Returns:
        (white_examples, black_examples) - separated for the two color-specific models
    """
    white_examples = []
    black_examples = []
    
    username_lower = username.lower()
    
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            
            # Determine which color the user played
            white_player = game.headers.get('White', '').lower()
            black_player = game.headers.get('Black', '').lower()
            
            if username_lower in white_player:
                user_color = chess.WHITE
            elif username_lower in black_player:
                user_color = chess.BLACK
            else:
                # User not in this game, skip
                continue
            
            game_id = game.headers.get('Link', filepath.stem)
            
            # Replay the game and extract positions before user's moves
            board = game.board()
            move_num = 0
            
            for move in game.mainline_moves():
                if board.turn == user_color:
                    # This is our move - create training example
                    try:
                        board_tensor = board_encoder.encode(board, user_color)
                        move_idx = move_encoder.encode(move)
                        
                        example = TrainingExample(
                            board_tensor=board_tensor,
                            move_index=move_idx,
                            move_uci=move.uci(),
                            game_id=game_id,
                            move_number=move_num,
                            color='white' if user_color == chess.WHITE else 'black'
                        )
                        
                        if user_color == chess.WHITE:
                            white_examples.append(example)
                        else:
                            black_examples.append(example)
                        
                        move_num += 1
                    except KeyError:
                        # Move not in vocabulary (shouldn't happen but just in case)
                        print(f"Warning: Move {move.uci()} not in vocabulary")
                
                board.push(move)
    
    return white_examples, black_examples


def process_all_pgns(pgn_dir: Path, username: str, output_dir: Path):
    """
    Process all PGN files and save the training datasets.
    """
    board_encoder = BoardEncoder()
    move_encoder = MoveEncoder()
    
    all_white_examples = []
    all_black_examples = []
    
    # Find all PGN files
    pgn_files = list(pgn_dir.glob('*.pgn'))
    
    if not pgn_files:
        print(f"No PGN files found in {pgn_dir}")
        print("Please add your chess.com PGN exports to this directory.")
        return
    
    print(f"Found {len(pgn_files)} PGN files")
    
    for pgn_file in tqdm(pgn_files, desc="Processing PGN files"):
        white_ex, black_ex = parse_pgn_file(pgn_file, username, board_encoder, move_encoder)
        all_white_examples.extend(white_ex)
        all_black_examples.extend(black_ex)
    
    all_combined = all_white_examples + all_black_examples
    
    # Print statistics
    print(f"\n{'='*50}")
    print("Dataset Statistics:")
    print(f"{'='*50}")
    print(f"Total games processed: {len(pgn_files)} files")
    print(f"White examples: {len(all_white_examples):,}")
    print(f"Black examples: {len(all_black_examples):,}")
    print(f"Combined examples: {len(all_combined):,}")
    print(f"Move vocabulary size: {move_encoder.vocab_size}")
    
    # Save datasets
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save move encoder (needed for inference)
    with open(output_dir / 'move_encoder.pkl', 'wb') as f:
        pickle.dump(move_encoder, f)
    
    # Convert to numpy arrays for efficient storage
    def examples_to_arrays(examples: List[TrainingExample]) -> Dict:
        if not examples:
            return {'boards': np.array([]), 'moves': np.array([]), 'meta': []}
        
        boards = np.stack([ex.board_tensor for ex in examples])
        moves = np.array([ex.move_index for ex in examples])
        meta = [(ex.game_id, ex.move_number, ex.color, ex.move_uci) for ex in examples]
        
        return {'boards': boards, 'moves': moves, 'meta': meta}
    
    # Save White dataset
    white_data = examples_to_arrays(all_white_examples)
    np.savez_compressed(output_dir / 'white_dataset.npz', 
                        boards=white_data['boards'], 
                        moves=white_data['moves'])
    with open(output_dir / 'white_meta.pkl', 'wb') as f:
        pickle.dump(white_data['meta'], f)
    
    # Save Black dataset
    black_data = examples_to_arrays(all_black_examples)
    np.savez_compressed(output_dir / 'black_dataset.npz',
                        boards=black_data['boards'],
                        moves=black_data['moves'])
    with open(output_dir / 'black_meta.pkl', 'wb') as f:
        pickle.dump(black_data['meta'], f)
    
    # Save Combined dataset
    combined_data = examples_to_arrays(all_combined)
    np.savez_compressed(output_dir / 'combined_dataset.npz',
                        boards=combined_data['boards'],
                        moves=combined_data['moves'])
    with open(output_dir / 'combined_meta.pkl', 'wb') as f:
        pickle.dump(combined_data['meta'], f)
    
    print(f"\nDatasets saved to {output_dir}")
    print("Files created:")
    print("  - white_dataset.npz")
    print("  - black_dataset.npz") 
    print("  - combined_dataset.npz")
    print("  - move_encoder.pkl")
    print("  - *_meta.pkl (game metadata)")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Process chess.com PGN files for training')
    parser.add_argument('--pgn-dir', type=str, default='./pgn',
                        help='Directory containing PGN files')
    parser.add_argument('--username', type=str, default='roy_russe11',
                        help='Your chess.com username')
    parser.add_argument('--output-dir', type=str, default='./processed',
                        help='Output directory for processed datasets')
    
    args = parser.parse_args()
    
    process_all_pgns(
        pgn_dir=Path(args.pgn_dir),
        username=args.username,
        output_dir=Path(args.output_dir)
    )
