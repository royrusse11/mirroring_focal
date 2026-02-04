"""
Stockfish Analysis Tool

Analyzes your bot's play by:
1. Running the bot against Stockfish at various skill levels
2. Comparing bot moves to Stockfish recommendations
3. Generating detailed weakness reports

Reports include:
- Centipawn loss by game phase
- Tactical pattern misses
- Opening performance
- Position type weaknesses
"""

import chess
import chess.engine
import chess.pgn
import torch
import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.network import ChessPolicyNet
from data.process_pgn import BoardEncoder, MoveEncoder


@dataclass
class MoveAnalysis:
    """Analysis of a single move."""
    move_number: int
    position_fen: str
    bot_move: str
    best_move: str
    bot_eval: float  # Centipawns after bot move
    best_eval: float  # Centipawns after best move
    centipawn_loss: float
    phase: str  # 'opening', 'middlegame', 'endgame'
    is_blunder: bool  # Loss > 100cp
    is_mistake: bool  # Loss > 50cp
    is_inaccuracy: bool  # Loss > 25cp
    tactical_motif: Optional[str] = None


@dataclass  
class GameAnalysis:
    """Analysis of a complete game."""
    game_id: str
    bot_color: str
    result: str  # '1-0', '0-1', '1/2-1/2'
    total_moves: int
    avg_centipawn_loss: float
    blunders: int
    mistakes: int
    inaccuracies: int
    opening_loss: float
    middlegame_loss: float
    endgame_loss: float
    move_analyses: List[MoveAnalysis]


@dataclass
class WeaknessReport:
    """Overall weakness report."""
    model_name: str
    games_analyzed: int
    total_positions: int
    
    # Overall stats
    avg_centipawn_loss: float
    blunder_rate: float  # per 100 moves
    mistake_rate: float
    inaccuracy_rate: float
    
    # By phase
    opening_avg_loss: float
    middlegame_avg_loss: float
    endgame_avg_loss: float
    
    # By position type (will be populated)
    worst_position_types: List[Dict]
    
    # Tactical weaknesses
    missed_tactics: Dict[str, int]
    
    # Opening stats
    opening_performance: Dict[str, Dict]
    
    timestamp: str


class StockfishAnalyzer:
    """Analyzes bot performance using Stockfish."""
    
    def __init__(self, 
                 model_path: str,
                 data_dir: str = './processed',
                 stockfish_path: str = '/opt/homebrew/bin/stockfish',
                 analysis_depth: int = 18):
        """
        Initialize analyzer.
        
        Args:
            model_path: Path to the trained model
            data_dir: Directory with move_encoder.pkl
            stockfish_path: Path to Stockfish executable
            analysis_depth: Depth for Stockfish analysis
        """
        self.analysis_depth = analysis_depth
        self.stockfish_path = stockfish_path
        
        # Setup device
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # Load model
        self.model = self._load_model(model_path)
        self.model_name = Path(model_path).stem
        
        # Load encoders
        self.board_encoder = BoardEncoder()
        with open(Path(data_dir) / 'move_encoder.pkl', 'rb') as f:
            self.move_encoder = pickle.load(f)
        
        # Stockfish engine (lazy loaded)
        self._engine = None
    
    def _load_model(self, path: str) -> ChessPolicyNet:
        """Load the trained model."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        model = ChessPolicyNet(
            num_moves=checkpoint['num_moves'],
            num_filters=checkpoint.get('num_filters', 128),
            num_residual_blocks=checkpoint.get('num_residual_blocks', 6)
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    @property
    def engine(self) -> chess.engine.SimpleEngine:
        """Lazy load Stockfish engine."""
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        return self._engine
    
    def close(self):
        """Close the Stockfish engine."""
        if self._engine:
            self._engine.quit()
            self._engine = None
    
    def _get_phase(self, board: chess.Board) -> str:
        """Determine game phase based on material and move number."""
        # Count material
        material = 0
        for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            material += len(board.pieces(piece_type, chess.WHITE)) * {
                chess.QUEEN: 9, chess.ROOK: 5, chess.BISHOP: 3, chess.KNIGHT: 3
            }[piece_type]
            material += len(board.pieces(piece_type, chess.BLACK)) * {
                chess.QUEEN: 9, chess.ROOK: 5, chess.BISHOP: 3, chess.KNIGHT: 3
            }[piece_type]
        
        move_count = board.fullmove_number
        
        if move_count <= 10:
            return 'opening'
        elif material <= 20:
            return 'endgame'
        else:
            return 'middlegame'
    
    def _detect_tactical_motif(self, board: chess.Board, best_move: chess.Move) -> Optional[str]:
        """Try to detect if the best move involves a tactical motif."""
        # Simple heuristics for common tactics
        
        # Check for forks (piece attacks two or more enemy pieces after move)
        test_board = board.copy()
        test_board.push(best_move)
        
        moving_piece = board.piece_at(best_move.from_square)
        if moving_piece is None:
            return None
        
        attacked_squares = test_board.attacks(best_move.to_square)
        valuable_attacks = 0
        
        for sq in attacked_squares:
            attacked_piece = test_board.piece_at(sq)
            if attacked_piece and attacked_piece.color != moving_piece.color:
                if attacked_piece.piece_type in [chess.QUEEN, chess.ROOK, chess.KING]:
                    valuable_attacks += 1
        
        if valuable_attacks >= 2:
            piece_name = chess.piece_name(moving_piece.piece_type)
            return f"{piece_name}_fork"
        
        # Check for discovered attacks
        # (simplified - just check if another piece now attacks something valuable)
        
        # Check for pins/skewers (very simplified)
        if moving_piece.piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
            # Check if move creates an attack through an enemy piece to something behind
            pass
        
        return None
    
    def _get_bot_move(self, board: chess.Board, bot_color: chess.Color, 
                      temperature: float = 0.3) -> chess.Move:
        """Get the bot's move for a position."""
        board_tensor = self.board_encoder.encode(board, bot_color)
        board_tensor = torch.from_numpy(board_tensor).unsqueeze(0).to(self.device)
        
        # Create legal move mask
        legal_moves = list(board.legal_moves)
        legal_move_mask = torch.zeros(1, self.move_encoder.vocab_size, dtype=torch.bool).to(self.device)
        
        legal_moves_in_vocab = []
        for move in legal_moves:
            try:
                idx = self.move_encoder.move_to_idx[move.uci()]
                legal_move_mask[0, idx] = True
                legal_moves_in_vocab.append(move)
            except KeyError:
                pass
        
        # Fallback if no legal moves in vocabulary
        if not legal_moves_in_vocab:
            import random
            return random.choice(legal_moves)
        
        with torch.no_grad():
            log_probs = self.model(board_tensor, legal_move_mask)
            
            if temperature == 0:
                move_idx = log_probs.argmax(dim=1).item()
            else:
                probs = (log_probs / temperature).exp()
                probs = probs / probs.sum()
                move_idx = torch.multinomial(probs, 1).item()
        
        move_uci = self.move_encoder.idx_to_move[move_idx]
        return chess.Move.from_uci(move_uci)
    
    def analyze_position(self, board: chess.Board, bot_color: chess.Color,
                        move_number: int) -> MoveAnalysis:
        """Analyze a single position."""
        # Get bot's move
        bot_move = self._get_bot_move(board, bot_color)
        
        # Get Stockfish's best move and evaluation
        result = self.engine.analyse(board, chess.engine.Limit(depth=self.analysis_depth))
        best_move = result['pv'][0]
        
        # Get evaluation after both moves
        board_after_bot = board.copy()
        board_after_bot.push(bot_move)
        bot_result = self.engine.analyse(board_after_bot, chess.engine.Limit(depth=self.analysis_depth - 2))
        
        board_after_best = board.copy()
        board_after_best.push(best_move)
        best_result = self.engine.analyse(board_after_best, chess.engine.Limit(depth=self.analysis_depth - 2))
        
        # Convert to centipawns from bot's perspective
        def score_to_cp(score, is_bot_turn):
            if score.is_mate():
                cp = 10000 if score.mate() > 0 else -10000
            else:
                cp = score.score()
            # Flip if it was opponent's perspective
            return cp if is_bot_turn else -cp
        
        is_bot_turn = board.turn == bot_color
        bot_eval = score_to_cp(bot_result['score'].relative, not is_bot_turn)
        best_eval = score_to_cp(best_result['score'].relative, not is_bot_turn)
        
        # Calculate loss (positive = bot did worse)
        centipawn_loss = best_eval - bot_eval
        centipawn_loss = max(0, centipawn_loss)  # Don't count if bot found better move
        
        # Detect tactical motif in best move (if bot missed it)
        tactical_motif = None
        if bot_move != best_move and centipawn_loss > 50:
            tactical_motif = self._detect_tactical_motif(board, best_move)
        
        return MoveAnalysis(
            move_number=move_number,
            position_fen=board.fen(),
            bot_move=bot_move.uci(),
            best_move=best_move.uci(),
            bot_eval=bot_eval,
            best_eval=best_eval,
            centipawn_loss=centipawn_loss,
            phase=self._get_phase(board),
            is_blunder=centipawn_loss >= 100,
            is_mistake=50 <= centipawn_loss < 100,
            is_inaccuracy=25 <= centipawn_loss < 50,
            tactical_motif=tactical_motif
        )
    
    def play_game_vs_stockfish(self, bot_color: chess.Color, 
                               stockfish_elo: int = 1500,
                               max_moves: int = 100) -> GameAnalysis:
        """Play a game between the bot and Stockfish."""
        # Configure Stockfish skill level
        # Note: Skill level to ELO mapping is approximate and varies by Stockfish version
        # This is a rough heuristic, not an exact correspondence
        skill_level = max(0, min(20, (stockfish_elo - 500) // 100))
        self.engine.configure({"Skill Level": skill_level})
        
        board = chess.Board()
        move_analyses = []
        move_number = 0
        
        while not board.is_game_over() and move_number < max_moves:
            if board.turn == bot_color:
                # Bot's turn - analyze it
                analysis = self.analyze_position(board, bot_color, move_number)
                move_analyses.append(analysis)
                move = chess.Move.from_uci(analysis.bot_move)
            else:
                # Stockfish's turn
                result = self.engine.play(board, chess.engine.Limit(time=0.1))
                move = result.move
            
            board.push(move)
            move_number += 1
        
        # Determine result
        if board.is_checkmate():
            result = '0-1' if board.turn == bot_color else '1-0'
        else:
            result = '1/2-1/2'
        
        # Calculate stats
        if move_analyses:
            avg_loss = np.mean([m.centipawn_loss for m in move_analyses])
            blunders = sum(1 for m in move_analyses if m.is_blunder)
            mistakes = sum(1 for m in move_analyses if m.is_mistake)
            inaccuracies = sum(1 for m in move_analyses if m.is_inaccuracy)
            
            opening_moves = [m for m in move_analyses if m.phase == 'opening']
            middlegame_moves = [m for m in move_analyses if m.phase == 'middlegame']
            endgame_moves = [m for m in move_analyses if m.phase == 'endgame']
            
            opening_loss = np.mean([m.centipawn_loss for m in opening_moves]) if opening_moves else 0
            middlegame_loss = np.mean([m.centipawn_loss for m in middlegame_moves]) if middlegame_moves else 0
            endgame_loss = np.mean([m.centipawn_loss for m in endgame_moves]) if endgame_moves else 0
        else:
            avg_loss = blunders = mistakes = inaccuracies = 0
            opening_loss = middlegame_loss = endgame_loss = 0
        
        return GameAnalysis(
            game_id=f"vs_stockfish_{stockfish_elo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            bot_color='white' if bot_color == chess.WHITE else 'black',
            result=result,
            total_moves=len(move_analyses),
            avg_centipawn_loss=avg_loss,
            blunders=blunders,
            mistakes=mistakes,
            inaccuracies=inaccuracies,
            opening_loss=opening_loss,
            middlegame_loss=middlegame_loss,
            endgame_loss=endgame_loss,
            move_analyses=move_analyses
        )
    
    def run_analysis(self, num_games: int = 10, stockfish_elo: int = 1500) -> WeaknessReport:
        """Run full analysis over multiple games."""
        print(f"\nRunning analysis: {num_games} games vs Stockfish (ELO {stockfish_elo})")
        print("=" * 60)
        
        all_analyses = []
        all_moves = []
        
        for i in range(num_games):
            # Alternate colors
            bot_color = chess.WHITE if i % 2 == 0 else chess.BLACK
            color_name = "White" if bot_color == chess.WHITE else "Black"
            
            print(f"\nGame {i+1}/{num_games} - Bot plays {color_name}...")
            
            game_analysis = self.play_game_vs_stockfish(bot_color, stockfish_elo)
            all_analyses.append(game_analysis)
            all_moves.extend(game_analysis.move_analyses)
            
            print(f"  Result: {game_analysis.result}")
            print(f"  Avg CP Loss: {game_analysis.avg_centipawn_loss:.1f}")
            print(f"  Blunders/Mistakes/Inaccuracies: {game_analysis.blunders}/{game_analysis.mistakes}/{game_analysis.inaccuracies}")
        
        # Aggregate statistics
        total_moves = len(all_moves)
        
        if total_moves == 0:
            print("No moves to analyze!")
            return None
        
        avg_loss = np.mean([m.centipawn_loss for m in all_moves])
        blunder_count = sum(1 for m in all_moves if m.is_blunder)
        mistake_count = sum(1 for m in all_moves if m.is_mistake)
        inaccuracy_count = sum(1 for m in all_moves if m.is_inaccuracy)
        
        # By phase
        opening = [m for m in all_moves if m.phase == 'opening']
        middlegame = [m for m in all_moves if m.phase == 'middlegame']
        endgame = [m for m in all_moves if m.phase == 'endgame']
        
        # Missed tactics
        missed_tactics = defaultdict(int)
        for m in all_moves:
            if m.tactical_motif:
                missed_tactics[m.tactical_motif] += 1
        
        report = WeaknessReport(
            model_name=self.model_name,
            games_analyzed=num_games,
            total_positions=total_moves,
            avg_centipawn_loss=avg_loss,
            blunder_rate=100 * blunder_count / total_moves,
            mistake_rate=100 * mistake_count / total_moves,
            inaccuracy_rate=100 * inaccuracy_count / total_moves,
            opening_avg_loss=np.mean([m.centipawn_loss for m in opening]) if opening else 0,
            middlegame_avg_loss=np.mean([m.centipawn_loss for m in middlegame]) if middlegame else 0,
            endgame_avg_loss=np.mean([m.centipawn_loss for m in endgame]) if endgame else 0,
            worst_position_types=[],  # Could add more position classification
            missed_tactics=dict(missed_tactics),
            opening_performance={},  # Could track specific openings
            timestamp=datetime.now().isoformat()
        )
        
        return report
    
    def generate_report(self, report: WeaknessReport, output_path: str):
        """Generate a human-readable report."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"CHESS MIRROR - WEAKNESS ANALYSIS REPORT")
        lines.append(f"Model: {report.model_name}")
        lines.append(f"Generated: {report.timestamp}")
        lines.append("=" * 70)
        
        lines.append(f"\n📊 OVERVIEW")
        lines.append(f"  Games analyzed: {report.games_analyzed}")
        lines.append(f"  Total positions: {report.total_positions}")
        lines.append(f"  Average centipawn loss: {report.avg_centipawn_loss:.1f}")
        
        lines.append(f"\n❌ ERROR RATES (per 100 moves)")
        lines.append(f"  Blunders (≥100cp loss):    {report.blunder_rate:.1f}")
        lines.append(f"  Mistakes (50-100cp loss):  {report.mistake_rate:.1f}")
        lines.append(f"  Inaccuracies (25-50cp):    {report.inaccuracy_rate:.1f}")
        
        lines.append(f"\n📈 PERFORMANCE BY GAME PHASE")
        phases = [
            ("Opening", report.opening_avg_loss),
            ("Middlegame", report.middlegame_avg_loss),
            ("Endgame", report.endgame_avg_loss)
        ]
        
        # Find worst phase
        worst_phase = max(phases, key=lambda x: x[1])
        
        for phase, loss in phases:
            indicator = "⚠️ " if phase == worst_phase[0] and loss > report.avg_centipawn_loss else "   "
            bar_length = int(loss / 5)
            bar = "█" * min(bar_length, 30)
            lines.append(f"  {indicator}{phase:12s}: {loss:6.1f} cp  {bar}")
        
        if report.missed_tactics:
            lines.append(f"\n⚔️ MISSED TACTICAL PATTERNS")
            for tactic, count in sorted(report.missed_tactics.items(), key=lambda x: -x[1]):
                lines.append(f"  {tactic.replace('_', ' ').title():20s}: {count} times")
        
        lines.append(f"\n💡 RECOMMENDATIONS")
        
        if report.endgame_avg_loss > report.middlegame_avg_loss * 1.3:
            lines.append("  • Focus on endgame study - your endgame play is significantly weaker")
        
        if report.opening_avg_loss > 40:
            lines.append("  • Review opening principles - consider learning a solid repertoire")
        
        if report.blunder_rate > 5:
            lines.append("  • Practice tactical puzzles to reduce blunder rate")
        
        if 'knight_fork' in report.missed_tactics:
            lines.append("  • Study knight fork patterns - you're missing these frequently")
        
        lines.append("\n" + "=" * 70)
        
        report_text = "\n".join(lines)
        
        # Print to console
        print(report_text)
        
        # Save to file
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        # Also save JSON for programmatic access
        json_path = output_path.replace('.txt', '.json')
        with open(json_path, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        
        print(f"\nReport saved to: {output_path}")
        print(f"JSON data saved to: {json_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze bot weaknesses with Stockfish')
    parser.add_argument('model', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--data-dir', type=str, default='./processed',
                        help='Directory containing move_encoder.pkl')
    parser.add_argument('--stockfish', type=str, default='/opt/homebrew/bin/stockfish',
                        help='Path to Stockfish executable')
    parser.add_argument('--games', type=int, default=10,
                        help='Number of games to play for analysis')
    parser.add_argument('--stockfish-elo', type=int, default=1500,
                        help='Stockfish ELO level')
    parser.add_argument('--output', type=str, default='./reports/weakness_report.txt',
                        help='Output path for report')
    parser.add_argument('--depth', type=int, default=18,
                        help='Analysis depth')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    analyzer = StockfishAnalyzer(
        model_path=args.model,
        data_dir=args.data_dir,
        stockfish_path=args.stockfish,
        analysis_depth=args.depth
    )
    
    try:
        report = analyzer.run_analysis(num_games=args.games, stockfish_elo=args.stockfish_elo)
        
        if report:
            analyzer.generate_report(report, args.output)
    finally:
        analyzer.close()


if __name__ == '__main__':
    main()
