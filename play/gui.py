"""
Chess Mirror - Play Interface

A graphical interface to play chess against your trained bot.
Uses pygame for the GUI and python-chess for game logic.
"""

import pygame
import chess
import torch
import numpy as np
import pickle
import sys
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.network import ChessPolicyNet
from data.process_pgn import BoardEncoder, MoveEncoder


# =============================================================================
# Configuration
# =============================================================================

WINDOW_SIZE = 640
BOARD_SIZE = 560
BOARD_OFFSET = 40
SQUARE_SIZE = BOARD_SIZE // 8

# Colors
LIGHT_SQUARE = (240, 217, 181)  # Tan
DARK_SQUARE = (181, 136, 99)   # Brown
HIGHLIGHT_COLOR = (186, 202, 68, 180)  # Yellow-green with alpha
LAST_MOVE_COLOR = (205, 210, 106, 150)  # Softer yellow
LEGAL_MOVE_DOT = (100, 100, 100, 128)  # Gray dot for legal moves
SELECTED_COLOR = (246, 246, 105, 180)  # Bright yellow
BG_COLOR = (49, 46, 43)  # Dark background
TEXT_COLOR = (255, 255, 255)
BUTTON_COLOR = (70, 70, 70)
BUTTON_HOVER = (90, 90, 90)


class GameState(Enum):
    PLAYING = 1
    WHITE_WINS = 2
    BLACK_WINS = 3
    DRAW = 4
    MENU = 5


@dataclass
class GameConfig:
    player_color: chess.Color
    model_path: str
    model_name: str
    temperature: float = 0.5  # Sampling temperature for bot moves


class ChessGUI:
    """Main GUI class for playing chess against the bot."""
    
    def __init__(self, data_dir: str = './processed', checkpoints_dir: str = './checkpoints'):
        pygame.init()
        pygame.display.set_caption("Chess Mirror - Play Your Bot")
        
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        self.clock = pygame.time.Clock()
        
        self.data_dir = Path(data_dir)
        self.checkpoints_dir = Path(checkpoints_dir)
        
        # Load pieces
        self.pieces = self._load_pieces()
        
        # Load encoders
        self.board_encoder = BoardEncoder()
        self.move_encoder = self._load_move_encoder()
        
        # Game state
        self.board: Optional[chess.Board] = None
        self.config: Optional[GameConfig] = None
        self.model: Optional[ChessPolicyNet] = None
        self.device = self._get_device()
        
        self.selected_square: Optional[int] = None
        self.legal_moves_from_selected: List[chess.Move] = []
        self.last_move: Optional[chess.Move] = None
        self.game_state = GameState.MENU
        
        # Font
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)
        
    def _get_device(self) -> torch.device:
        """Auto-detect best device."""
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    
    def _load_pieces(self) -> dict:
        """Load chess piece images from the web or generate fallback."""
        pieces = {}
        
        # Try to download piece images from a public source
        # Using the popular "cburnett" piece set from lichess
        base_url = "https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/cburnett"
        
        piece_names = {
            'K': 'wK', 'Q': 'wQ', 'R': 'wR', 'B': 'wB', 'N': 'wN', 'P': 'wP',
            'k': 'bK', 'q': 'bQ', 'r': 'bR', 'b': 'bB', 'n': 'bN', 'p': 'bP'
        }
        
        # Create pieces directory if it doesn't exist
        pieces_dir = Path(__file__).parent / 'pieces'
        pieces_dir.mkdir(exist_ok=True)
        
        import urllib.request
        import io
        
        for symbol, name in piece_names.items():
            piece_path = pieces_dir / f"{name}.svg"
            png_path = pieces_dir / f"{name}.png"
            
            # Try to load cached PNG first
            if png_path.exists():
                try:
                    img = pygame.image.load(str(png_path))
                    pieces[symbol] = pygame.transform.smoothscale(img, (SQUARE_SIZE, SQUARE_SIZE))
                    continue
                except:
                    pass
            
            # Try to download and convert SVG
            try:
                url = f"{base_url}/{name}.svg"
                # Download SVG
                with urllib.request.urlopen(url, timeout=5) as response:
                    svg_data = response.read()
                
                # Try using cairosvg if available
                try:
                    import cairosvg
                    png_data = cairosvg.svg2png(bytestring=svg_data, output_width=SQUARE_SIZE, output_height=SQUARE_SIZE)
                    img = pygame.image.load(io.BytesIO(png_data))
                    pieces[symbol] = img
                    # Cache the PNG
                    with open(png_path, 'wb') as f:
                        f.write(png_data)
                    continue
                except ImportError:
                    pass
                
            except Exception as e:
                print(f"Could not download piece {name}: {e}")
        
        # If we couldn't load images, fall back to drawing simple pieces
        if len(pieces) < 12:
            print("Using fallback drawn pieces...")
            pieces = self._draw_fallback_pieces()
        
        return pieces
    
    def _draw_fallback_pieces(self) -> dict:
        """Draw simple but clear chess pieces as fallback."""
        pieces = {}
        
        for symbol in ['K', 'Q', 'R', 'B', 'N', 'P', 'k', 'q', 'r', 'b', 'n', 'p']:
            surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            
            is_white = symbol.isupper()
            fill_color = (255, 255, 255) if is_white else (40, 40, 40)
            outline_color = (40, 40, 40) if is_white else (255, 255, 255)
            
            center_x = SQUARE_SIZE // 2
            center_y = SQUARE_SIZE // 2
            
            piece_type = symbol.upper()
            
            if piece_type == 'P':  # Pawn - simple circle with base
                # Base
                pygame.draw.ellipse(surface, fill_color, 
                    (center_x - 18, SQUARE_SIZE - 20, 36, 12))
                pygame.draw.ellipse(surface, outline_color, 
                    (center_x - 18, SQUARE_SIZE - 20, 36, 12), 2)
                # Body
                pygame.draw.polygon(surface, fill_color,
                    [(center_x - 12, SQUARE_SIZE - 18), (center_x + 12, SQUARE_SIZE - 18),
                     (center_x + 8, center_y + 5), (center_x - 8, center_y + 5)])
                pygame.draw.polygon(surface, outline_color,
                    [(center_x - 12, SQUARE_SIZE - 18), (center_x + 12, SQUARE_SIZE - 18),
                     (center_x + 8, center_y + 5), (center_x - 8, center_y + 5)], 2)
                # Head
                pygame.draw.circle(surface, fill_color, (center_x, center_y - 5), 12)
                pygame.draw.circle(surface, outline_color, (center_x, center_y - 5), 12, 2)
                
            elif piece_type == 'R':  # Rook - castle shape
                points = [
                    (center_x - 18, SQUARE_SIZE - 15),
                    (center_x + 18, SQUARE_SIZE - 15),
                    (center_x + 18, SQUARE_SIZE - 22),
                    (center_x + 14, SQUARE_SIZE - 22),
                    (center_x + 14, 20),
                    (center_x + 18, 20),
                    (center_x + 18, 12),
                    (center_x + 10, 12),
                    (center_x + 10, 18),
                    (center_x + 3, 18),
                    (center_x + 3, 12),
                    (center_x - 3, 12),
                    (center_x - 3, 18),
                    (center_x - 10, 18),
                    (center_x - 10, 12),
                    (center_x - 18, 12),
                    (center_x - 18, 20),
                    (center_x - 14, 20),
                    (center_x - 14, SQUARE_SIZE - 22),
                    (center_x - 18, SQUARE_SIZE - 22),
                ]
                pygame.draw.polygon(surface, fill_color, points)
                pygame.draw.polygon(surface, outline_color, points, 2)
                
            elif piece_type == 'N':  # Knight - horse head shape
                points = [
                    (center_x - 16, SQUARE_SIZE - 15),
                    (center_x + 16, SQUARE_SIZE - 15),
                    (center_x + 14, SQUARE_SIZE - 25),
                    (center_x + 8, center_y),
                    (center_x + 15, center_y - 15),
                    (center_x + 10, center_y - 25),
                    (center_x - 5, center_y - 20),
                    (center_x - 15, center_y - 25),
                    (center_x - 10, center_y - 15),
                    (center_x - 12, center_y - 5),
                    (center_x - 14, SQUARE_SIZE - 25),
                ]
                pygame.draw.polygon(surface, fill_color, points)
                pygame.draw.polygon(surface, outline_color, points, 2)
                # Eye
                eye_color = outline_color
                pygame.draw.circle(surface, eye_color, (center_x - 2, center_y - 18), 3)
                
            elif piece_type == 'B':  # Bishop - tall with slit
                points = [
                    (center_x - 16, SQUARE_SIZE - 15),
                    (center_x + 16, SQUARE_SIZE - 15),
                    (center_x + 12, SQUARE_SIZE - 22),
                    (center_x + 8, center_y + 10),
                    (center_x + 12, center_y - 5),
                    (center_x + 6, 15),
                    (center_x, 10),
                    (center_x - 6, 15),
                    (center_x - 12, center_y - 5),
                    (center_x - 8, center_y + 10),
                    (center_x - 12, SQUARE_SIZE - 22),
                ]
                pygame.draw.polygon(surface, fill_color, points)
                pygame.draw.polygon(surface, outline_color, points, 2)
                # Slit
                pygame.draw.line(surface, outline_color, 
                    (center_x - 4, center_y - 8), (center_x + 4, center_y - 16), 2)
                    
            elif piece_type == 'Q':  # Queen - crown with spikes
                # Base
                pygame.draw.ellipse(surface, fill_color,
                    (center_x - 18, SQUARE_SIZE - 20, 36, 14))
                pygame.draw.ellipse(surface, outline_color,
                    (center_x - 18, SQUARE_SIZE - 20, 36, 14), 2)
                # Body
                points = [
                    (center_x - 16, SQUARE_SIZE - 18),
                    (center_x + 16, SQUARE_SIZE - 18),
                    (center_x + 12, center_y),
                    (center_x + 16, 18),
                    (center_x + 10, 25),
                    (center_x + 8, 15),
                    (center_x + 4, 22),
                    (center_x, 12),
                    (center_x - 4, 22),
                    (center_x - 8, 15),
                    (center_x - 10, 25),
                    (center_x - 16, 18),
                    (center_x - 12, center_y),
                ]
                pygame.draw.polygon(surface, fill_color, points)
                pygame.draw.polygon(surface, outline_color, points, 2)
                # Ball on top
                pygame.draw.circle(surface, fill_color, (center_x, 10), 5)
                pygame.draw.circle(surface, outline_color, (center_x, 10), 5, 2)
                
            elif piece_type == 'K':  # King - crown with cross
                # Base
                pygame.draw.ellipse(surface, fill_color,
                    (center_x - 18, SQUARE_SIZE - 20, 36, 14))
                pygame.draw.ellipse(surface, outline_color,
                    (center_x - 18, SQUARE_SIZE - 20, 36, 14), 2)
                # Body
                points = [
                    (center_x - 16, SQUARE_SIZE - 18),
                    (center_x + 16, SQUARE_SIZE - 18),
                    (center_x + 14, center_y + 5),
                    (center_x + 18, center_y - 5),
                    (center_x + 10, center_y - 5),
                    (center_x + 10, 25),
                    (center_x - 10, 25),
                    (center_x - 10, center_y - 5),
                    (center_x - 18, center_y - 5),
                    (center_x - 14, center_y + 5),
                ]
                pygame.draw.polygon(surface, fill_color, points)
                pygame.draw.polygon(surface, outline_color, points, 2)
                # Cross
                pygame.draw.rect(surface, fill_color, (center_x - 3, 8, 6, 20))
                pygame.draw.rect(surface, outline_color, (center_x - 3, 8, 6, 20), 2)
                pygame.draw.rect(surface, fill_color, (center_x - 10, 14, 20, 6))
                pygame.draw.rect(surface, outline_color, (center_x - 10, 14, 20, 6), 2)
            
            pieces[symbol] = surface
        
        return pieces
    
    def _load_move_encoder(self) -> Optional[MoveEncoder]:
        """Load the move encoder."""
        encoder_path = self.data_dir / 'move_encoder.pkl'
        if encoder_path.exists():
            with open(encoder_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def _load_model(self, model_path: str) -> Optional[ChessPolicyNet]:
        """Load a trained model."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            model = ChessPolicyNet(
                num_moves=checkpoint['num_moves'],
                num_filters=checkpoint.get('num_filters', 128),
                num_residual_blocks=checkpoint.get('num_residual_blocks', 6)
            ).to(self.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            print(f"Loaded model: {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def _get_available_models(self) -> List[Tuple[str, str]]:
        """Get list of available model files."""
        models = []
        if self.checkpoints_dir.exists():
            for f in self.checkpoints_dir.glob('*_best.pt'):
                name = f.stem.replace('_best', '')
                models.append((str(f), name))
        return models
    
    def _square_to_coords(self, square: int, flip: bool = False) -> Tuple[int, int]:
        """Convert chess square to screen coordinates."""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        if flip:
            file = 7 - file
            rank = 7 - rank
        
        x = BOARD_OFFSET + file * SQUARE_SIZE
        y = BOARD_OFFSET + (7 - rank) * SQUARE_SIZE
        
        return x, y
    
    def _coords_to_square(self, x: int, y: int, flip: bool = False) -> Optional[int]:
        """Convert screen coordinates to chess square."""
        if not (BOARD_OFFSET <= x < BOARD_OFFSET + BOARD_SIZE and
                BOARD_OFFSET <= y < BOARD_OFFSET + BOARD_SIZE):
            return None
        
        file = (x - BOARD_OFFSET) // SQUARE_SIZE
        rank = 7 - (y - BOARD_OFFSET) // SQUARE_SIZE
        
        if flip:
            file = 7 - file
            rank = 7 - rank
        
        return chess.square(file, rank)
    
    def _draw_board(self, flip: bool = False):
        """Draw the chess board."""
        for rank in range(8):
            for file in range(8):
                is_light = (rank + file) % 2 == 1
                color = LIGHT_SQUARE if is_light else DARK_SQUARE
                
                display_file = 7 - file if flip else file
                display_rank = rank if flip else 7 - rank
                
                x = BOARD_OFFSET + display_file * SQUARE_SIZE
                y = BOARD_OFFSET + display_rank * SQUARE_SIZE
                
                pygame.draw.rect(self.screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))
        
        # Draw file/rank labels
        files = 'abcdefgh'
        ranks = '12345678'
        
        if flip:
            files = files[::-1]
            ranks = ranks[::-1]
        
        for i, f in enumerate(files):
            text = self.small_font.render(f, True, TEXT_COLOR)
            x = BOARD_OFFSET + i * SQUARE_SIZE + SQUARE_SIZE // 2 - text.get_width() // 2
            self.screen.blit(text, (x, BOARD_OFFSET + BOARD_SIZE + 5))
        
        for i, r in enumerate(ranks):
            text = self.small_font.render(r, True, TEXT_COLOR)
            y = BOARD_OFFSET + (7 - i) * SQUARE_SIZE + SQUARE_SIZE // 2 - text.get_height() // 2
            self.screen.blit(text, (BOARD_OFFSET - 20, y))
    
    def _draw_highlights(self, flip: bool = False):
        """Draw highlights for selected square, legal moves, and last move."""
        # Last move highlight
        if self.last_move:
            for sq in [self.last_move.from_square, self.last_move.to_square]:
                x, y = self._square_to_coords(sq, flip)
                surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                surf.fill(LAST_MOVE_COLOR)
                self.screen.blit(surf, (x, y))
        
        # Selected square highlight
        if self.selected_square is not None:
            x, y = self._square_to_coords(self.selected_square, flip)
            surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            surf.fill(SELECTED_COLOR)
            self.screen.blit(surf, (x, y))
        
        # Legal move indicators
        for move in self.legal_moves_from_selected:
            x, y = self._square_to_coords(move.to_square, flip)
            # Draw circle for legal move
            center = (x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2)
            
            # If capturing, draw ring instead of dot
            if self.board.piece_at(move.to_square):
                pygame.draw.circle(self.screen, (100, 100, 100), center, SQUARE_SIZE // 2 - 5, 4)
            else:
                pygame.draw.circle(self.screen, (100, 100, 100, 180), center, SQUARE_SIZE // 6)
    
    def _draw_pieces(self, flip: bool = False):
        """Draw all pieces on the board."""
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                x, y = self._square_to_coords(square, flip)
                self.screen.blit(self.pieces[piece.symbol()], (x, y))
    
    def _draw_status(self):
        """Draw game status."""
        # Draw whose turn it is
        if self.game_state == GameState.PLAYING:
            turn = "White" if self.board.turn == chess.WHITE else "Black"
            is_player = self.board.turn == self.config.player_color
            text = f"{turn} to move" + (" (You)" if is_player else " (Bot)")
        elif self.game_state == GameState.WHITE_WINS:
            text = "White wins!"
        elif self.game_state == GameState.BLACK_WINS:
            text = "Black wins!"
        elif self.game_state == GameState.DRAW:
            text = "Draw!"
        else:
            text = ""
        
        if text:
            rendered = self.font.render(text, True, TEXT_COLOR)
            x = (WINDOW_SIZE - rendered.get_width()) // 2
            self.screen.blit(rendered, (x, 5))
    
    def _get_bot_move(self) -> chess.Move:
        """Get the bot's move using the trained model."""
        # Encode board from bot's perspective
        bot_color = not self.config.player_color
        board_tensor = self.board_encoder.encode(self.board, bot_color)
        board_tensor = torch.from_numpy(board_tensor).unsqueeze(0).to(self.device)
        
        # Create legal move mask
        legal_moves = list(self.board.legal_moves)
        legal_move_mask = torch.zeros(1, self.move_encoder.vocab_size, dtype=torch.bool).to(self.device)
        
        for move in legal_moves:
            try:
                idx = self.move_encoder.move_to_idx[move.uci()]
                legal_move_mask[0, idx] = True
            except KeyError:
                pass  # Move not in vocabulary
        
        # Get model prediction
        with torch.no_grad():
            log_probs = self.model(board_tensor, legal_move_mask)
            
            if self.config.temperature == 0:
                # Greedy
                move_idx = log_probs.argmax(dim=1).item()
            else:
                # Sample with temperature
                probs = (log_probs / self.config.temperature).exp()
                probs = probs / probs.sum()
                move_idx = torch.multinomial(probs, 1).item()
        
        move_uci = self.move_encoder.idx_to_move[move_idx]
        return chess.Move.from_uci(move_uci)
    
    def _handle_click(self, pos: Tuple[int, int]):
        """Handle mouse click on the board."""
        flip = self.config.player_color == chess.BLACK
        square = self._coords_to_square(pos[0], pos[1], flip)
        
        if square is None:
            self.selected_square = None
            self.legal_moves_from_selected = []
            return
        
        # If we have a selected square, try to make a move
        if self.selected_square is not None:
            # Check if this is a legal move
            move = None
            for m in self.legal_moves_from_selected:
                if m.to_square == square:
                    move = m
                    break
            
            if move:
                # Handle promotion
                if move.promotion:
                    # For simplicity, always promote to queen
                    # Could add a promotion dialog later
                    pass
                
                self.board.push(move)
                self.last_move = move
                self.selected_square = None
                self.legal_moves_from_selected = []
                self._check_game_over()
                return
        
        # Select new square if it's our piece
        piece = self.board.piece_at(square)
        if piece and piece.color == self.config.player_color and self.board.turn == self.config.player_color:
            self.selected_square = square
            self.legal_moves_from_selected = [m for m in self.board.legal_moves if m.from_square == square]
        else:
            self.selected_square = None
            self.legal_moves_from_selected = []
    
    def _check_game_over(self):
        """Check if the game is over."""
        if self.board.is_checkmate():
            if self.board.turn == chess.WHITE:
                self.game_state = GameState.BLACK_WINS
            else:
                self.game_state = GameState.WHITE_WINS
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or \
             self.board.is_fifty_moves() or self.board.is_repetition():
            self.game_state = GameState.DRAW
    
    def _draw_menu(self):
        """Draw the game setup menu."""
        self.screen.fill(BG_COLOR)
        
        # Title
        title = self.font.render("Chess Mirror", True, TEXT_COLOR)
        self.screen.blit(title, ((WINDOW_SIZE - title.get_width()) // 2, 50))
        
        subtitle = self.small_font.render("Play against your trained bot", True, (180, 180, 180))
        self.screen.blit(subtitle, ((WINDOW_SIZE - subtitle.get_width()) // 2, 90))
        
        # Model selection
        models = self._get_available_models()
        y = 150
        
        if not models:
            text = self.small_font.render("No models found! Train a model first.", True, (255, 100, 100))
            self.screen.blit(text, ((WINDOW_SIZE - text.get_width()) // 2, y))
            return []
        
        text = self.small_font.render("Select a model:", True, TEXT_COLOR)
        self.screen.blit(text, (100, y))
        y += 40
        
        buttons = []
        for i, (path, name) in enumerate(models):
            rect = pygame.Rect(100, y, 200, 40)
            
            mouse_pos = pygame.mouse.get_pos()
            color = BUTTON_HOVER if rect.collidepoint(mouse_pos) else BUTTON_COLOR
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
            
            text = self.small_font.render(name, True, TEXT_COLOR)
            self.screen.blit(text, (rect.x + 20, rect.y + 10))
            
            buttons.append(('model', rect, path, name))
            y += 50
        
        # Color selection
        y += 20
        text = self.small_font.render("Play as:", True, TEXT_COLOR)
        self.screen.blit(text, (100, y))
        y += 40
        
        for color_name, color_value in [("White", chess.WHITE), ("Black", chess.BLACK)]:
            rect = pygame.Rect(100 + (0 if color_value == chess.WHITE else 110), y, 100, 40)
            
            mouse_pos = pygame.mouse.get_pos()
            btn_color = BUTTON_HOVER if rect.collidepoint(mouse_pos) else BUTTON_COLOR
            pygame.draw.rect(self.screen, btn_color, rect, border_radius=5)
            
            text = self.small_font.render(color_name, True, TEXT_COLOR)
            self.screen.blit(text, (rect.x + (rect.width - text.get_width()) // 2, rect.y + 10))
            
            buttons.append(('color', rect, color_value, color_name))
        
        return buttons
    
    def _start_game(self, model_path: str, model_name: str, player_color: chess.Color):
        """Start a new game."""
        self.model = self._load_model(model_path)
        if not self.model:
            return
        
        self.config = GameConfig(
            player_color=player_color,
            model_path=model_path,
            model_name=model_name
        )
        
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves_from_selected = []
        self.last_move = None
        self.game_state = GameState.PLAYING
    
    def run(self):
        """Main game loop."""
        running = True
        selected_model = None
        selected_color = chess.WHITE
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.game_state == GameState.MENU:
                        buttons = self._draw_menu()
                        pos = pygame.mouse.get_pos()
                        
                        for btn_type, rect, value, name in buttons:
                            if rect.collidepoint(pos):
                                if btn_type == 'model':
                                    selected_model = (value, name)
                                elif btn_type == 'color':
                                    selected_color = value
                                    if selected_model:
                                        self._start_game(selected_model[0], selected_model[1], selected_color)
                    
                    elif self.game_state == GameState.PLAYING:
                        if self.board.turn == self.config.player_color:
                            self._handle_click(event.pos)
                    
                    else:
                        # Game over - click to return to menu
                        self.game_state = GameState.MENU
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.game_state != GameState.MENU:
                            self.game_state = GameState.MENU
                        else:
                            running = False
                    
                    elif event.key == pygame.K_r and self.game_state != GameState.MENU:
                        # Restart game
                        if self.config:
                            self._start_game(self.config.model_path, self.config.model_name, 
                                           self.config.player_color)
            
            # Draw
            self.screen.fill(BG_COLOR)
            
            if self.game_state == GameState.MENU:
                self._draw_menu()
            else:
                flip = self.config.player_color == chess.BLACK
                self._draw_board(flip)
                self._draw_highlights(flip)
                self._draw_pieces(flip)
                self._draw_status()
                
                # Bot's turn
                if self.game_state == GameState.PLAYING and self.board.turn != self.config.player_color:
                    pygame.display.flip()
                    pygame.time.wait(500)  # Small delay for visual effect
                    
                    move = self._get_bot_move()
                    self.board.push(move)
                    self.last_move = move
                    self._check_game_over()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Play chess against your trained bot')
    parser.add_argument('--data-dir', type=str, default='./processed',
                        help='Directory containing move_encoder.pkl')
    parser.add_argument('--checkpoints-dir', type=str, default='./checkpoints',
                        help='Directory containing trained models')
    
    args = parser.parse_args()
    
    gui = ChessGUI(data_dir=args.data_dir, checkpoints_dir=args.checkpoints_dir)
    gui.run()


if __name__ == '__main__':
    main()
