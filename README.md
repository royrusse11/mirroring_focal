# Chess Mirror Bot 🪞♟️

Train a neural network to play chess exactly like YOU, then analyze your own weaknesses!

## Overview

This project:
1. Parses your chess.com game exports (PGN files)
2. Trains a CNN-based policy network on YOUR moves only
3. Creates three models: White-only, Black-only, and Combined
4. Lets you play against your "mirror" bot via a graphical interface
5. Analyzes weaknesses by comparing your bot to Stockfish

## Quick Start

### Step 1: Setup

```bash
# Navigate to project directory
cd ~/Developer/chess-mirror

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Add Your Games

1. Download your games from chess.com:
   - Go to chess.com → Your Profile → Games → Download
   - Download all months you want to include

2. Move the PGN files to the `pgn` folder:
```bash
mkdir -p pgn
mv ~/Desktop/pgn/*.pgn ./pgn/
```

### Step 3: Process Your Games

```bash
cd data
python process_pgn.py --pgn-dir ../pgn --username roy_russe11 --output-dir ../processed
```

This will:
- Parse all PGN files
- Extract positions where YOU made moves
- Save three datasets: white, black, and combined
- Show statistics about your games

### Step 4: Train the Models

```bash
cd ../training
python train.py --data-dir ../processed --output-dir ../checkpoints
```

Training options:
```bash
# Train only the combined model (fastest)
python train.py --model combined

# Adjust epochs and batch size
python train.py --epochs 100 --batch-size 64

# Train with different learning rate
python train.py --lr 0.0005
```

Expected training time on M4 Pro: ~30-60 minutes per model.

### Step 5: Play Against Your Bot!

```bash
cd ../play
python gui.py --checkpoints-dir ../checkpoints --data-dir ../processed
```

Controls:
- **Click** to select and move pieces
- **ESC** to return to menu
- **R** to restart the current game

### Step 6: Analyze Your Weaknesses

First, install Stockfish:
```bash
brew install stockfish
```

Then run the analysis:
```bash
cd ../analysis
python stockfish_analysis.py ../checkpoints/roy_combined_best.pt \
    --games 10 \
    --stockfish-elo 1500 \
    --output ../reports/weakness_report.txt
```

This generates a report showing:
- Average centipawn loss
- Blunder/mistake/inaccuracy rates
- Performance by game phase (opening/middlegame/endgame)
- Missed tactical patterns
- Personalized improvement recommendations

---

## Project Structure

```
chess-mirror/
├── pgn/                    # Your chess.com PGN exports go here
├── processed/              # Processed training data
│   ├── white_dataset.npz
│   ├── black_dataset.npz
│   ├── combined_dataset.npz
│   └── move_encoder.pkl
├── checkpoints/            # Trained model files
│   ├── roy_white_best.pt
│   ├── roy_black_best.pt
│   └── roy_combined_best.pt
├── reports/                # Analysis reports
├── data/
│   └── process_pgn.py      # PGN parsing and data processing
├── model/
│   └── network.py          # Neural network architecture
├── training/
│   └── train.py            # Training script
├── play/
│   └── gui.py              # Graphical chess interface
├── analysis/
│   └── stockfish_analysis.py  # Weakness analysis tool
├── requirements.txt
└── README.md
```

---

## How It Works

### Board Encoding

Each position is encoded as a 14-channel 8x8 tensor:
- Channels 0-5: Your pieces (Pawn, Knight, Bishop, Rook, Queen, King)
- Channels 6-11: Opponent pieces
- Channel 12: Whose turn it is
- Channel 13: Castling rights and en passant

The board is always oriented from YOUR perspective (your pieces at bottom).

### Neural Network

A CNN-based policy network inspired by AlphaZero:
- Input convolution: 14 → 128 channels
- 6 residual blocks with skip connections
- Policy head outputs probability over all legal moves

Total parameters: ~1.5 million (suitable for ~65k training examples)

### Training

- Loss function: Cross-entropy (predicting your actual move)
- Optimizer: AdamW with weight decay
- Learning rate scheduler: ReduceLROnPlateau
- Early stopping: Patience of 10 epochs

### Move Selection

During play, the bot:
1. Encodes the current board position
2. Gets probability distribution over all moves
3. Masks illegal moves
4. Samples from remaining legal moves (with temperature)

Temperature controls randomness:
- 0.0 = Always pick highest probability (deterministic)
- 0.5 = Some randomness (default, more human-like)
- 1.0 = More random exploration

---

## Understanding the Analysis Report

### Centipawn Loss
- Measures how much worse your move is vs the best move
- 100 centipawns = 1 pawn worth of advantage lost
- Average amateur: 50-80 cp loss
- Average 1300 player: 40-60 cp loss

### Error Classification
- **Blunder** (≥100cp): Serious mistake, loses significant material/position
- **Mistake** (50-100cp): Notable error that gives opponent advantage
- **Inaccuracy** (25-50cp): Suboptimal but not critical

### Game Phases
- **Opening** (moves 1-10): Development and piece placement
- **Middlegame**: Main battle, tactics, and strategy
- **Endgame**: Simplified position, king activity matters

---

## Tips for Better Results

### Getting More Training Data

If you want more games:
1. Play more on chess.com!
2. Download historical games going back further
3. Consider data augmentation (board reflections)

### Improving Model Performance

For a stronger bot that better mimics you:
```bash
# More training epochs
python train.py --epochs 100 --patience 20

# Larger model (if you have lots of games)
# Edit network.py: num_filters=256, num_residual_blocks=10
```

### Understanding Your Weaknesses

The analysis report identifies patterns like:
- "Endgame loss is 2x middlegame" → Study endgames
- "Missing knight forks frequently" → Do fork puzzles
- "High opening inaccuracy" → Learn opening principles

---

## Troubleshooting

### "No PGN files found"
Make sure your PGN files are in the `pgn/` directory and have `.pgn` extension.

### "Move not in vocabulary"
This is rare but can happen with unusual promotions. The move is skipped.

### Training is slow
- Ensure MPS is being used (check "Using Apple Silicon MPS acceleration" message)
- Reduce batch size if running out of memory
- Use fewer residual blocks for faster training

### Stockfish not found
Install with: `brew install stockfish`
Or specify path: `--stockfish /path/to/stockfish`

### GUI won't start
Make sure pygame is installed: `pip install pygame`

---

## Future Ideas

- [ ] Add opening book detection and tracking
- [ ] Compare White vs Black model performance
- [ ] Time-based analysis (how you play in time pressure)
- [ ] Opponent strength analysis (how you play vs different ratings)
- [ ] Export games played against the bot
- [ ] Web-based interface option

---

## License

MIT License - Use freely for personal projects!

---

*Built for Roy to study his own chess weaknesses* ♟️
