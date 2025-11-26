"""Deep-learning powered Asalto player for Team03.

Features
--------
* `Player.play_rebel` / `Player.play_officer` run a neural policy by default.
* `Player.train_self_play` plays short matches against itself, logs win rates,
  and trains a value network (stored in `team03_value_net.pt`).
* A small CLI (see `python Team03_learning.py --help`) can launch training
  before you plug the model back into `Asalto.py` for grading matches.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

Coord = Tuple[int, int]
Move = List[List[int]]


def initial_board() -> List[List[str]]:
  return [
    [' ', ' ', '.', '.', '.', ' ', ' '],
    [' ', ' ', '.', 'O', '.', ' ', ' '],
    ['R', 'R', '.', 'R', 'O', 'R', 'R'],
    ['R', 'R', 'R', '.', 'R', 'R', 'R'],
    ['R', 'R', 'R', 'R', 'R', 'R', 'R'],
    [' ', ' ', 'R', 'R', 'R', ' ', ' '],
    [' ', ' ', 'R', 'R', 'R', ' ', ' '],
  ]


class AsaltoRules:
  FORTRESS: frozenset[Coord] = frozenset(
    {
      (0, 2), (0, 3), (0, 4),
      (1, 2), (1, 3), (1, 4),
      (2, 2), (2, 3), (2, 4),
    }
  )
  DIRECTIONS: Tuple[Coord, ...] = (
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
  )

  @staticmethod
  def clone(board: Sequence[Sequence[str]]) -> List[List[str]]:
    return [list(row) for row in board]

  @staticmethod
  def is_on_board(board: Sequence[Sequence[str]], position: Coord) -> bool:
    row, col = position
    return 0 <= row < len(board) and 0 <= col < len(board[row]) and board[row][col] != ' '

  @staticmethod
  def diagonal_allowed(position: Coord) -> bool:
    return (position[0] % 2) == (position[1] % 2)

  @staticmethod
  def piece_positions(board: Sequence[Sequence[str]], piece: str) -> Iterable[Coord]:
    for row_idx, row in enumerate(board):
      for col_idx, value in enumerate(row):
        if value == piece:
          yield (row_idx, col_idx)

  def rebel_targets(self, board: Sequence[Sequence[str]], row: int, col: int) -> Iterable[Coord]:
    for dr, dc in self.DIRECTIONS:
      if dr == 0 and dc == 0:
        continue
      if dr > 0:
        continue
      if dr != 0 and dc != 0 and not self.diagonal_allowed((row, col)):
        continue
      target = (row + dr, col + dc)
      if not self.is_on_board(board, target):
        continue
      if board[target[0]][target[1]] != '.':
        continue
      if col < 3 and target[1] < col:
        continue
      if col > 3 and target[1] > col:
        continue
      yield target

  def officer_step_targets(self, board: Sequence[Sequence[str]], row: int, col: int) -> Iterable[Coord]:
    for dr, dc in self.DIRECTIONS:
      if dr == 0 and dc == 0:
        continue
      if dr != 0 and dc != 0 and not self.diagonal_allowed((row, col)):
        continue
      target = (row + dr, col + dc)
      if not self.is_on_board(board, target):
        continue
      if board[target[0]][target[1]] == '.':
        yield target

  def officer_capture_paths(self, board: Sequence[Sequence[str]], start: Coord) -> List[Move]:
    results: List[Move] = []

    def dfs(current_board: List[List[str]], current: Coord, path: List[Coord]) -> None:
      extended = False
      for dr, dc in self.DIRECTIONS:
        if dr == 0 and dc == 0:
          continue
        if dr != 0 and dc != 0 and not self.diagonal_allowed(current):
          continue
        mid = (current[0] + dr, current[1] + dc)
        landing = (current[0] + 2 * dr, current[1] + 2 * dc)
        if not self.is_valid_capture(current_board, mid, landing):
          continue
        new_board = self.clone(current_board)
        new_board[current[0]][current[1]] = '.'
        new_board[mid[0]][mid[1]] = '.'
        new_board[landing[0]][landing[1]] = 'O'
        dfs(new_board, landing, path + [landing])
        extended = True
      if not extended and len(path) > 1:
        results.append([[r, c] for r, c in path])

    dfs(self.clone(board), start, [start])
    return results

  def generate_rebel_moves(self, board: Sequence[Sequence[str]]) -> List[Move]:
    moves: List[Move] = []
    for row, col in self.piece_positions(board, 'R'):
      for target in self.rebel_targets(board, row, col):
        moves.append([[row, col], [target[0], target[1]]])
    return moves

  def generate_officer_moves(self, board: Sequence[Sequence[str]]) -> List[Move]:
    capture_moves: List[Move] = []
    for start in self.piece_positions(board, 'O'):
      capture_moves.extend(self.officer_capture_paths(board, start))
    if capture_moves:
      return capture_moves
    moves: List[Move] = []
    for row, col in self.piece_positions(board, 'O'):
      for target in self.officer_step_targets(board, row, col):
        moves.append([[row, col], [target[0], target[1]]])
    return moves

  def is_valid_capture(self, board: Sequence[Sequence[str]], mid: Coord, landing: Coord) -> bool:
    if not self.is_on_board(board, mid) or not self.is_on_board(board, landing):
      return False
    return board[mid[0]][mid[1]] == 'R' and board[landing[0]][landing[1]] == '.'

  def apply_move(self, board: Sequence[Sequence[str]], move: Move, is_rebel: bool) -> List[List[str]]:
    new_board = self.clone(board)
    start_row, start_col = move[0]
    new_board[start_row][start_col] = '.'
    prev_row, prev_col = start_row, start_col
    for step_row, step_col in move[1:]:
      if not is_rebel and (abs(step_row - prev_row) > 1 or abs(step_col - prev_col) > 1):
        mid_row = (step_row + prev_row) // 2
        mid_col = (step_col + prev_col) // 2
        new_board[mid_row][mid_col] = '.'
      prev_row, prev_col = step_row, step_col
    end_row, end_col = move[-1]
    new_board[end_row][end_col] = 'R' if is_rebel else 'O'
    return new_board

  def rebel_wins(self, board: Sequence[Sequence[str]]) -> bool:
    if all(board[row][col] == 'R' for row, col in self.FORTRESS):
      return True
    if not any(value == 'O' for row in board for value in row):
      return True
    if not self.officer_has_move(board):
      return True
    return False

  def officer_wins(self, board: Sequence[Sequence[str]]) -> bool:
    rebel_count = sum(value == 'R' for row in board for value in row)
    if rebel_count < 9:
      return True
    if not self.rebel_has_move(board):
      return True
    return False

  def rebel_has_move(self, board: Sequence[Sequence[str]]) -> bool:
    for row, col in self.piece_positions(board, 'R'):
      if any(True for _ in self.rebel_targets(board, row, col)):
        return True
    return False

  def officer_has_move(self, board: Sequence[Sequence[str]]) -> bool:
    for row, col in self.piece_positions(board, 'O'):
      if any(True for _ in self.officer_step_targets(board, row, col)):
        return True
      if self.count_immediate_captures(board, row, col) > 0:
        return True
    return False

  def count_immediate_captures(self, board: Sequence[Sequence[str]], row: int, col: int) -> int:
    count = 0
    for dr, dc in self.DIRECTIONS:
      if dr == 0 and dc == 0:
        continue
      if dr != 0 and dc != 0 and not self.diagonal_allowed((row, col)):
        continue
      mid = (row + dr, col + dc)
      landing = (row + 2 * dr, col + 2 * dc)
      if self.is_valid_capture(board, mid, landing):
        count += 1
    return count

  def outcome(self, board: Sequence[Sequence[str]]) -> int:
    if self.rebel_wins(board):
      return 1
    if self.officer_wins(board):
      return -1
    return 0


class ValueNet(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.encoder = nn.Sequential(
      nn.Conv2d(4, 32, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, padding=1),
      nn.ReLU(),
    )
    self.head = nn.Sequential(
      nn.Flatten(),
      nn.Linear(64 * 7 * 7, 256),
      nn.ReLU(),
      nn.Linear(256, 64),
      nn.ReLU(),
      nn.Linear(64, 1),
      nn.Tanh(),
    )

  def forward(self, x: Tensor) -> Tensor:
    return self.head(self.encoder(x)).squeeze(-1)


@dataclass
class ReplaySample:
  board_tensor: Tensor
  target: float


class Player:
  def __init__(
    self,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    temperature: float = 0.5,
    buffer_limit: int = 4096,
  ) -> None:
    self.rules = AsaltoRules()
    preferred_device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    self.device = torch.device(preferred_device)
    self.model = ValueNet().to(self.device)
    self.temperature = max(0.0, temperature)
    self.replay_buffer: List[ReplaySample] = []
    self.buffer_limit = buffer_limit
    self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
    self.loss_fn = nn.MSELoss()
    if model_path:
      self.load(model_path)

  def play_rebel(self, board: Sequence[Sequence[str]]) -> Move:
    return self._choose_move(board, True)

  def play_officer(self, board: Sequence[Sequence[str]]) -> Move:
    return self._choose_move(board, False)

  def _choose_move(self, board: Sequence[Sequence[str]], is_rebel: bool, epsilon: float = 0.0) -> Move:
    moves = self.rules.generate_rebel_moves(board) if is_rebel else self.rules.generate_officer_moves(board)
    if not moves:
      return [[0, 0], [0, 0]]

    if epsilon > 0 and random.random() < epsilon:
      return random.choice(moves)

    scores: List[float] = []
    for move in moves:
      next_board = self.rules.apply_move(board, move, is_rebel)
      next_turn_is_rebel = not is_rebel
      value = self._predict_value(next_board, next_turn_is_rebel)
      scores.append(value if is_rebel else -value)

    if self.temperature > 0 and len(moves) > 1:
      scaled = [s / max(1e-6, self.temperature) for s in scores]
      max_scaled = max(scaled)
      exp_scores = [math.exp(s - max_scaled) for s in scaled]
      total = sum(exp_scores)
      probs = [score / total for score in exp_scores]
      return random.choices(moves, weights=probs, k=1)[0]

    best_idx = max(range(len(moves)), key=lambda idx: scores[idx])
    return moves[best_idx]

  def _predict_value(self, board: Sequence[Sequence[str]], is_rebel_turn: bool) -> float:
    tensor = self._board_to_tensor(board, is_rebel_turn).unsqueeze(0).to(self.device)
    self.model.eval()
    with torch.no_grad():
      return float(self.model(tensor)[0].item())

  def _board_to_tensor(self, board: Sequence[Sequence[str]], is_rebel_turn: bool) -> Tensor:
    tensor = torch.zeros((4, 7, 7), dtype=torch.float32)
    for row_idx, row in enumerate(board):
      for col_idx, cell in enumerate(row):
        if cell == 'R':
          tensor[0, row_idx, col_idx] = 1.0
        elif cell == 'O':
          tensor[1, row_idx, col_idx] = 1.0
        if cell != ' ':
          tensor[2, row_idx, col_idx] = 1.0
    if is_rebel_turn:
      tensor[3, :, :] = 1.0
    return tensor

  def save(self, path: str) -> None:
    torch.save(self.model.state_dict(), path)

  def load(self, path: str) -> None:
    self.model.load_state_dict(torch.load(path, map_location=self.device))

  def train_self_play(
    self,
    num_iterations: int = 3,
    games_per_iter: int = 10,
    max_turns: int = 200,
    epsilon: float = 0.2,
    batch_size: int = 64,
    save_path: str = 'team03_value_net.pt',
  ) -> None:
    for iteration in range(1, num_iterations + 1):
      samples, stats = self._collect_self_play(games_per_iter, max_turns, epsilon)
      self._extend_buffer(samples)
      loss = self._update_model(batch_size)
      self._log_iteration(iteration, stats, loss)
      if save_path:
        self.save(save_path)

  def _collect_self_play(self, games: int, max_turns: int, epsilon: float) -> Tuple[List[ReplaySample], dict]:
    samples: List[ReplaySample] = []
    rebel_wins = 0
    officer_wins = 0
    total_turns: List[int] = []
    for game_idx in range(1, games + 1):
      board = initial_board()
      history: List[Tuple[Tensor, bool]] = []
      is_rebel_turn = True
      turns = 0
      winner = 0
      while turns < max_turns:
        turns += 1
        moves = self.rules.generate_rebel_moves(board) if is_rebel_turn else self.rules.generate_officer_moves(board)
        if not moves:
          winner = -1 if is_rebel_turn else 1
          break
        board_tensor = self._board_to_tensor(board, is_rebel_turn)
        history.append((board_tensor, is_rebel_turn))
        move = self._choose_move(board, is_rebel_turn, epsilon=epsilon)
        board = self.rules.apply_move(board, move, is_rebel_turn)
        outcome = self.rules.outcome(board)
        if outcome != 0:
          winner = outcome
          break
        is_rebel_turn = not is_rebel_turn
      else:
        winner = 0

      total_turns.append(turns)
      if winner > 0:
        rebel_wins += 1
      elif winner < 0:
        officer_wins += 1

      for board_tensor, rebel_to_move in history:
        target = float(winner if rebel_to_move else -winner)
        samples.append(ReplaySample(board_tensor, target))

      print(f"[SELF-PLAY] game={game_idx} winner={'R' if winner>0 else ('O' if winner<0 else 'draw')} turns={turns}")

    stats = {
      'games': games,
      'rebel_win_rate': rebel_wins / games if games else 0.0,
      'officer_win_rate': officer_wins / games if games else 0.0,
      'avg_turns': statistics.mean(total_turns) if total_turns else 0.0,
    }
    return samples, stats

  def _extend_buffer(self, samples: List[ReplaySample]) -> None:
    self.replay_buffer.extend(samples)
    if len(self.replay_buffer) > self.buffer_limit:
      excess = len(self.replay_buffer) - self.buffer_limit
      self.replay_buffer = self.replay_buffer[excess:]

  def _update_model(self, batch_size: int) -> float:
    if not self.replay_buffer:
      return 0.0
    self.model.train()
    batch = random.sample(self.replay_buffer, min(batch_size, len(self.replay_buffer)))
    boards = torch.stack([sample.board_tensor for sample in batch]).to(self.device)
    targets = torch.tensor([sample.target for sample in batch], dtype=torch.float32, device=self.device)
    preds = self.model(boards)
    loss = self.loss_fn(preds, targets)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return float(loss.item())

  def _log_iteration(self, iteration: int, stats: dict, loss: float) -> None:
    print(
      "[TRAIN] iter={:03d} games={} rebel_win={:.2f} officer_win={:.2f} avg_turns={:.1f} loss={:.4f}".format(
        iteration,
        stats['games'],
        stats['rebel_win_rate'],
        stats['officer_win_rate'],
        stats['avg_turns'],
        loss,
      )
    )


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Train the Team03 learning agent')
  parser.add_argument('--train', action='store_true', help='Run self-play training loop.')
  parser.add_argument('--iterations', type=int, default=3, help='Outer training iterations.')
  parser.add_argument('--games', type=int, default=10, help='Self-play games per iteration.')
  parser.add_argument('--epsilon', type=float, default=0.2, help='Exploration prob during training.')
  parser.add_argument('--load', type=str, default=None, help='Path to existing checkpoint to load.')
  parser.add_argument('--save', type=str, default='team03_value_net.pt', help='Where to save checkpoints.')
  parser.add_argument('--device', type=str, default=None, help='Force cpu/cuda device string.')
  return parser.parse_args()


def _main() -> None:
  args = _parse_args()
  player = Player(model_path=args.load, device=args.device)
  if args.train:
    player.train_self_play(
      num_iterations=args.iterations,
      games_per_iter=args.games,
      epsilon=args.epsilon,
      save_path=args.save,
    )
    print(f'Checkpoint saved to {args.save}')
  else:
    print('Module ready. Import Player in Asalto.py to use the learned agent.')


if __name__ == '__main__':
  _main()