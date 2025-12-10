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
      [' ', ' ', 'O', '.', 'O', ' ', ' '],
      ['R', 'R', '.', '.', '.', 'R', 'R'],
      ['R', 'R', 'R', 'R', 'R', 'R', 'R'],
      ['R', 'R', 'R', 'R', 'R', 'R', 'R'],
      [' ', ' ', 'R', 'R', 'R', ' ', ' '],
      [' ', ' ', 'R', 'R', 'R', ' ', ' ']
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
  def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # 先 dummy input 计算 flatten size
        dummy = torch.zeros(1,4,7,7)
        with torch.no_grad():
            enc_out = self.encoder(dummy)
            self.flat_dim = enc_out.numel() // enc_out.shape[0]

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_dim, 256),
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
      scaled = [s / max(1e-6, max(self.temperature, 0.1)) for s in scores]
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


  def _compute_potential(self, board: Sequence[Sequence[str]], is_rebel_turn: bool) -> float:
    # 1. 获取位置
    rebels_list = list(self.rules.piece_positions(board, 'R'))
    rebel_set = set(rebels_list) # <--- 关键优化：转为集合，极大加速后续查找
    officers = list(self.rules.piece_positions(board, 'O'))

    # 2. 输赢终局判定 (极高权重，防止被后续的分数冲淡)
    # 如果 Rebel 没了，或者 Officer 没了，直接返回最大值/最小值
    if not rebels_list: return -1.0 
    if not officers: return 1.0

    # --- 评分逻辑 ---

    # A. 基础分：保命 (Material)
    material_score = len(rebels_list) / 24.0

    # B. 进阶分：堵路 (Clustering/Blocking)
    # 这一项是 Rebel 能够逼平甚至战胜 Officer 的核心
    support_score = 0
    for r, c in rebels_list:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                # 检查周围是否有队友
                if (r+dr, c+dc) in rebel_set: # 这里用 set 查找非常快
                    support_score += 1
    # 归一化：假设每个兵平均有 2-3 个队友支持就是很好的阵型
    support_score /= max(1, len(rebels_list) * 8)

    # C. 进攻分：逼近要塞中心 (1, 3)
    distance_score = 0
    for r, c in rebels_list:
        # 曼哈顿距离：越小越好。最大距离约 8 (从6,6到1,3)
        dist = abs(r - 1) + abs(c - 3)
        distance_score += (10.0 - dist) 
    distance_score /= max(1, len(rebels_list) * 10)

    # --- 组合 ---
    # 早期训练最容易死，所以 Material 权重最高
    # 中期需要学会抱团，Support 权重次之
    phi = (0.4 * material_score + 
           0.4 * support_score + 
           0.2 * distance_score) 
           
    # 映射到 [-1, 1]
    return 2.0 * phi - 1.0

  
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
    num_iterations: int = 50,
    games_per_iter: int = 100,
    max_turns: int = 250,
    epsilon: float = 0.4,
    batch_size: int = 128,
    save_path: str = 'team03_value_net.pt',
    checkpoint_interval: int = 10,
    log_file: str = 'Log.txt',  # 新增日志文件参数
) -> None:
    epsilon_start = max(0.4, epsilon)
    epsilon_end = 0.1
    shaping_alpha = 1.5

    # 清空日志文件
    with open(log_file, 'w') as f:
        f.write("=== Self-Play Training Log ===\n")

    for iteration in range(1, num_iterations + 1):
        samples: List[ReplaySample] = []
        rebel_wins, officer_wins = 0, 0
        total_turns: List[int] = []

        for game_idx in range(1, games_per_iter + 1):
            board = initial_board()
            history: List[Tuple[Tensor, bool, float]] = []
            is_rebel_turn = True
            turns = 0
            winner = 0
            epsilon = epsilon_start - (epsilon_start - epsilon_end) * (iteration - 1) / num_iterations

            while turns < max_turns:
                turns += 1
                moves = self.rules.generate_rebel_moves(board) if is_rebel_turn else self.rules.generate_officer_moves(board)
                if not moves:
                    winner = -1 if is_rebel_turn else 1
                    break

                board_tensor = self._board_to_tensor(board, is_rebel_turn)
                move = self._choose_move(board, is_rebel_turn, epsilon)
                
                if is_rebel_turn:
                    captured_before = sum(row.count('O') for row in board)
                    board = self.rules.apply_move(board, move, is_rebel_turn)
                    captured_after = sum(row.count('O') for row in board)
                    capture_bonus = (captured_before - captured_after) * 0.5
                else:
                    board = self.rules.apply_move(board, move, is_rebel_turn)
                    capture_bonus = 0.0

                history.append((board_tensor, is_rebel_turn, capture_bonus if is_rebel_turn else 0.0))
                outcome = self.rules.outcome(board)
                if outcome != 0:
                    winner = outcome
                    break
                is_rebel_turn = not is_rebel_turn

            total_turns.append(turns)
            rebel_wins += winner > 0
            officer_wins += winner < 0

            for board_tensor, rebel_to_move, step_capture_bonus in history:
                bt = board_tensor.cpu().numpy()
                board_from_tensor = [[' ' for _ in range(7)] for _ in range(7)]
                for r in range(7):
                    for c in range(7):
                        if bt[0, r, c] > 0.5:
                            board_from_tensor[r][c] = 'R'
                        elif bt[1, r, c] > 0.5:
                            board_from_tensor[r][c] = 'O'
                        else:
                            board_from_tensor[r][c] = '.' if bt[2, r, c] > 0.5 else ' '

                phi_rebel = self._compute_potential(board_from_tensor, rebel_to_move)
                phi_current = phi_rebel if rebel_to_move else -phi_rebel

                rebel_terminal = 0.0
                if winner > 0:
                    rebel_terminal = 1.5
                elif winner < 0:
                    rebel_terminal = -0.5
                winner_val = rebel_terminal if rebel_to_move else -rebel_terminal

                target = winner_val + shaping_alpha * phi_current + (step_capture_bonus if rebel_to_move else 0.0)
                samples.append(ReplaySample(board_tensor, float(target)))

            msg = f"[SELF-PLAY] game={game_idx} winner={'R' if winner>0 else ('O' if winner<0 else 'draw')} turns={turns}"
            print(msg)
            with open(log_file, 'a') as f:
                f.write(msg + "\n")

        self._extend_buffer(samples)
        loss = self._update_model(batch_size)

        msg = (f"[TRAIN] iter={iteration:03d} games={games_per_iter} "
               f"rebel_win={rebel_wins/games_per_iter:.2f} "
               f"officer_win={officer_wins/games_per_iter:.2f} "
               f"avg_turns={statistics.mean(total_turns):.1f} loss={loss:.4f}")
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + "\n")

        if save_path and iteration % checkpoint_interval == 0:
            checkpoint_file = save_path.replace('.pt', f'_iter{iteration}.pt')
            self.save(checkpoint_file)
            msg = f"[CHECKPOINT] saved to {checkpoint_file}"
            print(msg)
            with open(log_file, 'a') as f:
                f.write(msg + "\n")

      # 每隔 checkpoint_interval 迭代保存一个带迭代编号的新文件
      # if iteration % checkpoint_interval == 0:
      #     checkpoint_name = save_path.replace('.pt', f'_iter{iteration}.pt')
      #     self.save(checkpoint_name)
      #     print(f"[CHECKPOINT] Saved checkpoint at iteration {iteration} -> {checkpoint_name}")
      #     print(
      #         f"[STATS] iter={iteration} games={stats['games']} "
      #         f"rebel_win={stats['rebel_win_rate']:.2f} "
      #         f"officer_win={stats['officer_win_rate']:.2f} "
      #         f"avg_turns={stats['avg_turns']:.1f} loss={loss:.4f}"
      #     )

  def _collect_self_play(self, games: int, max_turns: int, epsilon_start: float = 0.8, epsilon_end: float = 0.1) -> Tuple[List[ReplaySample], dict]:
    samples: List[ReplaySample] = []
    rebel_wins, officer_wins = 0, 0
    total_turns: List[int] = []

    shaping_alpha = 1.5

    for game_idx in range(1, games+1):
        board = initial_board()
        history: List[Tuple[Tensor,bool]] = []
        is_rebel_turn = True
        turns = 0
        winner = 0

        # epsilon decay
        epsilon = epsilon_start * (1 - (game_idx-1)/games)**0.5  # sqrt decay，更慢收敛，早期多探索

        while turns < max_turns:
            turns += 1
            moves = self.rules.generate_rebel_moves(board) if is_rebel_turn else self.rules.generate_officer_moves(board)
            if not moves:
                winner = -1 if is_rebel_turn else 1
                break

            board_tensor = self._board_to_tensor(board, is_rebel_turn)
            history.append((board_tensor, is_rebel_turn))

            move = self._choose_move(board, is_rebel_turn, epsilon)
            board = self.rules.apply_move(board, move, is_rebel_turn)

            outcome = self.rules.outcome(board)
            if outcome != 0:
                winner = outcome
                break

            is_rebel_turn = not is_rebel_turn
        else:
            winner = 0

        total_turns.append(turns)
        rebel_wins += winner>0
        officer_wins += winner<0

        for board_tensor, rebel_to_move in history:
          bt = board_tensor.cpu().numpy()  # bt.shape = (4,7,7)
          board_from_tensor = [[' ' for _ in range(7)] for _ in range(7)]
          for r in range(7):
              for c in range(7):
                  if bt[0, r, c] > 0.5:
                      board_from_tensor[r][c] = 'R'
                  elif bt[1, r, c] > 0.5:
                      board_from_tensor[r][c] = 'O'
                  else:
                      board_from_tensor[r][c] = '.' if bt[2, r, c] > 0.5 else ' '

          # 势函数：从 Rebel 视角定义，再转换到当前走棋方视角
          phi_rebel = self._compute_potential(board_from_tensor, rebel_to_move)
          phi_current = phi_rebel if rebel_to_move else -phi_rebel

          # 终局奖励同样对 Rebel 略有偏置
          rebel_terminal = 0.0
          if winner > 0:      # Rebel 赢
              rebel_terminal = 1.5
          elif winner < 0:    # Rebel 输
              rebel_terminal = -0.5
          winner_val = rebel_terminal if rebel_to_move else -rebel_terminal

          target = float(winner_val + shaping_alpha * phi_current)
          samples.append(ReplaySample(board_tensor, target))


        print(f"[SELF-PLAY] game={game_idx} winner={'R' if winner>0 else ('O' if winner<0 else 'draw')} turns={turns}")

    stats = {
        'games': games,
        'rebel_win_rate': rebel_wins/games,
        'officer_win_rate': officer_wins/games,
        'avg_turns': statistics.mean(total_turns) if total_turns else 0.0
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
  parser.add_argument('--checkpoint-interval', type=int, default=10,
                        help='Save a new checkpoint every N iterations.')
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
      checkpoint_interval=args.checkpoint_interval,
    )
    print(f'Checkpoint saved to {args.save}')
  else:
    print('Module ready. Import Player in Asalto.py to use the learned agent.')


if __name__ == '__main__':
  _main()