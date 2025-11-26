"""Asalto AI bot for coursework JC4004 Computational Intelligence 2025-26."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


Coord = Tuple[int, int]
Move = List[List[int]]


class Player:
  """Implements both rebel and officer strategies via alpha-beta search."""

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

  def __init__(self, rebel_depth: int = 3, officer_depth: int = 3) -> None:
    self.rebel_depth = rebel_depth
    self.officer_depth = officer_depth
    self._cache: dict[Tuple[Tuple[Tuple[str, ...], ...], int, bool], int] = {}

  # =================================================
  # Public API
  def play_rebel(self, board: Sequence[Sequence[str]]) -> Move:
    move = self._select_move(board, is_rebel=True)
    return move if move else [[0, 0], [0, 0]]

  def play_officer(self, board: Sequence[Sequence[str]]) -> Move:
    move = self._select_move(board, is_rebel=False)
    return move if move else [[0, 0], [0, 0]]

  # =================================================
  # Core search
  def _select_move(self, board: Sequence[Sequence[str]], is_rebel: bool) -> Move | None:
    depth = self.rebel_depth if is_rebel else self.officer_depth
    moves = self._generate_rebel_moves(board) if is_rebel else self._generate_officer_moves(board)
    if not moves:
      return None

    alpha = float('-inf')
    beta = float('inf')
    best_value = float('-inf') if is_rebel else float('inf')
    best_moves: List[Move] = []

    for move in moves:
      new_board = self._apply_move(board, move, is_rebel)
      value = self._search(new_board, depth - 1, not is_rebel, alpha, beta)
      if is_rebel:
        if value > best_value:
          best_value = value
          best_moves = [move]
        elif value == best_value:
          best_moves.append(move)
        alpha = max(alpha, best_value)
      else:
        if value < best_value:
          best_value = value
          best_moves = [move]
        elif value == best_value:
          best_moves.append(move)
        beta = min(beta, best_value)
      if alpha >= beta:
        break

    return min(best_moves, key=self._move_sort_key)

  def _search(
    self,
    board: Sequence[Sequence[str]],
    depth: int,
    is_rebel_turn: bool,
    alpha: float,
    beta: float,
  ) -> int:
    board_key = self._board_key(board)
    cache_key = (board_key, depth, is_rebel_turn)
    if cache_key in self._cache:
      return self._cache[cache_key]

    if depth <= 0 or self._is_terminal(board):
      value = self._evaluate(board)
      self._cache[cache_key] = value
      return value

    if is_rebel_turn:
      value = -1_000_000_000
      for move in self._generate_rebel_moves(board):
        new_board = self._apply_move(board, move, True)
        value = max(value, self._search(new_board, depth - 1, False, alpha, beta))
        alpha = max(alpha, value)
        if alpha >= beta:
          break
    else:
      value = 1_000_000_000
      for move in self._generate_officer_moves(board):
        new_board = self._apply_move(board, move, False)
        value = min(value, self._search(new_board, depth - 1, True, alpha, beta))
        beta = min(beta, value)
        if alpha >= beta:
          break

    self._cache[cache_key] = value
    return value

  # =================================================
  # Move generation helpers
  def _generate_rebel_moves(self, board: Sequence[Sequence[str]]) -> List[Move]:
    moves: List[Move] = []
    for row, col in self._piece_positions(board, 'R'):
      for target in self._rebel_targets(board, row, col):
        moves.append([[row, col], [target[0], target[1]]])
    return moves

  def _generate_officer_moves(self, board: Sequence[Sequence[str]]) -> List[Move]:
    capture_moves: List[Move] = []
    for start in self._piece_positions(board, 'O'):
      capture_moves.extend(self._officer_capture_paths(board, start))
    if capture_moves:
      return capture_moves

    moves: List[Move] = []
    for row, col in self._piece_positions(board, 'O'):
      for target in self._officer_step_targets(board, row, col):
        moves.append([[row, col], [target[0], target[1]]])
    return moves

  def _officer_capture_paths(self, board: Sequence[Sequence[str]], start: Coord) -> List[Move]:
    results: List[Move] = []

    def dfs(current_board: List[List[str]], current: Coord, path: List[Coord]) -> None:
      extended = False
      for dr, dc in self.DIRECTIONS:
        if dr == 0 and dc == 0:
          continue
        if dr != 0 and dc != 0 and not self._diagonal_allowed(current):
          continue
        mid = (current[0] + dr, current[1] + dc)
        landing = (current[0] + 2 * dr, current[1] + 2 * dc)
        if not self._is_valid_capture(board=current_board, mid=mid, landing=landing):
          continue
        new_board = self._clone_board(current_board)
        new_board[current[0]][current[1]] = '.'
        new_board[mid[0]][mid[1]] = '.'
        new_board[landing[0]][landing[1]] = 'O'
        dfs(new_board, landing, path + [landing])
        extended = True
      if not extended and len(path) > 1:
        results.append([[r, c] for r, c in path])

    dfs(self._clone_board(board), start, [start])
    return results

  # =================================================
  # Evaluation utilities
  def _evaluate(self, board: Sequence[Sequence[str]]) -> int:
    if self._rebel_wins(board):
      return 100000
    if self._officer_wins(board):
      return -100000

    rebels = list(self._piece_positions(board, 'R'))
    officers = list(self._piece_positions(board, 'O'))

    rebel_count = len(rebels)
    fortress_rebels = sum(1 for pos in rebels if pos in self.FORTRESS)
    progress = sum(6 - row for row, _ in rebels)
    centrality = sum(3 - abs(col - 3) for _, col in rebels)
    rebel_mobility = sum(1 for row, col in rebels for _ in self._rebel_targets(board, row, col))
    officer_mobility = sum(1 for row, col in officers for _ in self._officer_step_targets(board, row, col))
    capture_pressure = sum(self._count_immediate_captures(board, row, col) for row, col in officers)

    score = 0
    score += rebel_count * 80
    score += fortress_rebels * 120
    score += progress * 4
    score += centrality * 6
    score += rebel_mobility * 5
    score -= officer_mobility * 5
    score -= capture_pressure * 40
    score -= len(officers) * 150

    return score

  def _is_terminal(self, board: Sequence[Sequence[str]]) -> bool:
    return self._rebel_wins(board) or self._officer_wins(board)

  def _rebel_wins(self, board: Sequence[Sequence[str]]) -> bool:
    if all(board[row][col] == 'R' for row, col in self.FORTRESS):
      return True
    if not any(value == 'O' for row in board for value in row):
      return True
    if not self._officer_has_move(board):
      return True
    return False

  def _officer_wins(self, board: Sequence[Sequence[str]]) -> bool:
    rebel_count = sum(value == 'R' for row in board for value in row)
    if rebel_count < 9:
      return True
    if not self._rebel_has_move(board):
      return True
    return False

  # =================================================
  # Low-level helpers
  @staticmethod
  def _clone_board(board: Sequence[Sequence[str]]) -> List[List[str]]:
    return [list(row) for row in board]

  @staticmethod
  def _piece_positions(board: Sequence[Sequence[str]], piece: str) -> Iterable[Coord]:
    for row_idx, row in enumerate(board):
      for col_idx, value in enumerate(row):
        if value == piece:
          yield (row_idx, col_idx)

  @staticmethod
  def _is_on_board(board: Sequence[Sequence[str]], position: Coord) -> bool:
    row, col = position
    return 0 <= row < len(board) and 0 <= col < len(board[row]) and board[row][col] != ' '

  @staticmethod
  def _diagonal_allowed(position: Coord) -> bool:
    return (position[0] % 2) == (position[1] % 2)

  def _rebel_targets(self, board: Sequence[Sequence[str]], row: int, col: int) -> Iterable[Coord]:
    for dr, dc in self.DIRECTIONS:
      if dr == 0 and dc == 0:
        continue
      if dr > 0:
        continue  # rebels cannot move downwards
      if dr != 0 and dc != 0 and not self._diagonal_allowed((row, col)):
        continue
      target = (row + dr, col + dc)
      if not self._is_on_board(board, target):
        continue
      if board[target[0]][target[1]] != '.':
        continue
      if col < 3 and target[1] < col:
        continue
      if col > 3 and target[1] > col:
        continue
      yield target

  def _officer_step_targets(self, board: Sequence[Sequence[str]], row: int, col: int) -> Iterable[Coord]:
    for dr, dc in self.DIRECTIONS:
      if dr == 0 and dc == 0:
        continue
      if dr != 0 and dc != 0 and not self._diagonal_allowed((row, col)):
        continue
      target = (row + dr, col + dc)
      if not self._is_on_board(board, target):
        continue
      if board[target[0]][target[1]] == '.':
        yield target

  def _count_immediate_captures(self, board: Sequence[Sequence[str]], row: int, col: int) -> int:
    count = 0
    for dr, dc in self.DIRECTIONS:
      if dr == 0 and dc == 0:
        continue
      if dr != 0 and dc != 0 and not self._diagonal_allowed((row, col)):
        continue
      mid = (row + dr, col + dc)
      landing = (row + 2 * dr, col + 2 * dc)
      if self._is_valid_capture(board, mid, landing):
        count += 1
    return count

  def _officer_has_move(self, board: Sequence[Sequence[str]]) -> bool:
    for row, col in self._piece_positions(board, 'O'):
      if self._count_immediate_captures(board, row, col) > 0:
        return True
      if any(True for _ in self._officer_step_targets(board, row, col)):
        return True
    return False

  def _rebel_has_move(self, board: Sequence[Sequence[str]]) -> bool:
    for row, col in self._piece_positions(board, 'R'):
      if any(True for _ in self._rebel_targets(board, row, col)):
        return True
    return False

  def _is_valid_capture(self, board: Sequence[Sequence[str]], mid: Coord, landing: Coord) -> bool:
    if not self._is_on_board(board, mid) or not self._is_on_board(board, landing):
      return False
    return board[mid[0]][mid[1]] == 'R' and board[landing[0]][landing[1]] == '.'

  def _apply_move(self, board: Sequence[Sequence[str]], move: Move, is_rebel: bool) -> List[List[str]]:
    new_board = self._clone_board(board)
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

  @staticmethod
  def _board_key(board: Sequence[Sequence[str]]) -> Tuple[Tuple[str, ...], ...]:
    return tuple(tuple(row) for row in board)

  @staticmethod
  def _move_sort_key(move: Move) -> Tuple[int, ...]:
    return tuple(coord for step in move for coord in step)

# ==== End of file