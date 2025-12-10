# Asalto testing framework
# JC4004 Computational Intelligence 2025-26

import copy
import time


class Asalto:

  # =================================================
  # Initialise the board for a new game
  def __init__(self):

    # Initialise a new game
    self.board = [
      [' ', ' ', '.', '.', '.', ' ', ' '],
      [' ', ' ', 'O', '.', 'O', ' ', ' '],
      ['R', 'R', '.', '.', '.', 'R', 'R'],
      ['R', 'R', 'R', 'R', 'R', 'R', 'R'],
      ['R', 'R', 'R', 'R', 'R', 'R', 'R'],
      [' ', ' ', 'R', 'R', 'R', ' ', ' '],
      [' ', ' ', 'R', 'R', 'R', ' ', ' ']
    ]
    self.officer_illegal = 0
    self.rebel_illegal = 0
    self.rounds_played = 0
    self.winner = ''

  # =================================================
  # Pretty-print the current board state
  def print_board(self, heading=None):
    if heading:
      print(heading)
    print('  0 1 2 3 4 5 6')
    for row_idx, row in enumerate(self.board):
      line = str(row_idx) + ' '
      for cell in row:
        line += (cell if cell != ' ' else ' ') + ' '
      print(line.rstrip())
    print('')

  # =================================================
  # Increase counter for illegal moves
  def increase_illegal_moves(self, is_rebel):
    if is_rebel:
      self.rebel_illegal += 1
    else:
      self.officer_illegal += 1

  # =================================================
  # Check if the move is valid and update the board
  def is_valid_move(self, is_rebel, moves):

    # Round is counted even if the move is illegal
    self.rounds_played += 1

    # Test for valid number of moves and value of staring position
    if len(moves) < 2:
      self.increase_illegal_moves(is_rebel)
      return False
    elif moves[0][0] < 0 or moves[0][0] >= len(self.board) or moves[0][1] < 0 or moves[0][1] >= len(self.board):
      self.increase_illegal_moves(is_rebel)
      return False
    elif len(moves) > 2 and is_rebel:
      self.increase_illegal_moves(is_rebel)
      return False
    elif self.board[moves[0][0]][moves[0][1]] != 'R' and is_rebel == True:
      self.increase_illegal_moves(is_rebel)
      return False
    elif self.board[moves[0][0]][moves[0][1]] != 'O' and is_rebel == False:
      self.increase_illegal_moves(is_rebel)
      return False

    # Initialise
    new_board = copy.deepcopy(self.board)
    captured = False

    # Loop through moves to test their validity
    for i in range(len(moves) - 1):
      from_pos = moves[i]
      to_pos = moves[i+1]

      if from_pos[0] == to_pos[0] and from_pos[1] == to_pos[1]:
        self.increase_illegal_moves(is_rebel)
        return False
      if (from_pos[0] != to_pos[0] and from_pos[1] != to_pos[1]) and ((from_pos[0] % 2) != (from_pos[1] % 2)):
        self.increase_illegal_moves(is_rebel)
        return False
      elif to_pos[0] < 0 or to_pos[1] < 0 or to_pos[0] >= len(new_board) or to_pos[1] >= len(new_board):
        self.increase_illegal_moves(is_rebel)
        return False
      elif new_board[to_pos[0]][to_pos[1]] != '.':
        self.increase_illegal_moves(is_rebel)
        return False
      elif abs(from_pos[0] - to_pos[0]) > 2 or abs(from_pos[1] - to_pos[1]) > 2:
        self.increase_illegal_moves(is_rebel)
        return False
      elif abs(from_pos[0] - to_pos[0]) > 1 or abs(from_pos[1] - to_pos[1]) > 1:

        # You can only move more than one step if you are a fox capturing a rebel
        if is_rebel or (not captured and i > 0):
          self.increase_illegal_moves(is_rebel)
          return False
        elif (abs(from_pos[0] - to_pos[0]) + abs(from_pos[1] - to_pos[1])) % 2 != 0:
          self.increase_illegal_moves(is_rebel)
          return False
        mid_pos = [(from_pos[0] + to_pos[0]) // 2, (from_pos[1] + to_pos[1]) // 2]
        if new_board[mid_pos[0]][mid_pos[1]] != 'R':
          self.increase_illegal_moves(is_rebel)
          return False
        new_board[to_pos[0]][to_pos[1]] = new_board[from_pos[0]][from_pos[1]]
        new_board[mid_pos[0]][mid_pos[1]] = '.'
        new_board[from_pos[0]][from_pos[1]] = '.'
        captured = True

      else:

        # You can only chain moves if you are capturing consecutively
        if i > 0:
          self.increase_illegal_moves(is_rebel)
          return False

        # Test if rebel moving away from fortress
        if is_rebel:

          # Test if rebel moving downwards
          if to_pos[0] > from_pos[0]:
            self.increase_illegal_moves(is_rebel)
            return False

          # Test if rebel on the left of the middle column and moving further to left
          if to_pos[1] < from_pos[1] < 3:
            self.increase_illegal_moves(is_rebel)
            return False

          # Test if rebel on the right of the middle column and moving further to right
          if to_pos[1] > from_pos[1] > 3:
            self.increase_illegal_moves(is_rebel)
            return False

        # Test if officer COULD capture a rebel: in this case, remove (huff) officer!
        elif not captured:

          officer_pos = [[from_pos[0],from_pos[1]]]
          for i2 in range(len(self.board)):
            for j2 in range(len(self.board[i2])):
              if (i2 != from_pos[0] or j2 != from_pos[1]) and self.board[i2][j2] == 'O':
                officer_pos.append([i2, j2])

          # Can any of the officers capture?
          for k in range(len(officer_pos)):
            for i2 in range(max(officer_pos[k][0] - 1, 0), min(officer_pos[k][0] + 2, len(self.board))):
              for j2 in range(max(officer_pos[k][1] - 1, 0), min(officer_pos[k][1] + 2, len(self.board))):
                if (i2 == officer_pos[k][0] or j2 == officer_pos[k][1]) or (
                          (officer_pos[k][0] % 2) == (officer_pos[k][1] % 2)):
                  if self.board[i2][j2] == 'R':
                    capture_pos = [i2 + i2 - officer_pos[k][0], j2 + j2 - officer_pos[k][1]]
                    if (capture_pos[0] >= 0) and (capture_pos[0] < len(self.board)) and (capture_pos[1] >= 0) and (capture_pos[1] < len(self.board)):
                      if self.board[capture_pos[0]][capture_pos[1]] == '.':
                        self.board[officer_pos[k][0]][officer_pos[k][1]] = '.' # Huffing
                        return False  # Officer can capture a rebel

        # This move is legal
        new_board[to_pos[0]][to_pos[1]] = new_board[from_pos[0]][from_pos[1]]
        new_board[from_pos[0]][from_pos[1]] = '.'
        captured = False

    # Legal move(s): update the board
    self.board = new_board
    return True

  # =================================================
  # Check if the game is won by one player
  def check_win(self):

    # Checks if the game is won by the officers
    rebel_count = 0
    rebel_fortress_count = 0
    for i in range(len(self.board)):
      for j in range(len(self.board[i])):
        if self.board[i][j] == 'R':
          rebel_count += 1
          if i < 3 and 1 < j < 5:
             rebel_fortress_count += 1

    # If less than 9 rebels left, the officers wins
    if rebel_count < 9:
      self.winner = 'O'
      return True

    # Checks if the game is won by the rebels
    if rebel_fortress_count == 9:
      self.winner = 'R'
      return True

    officer_pos = []
    for i in range(len(self.board)):
      for j in range(len(self.board[i])):
        if self.board[i][j] == 'O':
          officer_pos.append([i,j])

    # Can any of the officers move?
    for k in range(len(officer_pos)):
      for i in range(max(officer_pos[k][0] - 1, 0), min(officer_pos[k][0] + 2, len(self.board))):
        for j in range(max(officer_pos[k][1] - 1, 0), min(officer_pos[k][1] + 2, len(self.board))):
          if (i == officer_pos[k][0] or j == officer_pos[k][1]) or ((officer_pos[k][0] % 2) == (officer_pos[k][1] % 2)):
            if self.board[i][j] == '.':
              return False  # Officer can move one step
            if self.board[i][j] == 'R':
              capture_pos = [i + i - officer_pos[k][0], j + j - officer_pos[k][1]]
              if (capture_pos[0] >= 0) and (capture_pos[0] < len(self.board)) and (capture_pos[1] >= 0) and (capture_pos[1] < len(self.board)):
                if self.board[capture_pos[0]][capture_pos[1]] == '.':
                  return False  # Officer can capture a rebel

    # The officers cannot move, the rebels win
    self.winner = 'R'
    return True

  # =================================================
  # Play one round of the game between players 1 (goose) and 2 (fox)
  def play(self, rebel_player, officer_player):

    # Play a round of the game
    game_over = False
    rounds_played = 0
    while not game_over:

      # The rebel plays first
      temp_board = copy.deepcopy(self.board)
      start_time = time.time()
      try:
        move = rebel_player.play_rebel(temp_board)
      except Exception as err:
        print("Exception caught for rebel!")
        self.winner = 'O'
        game_over = True
      else:
        used_time = time.time() - start_time
        if used_time > 10000: # This will 10 when testing the bots
          print("Rebel exceeded time limit!")
          self.winner = 'O'
          game_over = True
        elif not self.is_valid_move(True, move):
          print("Illegal move by rebel!")
        elif self.check_win():
          game_over = True

      if not game_over:

        # The officer plays second
        temp_board = copy.deepcopy(self.board)
        start_time = time.time()
        try:
          move = officer_player.play_officer(temp_board)
        except Exception as err:
          print("Exception caught for officer!")
          self.winner = 'R'
          game_over = True
        else:
          used_time = time.time() - start_time
          if used_time > 10000: # This will 10 when testing the bots
            print("Officer exceeded time limit!")
            self.winner = 'R'
            game_over = True
          elif not self.is_valid_move(False, move):
            print("Illegal move by officer!")
          elif self.check_win():
            game_over = True

      # Make sure that the game doesn't last forever (deadlock)
      rounds_played += 1
      print("Rounds played: " + str(rounds_played))
      if rounds_played > 999:
        game_over = True
        if self.rebel_illegal > self.rebel_illegal:
          self.winner = 'O'
        elif self.officer_illegal > self.officer_illegal:
          self.winner = 'R'

    # Game is over
    if self.winner == 'R':
      print("Rebels won!")
    elif self.winner == 'O':
      print("Officers won!")
    else:
      print("Game over, no winner!")
    print("Rounds played: " + str(rounds_played))

# ===================================================
# The main function demonstrates how to run a game
if __name__ == "__main__":
    import importlib

    NUM_GAMES = 50  # 可以修改局数
    rebel_wins = 0
    officer_wins = 0

    # 导入玩家模块
    rebel_module = importlib.import_module("Team03_learning_1")  # Rebel bot
    officer_module = importlib.import_module("Team03_learning_1")  # Officer bot

    for game_idx in range(1, NUM_GAMES + 1):
        print(f"\n===== Game {game_idx} =====")
        game = Asalto()
        rebel = getattr(rebel_module, "Player")()
        officer = getattr(officer_module, "Player")()

        # 这里临时关闭 print_board，只打印关键信息
        game.print_board = lambda heading=None: None

        game.play(rebel, officer)
        if game.winner == 'R':
            rebel_wins += 1
            print(f"Game {game_idx} winner: Rebels")
        elif game.winner == 'O':
            officer_wins += 1
            print(f"Game {game_idx} winner: Officers")
        else:
            print(f"Game {game_idx} ended in a draw")

    print("\n===== Summary =====")
    print(f"Total games: {NUM_GAMES}")
    print(f"Rebels won: {rebel_wins}")
    print(f"Officers won: {officer_wins}")


# ==== End of file

