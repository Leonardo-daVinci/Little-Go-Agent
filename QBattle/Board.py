import numpy as np
from copy import deepcopy

BOARD_SIZE = 5
ONGOING = -1
DRAW = 0
X_WIN = 1
O_WIN = 2


class Board:
    def __init__(self, board=None, show_board=False, show_result=False):
        if board is None:
            self.board = np.zeroes((5, 5), dtype=np.int)
        else:
            self.board = board.copy()
        self.game_result = ONGOING
        self.show_board = show_board
        self.show_result = show_result
        # adding for go
        self.died_pieces = []

    def set_show_board(self, show_board):
        self.show_board = show_board

    def getHash(self):
        return str(self.board.reshape(BOARD_SIZE * BOARD_SIZE))

    def reset(self):
        self.board.fill(0)
        self.game_result = ONGOING

    def valid_place_check(self, i, j, piece_type, previous_board, test_check=False):
        board = self.board

        # Check if the place is in the board range
        if not (0 <= i < len(board) and 0 <= j < len(board)):
            return False

        # Check if the place already has a piece
        if board[i][j] != 0:
            return False

        # Copy the board for testing
        test_go = deepcopy(self)
        test_board = test_go.board

        # Check if the place has liberty
        test_board[i][j] = piece_type
        test_go.board = test_board
        if test_go.find_liberty(i, j):
            return True

        # If not, remove the died pieces of opponent and check again
        test_go.remove_died_pieces(3 - piece_type)
        if not test_go.find_liberty(i, j):
            return False

        # Check special case: repeat placement causing the repeat board state (KO rule)
        else:
            if self.died_pieces and self.compare_board(previous_board, test_go.board):
                return False
        return True

    def find_liberty(self, i, j):
        board = self.board
        ally_members = self.ally_dfs(i, j)
        for member in ally_members:
            neighbors = self.detect_neighbor(member[0], member[1])
            for piece in neighbors:
                # If there is empty space around a piece, it has liberty
                if board[piece[0]][piece[1]] == 0:
                    return True
        # If none of the pieces in a allied group has an empty space, it has no liberty
        return False

    def ally_dfs(self, i, j):
        # returns a list of all allies using DFS search
        stack = [(i, j)]  # stack implementation of DFS
        ally_members = []
        while stack:
            piece = stack.pop()
            ally_members.append(piece)
            neighbor_allies = self.detect_neighbor_ally(piece[0], piece[1])
            for ally in neighbor_allies:
                if ally not in stack and ally not in ally_members:
                    stack.append(ally)
        return ally_members

    def detect_neighbor_ally(self, i, j):
        board = self.board
        neighbors = self.detect_neighbor(i, j)
        group_allies = []
        # Iterate through neighbors
        for piece in neighbors:
            # Add to allies list if having the same color
            if board[piece[0]][piece[1]] == board[i][j]:
                group_allies.append(piece)
        return group_allies

    def detect_neighbor(self, i, j):
        board = self.board
        neighbors = []
        if i > 0: neighbors.append((i - 1, j))
        if i < len(board) - 1: neighbors.append((i + 1, j))
        if j > 0: neighbors.append((i, j - 1))
        if j < len(board) - 1: neighbors.append((i, j + 1))
        return neighbors

    def remove_died_pieces(self, piece_type):
        '''
        Remove the dead stones in the board.

        :param piece_type: 1('X') or 2('O').
        :return: locations of dead pieces.
        '''

        died_pieces = self.find_died_pieces(piece_type)
        if not died_pieces: return []
        self.remove_certain_pieces(died_pieces)
        return died_pieces

    def compare_board(self, board1, board2):
        return np.array_equal(board1, board2)

    def find_died_pieces(self, piece_type):
        board = self.board
        died_pieces = []

        for i in range(len(board)):
            for j in range(len(board)):
                # Check if there is a piece at this position:
                if board[i][j] == piece_type:
                    # The piece die if it has no liberty
                    if not self.find_liberty(i, j):
                        died_pieces.append((i, j))
        return died_pieces

    def remove_certain_pieces(self, died_pieces):
        board = self.board
        for piece in died_pieces:
            board[piece[0]][piece[1]] = 0
            self.board = board

    def move(self, row, col, piece_type, previous_board):
        if not self.valid_place_check(row, col, piece_type, previous_board):
            print(row, col)
        else:
            raise ValueError("Invalid Move")

        self.board[row][col] = piece_type
        self.game_result = self._check_winner()

        if self.show_result:
            self.game_result_report()

        return self.board, self.game_result

    def _check_winner(self):
        # first we check if game has ended or not
        if self.game_end():
            count1 = self.score(1)
            count2 = self.score(2)
            if count1 > count2 + 2.5:
                return 1
            elif count1 < count2 + 2.5:
                return 2
            else:
                return 0
        else:
            return ONGOING

    def score(self, piece_type):
        board = self.board
        return np.count_nonzero(board == piece_type)

    def game_end(self):
        pass

    def game_result_report(self):
        if self.game_result is ONGOING:
            return
        print('=' * 30)
        if self.game_result is DRAW:
            print('Game Over : Draw'.center(30))
        elif self.game_result is X_WIN:
            print('Game Over : Winner Black'.center(30))
        elif self.game_result is O_WIN:
            print('Game Over : Winner White'.center(30))
        print('=' * 30)

    def game_over(self):
        return self.game_result != ONGOING
