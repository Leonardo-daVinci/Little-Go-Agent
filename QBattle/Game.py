import sys
from pathlib import Path
import json

sys.path.insert(1, str(Path.cwd()))

from Board import Board
from Agent import Agent
from RandomPlayer import RandomPlayer

PLAYER_X = 1
PLAYER_O = 2


def play(board, player1, player2, learn):
    # we need to create a board first and the with each move, output the board too.
    player1.set_side(PLAYER_X)
    player2.set_side(PLAYER_O)

    while not board.game_over():
        a, b, c, d = player1.move(board)
        e, f, g, h = player2.move(board)

        if (a == "PASS" and e == "PASS") and board.compare_board(b, f):
            board.game_result = board._check_winner()

    if learn:
        player1.learn(board)
        player2.learn(board)

    return board.game_result


def battle(board, player1, player2, iter, learn=False, show_result=True):
    p1_stats = [0, 0, 0]
    for i in range(0, iter):
        result = play(board, player1, player2, learn)
        p1_stats[result] += 1
        board.reset()

    p1_stats = [round(x / iter * 100.0, 1) for x in p1_stats]
    if show_result:
        print('_' * 60)
        print('{:>15}(X) | Wins:{}% Draws:{}% Losses:{}%'.format(player1.__class__.__name__, p1_stats[1], p1_stats[0],
                                                                 p1_stats[2]).center(50))
        print('{:>15}(O) | Wins:{}% Draws:{}% Losses:{}%'.format(player2.__class__.__name__, p1_stats[2], p1_stats[0],
                                                                 p1_stats[1]).center(50))
        print('_' * 60)
        print()

    return p1_stats


if __name__ == "__main__":
    qlearner = Agent()
    randomPlayer = RandomPlayer()
    NUM = qlearner.GAME_NUMS
    print('Training QLearner against RandomPlayer for {} times......'.format(NUM))
    board = Board()
    battle(board, randomPlayer, qlearner, NUM, learn=True, show_result=False)
    battle(board, qlearner, randomPlayer, NUM, learn=True, show_result=False)

    # once the battle is over we save the q-value dictionary

    print('Playing QLearner against RandomPlayer for 1000 times......')
    q_rand = battle(board, qlearner, randomPlayer, 20)
    rand_q = battle(board, randomPlayer, qlearner, 20)

    winning_rate_w_random_player = round(100 - (q_rand[2] + rand_q[1]) / 2, 2)
    print("Summary:")
    print("_" * 60)
    print("QLearner VS  RandomPlayer |  Win/Draw Rate = {}%".format(winning_rate_w_random_player))
    print("_" * 60)
