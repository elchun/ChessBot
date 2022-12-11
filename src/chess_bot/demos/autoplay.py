# Automatic playing for evaluating the error rate of the game station

import matplotlib.pyplot as plt

from plotly.offline import init_notebook_mode, iplot

from pydrake.all import (StartMeshcat, RigidTransform)

from chess_bot.utils.plotly_utils import multiplot
from chess_bot.stations.game_station import GameStation
from chess_bot.resources.board import Board

from stockfish import Stockfish

meshcat = StartMeshcat()
print('Meshcat url: ', meshcat.web_url())

stockfish_params = {
    'UCI_Elo': 1300,
}

total_turns = 0
# total_errors = 0
total_percept_errors = 0
total_other_errors = 0
for i in range(10):
    game_station = GameStation(meshcat)
    stockfish =  Stockfish('/opt/homebrew/bin/stockfish', parameters=stockfish_params)
    perception_errors, other_errors, turns, complete = game_station.auto_play_game(stockfish)
    total_turns += turns
    total_percept_errors += perception_errors
    total_other_errors += other_errors
    print('---------------------------------------------------------')
    print(f'Percept Errors: {perception_errors} | Other Errors: {other_errors} | '
     f'Turns: {turns} | Completed: {complete}')
    print('---------------------------------------------------------')

    print('---------------------------------------------------------')
    print(f'Total Percept Errors: {total_percept_errors} | Total Other Errors: {total_other_errors} | '
        f'Turns: {total_turns} | Total Errors Rate: {(total_percept_errors + total_other_errors) / total_turns}')
    print('---------------------------------------------------------')


