"""
Board is stored as a dictionary mapping board locations to piece types

Pieces are named:

    K --> King

    Q --> Queen

    R --> Rook

    B --> Bishop

    N --> Knight

    P --> Pawn
"""

import os.path as osp
import numpy as np

from chess_bot.utils.path_util import get_chessbot_src

class Board:
    starting_board = {
        'a1': 'RW',
        'b1': 'NW',
        'c1': 'BW',
        'd1': 'QW',
        'e1': 'KW',
        'f1': 'BW',
        'g1': 'NW',
        'h1': 'RW',

        'a2': 'PW',
        'b2': 'PW',
        'c2': 'PW',
        'd2': 'PW',
        'e2': 'PW',
        'f2': 'PW',
        'g2': 'PW',
        'h2': 'PW',

        'a8': 'RB',
        'b8': 'NB',
        'c8': 'BB',
        'd8': 'QB',
        'e8': 'KB',
        'f8': 'BB',
        'g8': 'NB',
        'h8': 'RB',

        'a7': 'PB',
        'b7': 'PB',
        'c7': 'PB',
        'd7': 'PB',
        'e7': 'PB',
        'f7': 'PB',
        'g7': 'PB',
        'h7': 'PB',
    }

    # x coord corresponds to letter, y coord corresponds to number
    # use Board.print_board() to see true shape
    starting_board_list = [
        ['RW', 'PW', '  ', '  ', '  ', '  ', 'PB', 'RB'],
        ['NW', 'PW', '  ', '  ', '  ', '  ', 'PB', 'NB'],
        ['BW', 'PW', '  ', '  ', '  ', '  ', 'PB', 'BB'],
        ['QW', 'PW', '  ', '  ', '  ', '  ', 'PB', 'QB'],
        ['KW', 'PW', '  ', '  ', '  ', '  ', 'PB', 'KB'],
        ['BW', 'PW', '  ', '  ', '  ', '  ', 'PB', 'BB'],
        ['NW', 'PW', '  ', '  ', '  ', '  ', 'PB', 'NB'],
        ['RW', 'PW', '  ', '  ', '  ', '  ', 'PB', 'RB'],
    ]

    piece_to_fn = {
        'BB': 'Bishop_B.urdf',
        'BW': 'Bishop_W.urdf',

        'KB': 'King_B.urdf',
        'KW': 'King_W.urdf',

        'NB': 'Knight_B.urdf',
        'NW': 'Knight_W.urdf',

        'PB': 'Pawn_B.urdf',
        'PW': 'Pawn_W.urdf',

        'QB': 'Queen_B.urdf',
        'QW': 'Queen_W.urdf',

        'RB': 'Rook_B.urdf',
        'RW': 'Rook_W.urdf',
    }

    letter_to_idx = {
        'a': 0,
        'b': 1,
        'c': 2,
        'd': 3,
        'e': 4,
        'f': 5,
        'g': 6,
        'h': 7
    }

    idx_to_letter = {
        0: 'a',
        1: 'b',
        2: 'c',
        3: 'd',
        4: 'e',
        5: 'f',
        6: 'g',
        7: 'h'
    }


    model_dir = osp.join(get_chessbot_src(), 'resources/models/')
    board_fn = 'SimpleBoard.urdf'
    # board_fn = 'Board.sdf'


    board_spacing = 0.0635  # This is tile spacing in meters (unit of Drake)

    def __init__(self):
        self.pieces_to_idxs = {
            'a1': (0, 0),
            'b1': (1, 0),
            'c1': (2, 0),
            'd1': (3, 0),
            'e1': (4, 0),
            'f1': (5, 0),
            'g1': (6, 0),
            'h1': (7, 0),

            'a2': (0, 1),
            'b2': (1, 1),
            'c2': (2, 1),
            'd2': (3, 1),
            'e2': (4, 1),
            'f2': (5, 1),
            'g2': (6, 1),
            'h2': (7, 1),

            'a8': (0, 7),
            'b8': (1, 7),
            'c8': (2, 7),
            'd8': (3, 7),
            'e8': (4, 7),
            'f8': (5, 7),
            'g8': (6, 7),
            'h8': (7, 7),

            'a7': (0, 6),
            'b7': (1, 6),
            'c7': (2, 6),
            'd7': (3, 6),
            'e7': (4, 6),
            'f7': (5, 6),
            'g7': (6, 6),
            'h7': (7, 6),
        }

    @staticmethod
    def location_to_coord(location):
        """
        Given location in algebraic notation, generate 0-indexed location of
        piece

        Args:
            location (str): Location in algebraic notation.
        """

        return Board.letter_to_idx[location[0]], int(location[1]) - 1


    def get_xy_location(self, location):
        x_idx, y_idx = self.location_to_coord(location)
        x = self.board_spacing / 2 + self.board_spacing * x_idx
        y = self.board_spacing / 2 + self.board_spacing * y_idx

        # Origin is in middle of board
        x -= self.board_spacing * 4
        y -= self.board_spacing * 4

        return x, y

    def get_xy_location_from_idx(self, x_idx, y_idx):
        x = self.board_spacing / 2 + self.board_spacing * x_idx
        y = self.board_spacing / 2 + self.board_spacing * y_idx

        # Origin is in middle of board
        x -= self.board_spacing * 4
        y -= self.board_spacing * 4

        return x, y

    def coord_to_index(self, coord: tuple[float]) -> tuple:
        """
        Get xy index from x, y coordinate pait

        Args:
            coord (tuple(float)): Real coordinate of piece (with board centered
                at (0, 0))
        Returns:
            tuple(int): index of piece, (x_idx, y_idx) where x goes left to right
                and y goes front to back
        """
        x, y = coord
        x += self.board_spacing * 4
        y += self.board_spacing * 4

        x_idx = x // self.board_spacing
        y_idx = y // self.board_spacing

        return int(x_idx), int(y_idx)
        # return Board.idx_to_letter[x_idx] + str(int(y_idx) + 1)

    @staticmethod
    def index_to_location(coord):
        x_idx, y_idx = coord
        return Board.idx_to_letter[x_idx] + str(int(y_idx) + 1)


    @staticmethod
    def get_move(prev_board, current_board):
        """
        Find the move that transforms previous board into current board.
        Expects boards as 8x8 list of strings where '  ' (two spaces) is empty.

        Args:
            prev_board (list): most recent board
            current_board (list): current board
        Returns:
            tuple(tuple(int)): start_position as (x, y) and end_position as (x, y)
        """
        # print('prev: ')
        # Board.print_board(prev_board)
        # print('current: ')
        # Board.print_board(current_board)
        start_pos = None
        end_pos = None
        start_pos_list = []
        end_pos_list = []
        for x_idx in range(8):
            for y_idx in range(8):
                old_piece = prev_board[x_idx][y_idx]
                new_piece = current_board[x_idx][y_idx]
                if old_piece != new_piece:
                    # Case 1, piece left
                    if new_piece == '  ':
                        start_pos = (x_idx, y_idx)
                        start_pos_list.append(start_pos)

                    # Case 2: piece landed
                    elif old_piece[-1] != new_piece[-1]:
                        end_pos = (x_idx, y_idx)
                        end_pos_list.append(end_pos)

                    else:
                        raise ValueError('Could not determine move!')

        assert start_pos is not None, 'Could not find start position'
        assert end_pos is not None, 'Could not find end position'

        # Check for castle --> Return move of king
        if len(start_pos_list) == 2 and len(end_pos_list) == 2:
            if (4, 0) in start_pos_list:
                start_pos = (4, 0)
            elif (4, 7) in start_pos:
                start_pos = (4, 7)

            for possible_end_pos in [(2, 0), (6, 0), (2, 7), (6, 7)]:
                if possible_end_pos in end_pos_list:
                    end_pos = possible_end_pos
                    break

        return start_pos, end_pos

    @staticmethod
    def print_board(board):
        """
        Print a board from white view.  Board is 8x8 list of lists.

        Args:
            board (list[list[str]]): Board as 8x8 list of lists
        """
        for row in range(7, -1, -1):
            for col in range(0, 7):
                print(board[col][row], end=' ')
            print()



    # def make_board(self, board_dict):
    #     """
    #     board_dict maps locations in algebraic notiation to piece types.  Locations not
    #     listed don't have pieces on them.
    #     """

    #     for location, piece in board_dict.items():
    #         x_idx, y_idx = self.location_to_coord(location)
    #         x = self.board_spacing / 2 + self.board_spacing * x_idx
    #         y = self.board_spacing / 2 + self.board_spacing * y_idx

    #         # Origin is in middle of board
    #         x -= self.board_spacing * 4
    #         y -= self.board_spacing * 4

    #         # print(location, piece)
    #         # print(self.location_to_coord(location))
    #         # print(x, y)