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

# Import some basic libraries and functions for this tutorial.
import numpy as np

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
        'RW': 'Rook_W.urdf'
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


    model_dir = '../models/'
    board_fn = 'SimpleBoard.urdf'
    # board_fn = 'Board.sdf'


    board_spacing = 0.0635  # This is tile spacing in meters (unit of Drake)

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

    def coord_to_location(self, coord: tuple[float]) -> tuple:
        """
        Get (chess) algebraic location from x, y coordinate pait

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