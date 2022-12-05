import os
import os.path as osp
import inspect

import chess_bot


def get_chessbot_src():
    return osp.dirname(inspect.getfile(chess_bot))

def get_chessbot_model_weights():
    return osp.join(get_chessbot_src(), 'weights')

