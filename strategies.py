import random
import numpy as np
import copy
import math
import MARRAKECH
#from utils import score_norm

board_dim=5


#RANDOM
def random_plan(board, n_playouts, use_score=False):
    dice_result = board.throw_dice()
    moves = board.legal_moves(dice_result)
    n = random.randint(0, len(moves)-1)
    return moves[n]

#FLAT monte carlo
def flatMC(board, n, use_score=True):
    """
    Get the move with greatest mean after `n` playouts.
    
    Args:
        n (int): Number of playouts.
        use_score (bool): Use the real score to compute the score if True, otherwise use the number of wins
    """

    dice_result = board.throw_dice()
    moves = board.legal_moves(dice_result)
    sumScores = [0.0 for x in range(len(moves))]
    nbVisits = [0 for x in range(len(moves))]
    current_player = board.current_player.id

    # For each playout
    for i in range(n):
        m = random.randint(0, len(moves)-1) # Choose a random move
        b = copy.deepcopy(board)
        b.play(moves[m]) # Play the move
        r = b.playout() # Result of the random game from moves[m]

        if use_score:
            score = b.score() # Score of the random game from moves[m] 
            sumScores[m] += score 
        else:
            if (current_player == 0 and r == 1) or (current_player == 1 and r == -1):
                sumScores[m] += 1
        nbVisits[m] += 1 

    # Get the move with the greatest mean
    bestScore = 0
    bestMove = 0
    for m in range(len(moves)):
        score = 0
        if nbVisits[m] > 0:
            score = sumScores[m] / nbVisits[m]
        # If we don't use score, then maximize
        # If we use score, max for player 0, min for player 1
        if use_score == False:
            if score > bestScore:
                bestScore = score
                bestMove = m
        elif current_player == 0:
            if score > bestScore:
                bestScore = score
                bestMove = m
        else:
            if score < bestScore:
                bestScore = score
                bestMove = m
    
    return moves[bestMove]

#UCB

def score_norm(score):
    score = np.clip(score, 40, -40)
    return (score - (-40))/(40-(-40))

def UCB(board, n, c=0.4, use_score=True):
    dice_result = board.throw_dice()
    moves = board.legal_moves(dice_result)
    sumScores = [0.0 for x in range(len(moves))]
    nbVisits = [0 for x in range(len(moves))]
    current_player = board.current_player.id
    for i in range(n):
        bestScore = 0
        bestMove = 0
        for m in range(len(moves)):
            score = 1000000
            if nbVisits[m] > 0:
                 score = sumScores[m] / nbVisits[m] + c * math.sqrt(math.log(i) / nbVisits[m])
            if score > bestScore:
                bestScore = score
                bestMove = m
        b = copy.deepcopy(board)
        b.play(moves[bestMove])
        r = b.playout()
        s = b.score()
        if use_score:
            if current_player == 0:
                sumScores[bestMove] += score_norm(s)
            else:
                sumScores[bestMove] += (1 - score_norm(s))
        else:
            if (current_player == 0 and r == 1) or (current_player == 1 and r == -1):
                sumScores[bestMove] += 1

        nbVisits[bestMove] += 1
    
    # Get the most visited move
    bestScore = 0
    bestMove = 0
    for m in range(len(moves)):
        score = nbVisits[m]
        if score > bestScore:
            bestScore = score
            bestMove = m
    return moves[bestMove]

#UCT

MaxLegalMoves = 4 * (board_dim * board_dim) * 12 # 4 orientations * (5x5) pawn move * 12 rug placement
Table = {}


        
def look(board):
    """Returns the entry of the board in the transposition table."""
    return Table.get(board.h, None)

def add(board):
    """Adds an empty entry for the board in the transposition table."""
    nplayouts = [0.0 for x in range(MaxLegalMoves)]
    nwins = [0.0 for x in range(MaxLegalMoves)]
    Table[board.h] = [0, nplayouts, nwins]

def UCT(board, c=0.4, use_score=True):
    if board.terminal():
        return board.score()
    current_player = board.current_player.id
    t = look(board) 
    if t != None: # If current config visited
        bestValue = -1000000.0
        bestMove = 0
        dice_result = board.throw_dice()
        moves = board.legal_moves(dice_result)
        for m in range(len(moves)):
            val = 1000000.0
            if t[1][m] > 0: 
                Q = t[2][m] / t[1][m] 
                if board.current_player.id == 1:
                    Q = 1 - Q
                val = Q + c * math.sqrt(math.log(t[0]) / t[1][m])
            if val > bestValue:
                bestValue = val
                bestMove = m
        board.play(moves[bestMove])
        res = UCT(board, c)
        t[0] += 1
        t[1][bestMove] += 1 # +1 playout
        if use_score:
            s = board.score()
            if current_player == 0:
                t[2][bestMove] += score_norm(s)
            else:
                t[2][bestMove] += (1 - score_norm(s))
        else:
            if (current_player == 0 and res == 1) or (current_player == 1 and res == -1):
                t[2][bestMove] += 1
        return res
    else:
        add(board)
        return board.playout()

def BestMoveUCT(board, n, c=0.4, use_score=True):
    global Table
    Table = {}
    for i in range(n):
        b1 = copy.deepcopy(board)
        res = UCT(b1, c, use_score)
    t = look(board)
    current_player = board.current_player.id
    dice_result = board.throw_dice()
    moves = board.legal_moves(dice_result)
    bestMove = 0
    bestValue = t[2][0]
    for m in range(1, len(moves)):
        if use_score == False:
            if t[2][m] > bestValue:
                bestValue = t[2][m]
                bestMove = m
        elif current_player == 0:
            if t[2][m] > bestValue:
                bestValue = t[2][m]
                bestMove = m
        else:
            if t[2][m] < bestValue:
                bestValue = t[2][m]
                bestMove = m

    return moves[bestMove]

