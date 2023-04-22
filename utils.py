import pandas as pd
from itertools import combinations
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
from strategies import random_plan, flatMC, BestMoveUCT, UCB
from MARRAKECH import Board

def score_norm(score):
    score = np.clip(score, 40, -40)
    return (score - (-40))/(40-(-40))




#game simulation
def game_simulation(n_games, strategy1, strategy2, n_playouts=None, use_score1=False, use_score2=False, verbose=True, use_tqdm=True):
    """Returns win rate for player 0"""
    n_wins1 = 0
    n_wins2 = 0
    draw = 0
    s = 0
    if use_tqdm:
        num_games = tqdm(range(n_games))
    else:
        num_games = range(n_games)
    for i in num_games:
        game = Board(verbose=False)
        i=1
        while(not game.terminal()):
            #print(f'Tour {i}')
            i+=1
            current_player=game.current_player.id
            #print(current_player)
            if current_player == 0:
                best_move = strategy1(game, n_playouts, use_score=use_score1)
            if current_player == 1:
                best_move = strategy2(game, n_playouts, use_score=use_score2)
            game.play(best_move)
        score = game.score()
        
        if score > 0: #victory for player 0
            n_wins1 += 1
        elif score < 0:
            n_wins2 +=1
        elif score == 0:
            draw +=1
        s += score
        #print(score, normalize(score))
    if verbose:
        print(f'Winrate of player 0 : {n_wins1/n_games}')
        print(f'Winrate of player 1 : {n_wins2/n_games}')
        print(f'Number of draw : {draw}')
        print(f'Mean score : {s/n_games}')
    
    return n_wins1/n_games, n_wins2/n_games, draw, s/n_games

str2func = {'random':random_plan, 'flat':flatMC, 'flat_score':flatMC, 'ucb':UCB, 'ucb_score':UCB, 'uct':BestMoveUCT, 'uct_score':BestMoveUCT}
list_strats = ['random', 'flat', 'flat_score', 'ucb', 'ucb_score', 'uct', 'uct_score'] 

#tournoument between strategies
def tournament(n_games, n_playouts):
    results = pd.DataFrame(columns=list_strats, index=list_strats)
    dict_mean_scores = dict()
    for game in list(combinations(list_strats, 2)):
        if game[0] != game[1]:
            if game[0].endswith('score'):
                use_score1=True
            else:
                use_score1=False
            if game[1].endswith('score'):
                use_score2=True
            else:
                use_score2=False
            print(f'########## {game[0]} VS {game[1]}. ##########')
            winrate_p1, winrate_p2, draw, mean_score = game_simulation(n_games, strategy1=str2func[game[0]], strategy2=str2func[game[1]], 
                                                                 n_playouts=n_playouts, use_score1=use_score1, use_score2=use_score2)
            results.at[game[0],game[1]] = winrate_p1
            results.at[game[1],game[0]] = winrate_p2
            dict_mean_scores[(game[0],game[1])] = mean_score
            save_pkl(results, f'results{n_playouts}')
            save_pkl(dict_mean_scores, f'dict_mean_scores{n_playouts}')
    return results, dict_mean_scores


#generate heatmap
def heatmap_gen(winRates_df, meanScores_dict, n_games, n_playouts, list_strats):
    winRates_df = winRates_df.astype(float)
    sns.heatmap(winRates_df, annot=True, cmap="crest")
    plt.title(f'Win rate pour {n_games} parties avec {n_playouts} playouts', fontsize = 15)
    plt.show()

    df_scores = pd.DataFrame(columns = list_strats, index = list_strats)
    for key, value in meanScores_dict.items():
        df_scores.at[key[0], key[1]]=value
        df_scores.at[key[1], key[0]]=-value

    df_scores = df_scores.astype(float)
    sns.heatmap(df_scores, annot=True, cmap="PiYG")
    plt.title(f'Score moyen pour {n_games} parties avec {n_playouts} playouts', fontsize = 15)
    plt.show()


def save_pkl(doc, filename):
    with open(f'./results/{filename}.pkl', 'wb') as handle:
        pickle.dump(doc, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pkl(filename):
    with open(f'./results/{filename}.pkl', 'rb') as handle:
         doc = pickle.load(handle)
    return doc