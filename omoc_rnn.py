import numpy as np
import random
import pandas as pd

from scipy.ndimage.interpolation import shift

from keras.models import Sequential
from keras.layers import *
from keras.regularizers import L1L2
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.optimizers import *


class Omok(object):
    def __init__(self):
        self.board = np.full((15, 15), 2)

    def toss(self):
        turn = np.random.randint(0, 2)
        if turn == 0:
            self.turn_monitor = 0

        else:
            self.turn_monitor = 1

        return self.turn_monitor

    def move(self, player, coord):
        if ((self.board[coord] != 2 or
             self.game_status() != "In Progress" or
             self.turn_monitor != player)):
            raise ValueError("Invalid move")

        self.board[coord] = player
        self.turn_monitor = 1 - player
        return self.game_status(), self.board

    def game_status(self):
        for i in range(self.board.shape[0]):
            if (('0 0 0 0 0' in str(self.board[i, :]) or
                 '1 1 1 1 1' in str(self.board[i, :]))):
                return "Won"

        for j in range(self.board.shape[1]):
            if (('0 0 0 0 0' in str(self.board[:, j]) or
                 '1 1 1 1 1' in str(self.board[:, j]))):
                return "Won"

        for k in range(-14, 15):
            if (('0 0 0 0 0' in str(np.diag(self.board, k=k)) or
                 '1 1 1 1 1' in str(np.diag(self.board, k=k)))):
                return "Won"

        for l in range(-14, 15):
            if (('0 0 0 0 0' in str(np.diag(np.fliplr(self.board), k=l)) or
                 '1 1 1 1 1' in str(np.diag(np.fliplr(self.board), k=l)))):
                return "Won"

        if 2 not in self.board:
            return "Drawn"

        else:
            return "In Progress"

    def show(self):
        game_board = '   0  1  2  3  4  5  6  7  8  9  10 11 12 13 14\n'
        for i in range(self.board.shape[0]):
            game_board += '{:>2s}'.format(str(i)) + ' '
            for j in list(self.board[i, :]):
                game_board += 'XO.'[j] + '  '
            game_board += '\n'

        print(game_board)

def legal_moves_generator(current_board_state, turn_monitor):
    legal_moves_dict = {}
    for i in range(current_board_state.shape[0]):
        for j in range(current_board_state.shape[1]):
            if current_board_state[i, j] == 2:
                board_state_copy = current_board_state.copy()
                board_state_copy[i, j] = turn_monitor
                legal_moves_dict[(i, j)] = board_state_copy.flatten()
    return legal_moves_dict


def move_selector(model, current_board_state, turn_monitor):
    tracker = {}
    legal_moves_dict = legal_moves_generator(current_board_state, turn_monitor)
    for legal_move_coord in legal_moves_dict:
        score = model.predict(legal_moves_dict[legal_move_coord].reshape(1,-1, 225))
        tracker[legal_move_coord] = score
    selected_move = max(tracker, key=tracker.get)
    new_board_state = legal_moves_dict[selected_move]
    score = tracker[selected_move]
    return selected_move, new_board_state, score


def train(model, print_progress=False):
    if print_progress:
        print("_______________________________________________________________")
        print("Starting a new game")

    game = Omok()
    game.toss()
    scores_list = []
    corrected_scores_list = []
    new_board_states_list = []

    while(1):
        if game.game_status() == "In Progress" and game.turn_monitor == 1:
            selected_move, new_board_state, score = move_selector(model, game.board,
                                                                  game.turn_monitor)
            scores_list.append(score[0][0])
            new_board_states_list.append(new_board_state)

            game_status, board = game.move(game.turn_monitor, selected_move)
            if print_progress:
                game.show()
                print("\n")

        elif game.game_status() == "In Progress" and game.turn_monitor == 0:
            selected_move, new_board_state, score = move_selector(model, game.board,
                                                                  game.turn_monitor)
            game_status, board = game.move(game.turn_monitor, selected_move)
            if print_progress == True:
                game.show()
                print("\n")

        else:
            break

    # Correct the scores, assigning 1/0/-1 to the winning/drawn/losing final board state,
    # and assigning the other previous board states the score of their next board state
    new_board_states_list = tuple(new_board_states_list)
    new_board_states_list = np.vstack(new_board_states_list)
    if game_status == "Won" and (1-game.turn_monitor) == 1:
        corrected_scores_list = shift(scores_list, -1, cval=1.0)
        result = "Won"

    if game_status == "Won" and (1-game.turn_monitor) != 1:
        corrected_scores_list = shift(scores_list, -1, cval=-1.0)
        result = "Lost"

    if game_status == "Drawn":
        corrected_scores_list = shift(scores_list, -1, cval=0.0)
        result = "Drawn"

    if print_progress:
        print("Player 1 has", result)
        print("\n Correnting the Scores and Updating the model weights:")
        print("_______________________________________________________________\n")
        model.summary()

    x = new_board_states_list
    y = corrected_scores_list

    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    # shuffle x and y in unison
    x, y = unison_shuffled_copies(x, y)
    x = x.reshape(-1, 225)

    # update the weights of the model, on record at a time

    model.fit(x, y, epochs=1, batch_size=1, verbose=0)
    return model, y, result


def main():
    model = Sequential()
    # model.add(Dense(512, input_dim=225, kernel_initializer='glorot_normal', activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(128, kernel_initializer='glorot_normal', activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(32, kernel_initializer='glorot_normal', activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(8, kernel_initializer='glorot_normal', activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(1, kernel_initializer='glorot_normal'))
    reg = L1L2(l1=0.2, l2=0.2)
    # model.add(Dense(512, input_dim=225, kernel_initializer='glorot_normal', activation='relu'))
    model.add(Bidirectional(GRU(units=256, input_shape=(-1,225), dropout=0.3, recurrent_regularizer=reg, activation='relu',
                                return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Bidirectional(GRU(units=64, dropout=0.3, recurrent_regularizer=reg, activation='relu',
                                return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Bidirectional(GRU(units=16, dropout=0.3, recurrent_regularizer=reg, activation='relu',
                                return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Bidirectional(GRU(units=4, dropout=0.3, recurrent_regularizer=reg, activation='relu',
                                return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(units=1))

    model.compile(loss='mean_squared_error', optimizer=Adadelta())
    game_counter = 1
    data_for_graph = pd.DataFrame()

    while(game_counter <= 2):
        model, y, result = train(model, print_progress=True)
        data_for_graph = data_for_graph.append({"game_counter": game_counter,
                                               "result": result}, ignore_index=True)
        if game_counter % 10000 == 0:
            print("Game# : ", game_counter)
        game_counter += 1

    bins = np.arange(1, game_counter/1) * 1
    data_for_graph['game_counter_bins'] = np.digitize(data_for_graph['game_counter'], bins, right=True)
    counts = data_for_graph.groupby(['game_counter_bins', 'result']).game_counter.count().unstack()
    ax = counts.plot(kind='bar', stacked=True, figsize=(17, 5))
    ax.set_xlabel('counts of Games in Bins of 1s')
    ax.set_ylabel('counts of Draws/Losses/Wins')
    ax.set_title('Distribution of Results vs Count of Games Played')

    model.save('my_model.h5')

if __name__ == '__main__':
    main()