#  Copyright (c) 2019. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.
#  Created by LEEJUNKI
#  Copyright © 2019 LEEJUNKI. All rights reserved.
#  github :: https://github.com/ljk423

import random
import numpy as np
import pandas as pd

from scipy.ndimage.interpolation import shift
from keras.applications.resnet50 import *
from keras.layers import (Activation, BatchNormalization, Convolution2D, Dense,
                          Flatten, Input, MaxPooling2D, ZeroPadding2D, Reshape, Bidirectional, GRU)
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D
from keras import layers
from keras.optimizers import *
from keras.regularizers import L1L2
from keras import backend as K

bn_axis = 3 if K.image_dim_ordering() == 'tf' else 1

class Omok:
    def __init__(self):
        self.board = np.full((15, 15), 0)

    def toss(self):
        turn = random.randint(1, 3)
        if turn == 1:
            self.turn_monitor = 1

        else:
            self.turn_monitor = 2

        return self.turn_monitor

    def move(self, player, coord):
        if ((self.board[coord] != 0 or
             self.game_status() != "In Progress" or
             self.turn_monitor != player)):
            raise ValueError("Invalid move")

        self.board[coord] = player
        self.turn_monitor = 3 - player
        return self.game_status(), self.board

    def game_status(self):
        for i in range(self.board.shape[0]):
            if (('1 1 1 1 1' in str(self.board[i, :]) or
                 '2 2 2 2 2' in str(self.board[i, :]))):
                return "Won"

        for j in range(self.board.shape[1]):
            if (('1 1 1 1 1' in str(self.board[:, j]) or
                 '2 2 2 2 2' in str(self.board[:, j]))):
                return "Won"

        for k in range(-14, 15):
            if (('1 1 1 1 1' in str(np.diag(self.board, k=k)) or
                 '2 2 2 2 2' in str(np.diag(self.board, k=k)))):
                return "Won"

        for l in range(-14, 15):
            if (('1 1 1 1 1' in str(np.diag(np.fliplr(self.board), k=l)) or
                 '2 2 2 2 2' in str(np.diag(np.fliplr(self.board), k=l)))):
                return "Won"

        if 0 not in self.board:
            return "Drawn"

        else:
            return "In Progress"

    def get_result(self, player):
        if self.game_status() == "Won":
            if 3 - self.turn_monitor == player:
                return 1
            else:
                return -1

        if self.game_status() == "Drawn":
            return 0

        assert False

    def show(self):
        game_board = '   0  1  2  3  4  5  6  7  8  9  10 11 12 13 14\n'
        for i in range(self.board.shape[0]):
            game_board += '{:>2s}'.format(str(i)) + ' '
            for j in list(self.board[i, :]):
                game_board += '·OX'[j] + '  '
            game_board += '\n'

        print(game_board)


def legal_moves_generator(current_board_state, turn_monitor):
    legal_moves_dict = {}
    for i in range(current_board_state.shape[0]):
        for j in range(current_board_state.shape[1]):
            if current_board_state[i, j] == 0:
                board_state_copy = current_board_state.copy()
                board_state_copy[i, j] = turn_monitor
                legal_moves_dict[(i, j)] = board_state_copy  # flatten
    return legal_moves_dict


def move_selector(model, current_board_state, turn_monitor):
    tracker = {}
    legal_moves_dict = legal_moves_generator(current_board_state, turn_monitor)
    for legal_move_coord in legal_moves_dict:
        score = model.predict(legal_moves_dict[legal_move_coord].reshape(1, 15, 15, 1))  # reshape
        tracker[legal_move_coord] = score
    selected_move = max(tracker, key=tracker.get)
    new_board_state = legal_moves_dict[selected_move].reshape(-1, 15, 15, 1)
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

        elif game.game_status() == "In Progress" and game.turn_monitor == 2:
            oppo_board = game.board.copy()
            oppo_board[oppo_board == 1] = 3
            oppo_board[oppo_board == 2] = 1
            oppo_board[oppo_board == 3] = 2
            selected_move, new_board_state, score = move_selector(model, oppo_board,
                                                                  3-game.turn_monitor)

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

    if game_status == "Won" and (3-game.turn_monitor) == 1:
        corrected_scores_list = shift(scores_list, -1, cval=1.0)
        result = "Won"

    if game_status == "Won" and (3-game.turn_monitor) != 1:
        corrected_scores_list = shift(scores_list, -1, cval=-1.0)
        result = "Lost"

    if game_status == "Drawn":
        corrected_scores_list = shift(scores_list, -1, cval=0.0)
        result = "Drawn"

    if print_progress:
        print("Player 1 has", result)
        print("\n Correnting the Scores and Updating the model weights:")
        print("_______________________________________________________________\n")

    x1 = new_board_states_list
    y1 = corrected_scores_list

    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    # shuffle x and y in unison
    x1, y1 = unison_shuffled_copies(x1, y1)
    x1 = x1

    # update the weights of the model, on record at a time

    model.fit(x1, y1, epochs=1, batch_size=1, verbose=0)
    return model, y1, result

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(1, 1)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def main():
    input_layer = Input(shape=(15, 15, 1))

    x = ZeroPadding2D((2, 2))(input_layer)
    x = Convolution2D(48, 5, 5, subsample=(1, 1), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)  # axis=bn_axis,
    x = Activation('relu')(x)
    # x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 5, [64, 64, 256], stage=2, block='a',
                   strides=(1, 1))  # (input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    x = identity_block(x, 5, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 5, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 5, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 5, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 5, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 5, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 5, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 5, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 5, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 5, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 5, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 5, [256, 256, 1024], stage=4, block='f')
    # x = Flatten()(x)
    x = Reshape(target_shape=((225, 1024)), name='reshape')(x)
    # x = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(x)  # (None, 32, 64)
    reg = L1L2(l1=0.2, l2=0.2)

    x = Bidirectional(GRU(units=256, input_shape=(-1, 225), dropout=0.3, recurrent_regularizer=reg, activation='relu',
                          return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Bidirectional(GRU(units=64, dropout=0.3, recurrent_regularizer=reg, activation='relu',
                          return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Bidirectional(GRU(units=16, dropout=0.3, recurrent_regularizer=reg, activation='relu',
                          return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Bidirectional(GRU(units=4, dropout=0.3, recurrent_regularizer=reg, activation='relu',
                          return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(units=256, activation='relu')(x)
    x = Dense(units=1, activation='tanh')(x)
    # x2 = Dense(units=1, activation ='softmax')(x)

    model = Model(input_layer, x)
    # model2 = Model(input_layer, x2)
    model.summary()
    # model1.compile(loss='mean_squared_error', optimizer=Adadelta())

    model.compile(loss='mean_squared_error', optimizer=Adadelta())

    game_counter = 1
    data_for_graph = pd.DataFrame()

    while(game_counter <= 200):
        model, y, result = train(model, print_progress=True)
        data_for_graph = data_for_graph.append({"game_counter": game_counter,
                                               "result": result}, ignore_index=True)
        print("Game# : ", game_counter)
        game_counter += 1

    bins = np.arange(1, game_counter/10) * 10
    data_for_graph['game_counter_bins'] = np.digitize(data_for_graph['game_counter'], bins, right=True)
    counts = data_for_graph.groupby(['game_counter_bins', 'result']).game_counter.count().unstack()
    ax = counts.plot(kind='bar', stacked=True, figsize=(17, 5))
    ax.set_xlabel('counts of Games in Bins of 1s')
    ax.set_ylabel('counts of Draws/Losses/Wins')
    ax.set_title('Distribution of Results vs Count of Games Played')

    model.save('testing.h5')


if __name__ == '__main__':
    main()
