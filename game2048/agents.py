import torch
import numpy as np
import torchvision.transforms as transforms
#from myRNN import RNN
from time import sleep
import sys
#from expectimax import board_to_move
#from game import Game
#from displays import Display
import csv, os

PATH3 = '/home/septem7/Documents/2048-api-master/model/rnn_model_14.pkl'
PATH2 = '/home/septem7/Documents/2048-api-master/model/rnn_model_20final.pkl'
PATH1 = '/home/septem7/Documents/2048-api-master/model/myRNN3Random05Model50rate:0.pkl'
dataSetFilename = 'Train.csv'

class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction

class getBoardFormExpect(ExpectiMaxAgent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def writeBoard(self, root = dataSetFilename, max_iter=np.inf, verbose=False):
        with open(root, "a") as dataSetFile:
            wrt = csv.writer(dataSetFile)
            if not (os.path.exists(root)):
                wrt.writerow(["11","12","13","14","21","22","23","24","31","32","33","34","41","42","43","44"])
                n_iter = 0
                while (n_iter < max_iter) and (not self.game.end):
                    direction = self.step()
                    board = np.where(self.game.board == 0, 1, self.game.board)
                    board = np.log2(board)
                    board = board.flatten()
                    board = board.tolist()
                    oneRow = np.int32(np.append(board, direction))
                    wrt.writerow(data)
                    self.game.move(direction)

class MyAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        # from .expectimax import board_to_move
        # self.search_func = board_to_move
        # self.search_func = np.random.randint(0, 4)

        self.model = torch.load(PATH, map_location='cpu')
        self.model.eval()

    def step(self):

        # tmp = self.game.board.flatten()
        # print(board)
        board = np.where(self.game.board == 0, 1, self.game.board)
        board = np.log2(board)
        # print(board)
        board = board.reshape((4, 4))
        # sleep(3600)
        board = board[:, :, np.newaxis]
        board = board / 11.0
        trans = transforms.Compose([transforms.ToTensor()])
        board = trans(board)
        board = torch.unsqueeze(board, dim=0)
        board = board.type(torch.float)
        out = self.model(board)
        direction = torch.max(out, 1)[1]
        return int(direction)

class MyOnehotAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        # from .expectimax import board_to_move
        # self.search_func = board_to_move
        # self.search_func = np.random.randint(0, 4)

        self.model = torch.load(PATH, map_location='cpu')
        self.model.eval()

    def step(self):

        board = self.game.board.flatten()
        print(board)
        print(type(board))
        board = one_hot(board)
        board = board[:, :, np.newaxis]
        board = board / 1.0
        trans = transforms.Compose([transforms.ToTensor()])
        board = trans(board)
        board = torch.unsqueeze(board, dim=0)
        board = board.type(torch.float)
        out = self.model(board)
        direction = torch.max(out, 1)[1]
        return int(direction)

def one_hot(board):
    ohboard = np.zeros([16, 16])
    for i in range(15):
        if int(board[i]) == 0:
            j = 0
        else:
            j = int(np.log2(int(board[i])))
        m = int(int((i + 1) / 4) * 4 + int((j + 1) / 4))
        n = int(int(i % 4) * 4 + int(j % 4))
        ohboard[m, n] = 1

    return ohboard
class MyRnnAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        # from .expectimax import board_to_move
        # self.search_func = board_to_move
        # self.search_func = np.random.randint(0, 4)

        self.model = torch.load(PATH, map_location='cpu')
        self.model.eval()

    def step(self):

        # tmp = self.game.board.flatten()
        # print(self.game.board.ndim)
        board = np.where(self.game.board == 0, 1, self.game.board)
        # print(board)
        board = np.log2(board)
        # print(board)
        # board = board.reshape((4, 4))
        # print(board)
        board = board[:, :, np.newaxis]
        # print(self.game.board)


        board = board/ 11.0
        # print("*******")
        # print(board)
        trans = transforms.Compose([transforms.ToTensor()])
        board = trans(board)
        # print("&&&&&&&")
        # print(board)
        # print(board)
        # board = torch.unsqueeze(board, dim=0)
        board = board.type(torch.float)
        # print("^^^^^^^")
        # print(board)
        out = self.model(board)

        # direction = torch.max(out, 1)[1]
        # direction = torch.max(out, 1)[1]
        # direction = torch.max(out, 1)[1]
        direction = torch.max(out, 1)[1]
        # sleep(3600)
        return int(direction)
class MyRnnVoteAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        # from .expectimax import board_to_move
        # self.search_func = board_to_move
        # self.search_func = np.random.randint(0, 4)

        self.model1 = torch.load(PATH1, map_location='cpu')
        self.model2 = torch.load(PATH1, map_location='cpu')
        self.model3 = torch.load(PATH1, map_location='cpu')
        self.model1.eval()
        self.model2.eval()
        self.model3.eval()

    def step(self):

        # tmp = self.game.board.flatten()
        # print(self.game.board.ndim)
        board = np.where(self.game.board == 0, 1, self.game.board)
        # print(board)
        board = np.log2(board)
        # print(board)
        # board = board.reshape((4, 4))
        # print(board)
        board = board[:, :, np.newaxis]
        # print(self.game.board)


        board = board/ 11.0
        # print("*******")
        # print(board)
        trans = transforms.Compose([transforms.ToTensor()])
        board = trans(board)
        # print("&&&&&&&")
        # print(board)
        # print(board)
        # board = torch.unsqueeze(board, dim=0)
        board = board.type(torch.float)
        # print("^^^^^^^")
        # print(board)
        out1 = self.model1(board)
        direction1 = torch.max(out1, 1)[1]
        out2 = self.model2(board)
        direction2 = torch.max(out2, 1)[1]
        out3 = self.model3(board)
        direction3 = torch.max(out3, 1)[1]

        if (direction1 == direction2):

            direction = direction2
        else:
            direction = direction3


        # sleep(3600)
        return int(direction)
