from game2048.game import Game
from game2048.displays import Display
from game2048.myRNN import RNN
from game2048.myPlanning import MyPlanning



def single_run(size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display=Display(), **kwargs)
    agent.play(verbose=True)
    return game.score
def single_run_getdata(size, writeFile, LB, HB, AgentClass, **kwargs):
    game = Game(size, HB)
    agent = AgentClass(game, display=None, **kwargs)
    agent.writeBoard(writeFile, LB, HB, verbose=True)
    return game.score

if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    N_TESTS = 50
    WRITEFILE = 'game2048/DATA.csv'
    LB = 64
    HB = 64

    '''====================
    Use your own agent here.'''
    # from game2048.agents import ExpectiMaxAgent as TestAgent
    from game2048.agents import MyRnnAgent as TestAgent
    # from game2048.agents import MyPlanningAgent as TestAgent
    # from game2048.agents import getBoardFormExpect as TestAgent
    # from game2048.agents import getBoardFromMyRnnAgent as TestAgent
    '''===================='''

    scores = []
    for _ in range(N_TESTS):
        score = single_run(GAME_SIZE, SCORE_TO_WIN,
                           AgentClass=TestAgent)
        # score = single_run_getdata(GAME_SIZE, WRITEFILE, LB, HB,
        #                    AgentClass=TestAgent)
        scores.append(score)
        # print(scores)
    print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))
