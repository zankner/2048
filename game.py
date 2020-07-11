import random
import math
import copy
import numpy as np


class Game:

    def __init__(self):
        self.observation_space = (1,16)
        self.action_space = 4

    def reset(self):
        self.board = [[0,0,0,0] for i in range(4)]
        firstRow = random.randint(0,len(self.board)-1)
        secondRow = random.randint(0,len(self.board)-1)
        firstCol = random.randint(0,len(self.board[0])-1)
        secondCol = random.randint(0,len(self.board[0])-1)
        while firstCol == secondCol:
            secondCol = random.randint(0,len(self.board[0])-1)
        self.board[firstRow][firstCol] = 2
        self.board[secondRow][secondCol] = 2
        self.reward = 0
        self.score = 0
        self.highest = 0
        self.mergeCount = 0
        self.highest = 0
        self.empty = 0
        return self.getState(), self.boardVector()

    def randomInsert(self):
        possibleCoords = []
        for i, row in enumerate(self.board):
            if 0 in row:
                for j, element in enumerate(row):
                    if element == 0:
                        possibleCoords.append([i,j])
        if len(possibleCoords) != 0:
            randomCoord = possibleCoords[random.randint(0,len(possibleCoords)) - 1]
            self.board[randomCoord[0]][randomCoord[1]] = 2


    def checkGameActive(self):
        possible = self.getPossible()
        if len(possible) == 0:
            return False
        else:
            return True


    def moveUp(self):
        for col in range(len(self.board[0])):
            colReal = []
            for r, row in enumerate(self.board):
                colReal.append(row[col])
            replace = self.merge(colReal, True)
            for i in range(len(self.board)):
                self.board[i][col] = replace[i]
        self.randomInsert()


    def canMoveUp(self):
        testState = copy.deepcopy(self.board)
        for col in range(len(testState[0])):
            colReal = []
            for r, row in enumerate(testState):
                colReal.append(row[col])
            replace = self.merge(colReal, False)
            for i in range(len(testState)):
                testState[i][col] = replace[i]
        if testState == self.board:
            return False
        else:
            return True


    def moveDown(self):
        for col in range(len(self.board[0])):
            colReal = []
            for r, row in enumerate(self.board):
                colReal.append(row[col])
            replace = self.merge(colReal, True)
            blankSpots = 0
            for i in reversed(replace):
                if i == 0:
                    blankSpots += 1
                else:
                    break
            propOrder = [0 for i in range(len(replace))]
            for i, el in enumerate(replace):
                propOrder[(i + blankSpots) % len(propOrder)] = el
            for i in range(len(self.board)):
                self.board[i][col] = propOrder[i]
        self.randomInsert()


    def canMoveDown(self):
        testState = copy.deepcopy(self.board)
        for col in range(len(testState[0])):
            colReal = []
            for r, row in enumerate(testState):
                colReal.append(row[col])
            replace = self.merge(colReal, False)
            blankSpots = 0
            for i in reversed(replace):
                if i == 0:
                    blankSpots += 1
                else:
                    break
            propOrder = [0 for i in range(len(replace))]
            for i, el in enumerate(replace):
                propOrder[(i + blankSpots) % len(propOrder)] = el
            for i in range(len(testState)):
                testState[i][col] = propOrder[i]
        if testState == self.board:
            return False
        else:
            return True


    def moveRight(self):
        for r, row in enumerate(self.board):
            replace = self.merge(row, True)
            blankSpots = 0
            for i in reversed(replace):
                if i == 0:
                    blankSpots += 1
                else:
                    break
            propOrder = [0 for i in range(len(replace))]
            for i, el in enumerate(replace):
                propOrder[(i + blankSpots) % len(propOrder)] = el
            self.board[r] = propOrder
        self.randomInsert()


    def canMoveRight(self):
        testState = self.board.copy()
        for r, row in enumerate(testState):
            replace = self.merge(row, False)
            blankSpots = 0
            for i in reversed(replace):
                if i == 0:
                    blankSpots += 1
                else:
                    break
            propOrder = [0 for i in range(len(replace))]
            for i, el in enumerate(replace):
                propOrder[(i + blankSpots) % len(propOrder)] = el
            testState[r] = propOrder
        if testState == self.board:
            return False
        else:
            return True


    def moveLeft(self):
        for i, row in enumerate(self.board):
            replace = self.merge(row, True)
            self.board[i] = replace
        self.randomInsert()


    def canMoveLeft(self):
        testState = self.board.copy()
        for i, row in enumerate(testState):
            replace = self.merge(row, False)
            testState[i] = replace
        if testState == self.board:
            return False
        else:
            return True


    def getPossible(self):
        possibleActions = []
        if self.canMoveUp() == True:
            possibleActions.append(0)
        if self.canMoveDown() == True:
            possibleActions.append(1)
        if self.canMoveRight() == True:
            possibleActions.append(2)
        if self.canMoveLeft() == True:
            possibleActions.append(3)
        return possibleActions


    def merge(self, nums, acting):
        prev = None
        store = []
        for next_ in nums:
            if not next_:
                continue
            if prev is None:
                prev = next_
            elif prev == next_:
                store.append(prev + next_)
                prev = None
            else:
                store.append(prev)
                prev = next_
        if prev is not None:
            store.append(prev)
        store.extend([0] * (len(nums) - len(store)))
        return store

    def getMerges(self):
        mergeCount = 0
        for row in self.board:
            for i in range(2):
                if row[(2*i)] == row[(2*i)+1]:
                    mergeCount +=1
        for col in range(len(self.board[0])):
            for i in range(2):
                if self.board[(2*i)][col] == self.board[(2*i)+1][col]:
                    mergeCount += 1
        self.mergeCount = mergeCount


    def getReward(self):
        highestEl = 0
        for row in self.board:
            for col in row:
                if col > highestEl:
                    highestEl = col
        reward = 0 
        highestEl = math.log(highestEl, 2)
        if highestEl > self.highest:
            reward += highestEl
        reward += self.mergeCount
        return reward


    def boardVector(self):
        state = []
        for row in self.board:
            for element in row:
                if element == 0:
                    state.append(0)
                else:
                    state.append(math.log(element, 2))
        return np.asarray(state)

    def getHighest(self):
        elements = [el for col in self.board for el in col]
        self.score += math.log(max(elements), 2)

    def emptyCells(self, board):
        count = 0
        for row in board:
            for el in row:
                if el == 0:
                    count += 1
        return count

    def newMax(self, curState, prevState):
        curState = [el for row in curState for el in row]
        prevState = [el for row in prevState for el in row]
        if max(curState) == max(prevState):
            return 0
        else:
            return math.log(max(curState), 2)

    def getState(self):
        power_mat = np.zeros(shape=(4,4,16),dtype=np.float32)
        for i in range(4):
            for j in range(4):
                if(self.board[i][j]==0):
                    power_mat[i][j][0] = 1.0
                else:
                    power = int(math.log(self.board[i][j],2))
                    power_mat[i][j][power] = 1.0
        return power_mat

    def getNpState(self):
        board = np.asarray(self.board)
        #board = np.reshape(board, [16])
        board = board / np.linalg.norm(board)
        return board


    def step(self, action):
        self.getMerges()
        prev_state = self.board.copy()
        if action == 0:
            self.moveUp()
        elif action == 1:
            self.moveDown()
        elif action == 2:
            self.moveRight()
        else:
            self.moveLeft()
        return self.getNpState(), self.getReward(), self.checkGameActive()
