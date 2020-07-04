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
        return self.getNpState()

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
        for row in self.board:
            if 0 in row:
                return True
        for row in range(len(self.board)):
            for col in range(len(self.board[0])-1):
                if self.board[row][col] == self.board[row][col+1]:
                    return True
        for col in range(len(self.board[0])):
            for row in range(len(self.board)-1):
                if self.board[row][col] == self.board[row+1][col]:
                    return True
        return False


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
            replace = replace[::-1]
            for i in range(len(self.board)):
                self.board[i][col] = replace[i]
        self.randomInsert()


    def canMoveDown(self):
        testState = copy.deepcopy(self.board)
        for col in range(len(testState[0])):
            colReal = []
            for r, row in enumerate(testState):
                colReal.append(row[col])
            replace = self.merge(colReal, False)
            replace = replace[::-1]
            for i in range(len(testState)):
                testState[i][col] = replace[i]
        if testState == self.board:
            return False
        else:
            return True


    def moveRight(self):
        for i, row in enumerate(self.board):
            replace = self.merge(row, True)
            replace = replace[::-1]
            self.board[i] = replace
        self.randomInsert()


    def canMoveRight(self):
        testState = self.board.copy()
        for i, row in enumerate(testState):
            replace = self.merge(row, False)
            replace = replace[::-1]
            testState[i] = replace
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

    def getReward(self):
        highestEl = 0
        for row in self.board:
            for col in row:
                if col > highestEl:
                    highestEl = col
        return math.log(highestEl, 2)


    def getNpState(self):
        return np.asarray(self.board)


    def step(self, action):
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
