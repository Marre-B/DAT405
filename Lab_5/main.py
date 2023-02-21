import numpy as np
            
def valueIteration(stateTable, tempTable, newStateTable, discount):
    for y in range(len(stateTable)):
        for x in range(len(stateTable[y])):
            tempTable[x][y] = calculateNewReward(stateTable, newStateTable, x, y, discount)
    newStateTable = tempTable
    return newStateTable

def calculateNewReward(stateTable, newStateTable, x, y, discount):
    pStay = 0.2
    pAction = 0.8
    
    # North
    try:
        north = (pAction*(stateTable[x][y+1] + discount * newStateTable[x][y+1])) + (pStay*(stateTable[x][y] + discount * newStateTable[x][y]))
    except:
        north = 0
        
    # East
    try:
        east = (pAction*(stateTable[x+1][y] + discount * newStateTable[x+1][y])) + (pStay*(stateTable[x][y] + discount * newStateTable[x][y]))
    except:
        east = 0

    # South
    try:
        south = (pAction*(stateTable[x][y-1] + discount * newStateTable[x][y-1])) + (pStay*(stateTable[x][y] + discount * newStateTable[x][y]))
    except:
        south = 0
        
    # West
    try:
        west = (pAction*(stateTable[x-1][y] + discount * newStateTable[x-1][y])) + (pStay*(stateTable[x][y] + discount * newStateTable[x][y]))
    except:
        west = 0
    
    return max(north, east, south, west, newStateTable[x][y])
    
def main():
    stateTable = [[0, 0, 0], [0, 10, 0], [0, 0, 0]] # Original Reward table
    newStateTable = [[0, 0, 0], [0, 0, 0], [0, 0, 0]] # New Reward table for each iteration
    lastIteration = [[0, 0, 0], [0, 0, 0], [0, 0, 0]] # Last iteration of the Reward table

    changeValue = 1 # Just a random value to start the loop

    discount = 0.9 # Discount factor
    epsilon = 0.0001 # Threshold

    while changeValue > epsilon:
        tempTable = [[0, 0, 0], [0, 0, 0], [0, 0, 0]] # Temporary table to store the new values
        newStateTable = valueIteration(stateTable, tempTable, newStateTable, discount) # Calculate the new values
        changeValue = sum(sum(np.subtract(np.array(newStateTable), np.array(lastIteration))))/9 # Calculate the change in values
        lastIteration = newStateTable 
    print(newStateTable)


if __name__ == "__main__":
    main()