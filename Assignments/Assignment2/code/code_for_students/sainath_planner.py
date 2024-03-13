import argparse
import numpy as np
from pulp import *
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--mdp")
parser.add_argument("--algorithm", default="vi", choices=["vi", "hpi", "lp"])
parser.add_argument("--policy")

args = parser.parse_args()

S = 0       #Number of states
A = 0       #Number of actions

mdpPath = args.mdp
algorithm = args.algorithm
policyPath = args.policy

file1 = open(mdpPath, 'r')
Lines = file1.readlines()
L = len(Lines)

S = int(Lines[0].strip().split()[1])
A = int(Lines[1].strip().split()[1])
endStates = Lines[2].strip().split()
endStates = endStates[1:]
endStates = [int(i) for i in endStates]

mdpType = Lines[L-2].strip().split()[1]
discount = float(Lines[L-1].strip().split()[1])

transitionMatrix = np.zeros((A, S, S))
rewardMatrix = np.zeros((A, S, S))

for i in range(3, L-2):
    splitLine = Lines[i].strip().split()
    s1 = int(splitLine[1])
    s2 = int(splitLine[3])
    ac = int(splitLine[2])
    prob = float(splitLine[5])
    reward = float(splitLine[4])
    transitionMatrix[ac][s1][s2] = prob
    rewardMatrix[ac][s1][s2] = reward

# considerStates = []

# if (mdpType == "episodic"):
#     for endState in endStates:
#         for action in range(A):
#             transitionMatrix[action][int(endState)][int(endState)] = 1.0
#             rewardMatrix[action][int(endState)][int(endState)] = 0
#     for i in range(S):
#         if not(i in endStates):
#             considerStates.append(i)
# else :
#     for i in range(S):
#         considerStates.append(i)

value = np.zeros((1, S))
optPolicy = np.zeros(S, dtype = np.int64)

epsilon = 1e-6

np.random.seed(0)

'''
Here policy is a list of size S. One action for each state
transitionMatrix is of shape AxSxS
rewardMatrix is of shape AxSxS
The function returns a list of size S. One value for each state
'''
def getValue(policy, transitionMatrix, rewardMatrix, discount):
    '''
    First construct the coesfficient matrix. Coefficient matrix is of size SxS
    RHS will be of size (S, 1)
    '''
    coeffMatrix = np.zeros((S, S), dtype = np.float64)
    RHS = np.zeros((S, 1), dtype = np.float64)
    for i in range(S):
        coeffMatrix[i] = -discount*transitionMatrix[policy[i]][i]
        coeffMatrix[i][i] += 1
        RHS[i][0] = np.sum(transitionMatrix[policy[i]][i]*rewardMatrix[policy[i]][i])
    ans = np.zeros((S, 1))
    valueList = np.matmul(np.linalg.inv(coeffMatrix), RHS)
    ans = list(valueList)
    return np.reshape(ans, (1, S))

if ((policyPath == None) and algorithm == "vi"):
    prevValue = np.zeros((1, S))
    compMatrix = transitionMatrix*(rewardMatrix + discount*value)
    sumOverStates = np.sum(compMatrix, axis = 2)
    optPolicy = np.argmax(sumOverStates, axis = 0)
    value = np.max(sumOverStates, axis = 0)
    value = np.reshape(value, (1, S))
    while (abs(np.max(value - prevValue)) > 1e-9):
        prevValue = value
        compMatrix = transitionMatrix*(rewardMatrix + discount*value)
        sumOverStates = np.sum(compMatrix, axis = 2)
        optPolicy = np.argmax(sumOverStates, axis = 0)
        value = np.max(sumOverStates, axis = 0)
        value = np.reshape(value, (1, S))
    for i in range(S):
        print("{:.7f}".format(value[0][i]), optPolicy[i])
elif ((policyPath == None) and algorithm == "hpi"):
    '''
    Generating a random policy(Initialisation)
    '''
    x = 0
    for i in range(S):
        optPolicy[i] = np.random.randint(0, A)
    value = getValue(optPolicy, transitionMatrix, rewardMatrix, discount)

    compMatrix = transitionMatrix*(rewardMatrix + discount*value)   #Shape (A, S, S)
    sumOverStates = np.sum(compMatrix, axis = 2)                    #Shape (A, S). Every pair of state and action
    noOfImprovableStates = 0
    maxActionValues = np.max(sumOverStates, axis = 0)               #Shape S
    maxActions = np.argmax(sumOverStates, axis = 0)
    for i in range(S):
        if (maxActionValues[i] > value[0][i]):
            noOfImprovableStates += 1
            optPolicy[i] = maxActions[i]
    while (noOfImprovableStates > 0):
        x += 1
        value = getValue(optPolicy, transitionMatrix, rewardMatrix, discount)
        compMatrix = transitionMatrix*(rewardMatrix + discount*value)   #Shape (A, S, S)
        sumOverStates = np.sum(compMatrix, axis = 2)                    #Shape (A, S). Every pair of state and action
        noOfImprovableStates = 0
        maxActionValues = np.max(sumOverStates, axis = 0)               #Shape S
        maxActions = np.argmax(sumOverStates, axis = 0)
        for i in range(S):
            if ((maxActionValues[i] - value[0][i]) > epsilon):
                noOfImprovableStates += 1
                optPolicy[i] = maxActions[i]
    for i in range(S):
        print("{:.7f}".format(value[0][i]), optPolicy[i])
elif ((policyPath == None) and algorithm == "lp"):
    prob = LpProblem("Linear Programming", LpMaximize)
    states = np.arange(S)
    vals = LpVariable.dict("v", states)
    values = np.array([vals[i] for i in vals])                      #Values. This a list of size S
    values = np.reshape(values, (1, S))
    compMatrix = transitionMatrix*(rewardMatrix + discount*values)
    sumOverStates = np.sum(compMatrix, axis = 2)
    '''
    Adding constraints
    '''
    for i in range(A):
        count = 0
        for j in range(S):
            if not(j in endStates):
                constraintName = "C_" + str(i) + "_" + str(j)
                prob += (
                    values[0][j]>=sumOverStates[i][j], constraintName,
                )
            else:
                constraintName = "C_" + str(i) + "_" + str(j)
                prob += (
                    values[0][j]==0, constraintName
                ) 
    '''
    Objective function
    '''
    prob += (
        -1*lpSum(np.sum(values))
    )
    prob.solve(PULP_CBC_CMD(msg=0))
    for name, constraint in prob.constraints.items():
        splitName = name.split("_")
        action = int(splitName[1])
        state = int(splitName[2])
        if (constraint.value() < 1e-7):
            optPolicy[state] = action
    for i in range(S):
        if not(i in endStates):
            print("{:.7f}".format(values[0][i].varValue), optPolicy[i])
        else :
            print("0.000000", 0)
elif (not(policyPath == None)):    
    policyFile = open(policyPath, 'r')
    policyLines = policyFile.readlines()
    policy = np.zeros(S, dtype = np.int64)
    for i in range(S):
        policy[i] = int(policyLines[i].strip().split()[0])
    value = getValue(policy, transitionMatrix, rewardMatrix, discount)
    for i in range(S):
        print("{:.7f}".format(value[0][i]), policy[i])
