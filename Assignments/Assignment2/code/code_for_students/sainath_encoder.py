import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--states")
parser.add_argument("--parameters")
parser.add_argument("--q")

args = parser.parse_args()
statesPath = args.states
parametersPath = args.parameters
q = float(args.q)

parametersFile = open(parametersPath, 'r')
parameterLines = parametersFile.readlines()

probs = np.zeros((5, 7))

for i in range(1, 6):
    parameterLine = parameterLines[i].strip().split()
    for j in range(1, 8):
        probs[i-1][j-1] = float(parameterLine[j])

statesFile = open(statesPath, 'r')
stateLines = statesFile.readlines()

stateLines = [i.strip().split()[0] for i in stateLines]
stateLinesA = [(i + "0") for i in stateLines]
stateLinesB = [(i + "1") for i in stateLines]
states = stateLinesA + stateLinesB
states += ["W", "L"]
S = len(states)
actions = np.array([0, 1, 2, 4, 6])
outcomes = np.array([-1, 0, 1, 2, 3, 4, 6])
boutcomes = np.array([-1, 0, 1])
A = len(actions)

mapStrToState = {}
mapStateToStr = {}
for i in range(len(states)):
    mapStrToState[states[i]] = i
    mapStateToStr[i] = states[i]

discount = 1.0
transitions = []

transitionMatrix = np.zeros((A, S, S))

for i in range(S-2):
    balls = int(states[i][0:2])
    runs = int(states[i][2:4])
    player = int(states[i][4])
    if (player == 0):
        for j in range(len(actions)):
            for k in range(len(outcomes)):
                if (probs[j][k] > 1e-6):
                    if ((outcomes[k] == -1) or ((balls == 1) and (runs > outcomes[k]))):
                        transitionMatrix[j][i][S-1] += probs[j][k] 
                    elif (runs <= outcomes[k]):
                        transitionMatrix[j][i][S-2] += probs[j][k]
                    else :
                        scored = int(outcomes[k])
                        ballsNew = balls - 1
                        runsNew = runs - scored
                        playerNew = player
                        if ((ballsNew%6 != 0 and scored%2 == 1) or (ballsNew%6 == 0 and scored%2 == 0)):
                            playerNew = 1 - player
                        ballsNew = (str(ballsNew)).zfill(2)
                        runsNew = (str(runsNew)).zfill(2)
                        newStateStr = ballsNew + runsNew + str(playerNew)
                        newState = mapStrToState[newStateStr]
                        transitionMatrix[j][i][newState] += probs[j][k]
    else:
        for j in range(len(actions)):
            for k in range(len(boutcomes)):
                if (boutcomes[k] == -1):
                    transitionMatrix[j][i][S-1] += q
                elif (((balls == 1) and (runs > boutcomes[k]))):
                    transitionMatrix[j][i][S-1] += (1-q)/2
                elif (runs <= boutcomes[k]):
                    transitionMatrix[j][i][S-2] += (1-q)/2
                else :
                    scored = int(boutcomes[k])
                    ballsNew = balls - 1
                    runsNew = runs - scored
                    playerNew = player
                    if ((ballsNew%6 != 0 and scored%2 == 1) or (ballsNew%6 == 0 and scored%2 == 0)):
                        playerNew = 1 - player
                    ballsNew = (str(ballsNew)).zfill(2)
                    runsNew = (str(runsNew)).zfill(2)
                    newStateStr = ballsNew + runsNew + str(playerNew)
                    newState = mapStrToState[newStateStr]
                    transitionMatrix[j][i][newState] += (1-q)/2

print("numStates", S)
print("numActions", len(actions))
print("end", S-1, S-2)

for i in range(A):
    for j in range(S):
        for k in range(S):
            reward = 0
            if (k == S-2):
                reward = 1
            print("transition", j, i, k, reward ,transitionMatrix[i][j][k])

print("mdptype episodic")
print("discount", 1.0)