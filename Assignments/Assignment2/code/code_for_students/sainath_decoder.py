import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--value-policy")
parser.add_argument("--states")

args = parser.parse_args()

valuePolicyPath = args.value_policy
statesPath = args.states

states = []
actions = [0, 1, 2, 4, 6]

statesFile = open(statesPath, 'r')
stateLines = statesFile.readlines()

valuePolicyFile = open(valuePolicyPath, 'r')
valuePolicyLines = valuePolicyFile.readlines()
numLines = len(valuePolicyLines)

for stateLine in stateLines:
    state = stateLine.strip().split()[0]
    states.append(state)

count = 0
for i in range(int((numLines-2)/2)):
    vp = valuePolicyLines[i].strip().split()
    print(states[count], actions[int(vp[1])], vp[0])
    count += 1