import argparse,warnings
parser = argparse.ArgumentParser()
warnings.filterwarnings("ignore")

import numpy as np


if __name__ == "__main__":

    parser.add_argument("--value-policy")
    parser.add_argument("--opponent")

    args = parser.parse_args()

    input_path = args.opponent
    input_file = open(input_path, 'r')
    input_lines = input_file.readlines()
    input_lines = input_lines[1:]

    index_to_state = dict()
    state_to_index = dict()

    for i in range(len(input_lines)):
        line = input_lines[i]
        line = line.strip().split()
        state = line[0]
        index_to_state[i] = state
        state_to_index[state] = i

    output_path = args.value_policy
    output_file = open(output_path, 'r')
    output_lines = output_file.readlines()


    for i in range(len(output_lines)-2):
        line = output_lines[i]
        line = line.strip().split()
        optimal_value = line[0]
        optimal_action = int(line[1])
        print(index_to_state[i] , optimal_action, optimal_value)