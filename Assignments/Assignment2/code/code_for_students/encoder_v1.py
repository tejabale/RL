import argparse,warnings
parser = argparse.ArgumentParser()
warnings.filterwarnings("ignore")
import numpy as np

import time


if __name__ == "__main__":
    
    start_time = time.time()
    
    parser.add_argument("--opponent")
    parser.add_argument("--p")
    parser.add_argument("--q")

    args = parser.parse_args()
    p = float(args.p)
    q = float(args.q)
    
    input_path = args.opponent
    input_file = open(input_path, 'r')
    input_lines = input_file.readlines()
    input_lines = input_lines[1:]
    
    R_policy = []
    index_to_state = dict()
    state_to_index = dict()

    for i in range(len(input_lines)):
        line = input_lines[i]
        line = line.strip().split()
        state = line[0]
        index_to_state[i] = state
        state_to_index[state] = i
        rpolicy = [ float(line[i+1]) for i in range(4)]
        R_policy.append(rpolicy)

    n_s = len(R_policy) + 2
    n_a = 10

    transitionMatrix = np.zeros((n_a,n_s,n_s))
    rewardsMatrix = np.zeros((n_a,n_s,n_s))

    for state_index in range(n_s-2):

        state_string = index_to_state[state_index]
        
           

        for rpolicy_idx in range(4):
            
            R_pos = int(state_string[4:6])
            R_row = int((R_pos-1)/4)
            R_col = int((R_pos-1)%4)
            R_prev_pos = R_pos
            
            if rpolicy_idx == 0:
                R_col -= 1
                R_pos -= 1
            elif rpolicy_idx == 1:
                R_col += 1
                R_pos += 1
            elif rpolicy_idx == 2:
                R_row -= 1
                R_pos -= 4
            elif rpolicy_idx == 3:
                R_row += 1
                R_pos += 4

            b_move_prob =  R_policy[state_index][rpolicy_idx]

            if b_move_prob == 0:
                continue

            for action in range(n_a):
                
                B1_pos = int(state_string[0:2])
                B2_pos = int(state_string[2:4])
                has_ball = int(state_string[6])

                B1_prev_pos = B1_pos
                B2_prev_pos = B2_pos
                has_ball_prev = has_ball
                
                B1_row = int((B1_pos-1)/4)
                B1_col = int((B1_pos-1)%4)

                B2_row = int((B2_pos-1)/4)
                B2_col = int((B2_pos-1)%4)

                
                if action == 0:
                    
                    B1_col -= 1
                    B1_pos -= 1

                    if B1_col < 0:
                        transitionMatrix[action,state_index,n_s-2] += b_move_prob

                    else:
                        next_state_str =  "{0:0=2d}".format(B1_pos) + "{0:0=2d}".format(B2_pos) + "{0:0=2d}".format(R_pos) + str(has_ball)
                        new_state_index = state_to_index[next_state_str]
                        if has_ball == 1:
                            if (B1_pos == R_pos) or (B1_prev_pos == R_pos and R_prev_pos == B1_pos):
                                transitionMatrix[action,state_index,new_state_index] += b_move_prob * (0.5-p)
                                transitionMatrix[action,state_index,n_s-2] += b_move_prob * (0.5+p)
                            else:
                                transitionMatrix[action,state_index,new_state_index] += b_move_prob * (1-2*p)
                                transitionMatrix[action,state_index,n_s-2] += b_move_prob * (2*p)
                        else:
                            transitionMatrix[action,state_index,new_state_index] += b_move_prob * (1-p)
                            transitionMatrix[action,state_index,n_s-2] += b_move_prob * (p)
                        
                        

                elif action == 1:

                    B1_col += 1
                    B1_pos += 1

                    if B1_col >= 4:
                        transitionMatrix[action,state_index,n_s-2] += b_move_prob

                    else:
                        next_state_str =  "{0:0=2d}".format(B1_pos) + "{0:0=2d}".format(B2_pos) + "{0:0=2d}".format(R_pos) + str(has_ball)
                        new_state_index = state_to_index[next_state_str]
                            
                        if has_ball == 1:
                            if B1_pos == R_pos or (B1_prev_pos == R_pos and R_prev_pos == B1_pos):
                                
                                transitionMatrix[action,state_index,new_state_index] += b_move_prob * (0.5-p)
                                transitionMatrix[action,state_index,n_s-2] += b_move_prob * (0.5+p)
                            else:
                                transitionMatrix[action,state_index,new_state_index] += b_move_prob * (1-2*p)
                                transitionMatrix[action,state_index,n_s-2] += b_move_prob * (2*p)
                        else:
                            transitionMatrix[action,state_index,new_state_index] += b_move_prob * (1-p)
                            transitionMatrix[action,state_index,n_s-2] += b_move_prob * (p)

                elif action == 2:

                    B1_row -= 1
                    B1_pos -= 4

                    if B1_row < 0:
                        transitionMatrix[action,state_index,n_s-2] += b_move_prob

                    else:
                        next_state_str =  "{0:0=2d}".format(B1_pos) + "{0:0=2d}".format(B2_pos) + "{0:0=2d}".format(R_pos) + str(has_ball)
                        new_state_index = state_to_index[next_state_str]
                        if has_ball == 1:
                            if B1_pos == R_pos or (B1_prev_pos == R_pos and R_prev_pos == B1_pos):
                                transitionMatrix[action,state_index,new_state_index] += b_move_prob * (0.5-p)
                                transitionMatrix[action,state_index,n_s-2] += b_move_prob * (0.5+p)
                            else:
                                transitionMatrix[action,state_index,new_state_index] += b_move_prob * (1-2*p)
                                transitionMatrix[action,state_index,n_s-2] += b_move_prob * (2*p)
                        else:
                            transitionMatrix[action,state_index,new_state_index] += b_move_prob * (1-p)
                            transitionMatrix[action,state_index,n_s-2] += b_move_prob * (p)


                elif action == 3:

                    B1_row += 1
                    B1_pos += 4

                    if B1_row >= 4:
                        transitionMatrix[action,state_index,n_s-2] += b_move_prob

                    else:
                        next_state_str =  "{0:0=2d}".format(B1_pos) + "{0:0=2d}".format(B2_pos) + "{0:0=2d}".format(R_pos) + str(has_ball)
                        new_state_index = state_to_index[next_state_str]
                        if has_ball == 1:
                            if B1_pos == R_pos or (B1_prev_pos == R_pos and R_prev_pos == B1_pos):
                                transitionMatrix[action,state_index,new_state_index] += b_move_prob * (0.5-p)
                                transitionMatrix[action,state_index,n_s-2] += b_move_prob * (0.5+p)
                            else:
                                transitionMatrix[action,state_index,new_state_index] += b_move_prob * (1-2*p)
                                transitionMatrix[action,state_index,n_s-2] += b_move_prob * (2*p)
                        else:
                            transitionMatrix[action,state_index,new_state_index] += b_move_prob * (1-p)
                            transitionMatrix[action,state_index,n_s-2] += b_move_prob * (p)

                elif action == 4:

                    B2_col -= 1
                    B2_pos -= 1

                    if B2_col < 0:
                        transitionMatrix[action,state_index,n_s-2] += b_move_prob

                    else:
                        next_state_str =  "{0:0=2d}".format(B1_pos) + "{0:0=2d}".format(B2_pos) + "{0:0=2d}".format(R_pos) + str(has_ball)
                        new_state_index = state_to_index[next_state_str]

                        if has_ball == 1:
                            transitionMatrix[action,state_index,new_state_index] += b_move_prob * (1-p)
                            transitionMatrix[action,state_index,n_s-2] += b_move_prob * (p)
                        else:
                            if B2_pos == R_pos or (B2_prev_pos == R_pos and R_prev_pos == B2_pos):
                                transitionMatrix[action,state_index,new_state_index] += b_move_prob * (0.5-p)
                                transitionMatrix[action,state_index,n_s-2] += b_move_prob * (0.5+p)
                            else:
                                transitionMatrix[action,state_index,new_state_index] += b_move_prob * (1-2*p)
                                transitionMatrix[action,state_index,n_s-2] += b_move_prob * (2*p)

                
                elif action == 5:

                    B2_col += 1
                    B2_pos += 1

                    if B2_col >= 4:
                        transitionMatrix[action,state_index,n_s-2] += b_move_prob

                    else:
                        next_state_str =  "{0:0=2d}".format(B1_pos) + "{0:0=2d}".format(B2_pos) + "{0:0=2d}".format(R_pos) + str(has_ball)
                        new_state_index = state_to_index[next_state_str]

                        if has_ball == 1:
                            transitionMatrix[action,state_index,new_state_index] += b_move_prob * (1-p)
                            transitionMatrix[action,state_index,n_s-2] += b_move_prob * (p)
                        else:
                            if B2_pos == R_pos or (B2_prev_pos == R_pos and R_prev_pos == B2_pos):
                                transitionMatrix[action,state_index,new_state_index] += b_move_prob * (0.5-p)
                                transitionMatrix[action,state_index,n_s-2] += b_move_prob * (0.5+p)
                            else:
                                transitionMatrix[action,state_index,new_state_index] += b_move_prob * (1-2*p)
                                transitionMatrix[action,state_index,n_s-2] += b_move_prob * (2*p)


                elif action == 6:

                    B2_row -= 1
                    B2_pos -= 4

                    if B2_row < 0:
                        transitionMatrix[action,state_index,n_s-2] += b_move_prob

                    else:
                        next_state_str =  "{0:0=2d}".format(B1_pos) + "{0:0=2d}".format(B2_pos) + "{0:0=2d}".format(R_pos) + str(has_ball)
                        new_state_index = state_to_index[next_state_str]

                        if has_ball == 1:
                            transitionMatrix[action,state_index,new_state_index] += b_move_prob * (1-p)
                            transitionMatrix[action,state_index,n_s-2] += b_move_prob * (p)
                        else:
                            if B2_pos == R_pos or (B2_prev_pos == R_pos and R_prev_pos == B2_pos):
                                transitionMatrix[action,state_index,new_state_index] += b_move_prob * (0.5-p)
                                transitionMatrix[action,state_index,n_s-2] += b_move_prob * (0.5+p)
                            else:
                                transitionMatrix[action,state_index,new_state_index] += b_move_prob * (1-2*p)
                                transitionMatrix[action,state_index,n_s-2] += b_move_prob * (2*p)


                elif action == 7:

                    B2_row += 1
                    B2_pos += 4

                    if B2_row >= 4:
                        transitionMatrix[action,state_index,n_s-2] += b_move_prob

                    else:
                        next_state_str =  "{0:0=2d}".format(B1_pos) + "{0:0=2d}".format(B2_pos) + "{0:0=2d}".format(R_pos) + str(has_ball)
                        new_state_index = state_to_index[next_state_str]

                        if has_ball == 1:
                            transitionMatrix[action,state_index,new_state_index] += b_move_prob * (1-p)
                            transitionMatrix[action,state_index,n_s-2] += b_move_prob * (p)
                        else:
                            if B2_pos == R_pos or (B2_prev_pos == R_pos and R_prev_pos == B2_pos):
                                transitionMatrix[action,state_index,new_state_index] += b_move_prob * (0.5-p)
                                transitionMatrix[action,state_index,n_s-2] += b_move_prob * (0.5+p)
                            else:
                                transitionMatrix[action,state_index,new_state_index] += b_move_prob * (1-2*p)
                                transitionMatrix[action,state_index,n_s-2] += b_move_prob * (2*p)

                elif action == 8:
                    
                    if has_ball == 1:
                        new_ball_pos = 2
                    else:
                        new_ball_pos = 1
                    
                    passing_prob = q-(0.1 * max(abs(B1_row - B2_row) , abs(B1_col - B2_col)))

                    if B1_col == B2_col and R_col == B1_col and ( (B1_row <= R_row and R_row <= B2_row and B1_row <= B2_row) or  (B2_row <= R_row and R_row <= B1_row and B2_row <= B1_row)):
                        passing_prob = passing_prob/2
                    
                    elif B1_row == B2_row and R_row == B1_row and ( (B1_col <= R_col and R_col <= B2_col and B1_col <= B2_col) or  (B2_col <= R_col and R_col <= B1_col and B2_col <= B1_col)):
                        passing_prob = passing_prob/2

                    else:
                        c1 = B1_row + B1_col
                        c2 = B2_row + B2_col
                        c3 = R_row + R_col

                        d1 = B1_row - B1_col
                        d2 = B2_row - B2_col
                        d3 = R_row - R_col

                        btw_row = ((abs(B1_row - R_row) + abs(R_row - B2_row)) == abs(B1_row - B2_row) )
                        btw_col = ((abs(B1_col - R_col) + abs(R_col - B2_col)) == abs(B1_col - B2_col) )

                        if c1 == c2 and c2 == c3 and btw_row and btw_col:
                            passing_prob = passing_prob/2

                        elif d1 == d2 and d2 == d3 and btw_row and btw_col:
                            passing_prob = passing_prob/2

                    next_state_str =  "{0:0=2d}".format(B1_pos) + "{0:0=2d}".format(B2_pos) + "{0:0=2d}".format(R_pos) + str(new_ball_pos)
                    new_state_index = state_to_index[next_state_str]

                    transitionMatrix[action,state_index,new_state_index] += b_move_prob * passing_prob
                    transitionMatrix[action,state_index,n_s-2] += b_move_prob * (1-passing_prob)

                elif action == 9:

                    if has_ball == 1:
                        goal_prob = q - (0.2 * (3 - B1_col ))
                    else:
                        goal_prob = q - (0.2 * (3 - B2_col ))

                    if R_pos == 8 or R_pos == 12:
                        goal_prob = goal_prob/2

                    transitionMatrix[action,state_index,n_s-2] += b_move_prob * (1-goal_prob)
                    transitionMatrix[action,state_index,n_s-1] += b_move_prob * goal_prob
                    rewardsMatrix[action,state_index, n_s-1] = 1


    print(f"numStates {n_s}")
    print(f"numActions {n_a}")
    print(f"end {n_s-2} {n_s-1}")

    for i in range(n_a):
        for j in range(n_s):
            for k in range(n_s):
                if transitionMatrix[i][j][k] != 0:
                    print(f"transition {j} {i} {k} {rewardsMatrix[i][j][k]} {transitionMatrix[i][j][k]}")
    

    print("mdptype episodic")
    print("discount", 1.0)
    
    end_time = time.time()
    print("Time taken by the encoder is", end_time-start_time)
    





