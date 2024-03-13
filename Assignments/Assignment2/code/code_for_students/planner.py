import argparse,warnings
parser = argparse.ArgumentParser()
warnings.filterwarnings("ignore")

from pulp import *
import numpy as np
np.random.seed(42)


def get_MDP(args):
    
    with open(args.mdp, 'r') as file:

        for line in file:
            line = line.strip().split()
            if line[0] == 'numStates':
                n_s = int(line[1])
            elif line[0] == 'numActions':
                n_a = int(line[1])
                transition = np.zeros((n_a,n_s,n_s))
                rewards = np.zeros((n_a,n_s,n_s))
            elif line[0] == 'end':
                end = line[1:]
                end = [int(i) for i in end]
            elif line[0] == 'transition':
                transition[int(line[2])][int(line[1])][int(line[3])] = float(line[5])
                rewards[int(line[2])][int(line[1])][int(line[3])] = float(line[4])
            elif line[0] == 'mdptype':
                mdptype = line[1]
            elif line[0] == 'discount':
                discount = float(line[1])

    return n_s,n_a,transition, rewards, end, mdptype, discount


def get_values(transition, rewards, policy, discount):

    n_a, n_s,n_s = transition.shape
    coefficients = np.zeros((n_s, n_s), dtype=float)
    _list = []

    for i in range(n_s):
        coefficients[i] = - discount * transition[policy[i], i]
        coefficients[i, i] += 1
        _list.append(  np.sum( transition[policy[i], i] * rewards[policy[i], i] ))

    _list = np.array(_list).reshape((-1,1))
    values = np.linalg.solve(coefficients, _list).reshape((1,-1))

    return values

  

if __name__ == "__main__":
    
    parser.add_argument("--mdp", type=str)
    parser.add_argument("--algorithm",  choices=["vi", "hpi", "lp"], type = str, default="vi")
    parser.add_argument("--policy")

    args = parser.parse_args()
    n_s,n_a,transition, rewards, end, mdptype, discount = get_MDP(args)

    
    if not args.policy == None:
        given_policy_list = []
        with open(args.policy, 'r') as file:
            for line in file:
                action = line.strip().split()[0]
                given_policy_list.append(int(action))
        given_policy_array = np.array(given_policy_list)
        optimal_values = get_values(transition, rewards, given_policy_array, discount)

        for i in range(n_s):
            print("{:.10f}".format(optimal_values[0][i]), given_policy_array[i])

    else:
        if args.algorithm == 'vi':
            v_t = np.zeros((1, n_s))

            while 1:
                summations = np.sum( transition*(rewards + discount*v_t) , axis = 2)
                optimal_policy = np.argmax(summations, axis = 0)
                v_t_1 = np.max(summations, axis = 0).reshape((1,n_s))

                if abs(np.max(v_t_1 - v_t)) < 1e-9:
                    break
                else:
                    v_t = v_t_1

            for i in range(n_s):
                print("{:.10f}".format(v_t_1[0][i]), optimal_policy[i])

        
        elif args.algorithm == 'hpi':

            optimal_policy = [np.random.randint(0, n_a) for i in range(n_s)]
            optimal_policy = np.array(optimal_policy , dtype=int)

            while 1:
                values_policy = get_values(transition, rewards, optimal_policy, discount)
                summations = np.sum( transition*(rewards + discount*values_policy) , axis = 2)

                improvable_states = 0
                actions_max_values = np.max(summations, axis = 0)
                actions_max_values_indices = np.argmax(summations, axis = 0)

                for i in range(n_s):
                    if (actions_max_values[i] - values_policy[0][i] > 1e-6):
                        improvable_states += 1
                        optimal_policy[i] = actions_max_values_indices[i]
                
                if improvable_states == 0:
                    break
            
            for i in range(n_s):
                print("{:.10f}".format(values_policy[0][i]), optimal_policy[i])

            
        elif args.algorithm == 'lp':

            optimal_policy = np.zeros(n_s, dtype = int)
            problem = LpProblem("Linear Programming", LpMaximize)
            vals = LpVariable.dict("v", np.arange(n_s))
            optimal_values = np.array([vals[i] for i in vals]).reshape((1,n_s))
            summations = np.sum( transition*(rewards + discount*optimal_values) , axis = 2)

            for i in range(n_a):
                for j in range(n_s):
                    name_cons =  str(i) + "|" + str(j)
                    if not(j in end):
                        problem += ( optimal_values[0][j]>=summations[i][j], name_cons)
                    else:
                        problem += (optimal_values[0][j]==0, name_cons) 

            problem += (-1*lpSum(np.sum(optimal_values)))

            problem.solve(PULP_CBC_CMD(msg=0))

            for name, constraint in problem.constraints.items():
                if constraint.value() < 1e-7:
                    A = int(name.split("|")[0])
                    S = int(name.split("|")[1])
                    optimal_policy[S] = A

            for i in range(n_s):
                if not(i in end):
                    print("{:.10f}".format(optimal_values[0][i].varValue), optimal_policy[i])
                else :
                    print("0.000000", 0)



        

    