"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value

# START EDITING HERE
# You can use this space to define any helper functions that you need


def divergence_dist(p,q):

    if (abs(p-q) < 1e-6):
        return 0
    elif (abs(1-q) < 1e-6):
        return np.inf
    elif (abs(p) < 1e-6 and abs(1-q) > 1e-6):
        return -math.log(1-q)
    elif (abs(1-p) < 1e-6):
        return p*(math.log(p/q))
    else:
        return p*(math.log(p/q)) + (1-p)*(math.log((1-p)/(1-q)))

def get_klucb( N ,p ,t, c, num_arms):

    klucb = np.zeros(num_arms)
    upper_bound = math.log(t) + c*math.log(math.log(t))
    for i in range(num_arms):
        low = p[i]
        high = 1
        while (high - low > 1e-6):
            mid = (high+low)/2
            div_dist = divergence_dist(p[i], mid)
            difference = div_dist - upper_bound/N[i]

            if (abs(difference) < 1e-6):
                break
            if (difference > 0):
                high = mid
            else:
                low = mid

        klucb[i] = (low + high)/2

    return klucb


# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # START EDITING HERE
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.ucb = np.zeros(num_arms)
        self.total_num_pulls = 0
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE

        if self.total_num_pulls < self.num_arms:
            return self.total_num_pulls
        
        self.ucb = self.values + np.sqrt( (2*math.log(self.total_num_pulls + 1))/ self.counts  )
        return np.argmax(self.ucb)
        # END EDITING HERE  
        
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.total_num_pulls += 1
        value = self.values[arm_index]
        total_rewards_arm =  value*self.counts[arm_index]
        new_total_rewards_arm = total_rewards_arm + reward
        self.counts[arm_index] += 1
        self.values[arm_index] = new_total_rewards_arm / self.counts[arm_index]
        # END EDITING HERE


class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE

        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.total_reward = np.zeros(num_arms)
        self.klucb = np.zeros(num_arms)
        self.c = 0
        self.total_num_pulls = 0

        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        
        if self.total_num_pulls < self.num_arms:
            return self.total_num_pulls

        t = self.total_num_pulls + 1
        self.klucb = get_klucb(self.counts , self.values,  t, self.c, self.num_arms)
        return np.argmax(self.klucb)

        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE

        self.total_num_pulls += 1
        value = self.values[arm_index]
        total_rewards_arm =  value*self.counts[arm_index]
        new_total_rewards_arm = total_rewards_arm + reward
        self.counts[arm_index] += 1
        self.values[arm_index] = new_total_rewards_arm / self.counts[arm_index]


        # END EDITING HERE

class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.alphas = np.ones(num_arms)
        self.betas = np.ones(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        samples_drawn = np.random.beta(self.alphas , self.betas )
        return np.argmax(samples_drawn)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        if reward == 1:
            self.alphas[arm_index] += 1
        else:
            self.betas[arm_index] += 1

        # END EDITING HERE
