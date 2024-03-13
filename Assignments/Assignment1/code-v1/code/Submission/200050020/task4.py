"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the MultiBanditsAlgo class. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, set_pulled, reward): This method is called 
        just after the give_pull method. The method should update the 
        algorithm's internal state based on the arm that was pulled and the 
        reward that was received.
        (The value of arm_index is the same as the one returned by give_pull 
        but set_pulled is the set that is randomly chosen when the pull is 
        requested from the bandit instance.)
"""

import numpy as np

# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE


class MultiBanditsAlgo:
    def __init__(self, num_arms, horizon):
        # You can add any other variables you need here
        self.num_arms = num_arms
        self.horizon = horizon


        # START EDITING HERE
        self.alphas = [np.ones(num_arms),np.ones(num_arms)]
        self.betas = [np.ones(num_arms), np.ones(num_arms)]
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        samples_drawn1 = np.random.beta(self.alphas[0] , self.betas[0] )
        samples_drawn2 = np.random.beta(self.alphas[1] , self.betas[1] )
        samples_drawn = (samples_drawn1 + samples_drawn2)/2

        return np.argmax(samples_drawn)
        # END EDITING HERE
    
    def get_reward(self, arm_index, set_pulled, reward):
        # START EDITING HERE
        if reward == 1:
            self.alphas[set_pulled][arm_index] += 1
        else:
            self.betas[set_pulled][arm_index] += 1
        # END EDITING HERE

