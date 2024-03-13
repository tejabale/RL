import os
import sys
import random 
import json
import math
import utils
import time
import config
import numpy as np
random.seed(73)

class Agent:
    def __init__(self, table_config) -> None:
        self.table_config = table_config
        self.prev_action = None
        self.curr_iter = 0
        self.state_dict = {}
        self.holes =[]
        self.ns = utils.NextState()


    def set_holes(self, holes_x, holes_y, radius):
        for x in holes_x:
            for y in holes_y:
                self.holes.append((x[0], y[0]))
        self.ball_radius = radius
        
    def get_target(self, white_ball, colors_balls ):
        
        distances = np.linalg.norm(colors_balls - white_ball, axis=1)
        closest_color_ball_index = np.argmin(distances)
        target = colors_balls[closest_color_ball_index]
        return target
    
    def get_fake_target(self, target, holes, radius = config.ball_radius):
        distances = np.linalg.norm(holes - target, axis=1)
        closest_hole_indices = np.argsort(distances)[:2]
        fake_targets = []
        for i in reversed(range(len(closest_hole_indices))):
            closest_hole_index = closest_hole_indices[i]
            hole_target = holes[closest_hole_index]
        
            x2,y2 = target
            x3,y3 = hole_target
            dist_t_h = distances[closest_hole_index]
            
            h = x2 - ((2*radius*(x3-x2))/(dist_t_h))
            k = y2 - ((2*radius*(y3-y2))/(dist_t_h))
            fake_targets.append([h,k])
        return np.array(fake_targets)
        
    def get_angle(self, white_ball, target):
        x1,y1 = white_ball
        x2,y2 = target
        
        if x1 == x2:
            if y2 > y1: return 1
            else: return 0
            
        if y1 == y2:
            if x2 > x1: return -0.5
            else: return 0.5
            
        if x2 > x1 and y1 > y2:
            angle =  -np.arctan((x2-x1)/(y1-y2))
        
        elif x2 < x1 and y1 > y2:
            angle =  np.arctan((x1-x2)/(y1-y2))

        elif x2 < x1 and y1 < y2:
            angle =  np.arctan((y2-y1)/(x1-x2))
            angle += np.pi/2
            
        elif x2 > x1 and y1 < y2:
            angle =  -np.arctan((y2-y1)/(x2-x1))
            angle -= np.pi/2
            
        angle /= np.pi
        return angle
    
    
        
        
    
    def action(self, ball_pos=None):
        ## Code you agent here ##
        ## You can access data from config.py for geometry of the table, configuration of the levels, etc.
        ## You are NOT allowed to change the variables of config.py (we will fetch variables from a different file during evaluation)
        ## Do not use any library other than those that are already imported.
        ## Try out different ideas and have fun!
        
        holes = np.array(self.holes)
        colors_balls_pos = {key: value for key, value in ball_pos.items() if key not in (0, 'white')}
        colors_balls = np.array(list(colors_balls_pos.values()))
        white_ball = np.array(ball_pos['white'])
        
        best_angle  = None

        for i in range(min(len(colors_balls), 15)):
            target = colors_balls[i]
            fake_targets = self.get_fake_target(target, holes)
            for j in range(len(fake_targets)):
                shooting_angle = self.get_angle(white_ball, fake_targets[j])
                force = 0.75
                    
                n_target = self.ns.get_next_state(ball_pos, (shooting_angle, force), 42)
                
                if(len(n_target) < len(ball_pos)):
                    best_angle = shooting_angle
                    break

        if best_angle:
            return ( best_angle , force)
        else:
            return ( shooting_angle , force)

        # return (2*random.random() - 1, random.random())
