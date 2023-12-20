import numpy as np

class Environment:
    '''
    Environment class which specifies how the agent moves through a given maze
    '''

    def __init__(self, **p):
        '''
        Initialise the class instance 
        '''

        self.__dict__.update(**p)
        self.num_states = self.num_x_states * self.num_y_states
        self._generate_env_model()

        return None

    def _generate_env_model(self):

        '''
        Generate transition and reward models
        '''

        self.T = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.R = np.zeros((self.num_states, self.num_actions), dtype=int)

        for s in range(self.num_states):
            for a in range(self.num_actions):

                s1, r = self._get_new_state(s, a)
                self.T[s, a, s1] += 1
                self.R[s, a] = r

        # normalise
        for s in range(self.num_states):
            for a in range(self.num_actions):
                if ~np.all(self.T[s, a, :] == 0):
                    self.T[s, a, :] /= np.sum(self.T[s, a, :])

        return None

    def _get_new_state(self, s, a):
        '''
        Returns next state and reward
        params:
            s: state
            a: action 
        '''

        i, j  = self._convert_state_to_coords(s)
        s1, r = None, 0

        # check boundaries
        if i == 0: # top end
            if a == 0: # action up
                return s, r
        elif i == self.num_y_states - 1: # bottom end
            if a == 1: # action down
                return s, r
        if j == 0: # left end
            if a == 2: # move left
                return s, r
        elif j == self.num_x_states - 1: # right end
            if a == 3: # move right
                return s, r

        if a == 0: # move up
            ni, nj = i-1, j
        elif a == 1: # move down
            ni, nj = i+1, j
        elif a == 2: # move left
            ni, nj = i, j-1
        else: # move right
            ni, nj = i, j+1

        s1 = self._convert_coords_to_state(ni, nj)

        # check blocked states
        if s1 in self.blocked_states:
            return s, r
        
        # check if goal
        if s1 == self.goal_state:
            r = self.reward_at_goal

        return s1, r
    
    def _convert_state_to_coords(self, s):
        '''
        Convert state to coordinates to index the maze
        params:
            s: state
        '''

        return s//self.num_x_states, s%self.num_x_states
    
    def _convert_coords_to_state(self, i, j):
        '''
        Convert coordinates to state
        params:
            i: y coordinate
            j: x coordinate
        '''

        return np.arange(self.num_y_states*self.num_x_states).reshape(self.num_y_states, self.num_x_states)[i][j]
    

class TwoStepEnv:

    def __init__(self):

        self.num_states = 3
        self.num_actions = 2
        self._reset()
        self.init_rewardp()
        self.rewardp_history = np.empty((0,4))

        return None
    
    def init_rewardp(self):

        self.rewardp = 0.5 * np.ones(4)

        return None
    
    def _reset(self):

        self.s = 0 # current state

        return None
    
    def _update_rewardp(self):

        walk = np.random.normal(0, 0.025, size=4)
        self.rewardp = 2 * np.clip(self.rewardp + walk, 0.25, 0.75) - self.rewardp - walk

        return None

    def step(self, a):
        # if we are in initial state transition to next state
        if self.s == 0:
            r = 0
            p = np.random.uniform()
            if p <= 0.7:
                self.s += a + 1
            else:
                self.s += 2 - a
        
        # if we are not in initial state deliver rewards and reset environment
        else:
            if np.random.uniform() <= self.rewardp[self.s*self.num_actions+a-2]:
                r = 1
            else:
                r = 0
            self._update_rewardp()
            self._reset()
        
        return r



