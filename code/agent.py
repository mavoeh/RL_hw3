import numpy as np
from scipy import stats # for gaussian noise
from environment import Environment, TwoStepEnv

class DynaAgent(Environment):

    def __init__(self, alpha, gamma, epsilon):

        '''
        Initialise the agent class instance
        Input arguments:
            alpha   -- learning rate \in (0, 1]
            gamma   -- discount factor \in (0, 1)
            epsilon -- controls the influence of the exploration bonus
        '''

        self.alpha   = alpha
        self.gamma   = gamma 
        self.epsilon = epsilon

        return None

    def init_env(self, **env_config):

        '''
        Initialise the environment
        Input arguments:
            **env_config -- dictionary with environment parameters
        '''

        Environment.__init__(self, **env_config)

        return None

    def _init_q_values(self):

        '''
        Initialise the Q-value table
        '''

        self.Q = np.zeros((self.num_states, self.num_actions))

        return None

    def _init_experience_buffer(self):

        '''
        Initialise the experience buffer
        '''

        self.experience_buffer = np.zeros((self.num_states*self.num_actions, 4), dtype=int)
        for s in range(self.num_states):
            for a in range(self.num_actions):
                self.experience_buffer[s*self.num_actions+a] = [s, a, 0, s]

        return None

    def _init_history(self):

        '''
        Initialise the history
        '''

        self.history = np.empty((0, 4), dtype=int)

        return None
    
    def _init_action_count(self):

        '''
        Initialise the action count
        '''

        self.action_count = np.zeros((self.num_states, self.num_actions), dtype=int)

        return None

    def _update_experience_buffer(self, s, a, r, s1):

        '''
        Update the experience buffer (world model)
        Input arguments:
            s  -- initial state
            a  -- chosen action
            r  -- received reward
            s1 -- next state
        '''

        self.experience_buffer[s*self.num_actions+a] = [s, a, r, s1]

        return None

    def _update_qvals(self, s, a, r, s1, bonus=False):

        '''
        Update the Q-value table
        Input arguments:
            s     -- initial state
            a     -- chosen action
            r     -- received reward
            s1    -- next state
            bonus -- True / False whether to use exploration bonus or not
        '''

        self.Q[s,a] += self.alpha * ( r + self.epsilon * bonus * np.sqrt(self.action_count[s,a]) + self.gamma * self.Q[s1,:].max() - self.Q[s,a] )

        return None

    def _update_action_count(self, s, a):

        '''
        Update the action count
        Input arguments:
            Input arguments:
            s  -- initial state
            a  -- chosen action
        '''

        self.action_count += 1
        self.action_count[s,a] = 0

        return None

    def _update_history(self, s, a, r, s1):

        '''
        Update the history
        Input arguments:
            s     -- initial state
            a     -- chosen action
            r     -- received reward
            s1    -- next state
        '''

        self.history = np.vstack((self.history, np.array([s, a, r, s1])))

        return None

    def _policy(self, s):

        '''
        Agent's policy 
        Input arguments:
            s -- state
        Output:
            a -- index of action to be chosen
        '''

        Q_exp = self.Q[s,:] + self.epsilon * np.sqrt(self.action_count[s,:])
        a = np.random.choice( np.flatnonzero(Q_exp == Q_exp.max()) )

        return a

    def _plan(self, num_planning_updates):

        '''
        Planning computations
        Input arguments:
            num_planning_updates -- number of planning updates to execute
        '''

        for _ in range(num_planning_updates):
            i = np.random.randint(self.experience_buffer.shape[0])
            s, a, r, s1 = self.experience_buffer[i,:]
            self._update_qvals(s, a, r, s1, bonus=True)

        return None

    def get_performace(self):

        '''
        Returns cumulative reward collected prior to each move
        '''

        return np.cumsum(self.history[:, 2])

    def simulate(self, num_trials, reset_agent=True, num_planning_updates=None):

        '''
        Main simulation function
        Input arguments:
            num_trials           -- number of trials (i.e., moves) to simulate
            reset_agent          -- whether to reset all knowledge and begin at the start state
            num_planning_updates -- number of planning updates to execute after every move
        '''

        if reset_agent:
            self._init_q_values()
            self._init_experience_buffer()
            self._init_action_count()
            self._init_history()

            self.s = self.start_state

        for _ in range(num_trials):

            # choose action
            a  = self._policy(self.s)
            # get new state
            s1 = np.random.choice(np.arange(self.num_states), p=self.T[self.s, a, :])
            # receive reward
            r  = self.R[self.s, a]
            # learning
            self._update_qvals(self.s, a, r, s1, bonus=False)
            # update world model 
            self._update_experience_buffer(self.s, a, r, s1)
            # reset action count
            self._update_action_count(self.s, a)
            # update history
            self._update_history(self.s, a, r, s1)
            # plan
            if num_planning_updates is not None:
                self._plan(num_planning_updates)

            if s1 == self.goal_state:
                self.s = self.start_state
            else:
                self.s = s1

        return None
    

    
class MyDynaAgent(DynaAgent):


    def _init_action_count(self):

        '''
        Initialise the action count
        '''

        self.action_count = np.zeros((self.num_states, self.num_actions), dtype=float)

        return None

    def _update_action_count(self, s, a):

        '''
        Update the action count
        Input arguments:
            Input arguments:
            s  -- initial state
            a  -- chosen action
        '''
        coords = Environment._convert_state_to_coords(self, np.arange(self.num_states))
        distance = np.sqrt( (coords[0]-coords[0][s])**2 + (coords[1]-coords[1][s])**2 )
        self.action_count += distance.reshape(self.num_states,1)
        self.action_count[s,a] = 0










    
class TwoStepAgent(TwoStepEnv):

    def __init__(self, env, alpha1, alpha2, beta1, beta2, lam, w, p):

        '''
        Initialise the agent class instance
        Input arguments:
            env    -- environment
            alpha1 -- learning rate for the first stage \in (0, 1]
            alpha2 -- learning rate for the second stage \in (0, 1]
            beta1  -- inverse temperature for the first stage
            beta2  -- inverse temperature for the second stage
            lam    -- eligibility trace parameter
            w      -- mixing weight for MF vs MB \in [0, 1] 
            p      -- perseveration strength
        '''

        self.env = env
        self.num_actions = env.num_actions
        self.num_states = env.num_states
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2
        self.lam    = lam
        self.w      = w
        self.p      = p

        return None
    
    def _init_qvals(self):
        self.QTD = np.zeros((self.num_states, self.num_actions))
        self.QMB = np.zeros((self.num_states, self.num_actions))
        self.Qnet = np.zeros((self.num_states, self.num_actions))
        
    def _init_history(self):

        '''
        Initialise history to later compute stay probabilities
        '''

        self.history = np.empty((0, 3), dtype=int)

        return None
    
    def _update_history(self, a, s1, r1):

        '''
        Update history
        Input arguments:
            a  -- first stage action
            s1 -- second stage state
            r1 -- second stage reward
        '''

        self.history = np.vstack((self.history, [a, s1, r1]))

        return None
    
    def get_stay_probabilities(self):

        '''
        Calculate stay probabilities
        '''

        common_r      = 0
        num_common_r  = 0
        common_nr     = 0
        num_common_nr = 0
        rare_r        = 0
        num_rare_r    = 0
        rare_nr       = 0
        num_rare_nr   = 0

        num_trials = self.history.shape[0]
        for idx_trial in range(num_trials-1):
            a, s1, r1 = self.history[idx_trial, :]
            a_next    = self.history[idx_trial+1, 0]

            # common
            if (a == 0 and s1 == 1) or (a == 1 and s1 == 2):
                # rewarded
                if r1 == 1:
                    if a == a_next:
                        common_r += 1
                    num_common_r += 1
                else:
                    if a == a_next:
                        common_nr += 1
                    num_common_nr += 1
            else:
                if r1 == 1:
                    if a == a_next:
                        rare_r += 1
                    num_rare_r += 1
                else:
                    if a == a_next:
                        rare_nr += 1
                    num_rare_nr += 1

        return np.array([common_r/num_common_r, rare_r/num_rare_r, common_nr/num_common_nr, rare_nr/num_rare_nr])
    
    def _update_QTD(self, s, a, r, s1):
        
        if s == 0:
            self.QTD[0,a] += self.alpha1 * (self.QTD[s1,:].max() - self.QTD[0,a])
        else:
            self.QTD[s,a] += self.alpha2 * (r - self.QTD[s,a])
            self.QTD[0,self.history[-1,0]] += self.alpha1 * self.lam * (r - self.QTD[s,a])

        return None
    
    def _update_QMB(self, s):
        
        if s == 0:
            self.QMB[s,:] = np.array([0.7,0.3]) * self.QTD[1,:].max() + np.array([0.3,0.7]) * self.QTD[2,:].max()
        else:
            self.QMB[s,:] = self.QTD[s,:]

        return None
    
    def _update_Qnet(self):

        self.Qnet = self.w * self.QMB + (1 - self.w) * self.QTD

        return None
    
    def _learn(self, s, a, r, s1):

        self._update_QTD(s, a, r, s1)
        self._update_QMB(s)
        self._update_Qnet()
        
        return None

    
    def _policy(self, s):

        if s == 0:
            if len(self.history[:,0]) >= 1:
                rep = np.array([[1,0],[0,1]][self.history[-1,0]])
            else:
                rep = 0
            beta = self.beta1
        else:
            rep = 0
            beta = self.beta2
        bf = np.exp(beta * self.Qnet[s,:] + self.p * rep)

        return np.random.choice([0,1], p = bf/bf.sum())

    def simulate(self, num_trials):

        '''
        Main simulation function
        Input arguments:
            num_trials -- number of trials to simulate
        '''
        self.env.init_rewardp()
        self._init_history()
        self._init_qvals()
        for _ in range(num_trials):
            
            # choose initial action
            s = self.env.s
            a = self._policy(s)
            # step environment
            self.env.step(a)
            # get new state
            s1 = self.env.s
            # choose second action
            a1 = self._policy(s1)
            # get reward
            r = self.env.step(a1)
            # update history
            self._update_history(a, s1, r)
            # learning
            self._learn(s, a, 0, s1)
            self._learn(s1, a1, r, None)
            # reset environment
            self.env._reset()

        return None