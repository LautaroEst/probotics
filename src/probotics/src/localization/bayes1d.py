
import numpy as np

class BayesFilter1D:

    def __init__(self, initial_belif, Ncells=20):
        self.current_belief = initial_belif
        self.Ncells = Ncells
        self._create_action_probs_matrices()

    def _create_action_probs_matrices(self):
        # action_probs_forward[i,j] = P(j|forward,i)
        action_probs_forward = np.zeros((self.Ncells, self.Ncells))
        for i in range(self.Ncells-2):
            action_probs_forward[i,i] = 0.25
            action_probs_forward[i,i+1] = 0.5
            action_probs_forward[i,i+2] = 0.25
        action_probs_forward[self.Ncells-2,self.Ncells-2] = 0.25
        action_probs_forward[self.Ncells-2,self.Ncells-1] = 0.75
        action_probs_forward[self.Ncells-1,self.Ncells-1] = 1.
        self.action_probs_forward = action_probs_forward

        # action_probs_backward[i,j] = P(j|backward,i)
        action_probs_backward = np.zeros((self.Ncells, self.Ncells))
        for i in range(2,self.Ncells):
            action_probs_backward[i,i] = 0.25
            action_probs_backward[i,i-1] = 0.5
            action_probs_backward[i,i-2] = 0.25
        action_probs_backward[1,1] = 0.25
        action_probs_backward[1,0] = 0.75
        action_probs_backward[0,0] = 1.
        self.action_probs_backward = action_probs_backward

    def forward_step(self, belief):
        # new_belief[j] = sum_j P(j|forward,i) * current_belief[i]
        new_belief = np.dot(belief, self.action_probs_forward)
        return new_belief

    def backward_step(self, belief):
        # new_belief[j] = sum_j P(j|backward,i) * current_belief[i]
        new_belief = np.dot(belief, self.action_probs_backward)
        return new_belief

    def step(self, belief, action):
        if action == 'forward':
            new_belief = self.forward_step(belief)
        elif action == 'backward':
            new_belief = self.backward_step(belief)
        else:
            raise ValueError("Invalid action")
        return new_belief

    def fit(self, actions):
        belief = self.current_belief
        history = [belief]
        for action in actions:
            belief = self.step(belief, action)
            history.append(belief)
        return history