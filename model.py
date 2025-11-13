import numpy as np

class HMM:
    def __init__(self, hidden_states, label):
        self.hidden_states = hidden_states
        self.PI = np.random.rand(self.hidden_states)
        self.A = np.random.rand(self.hidden_states, self.hidden_states)
        self.B = np.random.rand(self.hidden_states)
        self.B_parameters = np.random.rand(self.hidden_states, 2)
        self.label = label

    # Caution : This resets the parameters!! --> Requires training from scratch
    def set_hidden_states(self, size):
        self.hidden_states = size
        self.PI = np.random.rand(size)
        self.A = np.random.rand(size, size)
        self.B = np.random.rand(size, 2)

    def get_hidden_states(self):
        return self.hidden_states
    
    def get_label(self):
        return self.label

    # calculates the forward pass
    # return alpha matrix
    def forward(self, sequence):
        
        alpha = np.zeros((len(sequence), self.hidden_states), dtype=float)

        # initialize (t = 1), first row
        index = 0
        for entry in alpha[0]:
            
            alpha[0][index] = self.PI[index] * self.B[index]
            index += 1

        # recursive step (t = 2...N)
        for t in range(1, len(sequence)):
            for i in range(self.hidden_states):
                
                temp = 0 
                for state in range(self.hidden_states):
                    temp += (alpha[t-1][state] * self.A[state][i])

                alpha[t][i] = self.B[i] * temp


            # normalize values
            for k in range(self.hidden_states):
                alpha[t][k] /= sum(alpha[t])

        return alpha



    def backwards(self):
        pass

    def calc_evidence(self):
        pass

    def posterior(self):
        pass

    def transition_probability(self):
        pass

    def init_params(self):
        pass

    def train(self):
        pass

    def classify(self):
        pass

    def get_B(self):
        pass

