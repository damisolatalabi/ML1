import numpy as np
import math

class HMM:
    def __init__(self, hidden_states, label):
        self.hidden_states = hidden_states
        self.PI = np.random.rand(self.hidden_states)
        self.A = np.random.rand(self.hidden_states, self.hidden_states)
        self.mu = np.random.rand(self.hidden_states, 3)
        self.var = np.random.rand(self.hidden_states, 3, 3)
        self.label = label

    def get_hidden_states(self):
        return self.hidden_states
    
    def get_label(self):
        return self.label

    # calculates the forward pass
    # return alpha matrix
    def forward(self, sequence):
        
        alpha = np.zeros((len(sequence), self.hidden_states), dtype=float)

        # initialize (t = 1), first row
        for index in range(self.hidden_states):
            alpha[0][index] = self.PI[index] * self.B(sequence[0])


        # recursive step (t = 2...N)
        for t in range(1, len(sequence)):
            for i in range(self.hidden_states):
                
                temp = 0 
                for state in range(self.hidden_states):
                    temp += (alpha[t-1][state] * self.A[state][i])

                alpha[t][i] = self.B(sequence[t]) * temp


            # normalize values
            for k in range(self.hidden_states):
                alpha[t][k] /= sum(alpha[t])

        return alpha

    def backward(self, sequence):
        
        beta = np.zeros((len(sequence), self.hidden_states), dtype=float)

        # initialize t = T (last row)
        for index in range(self.hidden_states):
            beta[len(sequence)-1][index] = 1

        # recursive step
        for t in range(len(sequence)-2, 1):
            for i in range(self.hidden_states):

                temp = 0
                for state in range(self.hidden_states):
                    temp += self.A[i][state] * self.B(sequence[t+1], self.mu[state], self.var[state]) * beta[t+1][state]

                beta[t][i] = temp

            # normalization
            for k in range(self.hidden_states):
                beta[t][k] /= sum(beta[t+1])

        return beta

    def calc_evidence(self, alpha):
        return sum(alpha[alpha.length-1])

    def posterior(self, state, time, alpha, beta, evidence):
        return (alpha[time][state] * beta[time][state]) / evidence

    def transition_probability(self, state_i, state_j, alpha, beta, evidence, time, sequence):
        return (alpha[time][state_i] * self.A[state_i][state_j] * self.B(sequence[time+1], self.mu[state_j], self.var[state_j]) * beta[time+1][state_j]) / evidence

    def init_params(self):
        pass

    def update_parameters(self, alpha, beta, evidence, sequence):

        # Update initial state probabilities
        for i in range(self.hidden_states):
            self.PI[i] = self.posterior(i, 1, alpha, beta, evidence)

        # update transition matrix A
        for i in range(self.hidden_states):
            for j in range(self.hidden_states):

                transitions = 0
                posts = 0
                for t in range(len(sequence)-2):
                    transitions += self.transition_probability(i, j, alpha, beta, evidence, t, sequence)
                    posts += self.posterior(i, t, alpha, beta, evidence)

                self.A[i][j] = transitions/posts

        # update emission probability parameters
        for i in range(self.hidden_states):

            mean_num = den = var_num = var_den = 0

            # new mean parameter
            for t in range(1, len(sequence)-1):
                post = self.posterior(i, t, alpha, beta, evidence)
                
                mean_num += (post*sequence[t])
                den += post

            self.mu[i] = mean_num/den

            # new variance parameter
            for t in range(1, len(sequence)-1):
                post = self.posterior(i, t, alpha, beta, evidence)
                
                var_num += post*(sequence[t]-self.mu)*(sequence[t]-self.mu)

            self.var[i] = var_num/den

    def train(self):
        pass

    def classify(self):
        pass

    def B(self, observation):

        v = np.array([observation-self.mu])

        denominator = math.pow(2*math.pi,3/2) * np.linalg.det(self.var)

        power = -0.5*v.dot(np.linalg.inv(self.var))
        power = power.dot(np.transpose(v))

        power = power[0][0]

        return math.exp(power)/denominator


    

