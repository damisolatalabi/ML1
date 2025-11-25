import numpy as np
import math
import random

class HMM:
    def __init__(self, hidden_states, label):
        self.hidden_states = hidden_states

        self.PI = np.random.dirichlet(alpha=np.ones(self.hidden_states))
        self.PI = np.log(self.PI)

        self.A = np.random.rand(self.hidden_states, self.hidden_states)
        self.A /= self.A.sum(axis=1, keepdims=True)

        self.mu = np.random.rand(self.hidden_states, 3)

        # Identity matrices
        self.var = np.array([np.eye(3) for _ in range(self.hidden_states)])

        self.label = label

    def get_hidden_states(self):
        return self.hidden_states
    
    def get_label(self):
        return self.label

    # calculates the forward pass
    # return alpha matrix
    def forward(self, sequence):
        
        alpha = np.zeros((len(sequence), self.hidden_states), dtype=float)

        log_A = np.log(self.A)

        # initialize (t = 1), first row
        for index in range(self.hidden_states):
            alpha[0][index] = self.PI[index] + self.B(sequence[0], index)

        # recursive step (t = 2...N)
        for t in range(1, len(sequence)):
            for i in range(self.hidden_states):

                prev_logs = np.array([alpha[t-1][state] + log_A[state][i] for state in range(self.hidden_states)])

                alpha[t][i] = self.B(sequence[t], i) + self.logsumexp(prev_logs)


        return alpha

    def backward(self, sequence):
        
        beta = np.zeros((len(sequence), self.hidden_states), dtype=float)

        log_A = np.log(self.A)

        # initialize t = T (last row) --> log space --> all zeros
        for index in range(self.hidden_states):
            beta[-1][index] = 0

        # recursive step
        for t in range(len(sequence)-2, -1, -1): # to go backwards
            for i in range(self.hidden_states):

                terms = np.array([log_A[i][state] + self.B(sequence[t+1], state) + beta[t+1][state] for state in range(self.hidden_states)])

                beta[t][i] = self.logsumexp(terms)

        return beta

    def calc_evidence(self, alpha):
        return np.sum(alpha[-1])

    def posterior(self, state, time, alpha, beta, evidence):
        return (alpha[time][state] + beta[time][state]) - evidence

    def transition_probability(self, state_i, state_j, alpha, beta, evidence, time, sequence):
        log_A = np.log(self.A)
        return (alpha[time][state_i] + log_A[state_i][state_j] + self.B(sequence[time+1],state_j) + beta[time+1][state_j]) - evidence

    def update_parameters(self, alpha, beta, evidence, sequence):

        sequence = np.array(sequence)

        gamma = np.exp(alpha + beta - evidence)

        # Update initial state probabilities
        self.PI = gamma[0] / np.sum(gamma[0])

        # update transition matrix A
        for i in range(self.hidden_states):
            for j in range(self.hidden_states):
                
                transitions = []
                posts = []

                for t in range(len(sequence)-1):

                    xi_ij = self.transition_probability(i,j,alpha,beta,evidence,t,sequence)
                    transitions.append(xi_ij)

                    gamma_i = self.posterior(i, t, alpha, beta, evidence)
                    posts.append(gamma_i)

                log_sum = self.logsumexp(transitions)
                post_sum = self.logsumexp(posts)

                self.A[i][j] = max(np.exp(log_sum - post_sum), 1e-10)

        # update emission probability parameters
        for i in range(self.hidden_states):


            mean_num = np.zeros(3)
            den = 0.0

            # new mean parameters
            for t in range(len(sequence)):
                post = np.exp(self.posterior(i, t, alpha, beta, evidence))

                mean_num += (post*sequence[t])
                den += post

            den = max(den, 1e-10)
            self.mu[i] = mean_num/den

            # new variance parameter

            var_num = np.zeros((3, 3))

            for t in range(len(sequence)):
                post = np.exp(self.posterior(i, t, alpha, beta, evidence))
                diff = sequence[t] - self.mu[i]
                
                var_num += post*np.outer(diff, diff)

            self.var[i] = var_num/den + np.eye(sequence.shape[1]) * 1e-3 #for stability

    def train(self, sequences, max_iterations=20): #stops after max_iterations or when evidence stops improving significantly

        prev_evidence = 0  # Initialize here

        all_data = np.vstack(sequences)
        indices = np.random.choice(all_data.shape[0], self.hidden_states, replace=False)
        self.mu = all_data[indices]

        d = all_data.shape[1]  # dimension of points
        global_var = np.var(all_data, axis=0)  # variance along each dimension

        # use diagonal covariance
        eps = 1e-3  # variance floor
        self.var = np.array([np.diag(global_var + eps) for _ in range(self.hidden_states)])
        
        for iteration in range(max_iterations):
            total_evidence = 0 
            
            for sequence in sequences:

                # E-step: Forward-Backward
                alpha = self.forward(sequence)
                beta = self.backward(sequence)
                evidence = self.logsumexp(alpha[-1])  # Use this instead of calc_evidence
                total_evidence += evidence
                
                # M-step: Update parameters
                self.update_parameters(alpha, beta, evidence, sequence)
            
            #print(f"Iteration {iteration}, Total Evidence: {total_evidence}")

            
            # Check if evidence stopped improving significantly
            if iteration > 0 and abs(total_evidence - prev_evidence) < 0.0001:
                #print("Converged!")
                #print(abs(total_evidence - prev_evidence))
                break
            
            prev_evidence = total_evidence  # Update for next iteration

    def classify(self, sequence):
        
        alpha = self.forward(sequence)
        return self.logsumexp(alpha[-1])

    def B(self, observation, hidden_state):  

        cov = self.var[hidden_state] + np.eye(self.var.shape[1]) * 1e-6 # added for numerical stability

        diff = observation - self.mu[hidden_state]

        m = -0.5 * 3 * np.log(2 * math.pi) - 0.5 * np.log(np.linalg.det(cov))

        solved = np.linalg.solve(cov, diff)

        exponent_value = diff.T.dot(solved)

        e = -0.5 * exponent_value    

        return m+e

    def logsumexp(self, x):
        m = np.max(x)
        return m + np.log(np.sum(np.exp(x-m)))

        

