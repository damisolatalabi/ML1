import numpy as np
import math
import random
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

class HMM:
    def __init__(self, hidden_states, label):
        self.hidden_states = hidden_states

        self.PI = np.random.dirichlet(alpha=np.ones(self.hidden_states))
        self.A = np.random.rand(self.hidden_states, self.hidden_states)

        self.mu = np.zeros((self.hidden_states, 6))
        self.var = np.array([np.eye(6) for _ in range(self.hidden_states)])

        self.label = label

    def get_hidden_states(self):
        return self.hidden_states
    
    def get_label(self):
        return self.label

    def model_info(self):
        return [self.label, self.hidden_states, self.PI, self.A, self.mu, self.var]

    # calculates the forward pass
    # return alpha matrix
    def forward(self, sequence):
        
        alpha = np.zeros((len(sequence), self.hidden_states), dtype=float)

        # initialize (t = 1), first row
        for index in range(self.hidden_states):
            alpha[0][index] = self.PI[index] + self.B(sequence[0], index)

        # recursive step (t = 2...N)
        for t in range(1, len(sequence)):
            for i in range(self.hidden_states):

                prev_logs = 0

                for state in range(self.hidden_states):
                    prev_logs += alpha[t-1][state] * self.A[state][i]

                alpha[t][i] = self.B(sequence[t], i) + prev_logs

            alpha[t] /= np.sum(alpha[t])

        return alpha

        
    def backward(self, sequence, alpha):
        
        beta = np.zeros((len(sequence), self.hidden_states), dtype=float)

        # initialize t = T (last row)
        for index in range(self.hidden_states):
            beta[-1][index] = 1

        # recursive step
        for t in range(len(sequence)-2, -1, -1): # to go backwards
            for i in range(self.hidden_states):
                terms = 0
                for state in range(self.hidden_states):
                    terms += self.A[i][state] * self.B(sequence[t+1], state) * beta[t+1][state]

                beta[t][i] = terms
                
            beta[t] /= np.sum(alpha[t+1])

        return beta

    def calc_evidence(self, alpha):
        return np.sum(alpha[-1])

    def kmeans_cluster(self, data, K):
        indices = np.random.choice(len(data[0]), K, replace=False)
        centers = data[indices].copy()

        for _ in range(50):
            # assign points to closest center
            distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)

            # compute new centers
            new_centers = np.zeros_like(centers)
            for k in range(K):
                cluster = data[labels == k]

                if len(cluster) > 0:
                    new_centers[k] = cluster.mean(axis=0)
                else:
                    # random empty cluster
                    new_centers[k] = data[np.random.randint(len(data))]

            # check if converged
            if np.allclose(centers, new_centers):
                break

            centers = new_centers

        return centers

    def add_velocity(self, sequences):

        seq_v = []

        for seq in sequences:
            seq = np.array(seq)
            T = seq.shape[0]

            velocity = np.zeros_like(seq)
            velocity[1:] = seq[1:] - seq[:-1]

            seq_aug = np.hstack([seq, velocity])
            seq_v.append(seq_aug)

        return seq_v
 

    def posterior(self, state, time, alpha, beta, evidence):
        return (alpha[time][state] * beta[time][state]) / evidence

    def transition_probability(self, state_i, state_j, alpha, beta, evidence, time, sequence):
        return (alpha[time][state_i] * log_A[state_i][state_j] * self.B(sequence[time+1],state_j) * beta[time+1][state_j]) / evidence

    def train(self, sequences, max_iterations=20): #stops after max_iterations or when evidence stops improving significantly

        sequences = self.add_velocity(sequences)

        # Use clusters for mean
        all_data = np.vstack(sequences)
        self.mu = self.kmeans_cluster(all_data, self.hidden_states)

        # use cluster labels for variances
        labels = np.argmin(
            np.linalg.norm(all_data[:, None, :] - self.mu[None, :, :], axis=2),
            axis=1
        )

        self.var = []
        for k in range(self.hidden_states):
            cluster = all_data[labels == k]

            if len(cluster) > 5:
                cov = np.cov(cluster.T)
            else:
                cov = np.eye(6) * 0.1

            cov += np.eye(6) * 1e-3
            self.var.append(cov)

        self.var = np.array(self.var)

        prev_evidence = -np.inf
        
        
        for iteration in range(max_iterations):

            seq_evidence = self.update_parameters_all_sequences(sequences)

            # check convergence
            if iteration > 0 and abs(seq_evidence - prev_evidence) < 1e-4:
                break
            prev_evidence = seq_evidence

    def update_parameters_all_sequences(self, sequences):

        # accumulators
        mean_num_total = np.zeros((self.hidden_states, 6))
        var_num_total = np.zeros((self.hidden_states, 6, 6))
        den_total = np.zeros(self.hidden_states)
        PI_accum = np.zeros(self.hidden_states)
        A_accum = np.zeros((self.hidden_states, self.hidden_states))
        
        seq_evidence = 0
        epsilon = 1e-10

        for sequence in sequences:
            sequence = np.array(sequence)
            alpha = self.forward(sequence)
            beta = self.backward(sequence, alpha)
            evidence = logsumexp(alpha[-1])
            seq_evidence += evidence

            # posterior
            log_gamma = alpha + beta - logsumexp(alpha[-1])
            log_gamma -= np.max(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma - logsumexp(log_gamma, axis=1, keepdims=True))

            # accumulate PI and mean/variance
            PI_accum += gamma[0]
            
            # accumulate mu, var, and expected transitions
            for t in range(len(sequence)):
                for i in range(self.hidden_states):
                    mean_num_total[i] += gamma[t,i] * sequence[t]
                    den_total[i] += gamma[t,i]
    
                if t < len(sequence)-1:
                    for i in range(self.hidden_states):
                        for j in range(self.hidden_states):
                            # xi_ij in log-space
                            log_xi = alpha[t,i] + np.log(self.A[i,j]+epsilon) + self.B(sequence[t+1], j) + beta[t+1,j] - evidence
                            xi_ij = np.exp(log_xi)
                            A_accum[i,j] += xi_ij
    
            # accumulate variance
            for i in range(self.hidden_states):
                for t in range(len(sequence)):
                    diff = sequence[t] - mean_num_total[i]/max(den_total[i], epsilon)
                    var_num_total[i] += gamma[t,i] * np.outer(diff, diff)
    
        # update parameters
        den_total_safe = np.maximum(den_total, epsilon)
        self.mu = mean_num_total / den_total_safe[:, None]
        self.var = var_num_total / den_total_safe[:, None, None] + np.eye(6)*1e-3
        self.PI = PI_accum / PI_accum.sum()
        
        row_sums = A_accum.sum(axis=1, keepdims=True) + epsilon
        self.A = A_accum / row_sums
    
        return seq_evidence
            

    def classify(self, sequence):

        # add velocity
        seq = np.array(sequence)
        velocity = np.diff(seq, axis=0) 
        velocity = np.vstack([velocity[0], velocity]) 
        seq_augmented = np.hstack([seq, velocity])
        
        alpha = self.forward(seq_augmented)
        return logsumexp(alpha[-1])

    def B(self, observation, hidden_state):  

        cov = self.var[hidden_state] + np.eye(self.var.shape[1])*1e-6
        log_prob = multivariate_normal.logpdf(observation, mean=self.mu[hidden_state], cov=cov)
        return log_prob



        

