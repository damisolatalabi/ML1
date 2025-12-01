import numpy as np
import math
import random
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
import pickle

class HMM:
    def __init__(self, hidden_states, label, velocity):
        self.hidden_states = hidden_states

        self.PI = np.random.dirichlet(alpha=np.ones(self.hidden_states))
        self.A = np.random.rand(self.hidden_states, self.hidden_states)

        self.mu = None
        self.var = None

        self.label = label
        self.velocity = velocity

        if self.velocity:
            self.N = 6
        else:
            self.N = 3

    def save(self, filepath):
        params = {
            "hidden_states": self.hidden_states,
            "label": self.label,
            "PI": self.PI,
            "A": self.A,
            "mu": self.mu,
            "var": self.var,
            "velocity": self.velocity
        }

        with open(filepath, "wb") as f:
            pickle.dump(params, f)

    def with_velocity(self):
        return self.velocity

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            params = pickle.load(f)

        hmm = cls(params['hidden_states'], params['label'], params['velocity'])
        hmm.PI = params['PI']
        hmm.A = params['A']
        hmm.mu = params['mu']
        hmm.var = params['var']
        
        return hmm

    def get_hidden_states(self):
        return self.hidden_states
    
    def get_label(self):
        return self.label

    def model_info(self):
        return [self.label, self.hidden_states, self.PI, self.A, self.mu, self.var]

    # calculates the forward pass
    # return alpha matrix
    def forward(self, sequence):
        
        log_alpha = np.zeros((len(sequence), self.hidden_states), dtype=float)

        log_PI = np.log(self.PI+ 1e-10)
        log_A = np.log(self.A+ 1e-10)

        # initialize (t = 1), first row
        for i in range(self.hidden_states):
            log_alpha[0][i] = log_PI[i] + self.B(sequence[0], i)

        # recursive step (t = 2...N)
        for t in range(1, len(sequence)):
            for i in range(self.hidden_states):


                log_alpha[t][i] = (
                    logsumexp(log_alpha[t-1] + log_A[:, i]) + self.B(sequence[t], i)
                )

        return log_alpha
    
    def backward(self, sequence):

        log_A = np.log(self.A)
        
        log_beta = np.zeros((len(sequence), self.hidden_states), dtype=float)

        # initialize t = T (last row)
        for index in range(self.hidden_states):
            log_beta[-1][index] = 0

        # recursive step
        for t in range(len(sequence)-2, -1, -1): # to go backwards
            for i in range(self.hidden_states):

                terms = log_A[i] + np.array([self.B(sequence[t+1], j) for j in range(self.hidden_states)]) + log_beta[t+1]

                log_beta[t][i] = logsumexp(terms)

        return log_beta

    def calc_evidence(self, log_alpha):
        return logsumexp(log_alpha[-1])

    def kmeans_cluster(self, data, K):
        indices = np.random.choice(len(data), K, replace=False)
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
 
    def posterior(self, log_alpha, log_beta, log_evidence):
        log_gamma = log_alpha + log_beta - log_evidence

        log_gamma = log_gamma - logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        return gamma

    def transition_probability(self, log_alpha, log_beta, log_evidence, sequence):

        T = len(sequence)
        log_A = np.log(self.A+1e-10)

        xi_all = np.zeros((T-1, self.hidden_states, self.hidden_states))

        for t in range(T-1):
            log_xi = np.zeros((self.hidden_states, self.hidden_states))
            for i in range(self.hidden_states):
                for j in range(self.hidden_states):
                    log_B_j = self.B(sequence[t+1], j)
                    log_xi[i][j] = log_alpha[t][i] + log_A[i][j] + log_B_j + log_beta[t+1][j] - log_evidence

            max_log_xi = np.max(log_xi)
            sum_exp = 0.0

            for i in range(self.hidden_states):
                for j in range(self.hidden_states):
                    sum_exp += np.exp(log_xi[i][j] - max_log_xi)

            log_xi -= max_log_xi + np.log(sum_exp)

            for i in range(self.hidden_states):
                for j in range(self.hidden_states):
                    xi_all[t][i][j] = np.exp(log_xi[i][j])

        return xi_all

    def train(self, sequences, max_iterations=20): #stops after max_iterations or when evidence stops improving significantly

        if(self.velocity):
            sequences = self.add_velocity(sequences)

        # Use clusters for mean
        all_data = np.vstack(sequences)
        self.mu = self.kmeans_cluster(all_data, self.hidden_states)

        # use cluster labels for variances
        labels = np.argmin(
            np.linalg.norm(all_data[:, None, :] - self.mu[None, :, :], axis=2),
            axis=1
        )

        # add velocity to each sequence
        self.var = []
        for k in range(self.hidden_states):
            cluster = all_data[labels == k]

            if len(cluster) >= 2:
                cov = np.cov(cluster.T) + np.eye(self.N)*1e-3
            else:
                cov = np.eye(self.N) * 0.1

            
            self.var.append(cov)

        self.var = np.array(self.var)

        prev_evidence = -np.inf
        
        for iteration in range(max_iterations):

            total_evidence = 0

            for sequence in sequences:

                log_alpha = self.forward(sequence)
                log_beta = self.backward(sequence)
                log_evidence = self.calc_evidence(log_alpha)
                total_evidence += np.exp(log_evidence)
                
                posterior = self.posterior(log_alpha, log_beta, log_evidence)
                transition_probabilities = self.transition_probability(log_alpha, log_beta, log_evidence, sequence)

                self.update_parameters(posterior, sequence, transition_probabilities)

            # check convergence
            if iteration > 0 and abs(total_evidence - prev_evidence) < 1e-4:
                break
            prev_evidence = total_evidence

    def update_parameters(self, posterior, sequence, trans_prob):

        # update PI
        self.PI = posterior[0] / posterior[0].sum()

        # update A
        for i in range(self.hidden_states):
            for j in range(self.hidden_states):
                
                t_probs = 0
                posts = 0
                for t in range(len(sequence)-1):
                    t_probs += trans_prob[t][i][j]
                    posts += posterior[t][i]

                self.A[i][j] = t_probs/max(posts, 1e-10)

        # normalize rows
        self.A += 1e-10
        row_sums = self.A.sum(axis=1, keepdims=True)
        self.A /= row_sums

    def classify(self, sequence):

        # add velocity
        if self.velocity:

            seq = np.array(sequence)
            velocity = np.diff(seq, axis=0) 
            velocity = np.vstack([velocity[0], velocity]) 
            seq_augmented = np.hstack([seq, velocity])
            alpha = self.forward(seq_augmented)

        else:
            alpha = self.forward(sequence)

        return logsumexp(alpha[-1])

    def B(self, observation, hidden_state):  

        cov = self.var[hidden_state] + np.eye(self.var.shape[1])*1e-6
        log_prob = multivariate_normal.logpdf(observation, mean=self.mu[hidden_state], cov=cov)
        return log_prob



        

