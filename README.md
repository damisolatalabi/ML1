## Description
Hidden markov Models are used to model and classify sequences representing movements.
5 HMM's are trained, each modeling one class.
To classify, the sequence goes through all 5 HMM's, the highest probabilty dictates the class.


### The Model

#### Sample Size
After all data segmentation is done, we have approximately 4000 sequences for 5 classes where each sequence contains 200-300 points

- Cross-validation will be required to find best minimal data size for training.


#### $Hyperparameters$ :

- Number of hidden states --> requires cross-validation for optimal decision (can start at 3 and work up)

- $\Lambda=(\Pi,A,B)$
- $\Pi$ : initial state probabilities (probability of first state being $S_1=i$)
- $A$ : Transition probability matrix (probability of seeing transition $i\rightarrow j$) 
- $B$ : Observation probability matrix (probability of seeing observation $O_t$ at state $i$)


#### Variables and Data types


| Variables            |                     |
| -------------------- | ------------------- |
| Sequence             | $O=\{O_1,...,O_T\}$ |
| Sequence length      | $T$                 |
| Number Hidden states | $N$                 |


| Data types |                                                                                                                                                                                                                  |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $\alpha$   | $T\times N$ matrix : stores forward probabilities for each state $i$ at each time step $t$.                                                                                                                      |
|            | $\alpha[t][i]$ : Probability of observing partial sequence $\{O_1,..,O_t\}$ until time $t$ and being state $i$ at time $t$ given $\Lambda=(A,B,\Pi)$                                                             |
|            |                                                                                                                                                                                                                  |
| $\beta$    | $T\times N$ matrix : stores backward probability for each state $i$ at each time step $t$                                                                                                                        |
|            | $\beta[t][i]$ : Probability of observing observing remaining partial sequence $\{O_{t+1},...,O_T\}$  given state $i$ at time $t$                                                                                 |
|            |                                                                                                                                                                                                                  |
| $O$        | $T\times 3$ matrix : represents the sequence made up of 3D points.                                                                                                                                               |
|            |                                                                                                                                                                                                                  |
| $A$        | $N\times N$ matrix : contains transition probabilities.                                                                                                                                                          |
|            | $A[i][j]$ : probability of being state $j$ at time $t+1$, given state $i$ at time $t$                                                                                                                            |
|            |                                                                                                                                                                                                                  |
| $\Pi$      | $N\times 1$ vector : initial state probabilities.                                                                                                                                                                |
|            | $\Pi[i]$ : probability of starting sequence in state $i$                                                                                                                                                         |
|            |                                                                                                                                                                                                                  |
| $B$        | $Function$ : models the probability of seeing an observation in state $i$ and time $t$ (returns scalar value)                                                                                                    |
|            | $Function\ Arguments$ : <br> $O_t$ = point in sequence at time $t$ <br> $Mean_{state}$ = average of a state $i$ (across all time steps)<br> $Variance_{state}$ = variance of a state $i$ (across all time steps) |
|            | $Mean_{state}$ : $N\times 3$ matrix : <br> where $Mean_{state}[i]$ is the average value for [x,y,z] of state $i$ across all time steps                                                                           |
|            | $Variance_{state}$ : $N\times 3\times 3$ matrix : <br> where $Variance_{state}[i]$ is the variance between [x,y,z] of state $i$ across all time steps                                                            |
|            | $Calculation\ function\ B$ : <br>(Given $O[t],\ \hat{\mu}_i = Mean_{state}[i],\ \hat{\Sigma}_i = Variance_{state}[i]$)                                                                                           |
|            | $B=\frac{1}{(2\pi)^{\frac{d}{2}}det(\hat{\Sigma}_i)}e^{-\frac{1}{2}(O[t]-\hat{\mu}_i)^T\hat{\Sigma}^{-1}_i(O[t]-\hat{\mu}_i)}$                                                                                   |


### Training


##### Forward-Backward method:

-  **Compute** $\alpha _t(i)$ and $\beta_t(i)$ 

- $\alpha_t(i)$ 
	1) Initiation ($t=1$) : 
		- For each state $i=1\ to\ N$ : 
			 $\alpha_1(i)=\pi_i\cdot b_i(O_1)$ 

	2) Recursion (t=2..N) :
		- For each time step $t=2\ to\ T$ and state $i=1\ to\ N$ :
			 $\alpha_t(i)=[\sum_{j=1}^N\alpha_{t-1}(j)\cdot A_{ji}]\cdot b_i(O_t)$ 
			 OR
			 $\alpha_t(j)=[\sum_{i=1}^N\alpha_{t}(i)\cdot A_{ij}]\cdot b_j(O_{t+1})$  

- $\beta_t(i)$
	1) Initiation ($t=T$) :
		- For each state $i=1\ to\ N$ :
			- $\beta_{T}(i) =1$

	2) Recursion ($t=(T-1)\ to\ 1$) :
		- For each time step $t=(T-1)\ to\ 1$ and state $i=1\ to\ N$ :
			- $\beta_t(i)=\sum_{j=1}^NA[i][j]*B(O[t], \hat{\mu_i}, \hat{\Sigma_i})*\beta_{t+1}(i)$ 



- **Normalize** both by (to prevent underflow): 
	- $c_t=\frac{1}{\sum_{i=1}^N\alpha _t(i)}$ for $\alpha _t(i)$  --> $\hat{\alpha}_t(i)=c_t\cdot \alpha_t(i)$  
	- $c_{t+1}=\frac{1}{\sum_{i=1}^N\alpha _{t+1}(i)}$ for $\beta _t(i)$  --> $\hat{\beta}_t(i)=c_{t+1}\cdot \beta_t(i)$  


- **Calculate** $P(O)$ : Probability of entire observation sequence $O$ 
	- $P(O)=\sum_{i=1}^N\alpha_T(i)$ or $=\sum_{i=1}^N\alpha_t(i)\cdot\beta_t(i)$ (for any time t) 



- **Calculate** Posterior: probability that state was $i$ at time $t$

	- $\gamma_t(i)=\frac{\alpha _t(i)\cdot \beta_t(i)}{P(O)}$   


- **Calculate** probability of transition $i\rightarrow j$ at time $t$ 

	- $\xi_t(i,j)=\frac{\alpha_t(i)A_{ij}B_j(O_{t+1})\beta_{t+1}(j)}{P(O)}$ 

		- Where:
			- $A_{ij}$ : probability of transitioning $i\rightarrow j$ 
			- $B_j(O_{t+1})$ : probability of observing $O_{t+1}$ given state $j$ 


--> Potential improvements: log for multiplications

##### Updating Parameters : 

- **Initial state probabilities** : $\Pi$ 
	- $\hat{\pi_i}=\gamma_1(i)$ 

- **Transition Matrix** : $A$ 
	- $\hat{A_{ij}}=\frac{\sum_{t=1}^{T-1}\xi_t(i,j)}{\sum_{t=1}^{T-1}\gamma_t(i)}$ 
		- Update how likely a specific transition is

- **Emission Probabilities**    
	- For each state $i$ :

		- Mean vector $\mu_i$
		- Covariance matrix $\Sigma_i$ 

		- Update by using average from the data

			- $\hat{\mu_i}=\frac{\sum_{t=1}^T\gamma_t(i)O_t}{\sum_{t=1}^T\gamma_t(i)}$

			 - $\hat{\Sigma_i}=\frac{\sum_{t=1}^T\gamma_t(i)(O_t-\hat{\mu_i})(O_t-\hat{\mu_i})^T}{\sum_{t=1}^T\gamma_t(i)}$   




--> **Repeat till convergence/max iterations** 


#### Hyperparameter Initialization

--> currently set to random: more advanced techniques to be explored (clustering etc...)

#### Hidden states

--> 5-10 states : represent a specific part of movement (ex. circle: 4 states --> 4 quarters) 

#### Testing - Classification

--> The classifier only performs computation of $\alpha _t(i)$ 

- $\sum_{i=1}^N\alpha_T(i)$ (sum last time step row)

- After output from all 5 HMM classifiers --> Highest value --> classify to this class.




### Uncertainties/Doubts

- This setup uses statistical information only and does not 'learn' the weights like a neural network with SGD would do.

- Initialization of hyperparameters is tricky and needs thorough testing.

- Might need to reduce computational complexity <--> accuracy trade-off 

- Fixed sequence size?



### The program cycle

--> Create HMM object (#hidden states, label)

--> **Train**(sequence)

- Forward-backward
- Update parameters
- repeat

--> **Classify**(sequence)

- Forward: return likelihood
