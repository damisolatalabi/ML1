import model
import numpy as np

model = model.HMM(3, 'test')

sequence = [
    [10,15,-5],
    [19,16,-4],
    [20,17,-3],
    [30,18,-2],
    [40,19,-1],
    [50,20, 0],
    [61,22, 1],
    [83,29, 2]
]

sequence = np.array(sequence)


model.mu = np.mean(sequence, axis=0)
model.var = np.cov(sequence, rowvar=False)



model.forward(sequence)



