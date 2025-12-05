import model
import numpy as np
import os
import time
from multiprocessing import Process
import random
import shutil

def prepare_training_set(type, source):
    folder = f'{source}/{type}'
    samples = os.listdir(folder)

    train_sequences = []

    for sample in samples:

        sequence = []
    
        f = open(os.path.join(folder,sample))
        text = f.read()

        text = text.split('\n')

        for point in text:
            if len(point) < 2:
                continue

            point = point.replace('(',"")
            point = point.replace(')',"")

            point = point.split(',')
            point[0] = float(point[0])
            point[1] = float(point[1])
            point[2] = float(point[2])

            sequence.append(point)

        train_sequences.append(sequence)

    return train_sequences

def model_info(model):
    info = model.model_info()
    print(f"Model : {info[0]}")
    print(f"Hidden states : {info[1]}")
    print(f"PI : {info[2]}")
    print(f"A : {info[3]}")
    print(f"Mean : {info[4]}")
    print(f"Variance : {info[5]}")

def train(model, set):
    model.train(set) 

# Initialize models + number of hidden states
hidden_states = 4
source_training = 'training_set'

model_set = [
    model.HMM(hidden_states, 'circle', False),
    model.HMM(hidden_states, 'diagonal_left', False),
    model.HMM(hidden_states, 'diagonal_right', False),
    model.HMM(hidden_states, 'horizontal', False),
    model.HMM(hidden_states, 'vertical', False)
]

training_sets = [
    prepare_training_set('circle', source_training),
    prepare_training_set('diagonal_left', source_training),
    prepare_training_set('diagonal_right', source_training),
    prepare_training_set('horizontal', source_training),
    prepare_training_set('vertical', source_training)
]

# train models
start = time.time()

for model_obj, train_set in zip(model_set, training_sets):

    print(f"Training {model_obj.get_label()}")
    model_obj.train(train_set)

end = time.time()

print(f"Training finished in {end - start:.2f}s\n")

if os.path.exists(f'model_parameters{hidden_states}'):
    shutil.rmtree(f'model_parameters{hidden_states}')

os.makedirs(f'model_parameters{hidden_states}', exist_ok=True)

# save model parameters
for model in model_set:
    model.save(f'model_parameters{hidden_states}/{model.get_label()}.pkl')


