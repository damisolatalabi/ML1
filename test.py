import os
import numpy as np
import model
import random


def prepare_test_set(type, source):
    # test data
    folder = f'{source}/{type}'
    samples = os.listdir(folder)

    test_sequences = []

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

        test_sequences.append([sequence, type])
        
    return test_sequences

def model_info(model):
    info = model.model_info()
    print(f"Model : {info[0]}")
    print(f"Hidden states : {info[1]}")
    print(f"PI : {info[2]}")
    print(f"A : {info[3]}")
    print(f"Mean : {info[4]}")
    print(f"Variance : {info[5]}")

def test(models, sequence, true_label):
    best = -np.inf
    label = ''

    print(10*"*")
    for model in models:
        score = model.classify(sequence)
        print(f"Model : {model.get_label()} | score = {score} | for class : {true_label}")
        if score > best:
            best = score
            label = model.get_label()
    print(10*"*")

    return true_label == label

hidden_states = 3
source_test = 'validation_set'

model_set = [
    model.HMM.load(f"model_parameters{hidden_states}/circle.pkl"),
    model.HMM.load(f"model_parameters{hidden_states}/diagonal_left.pkl"),
    model.HMM.load(f"model_parameters{hidden_states}/diagonal_right.pkl"),
    model.HMM.load(f"model_parameters{hidden_states}/horizontal.pkl"),
    model.HMM.load(f"model_parameters{hidden_states}/vertical.pkl")
]

test_sets = [
    prepare_test_set('circle', source_test),
    prepare_test_set('diagonal_left', source_test),
    prepare_test_set('diagonal_right', source_test),
    prepare_test_set('horizontal', source_test),
    prepare_test_set('vertical', source_test)
]

test_set = [item for sublist in test_sets for item in sublist] 
random.shuffle(test_set)


# test models
correct = 0
for sequence, true_label in test_set:
    if test(model_set, sequence, true_label):
        correct += 1

accuracy = correct / len(test_set) * 100
print(f"\nAccuracy: {accuracy:.2f}%")