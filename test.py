import model
import numpy as np
import os

def get_data(type, hidden_states):
    folder = f'augmented_data_gaussian/{type}'
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

    
    folder = f'clean_data/{type}'
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

        test_sequences.append(sequence)

    return model.HMM(hidden_states, type), train_sequences, test_sequences


def train(models):
    for model in models:
        model[0].train(model[1])
        print("Finished training: ", model[0].get_label())

def test(models, sequence, true_label):
    max = 0
    label = ''
    for i in range(len(models)):
        
        res = models[i].classify(sequence)

        if i == 0:
            max = res
            label = models[0].get_label()
        else:
            if res > max:
                max = res
                label = models[i].get_label()

    print("True : ", true_label, " | Predicted : ", label)

    

HMM_circle, circle_training_set, circle_test_set = get_data('circle', 3)
HMM_diagonal_left, diagonal_left_training_set, diagonal_left_test_set = get_data('diagonal_left', 3)
HMM_diagonal_right, diagonal_right_training_set, diagonal_right_test_set = get_data('diagonal_right', 3)
HMM_horizontal, horizontal_training_set, horizontal_test_set = get_data('horizontal', 3)
HMM_vertical, vertical_training_set, vertical_test_set = get_data('vertical', 3)

model_set = [HMM_circle, HMM_diagonal_left, HMM_diagonal_right, HMM_horizontal, HMM_vertical]

train([[HMM_circle, circle_training_set], [HMM_diagonal_left, diagonal_left_training_set], [HMM_diagonal_right, diagonal_right_training_set], [HMM_horizontal, horizontal_training_set], [HMM_vertical, vertical_training_set]])

test(model_set, np.array(circle_test_set[0]), 'circle')
test(model_set, np.array(diagonal_left_test_set[0]), 'diagonal left')
test(model_set, np.array(diagonal_right_test_set[0]), 'diagonal right')
test(model_set, np.array(horizontal_test_set[0]), 'horizontal')
test(model_set, np.array(vertical_test_set[0]), 'vertical right')