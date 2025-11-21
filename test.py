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

    
    # test data
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

        test_sequences.append([sequence, type])

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


    return true_label == label

    

HMM_circle, circle_training_set, circle_test_set = get_data('circle', 6)
HMM_diagonal_left, diagonal_left_training_set, diagonal_left_test_set = get_data('diagonal_left', 6)
HMM_diagonal_right, diagonal_right_training_set, diagonal_right_test_set = get_data('diagonal_right', 6)
HMM_horizontal, horizontal_training_set, horizontal_test_set = get_data('horizontal', 6)
HMM_vertical, vertical_training_set, vertical_test_set = get_data('vertical', 6)

model_set = [HMM_circle, HMM_diagonal_left, HMM_diagonal_right, HMM_horizontal, HMM_vertical]
test_set = [circle_test_set+diagonal_left_test_set+diagonal_right_test_set+horizontal_test_set+vertical_test_set]

train([[HMM_circle, circle_training_set], [HMM_diagonal_left, diagonal_left_training_set], [HMM_diagonal_right, diagonal_right_training_set], [HMM_horizontal, horizontal_training_set], [HMM_vertical, vertical_training_set]])


correct = 0
for test_sequences in test_set:
    
    for sequence in test_sequences:
        
        if(test(model_set, np.array(sequence[0]), sequence[1])):
            correct += 1

print("Accuracy : ",correct/len(test_set),"%")

