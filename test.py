import model
import numpy as np
import os
import time
from multiprocessing import Process

def get_data(type, hidden_states):
    folder = f'aug5/{type}'
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
    folder = f'test_set/{type}'
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


def train(model, set):
    model.train(set)
    print("Finished training: ", model.get_label())

def test(models, sequence, true_label):
    best = -np.inf
    label = ''
    
    for model in models:
        score = model.classify(sequence)
        if score > best:
            best = score
            label = model.get_label()

    return true_label == label

    

if __name__ == "__main__":
    HMM_circle, circle_training_set, circle_test_set = get_data('circle', 3)
    HMM_diagonal_left, diagonal_left_training_set, diagonal_left_test_set = get_data('diagonal_left', 3)
    HMM_diagonal_right, diagonal_right_training_set, diagonal_right_test_set = get_data('diagonal_right', 3)
    HMM_horizontal, horizontal_training_set, horizontal_test_set = get_data('horizontal', 3)
    HMM_vertical, vertical_training_set, vertical_test_set = get_data('vertical', 3)

    model_set = [HMM_circle, HMM_diagonal_left, HMM_diagonal_right, HMM_horizontal, HMM_vertical]
    test_set = circle_test_set+diagonal_left_test_set+diagonal_right_test_set+horizontal_test_set+vertical_test_set



    processes = [
        Process(target=train, args=(HMM_circle, circle_training_set)),
        Process(target=train, args=(HMM_diagonal_left, diagonal_left_training_set)),
        Process(target=train, args=(HMM_diagonal_right, diagonal_right_training_set)),
        Process(target=train, args=(HMM_horizontal, horizontal_training_set)),
        Process(target=train, args=(HMM_vertical, vertical_training_set)),
    ]


    start = time.time()

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    end = time.time()


    correct = 0
    for sequence in test_set:
        
        if test(model_set, np.array(sequence[0]), sequence[1]):
            correct += 1

    accuracy = correct/len(test_set)*100
    print("Accuracy : ",accuracy,"%")
    print("Training time: ",(end-start),"s")

