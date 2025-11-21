import model
import numpy as np
import os

model = model.HMM(3, 'circle')

source = 'augmented_data_gaussian/circle'
source_list = os.listdir(source)

sequences = []


for sample in source_list:

    sequence = []
    
    f = open(os.path.join(source,sample))
    text = f.read()

    text = text.split('\n')

    for point in text:
        if len(point) < 2:
            continue

        point = point.replace('(',"")
        point = point.replace(')',"")

        point = point.split(',')
        point[0] = int(point[0])
        point[1] = int(point[1])
        point[2] = int(point[2])

        sequence.append(point)

    sequences.append(sequence)


for seq in sequences:
    O = np.array(seq)
    model.init_params(O)

    print(model.classify(O))

print(len(sequences))

#sequence = np.array(sequence)
#model.init_params(sequence)







