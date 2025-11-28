import os
import numpy as np

def clean():

    source = 'data'
    dest = 'clean_data'

    try:
        source_list = os.listdir(source)
    except:
        print('Could not find data set')
        exit(1)

    try:
        os.makedirs('clean_data')
    except:
        pass

    # Clean Data points -> store in clean_data folder
    for type in source_list:

        t = os.path.join(source,type)
        samples = os.listdir(t)

        try:
            os.mkdir(os.path.join(dest,type))
        except:
            pass


        for sample in samples:

            f = open(os.path.join(source,type,sample))
            text = f.read()

            target = os.path.join(dest,type,sample)

            with open(target, "w") as destination:

                for line in text.split('#'):
                    if(len(line) > 10):
                        temp = line.split(',')[6].replace('/',',')
                        destination.write('('+temp+')'+'\n')



def augment(s, d):
    # Created augmented data using clean data -> store in augmented_data
    dest = d
    source = s

    try:
        source_list = os.listdir(source)
    except:
        print("Could not find clean data set")
        exit(1)

    try:
        os.makedirs(dest)
    except:
        pass

    for type in source_list:

        t = os.path.join(source,type)
        samples = os.listdir(t)

        try:
            os.mkdir(os.path.join(dest,type))
        except:
            pass

        file_counter = 600

        for sample in samples:

            f = open(os.path.join(source,type,sample), "r")
            text = f.read()

            for i in range(3):

                with open(os.path.join(dest,type,str(file_counter))+'.txt', "w") as destination:
                    file_counter += 1

                    for i in text.split('\n'):

                        if len(i) > 1:
                            i = i.replace("(", "")
                            i = i.replace(")", "")
                            i = i.split(',')

                            point = [
                                int(i[0]) + np.random.normal(0, 1),
                                int(i[1]) + np.random.normal(0, 1),
                                int(i[2]) + np.random.normal(0, 1)
                            ]
                            
                            destination.write('('+str(point[0])+','+str(point[1])+','+str(point[2])+')\n')
                
                            
        file_counter = 600

        


augment('clean_data2', 'aug5')