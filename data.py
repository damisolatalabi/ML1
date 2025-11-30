import os
import numpy as np
import random
import shutil

def clean(source):

    source = source
    dest = f'{source}_processed'

    try:
        source_list = os.listdir(source)
    except:
        print('Could not find data set')
        exit(1)

    try:
        os.makedirs(f'{source}_processed')
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

def augment(src, clean, counter):
    # Created augmented data using clean data -> store in augmented_data
    dest = f'{src}_augmented'
    source = src

    try:
        source_list = os.listdir(source)
    except:
        print("Could not find clean data set")
        exit(1)

    try:
        os.makedirs(f'{src}_augmented')
    except:
        pass

    for type in source_list:

        t = os.path.join(source,type)
        samples = os.listdir(t)

        try:
            os.mkdir(os.path.join(f'{src}_augmented',type))
        except:
            pass

        file_counter = counter
        
        for sample in samples:
            
            f = open(os.path.join(source,type,sample), "r")
            text = f.read()

            if clean:
                r = 2
            else:
                r = 1

            for i in range(r):

                with open(os.path.join(f'{src}_augmented',type,str(file_counter))+'.txt', "w") as destination:
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
                
                            
        file_counter = counter

def create_sets(train_folders, validation_folders, ratio=0.8, seed=42):

    random.seed(seed)

    classes = ['circle', 'diagonal_left', 'diagonal_right', 'horizontal', 'vertical']

    os.makedirs("training_set", exist_ok=True)
    os.makedirs("validation_set", exist_ok=True)

    for cls in classes:

        # training set
        train_files = []
        for folder in train_folders:
            class_folder = os.path.join(folder, cls)
            if not os.path.exists(class_folder):
                continue

            files = [os.path.join(class_folder, f) for f in os.listdir(class_folder) if os.path.isfile(os.path.join(class_folder, f))]
            train_files.extend(files)
            
        random.shuffle(train_files)
        split_idx = int(len(train_files) * ratio)
        train_split = train_files[:split_idx]


        # Validation set
        val_files = []
        for folder in validation_folders:
            class_folder = os.path.join(folder, cls)
            if not os.path.exists(class_folder):
                continue
            files = [os.path.join(class_folder, f) for f in os.listdir(class_folder) if os.path.isfile(os.path.join(class_folder, f))]
            val_files.extend(files)


        os.makedirs(os.path.join("training_set", cls), exist_ok=True)
        os.makedirs(os.path.join("validation_set", cls), exist_ok=True)

        for f in train_split:
            shutil.copy(f, os.path.join("training_set", cls, os.path.basename(f)))
        for f in val_files:
            shutil.copy(f, os.path.join("validation_set", cls, os.path.basename(f)))



    # 80-20 separation

    # Create training set
    # use clean_set_augmented + noisy_set + noisy_set_augmented

    # create validation set -> only non-augmented samples!
    # use clean_set + noisy_set

    for cls in classes:

        all_files = []


folders_train = [
    "clean_dataset_processed_augmented",  # 2
    "noisy_dataset_processed",    # 3
    "noisy_dataset_processed_augmented"   # 4
]

# Validation: 1, 3
folders_val = [
    "clean_dataset_processed",    # 1
    "noisy_dataset_processed"     # 3
]

clean_input = 'clean_dataset'
noisy_input = 'noisy_dataset'

clean(clean_input)
clean(noisy_input)

augment(f'{clean_input}_processed', True, 100)
augment(f'{noisy_input}_processed', False, 500)

create_sets(folders_train, folders_val)