import os
import pandas as pd

def create_labels_csv(folder_path, output_csv):
    filenames = os.listdir(folder_path)
    data = []

    for file in filenames:
        if "autistic" in file.lower():
            label = 1  # autistic
        elif "non_autistis" in file.lower():
            label = 0  # non-autistic
        else:
            continue  # skip if filename is unclear

        data.append([file, label])

    df = pd.DataFrame(data, columns=['filename', 'label'])
    df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")

# Generate for train
create_labels_csv('data/Train', 'data/train_labels.csv')

# Generate for test
create_labels_csv('data/test', 'data/test_labels.csv')
