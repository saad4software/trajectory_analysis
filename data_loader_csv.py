import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import math
import random
from model import create_model


def clean_data(df):
    file_data = []
    for index, row in df.iterrows():
        x = row['x']
        y = row['y']
        pt1 = [x, y]
        pt2 = file_data[index-1][:2] if index > 1 else [x, y]
        dist = math.dist(pt1, pt2)
#         print(f"distance between {pt1} and {pt2} is {dist}")
        # print(dist)

        file_data += [[x, y, dist]]

    file_data = np.array(file_data)

    Q1_x = np.quantile(file_data[:, 0], 0.25)
    Q3_x = np.quantile(file_data[:, 0], 0.75)
    IQR_x = Q3_x - Q1_x
    Q1_y = np.quantile(file_data[:, 1], 0.25)
    Q3_y = np.quantile(file_data[:, 1], 0.75)
    IQR_y = Q3_y - Q1_y

    data_cleaned_iqr = file_data[
        ~((file_data[:, 0] < (Q1_x - 1.5 * IQR_x)) | (file_data[:, 0] > (Q3_x + 1.5 * IQR_x))) &
        ~((file_data[:, 1] < (Q1_y - 1.5 * IQR_y)) | (file_data[:, 1] > (Q3_y + 1.5 * IQR_y)))
    ]

    avg = data_cleaned_iqr[:, 2].sum()/len(file_data)
    data_cleaned_distance = data_cleaned_iqr[(data_cleaned_iqr[:, 2] > avg)]

    return data_cleaned_distance


def get_df_data(df, step, class_num=1, overlap=1, augment=True):
    df_data = []
    rows_count = len(df.index)
    
    # clean data here
    cleaned_data = clean_data(df)
    
    theta = random.random() * math.pi/6
    transformation_matrix = np.array([
        [1, 0, 0, 1], 
        [1, 0, 0, -1], 
        [-1, 0, 0, 1], 
        # [0, 1, 1, 0], 
        # [0, -1, 1, 0], 
        # [0, 1, -1, 0],
        [math.cos(theta), -math.sin(theta), math.sin(theta), math.cos(theta)],
        ]) if augment else np.array([[1, 0, 0, 1]])


    for tm in transformation_matrix:
        tmp = []
        [a, b, c, d] = tm

                
        for point in cleaned_data:
            x0 = point[0]
            y0 = point[1]

            x = a*(x0-0.5) + b*(y0-0.5) + 0.5
            y = c*(x0-0.5) + d*(y0-0.5) + 0.5

            tmp += [[x, y]]

            if len(tmp) == step:
                df_data += [tmp]
                tmp = tmp[overlap:]            
        


    y_shape = np.array(df_data).shape[0]
    y = [class_num for x in range(y_shape)]
    
    return df_data, y


def get_lstm_data(path, step=30, class_num=1):
    files = glob.glob(path)
    train_data = []
    y_data = []


    for file in files:

        df = pd.read_csv(file)
        df = df.reset_index()
        
        track_data, track_y = get_df_data(df, step, class_num=class_num, overlap=1, augment=True)
        train_data += track_data
        y_data += track_y
        
    
    # print(result)

#   y_shape = np.array(train_data).shape[0]
#   y = [class_num for x in range(y_shape)]
    # y = np.array(y)

    return train_data, y_data


if __name__ == '__main__':
    airplane_path = "/home/saad/Projects/data_prepare/tracks_with_noise_v2/airplane/*.csv"
    bird_path = "/home/saad/Projects/data_prepare/tracks_with_noise_v2/bird/*.csv"
    drone_path = "/home/saad/Projects/data_prepare/tracks_with_noise_v2/drone/*.csv"
    noise_path = "/home/saad/Projects/data_prepare/tracks_with_noise_v2/noise/*.csv"

    airplane_train, y_airplane = get_lstm_data(airplane_path, class_num=0)
    bird_train, y_bird = get_lstm_data(bird_path, class_num=1)
    drone_train, y_drone = get_lstm_data(drone_path, class_num=2)
    # noise_train, y_noise = get_lstm_data(noise_path, class_num=1)

    # print("shapes")
    # print(np.array(airplane_train).shape)
    # print(np.array(bird_train).shape)
    # print(np.array(drone_train).shape)


    train_data =  bird_train + drone_train + airplane_train #+ noise_train
    y = y_bird + y_drone + y_airplane #+ y_noise

    x_train = np.array(train_data)
    y_train = np.array(y)

    # shuffle the data

    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]


    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

    print(x_train.shape)

    model = create_model(3)

    p = model.predict(np.array([x_train[0]]))
    print(p)
    print(p[0]==max(p[0]))
