import numpy as np
from data_loader_csv import get_lstm_data
from model import create_model
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    db_path = "/home/saad/Projects/data_prepare/tracks_with_noise_v2"
    airplane_path = f"{db_path}/airplane/*.csv"
    bird_path = f"{db_path}/bird/*.csv"
    drone_path = f"{db_path}/drone/*.csv"
    noise_path = f"{db_path}/noise/*.csv"

    airplane_train, y_airplane = get_lstm_data(airplane_path, class_num=0)
    bird_train, y_bird = get_lstm_data(bird_path, class_num=1)
    drone_train, y_drone = get_lstm_data(drone_path, class_num=2)

    train_data =  bird_train + drone_train + airplane_train #+ noise_train
    y = y_bird + y_drone + y_airplane #+ y_noise

    x_train = np.array(train_data)
    y_train = np.array(y)

    # shuffle the data

    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]


    # test split 20% of the data
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

    model = create_model(number_of_classes=3)


    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        batch_size=1024,
        epochs=300,
    )

    model.evaluate(x_test, y_test, verbose=1)

    model.save("target_classes_model.keras")