from data import *
from models import *
import numpy as np

if __name__ == "__main__":
    my_data = data_operations()
    my_data.read_csv()
    my_data.preprocess_data()
    # print(my_data.label_encoder['0'])
    my_data.remove_open_response()
    for i in range(len(my_data.predict_indexes)):
        index = my_data.predict_indexes[i]
        print("NOW: " + str(index) + "-------------")
        x_train, y_train, x_test, y_test = my_data.split_train_test(index)
        if len(x_train) == 0:
            my_data.combine_train_test(index, x_train, y_train, x_test, y_test)
            continue
        clf = LinearRegression()
        w, b = clf.lassoRidge(x_train, y_train, 0.1)
        y_prediction = clf.predict_all(x_test)
        my_data.combine_train_test(index, x_train, y_train, x_test, y_prediction)
    my_data.reconvert_string()
    my_data.rewrite_file()

    # my_data.generate_realistic()
    # my_data.compare_real()
