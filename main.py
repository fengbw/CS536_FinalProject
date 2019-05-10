from data import *
from models import *

if __name__ == "__main__":
    my_data = data_operations()
    my_data.read_csv()
    my_data.preprocess_data()
    # print(my_data.label_encoder['0'])
