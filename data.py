import csv
import pandas as pd
import math
import openpyxl
import copy
import numpy as np
import random

class data_operations:
    def __init__(self):
        self.header = []
        self.content = []
        self.open_response_content = []
        self.col = 0
        self.na_threshold = 0
        self.digital_indexes = []
        self.string_indexes = []
        self.open_response = []
        self.open_response_indexes = []
        self.label_encoder = {}
        self.predict_indexes = []
        self.find_open_source()

    def read_csv(self):
        with open("ML3/ML3ALLSites.csv", encoding = "utf8", errors = "ignore") as f:
            csv_file = csv.reader(f)
            print(csv_file)
            index = 0
            for line in csv_file:
                if index == 0:
                    self.header = line
                    index += 1
                    continue
                self.content.append(line)
            self.col = len(self.content[0])
            self.na_threshold = math.ceil(self.col * 2 / 3)

    def find_open_source(self):
        wb = openpyxl.load_workbook("ML3/ML3 Variable Codebook.xlsx")
        sheet = wb.get_sheet_by_name("Sheet1")
        for i in range(3, 109):
            if sheet["E" + str(i)].value == "open response":
                self.open_response.append(sheet["A" + str(i)].value)

    def preprocess_data(self):
        #del data if there are too many NA
        del_indexes = []
        for i in range(len(self.content)):
            na_counter = 0
            for j in range(self.col):
                if self.content[i][j] == "NA":
                    na_counter += 1
            if na_counter > self.na_threshold:
                del_indexes.append(i)
        for i in range(len(del_indexes)):
            del self.content[del_indexes[i]]
            for j in range(len(del_indexes)):
                del_indexes[j] -= 1
        #open response index
        for i in range(self.col):
            if self.header[i] in self.open_response:
                self.open_response_indexes.append(i)
        #transform int and float data
        transform_vol_counter = [0] * self.col
        for i in range(len(self.content)):
            for j in range(self.col):
                if j in self.open_response_indexes:
                    continue
                if self.content[i][j] == "NA":
                    continue
                try:
                    self.content[i][j] = int(self.content[i][j])
                    transform_vol_counter[j] += 1
                except ValueError:
                    try:
                        self.content[i][j] = float(self.content[i][j])
                        transform_vol_counter[j] += 1
                    except ValueError:
                        pass
        # print(transform_vol_counter)
        for i in range(self.col):
            if i in self.open_response_indexes:
                continue
            if transform_vol_counter[i] > 1000:
                self.digital_indexes.append(i)
            else:
                self.string_indexes.append(i)
        for i in range(len(self.content)):
            for j in range(self.col):
                if j in self.open_response_indexes:
                    continue
                if j in self.digital_indexes:
                    if not isinstance(self.content[i][j], (int, float)):
                        self.content[i][j] = -1
                elif j in self.string_indexes:
                    if not isinstance(self.content[i][j], str):
                        self.content[i][j] = str(self.content[i][j])
        # print(self.digital_indexes)
        # print(self.string_indexes)
        # print(self.open_response_indexes)
        # print(len(self.digital_indexes)+len(self.string_indexes)+len(self.open_response_indexes))

        #label encoder
        for i in range(len(self.string_indexes)):
            if self.string_indexes[i] not in self.label_encoder:
                self.label_encoder[str(self.string_indexes[i])] = []
        # print(label_encoder)
        for i in range(len(self.string_indexes)):
            index = self.string_indexes[i]
            for j in range(len(self.content)):
                if self.content[j][index] == "NA":
                    self.content[j][index] = -1
                    continue
                if self.content[j][index] not in self.label_encoder[str(index)]:
                    self.label_encoder[str(index)].append(self.content[j][index])
                self.content[j][index] = self.label_encoder[str(index)].index(self.content[j][index]) + 1

    def remove_open_response(self):
        use_indexes = copy.deepcopy(self.open_response_indexes)
        # del_data = np.array([len(self.content), len(self.open_response_indexes)], dtype = str)
        for i in range(len(use_indexes)):
            index = use_indexes[i]
            for j in range(len(self.content)):
                # del_data[j][i] = self.content[j][index]
                del self.content[j][index]
            for k in range(len(use_indexes)):
                use_indexes[k] -= 1
        for i in range(len(self.content[0])):
            if i == 0:
                continue
            self.predict_indexes.append(i)
        # self.open_response_content = del_data.tolist()

    def split_train_test(self, col):
        x = []
        y = []
        use_content = copy.deepcopy(self.content)
        for i in range(len(use_content)):
            y.append(use_content[i][col])
            del use_content[i][col]
            x.append(use_content[i])
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for i in range(len(y)):
            if y[i] == -1:
                x_test.append(x[i])
                y_test.append(y[i])
            else:
                x_train.append(x[i])
                y_train.append(y[i])
        return x_train, y_train, x_test, y_test

    def combine_train_test(self, col, x_train, y_train, x_test, y_test):
        for i in range(len(x_train)):
            x_train[i].insert(col, y_train[i])
        for i in range(len(x_test)):
            x_test[i].insert(col, y_test[i])
        self.content = []
        for i in range(len(x_train)):
            self.content.append(x_train[i])
        for i in range(len(x_test)):
            self.content.append(x_test[i])

    def reconvert_string(self):
        string_indexes = copy.deepcopy(self.string_indexes)
        open_response_indexes = copy.deepcopy(self.open_response_indexes)
        for i in range(len(open_response_indexes)):
            index = open_response_indexes[i]
            for j in range(len(string_indexes)):
                if string_indexes[j] > index:
                    string_indexes[j] -= 1
            for k in range(len(open_response_indexes)):
                open_response_indexes[k] -= 1
        for i in range(len(string_indexes)):
            values_counter = len(self.label_encoder[str(self.string_indexes[i])])
            # print(values_counter)
            # print(str(self.string_indexes[i]))
            # print(self.label_encoder[str(self.string_indexes[i])])
            # print(self.header[self.string_indexes[i]])
            # print("-------------")
            index = string_indexes[i]
            for j in range(len(self.content)):
                choose_number = self.content[j][index]
                if (choose_number < 1) or (choose_number > values_counter):
                    if values_counter == 0:
                        self.content[j][index] = "NA"
                    else:
                        random_pick = random.randint(0, values_counter - 1)
                        self.content[j][index] = self.label_encoder[str(self.string_indexes[i])][random_pick]
                else:
                    self.content[j][index] = self.label_encoder[str(self.string_indexes[i])][choose_number - 1]

    def rewrite_file(self):
        open_response_indexes = copy.deepcopy(self.open_response_indexes)
        for i in range(len(open_response_indexes)):
            index = open_response_indexes[i]
            for j in range(len(self.header)):
                if j == index:
                    del self.header[j]
            for k in range(len(open_response_indexes)):
                open_response_indexes[k] -= 1
        self.content.insert(0, self.header)
        with open('result.csv', 'w', newline='') as csvfile:
            writer  = csv.writer(csvfile)
            for row in self.content:
                writer.writerow(row)

    def generate_realistic(self):
        self.read_csv()
        del_indexes = []
        for i in range(len(self.content)):
            na_counter = 0
            for j in range(self.col):
                if self.content[i][j] == "NA":
                    na_counter += 1
            if na_counter > self.na_threshold:
                del_indexes.append(i)
        for i in range(len(del_indexes)):
            del self.content[del_indexes[i]]
            for j in range(len(del_indexes)):
                del_indexes[j] -= 1
        #open response index
        for i in range(self.col):
            if self.header[i] in self.open_response:
                self.open_response_indexes.append(i)
        #transform int and float data
        transform_vol_counter = [0] * self.col
        for i in range(len(self.content)):
            for j in range(self.col):
                if j in self.open_response_indexes:
                    continue
                if self.content[i][j] == "NA":
                    continue
                try:
                    self.content[i][j] = int(self.content[i][j])
                    transform_vol_counter[j] += 1
                except ValueError:
                    try:
                        self.content[i][j] = float(self.content[i][j])
                        transform_vol_counter[j] += 1
                    except ValueError:
                        pass
        # print(transform_vol_counter)
        for i in range(self.col):
            if i in self.open_response_indexes:
                continue
            if transform_vol_counter[i] > 1000:
                self.digital_indexes.append(i)
            else:
                self.string_indexes.append(i)
        for i in range(len(self.content)):
            for j in range(self.col):
                if j in self.open_response_indexes:
                    continue
                if j in self.digital_indexes:
                    if not isinstance(self.content[i][j], (int, float)):
                        self.content[i][j] = -1
                elif j in self.string_indexes:
                    if not isinstance(self.content[i][j], str):
                        self.content[i][j] = str(self.content[i][j])

    def compare_real(self):
        string_indexes = copy.deepcopy(self.string_indexes)
        open_response_indexes = copy.deepcopy(self.open_response_indexes)
        for i in range(len(open_response_indexes)):
            index = open_response_indexes[i]
            for j in range(len(string_indexes)):
                if string_indexes[j] > index:
                    string_indexes[j] -= 1
            for k in range(len(open_response_indexes)):
                open_response_indexes[k] -= 1
        for i in range(len(string_indexes)):
            values_counter = len(self.label_encoder[str(self.string_indexes[i])])
            index = string_indexes[i]
            for j in range(len(self.content)):
                choose_number = self.content[j][index]
                if (choose_number < 1) or (choose_number > values_counter):
                    if values_counter == 0:
                        self.content[j][index] = "NA"
                    else:
                        random_pick = random.randint(0, values_counter - 1)
                        self.content[j][index] = self.label_encoder[str(self.string_indexes[i])][random_pick]
                else:
                    self.content[j][index] = self.label_encoder[str(self.string_indexes[i])][choose_number - 1]
        for i in range(len(open_response_indexes)):
            index = open_response_indexes[i]
            for j in range(len(self.header)):
                if j == index:
                    del self.header[j]
            for k in range(len(open_response_indexes)):
                open_response_indexes[k] -= 1
