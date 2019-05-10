import csv
import pandas as pd
import math
import openpyxl

class data_operations:
    def __init__(self):
        self.header = []
        self.content = []
        self.vol = 0
        self.na_threshold = 0
        self.digital_indexes = []
        self.string_indexes = []
        self.open_response = []
        self.open_response_indexes = []
        self.label_encoder = {}
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
            self.vol = len(self.content[0])
            self.na_threshold = math.ceil(self.vol * 2 / 3)

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
            for j in range(self.vol):
                if self.content[i][j] == "NA":
                    na_counter += 1
            if na_counter > self.na_threshold:
                del_indexes.append(i)
        for i in range(len(del_indexes)):
            del self.content[del_indexes[i]]
            for j in range(len(del_indexes)):
                del_indexes[j] -= 1
        #open response index
        for i in range(self.vol):
            if self.header[i] in self.open_response:
                self.open_response_indexes.append(i)
        #transform int and float data
        transform_vol_counter = [0] * self.vol
        for i in range(len(self.content)):
            for j in range(self.vol):
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
        for i in range(self.vol):
            if i in self.open_response_indexes:
                continue
            if transform_vol_counter[i] > 1000:
                self.digital_indexes.append(i)
            else:
                self.string_indexes.append(i)
        for i in range(len(self.content)):
            for j in range(self.vol):
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
                self.content[j][index] = self.label_encoder[str(index)].index(self.content[j][index])
