import csv
import matplotlib.pyplot as plt
import numpy as np

def plot_cpu():
    data = {}
    with open('results/attention_benchmark.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                data[row[0]] = {
                    'params': row