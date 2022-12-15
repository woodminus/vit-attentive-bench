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
                    'params': row[1],
                    'flops': row[2],
                    'cpu': int(row[3]),
                    'gpu': int(row[4]),
                }

    gpu_speeds = np.array([v['cpu'] for k, v in data.items()])
    names = np.array(list(data.keys()))

    indices = np.argsort(gpu_speeds)[::-1]
    gpu_speeds = gpu_speeds[indices]
    names = names[indices]

    fig, ax 