import numpy as np
import os
import matplotlib.pyplot as plt

#adding 4%(gaussian distribution: +- 0.04%)
path = "C:/Users/h/Desktop/PyCharm Community Edition 2020.3/PycharmProjects/MassSpringSystemDL/LongTimePrediction/noise_30%"
datanames = os.listdir(path)

np.random.seed(5)
for file in datanames[:-1]:
    source = np.loadtxt(file)
    if file[-5:] == "input":
        noise = 1.0 + np.random.randn(300)*0.3
    if file[-5:] == "state":
        noise = 1.0 + np.random.randn(301,2) * 0.3
    output = source * noise
    np.savetxt(file + "_30%", output)


