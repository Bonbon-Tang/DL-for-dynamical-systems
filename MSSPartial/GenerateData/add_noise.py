import numpy as np
import os
import matplotlib.pyplot as plt


path = "C:/Users/h/Desktop/PyCharm Community Edition 2020.3/PycharmProjects/MassSpringPartialObservables/GenerateData"
datanames = os.listdir(path)

np.random.seed(4)
for file in datanames[:-1]:
    source = np.loadtxt(file)
    if file[-5:] == "state":
        noise = 1.0 + np.random.randn(201,2) * 0.2
        output = source * noise
        np.savetxt(file + "_20%", output)


