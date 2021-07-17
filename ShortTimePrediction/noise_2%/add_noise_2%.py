import numpy as np
import os
import matplotlib.pyplot as plt


path = "C:/Users/h/Desktop/PyCharm Community Edition 2020.3/PycharmProjects/MassSpringSystemDL/ShortTimePrediction/noise_2%"
datanames = os.listdir(path)

# np.random.seed(2)
# for file in datanames[:-1]:
#     source = np.loadtxt(file)
#     if file[-5:] == "input":
#         noise = 1.0 + np.random.randn(300)*0.02
#     if file[-5:] == "state":
#         noise = 1.0 + np.random.randn(301,2) * 0.02
#     output = source * noise
#     np.savetxt(file + "_2%", output)


