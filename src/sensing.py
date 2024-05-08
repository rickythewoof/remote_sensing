import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn import linear_model
import os

os.chdir("../")
df = pd.read_csv("data/stockats.tar.gz")
