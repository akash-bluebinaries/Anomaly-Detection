import os,sys
import pandas as pd
import numpy as np
# from pycaret.classification import *
from imblearn.over_sampling import RandomOverSampler
import mlflow
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='_distutils_hack')


def load_data(path):
    data = pd.read_csv(path)
    return data

def data_cleaning(data):
    print("NA values available in data \n")
    print(data.isna().sum())
    data = data.dropna()
    print("After dropping NA values \n")
    print(data.isna().sum())
    return data