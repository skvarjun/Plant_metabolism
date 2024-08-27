# Author: Arjun S Kulathuvayal. Intellectual property. Copyright strictly restricted
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, re

font = {'family': 'serif',
        'weight': 'normal',
        'size': 12}
plt.rc('font', **font)
plt.rcParams["figure.figsize"] = [8, 6]


def MLDiCE():
    project_root = os.path.dirname(os.path.abspath(__file__))
    vgout = open("Output.md", "w")
    print("# AI Driven Data Analysis on Nanoplastic Influence on Metabolites in Lettuce PLant\n", file=vgout)



if __name__ == "__main__":
    MLDiCE()
