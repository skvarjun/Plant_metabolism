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


def Main():
    df = pd.read_excel('Utah_Lettuce_results_unit_adjusted.xlsx', skiprows=2)
    metabolites, labels, label_col = dict(), 0, []
    for index, row in df.iterrows():
        if row.isna().any():
            metabolites[row.iloc[0]] = labels
            labels += 1
        if row.isna().any():
            pass
        else:
            label_col.append(labels-1)

    df = df.dropna()
    df.insert(loc=0, column='Metabolites', value=label_col)

    samples, j, new_col = ['soil_alone', 'soil_nanoplatic', 'soil_biochar', 'soil_biochar_nanoplastic', 'soil_Fe-biochar', 'soil_Fe-biochar_nanoplastic'], 0, []
    for i, col in enumerate(df.iloc[0:, 2:].columns):
        s_number = re.findall(r'\d+', col)[0]
        if s_number.isdigit() and int(s_number) % 4 == 0:
            sample = samples[j]
            j +=1
        else:
            sample = samples[j]

        new_col.append(f"S_{i+1}_{sample}")
    new_col = ['Metabolites_labels', 'Metabolites'] + new_col
    df.columns = new_col
    df = df.replace('-', '0')
    df.to_csv('master_data.csv', index=False)

    print(f"Data have been reformated and save to master_data.csv, with metabolites with labels {metabolites}")



if __name__ == "__main__":
    Main()
