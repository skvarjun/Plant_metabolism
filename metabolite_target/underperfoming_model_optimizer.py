# Author: Arjun S Kulathuvayal. Intellectual property. Copyright strictly restricted
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, re

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from data import MasterData
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from itertools import combinations
from metabolite_target.metabolites_train import col_filter
from lolopy.learners import RandomForestRegressor as LoLoRF
pd.set_option('display.max_columns', 9)
pd.set_option('display.width', 10000)
font = {'family': 'serif',
        'weight': 'normal',
        'size': 12}
plt.rc('font', **font)
plt.rcParams["figure.figsize"] = [8, 6]


def generate_combinations(lst, min_length=3):
    all_combinations = []
    for length in range(min_length, len(lst) + 1):
        comb = combinations(lst, length)
        all_combinations.extend(comb)
    return all_combinations


def Main():
    p_df = pd.read_csv('top_per_df_std_scaler.csv')
    tb_tuned = p_df[(p_df['r2_score'] > 0.0) & (p_df['r2_score'] < 0.5)].reset_index(drop=True)
    under_p = p_df[(p_df['r2_score'] < 0.0)].reset_index(drop=True)


    df = MasterData().unscaled()
    df.rename(columns={'Metabolites_labels': 'Metabolites_type_labels'}, inplace=True)
    df['Metabolites_labels'] = LabelEncoder().fit_transform(df['Metabolites'])
    df_ = df.drop(columns=['Metabolites_labels']).transpose().reset_index()
    df_.columns = ['soil_type'] + df['Metabolites'].to_list()
    df = df_.drop(index=1).reset_index(drop=True)

    metabolites, r2_, features = [], [], []
    for i, eachMetaType in enumerate(metanolites_cat):
        print(eachMetaType)
        metabolite_df = df[df.columns[df.iloc[0] == i]].iloc[1:, :]
        for eachMeta in metabolite_df.columns:
            if eachMeta in under_p['metabolites'].to_list():
                y = metabolite_df[eachMeta].to_numpy()
                y = MinMaxScaler().fit_transform(y.reshape(-1, 1))

                for comb in generate_combinations(metabolite_df.drop(eachMeta, axis=1).columns.to_list(), 3):
                    comb_of_three_features = metabolite_df[list(comb)].to_numpy()
                    X = MinMaxScaler().fit_transform(comb_of_three_features)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = LoLoRF()
                    model.fit(X_train, y_train)
                    LoLo_y_pred, LoLo_y_uncer = model.predict(X_test, return_std=True)  # , range(14)
                    LoLo_r2 = r2_score(y_test, LoLo_y_pred)
                    y_resid = LoLo_y_pred - y_test
                    LoLo_mse = mean_squared_error(y_test, LoLo_y_pred)
                    LoLo_mae = mean_absolute_error(y_test, LoLo_y_pred)
                    print(f"For metabolite type {eachMetaType} with target {eachMeta} | R2: {LoLo_r2}, MAE: {LoLo_mae}, MSE {LoLo_mse}")
                    if 0.5 < LoLo_r2 < 1:
                        metabolites.append(eachMeta)
                        r2_.append(LoLo_r2)
                        features.append(list(comb))
    top_per_df = pd.DataFrame(dict(zip(['metabolites', 'r2_score', 'features_used'], [metabolites, r2_, features])))
    top_per_df.to_csv("under_perform_optimizer.csv", index=False)


if __name__ == "__main__":
    metanolites_cat = ['Amino_acid', 'Antioxidant', 'Fatty_acids', 'Nucleobase_or_side_or_tide', 'Organic_acid_or_phenolics', 'Vitamin_B', 'Sugar_or_Sugar_alcohol']
    Main()
