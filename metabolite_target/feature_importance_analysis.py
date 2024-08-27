import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from data import MasterData
from lolopy.learners import RandomForestRegressor as LoLoRF
from sklearn.ensemble import RandomForestRegressor as SKRF
from sklearn.preprocessing import MinMaxScaler, StandardScaler
pd.set_option('display.max_columns', 9)
pd.set_option('display.width', 10000)
font = {'family': 'serif',
        'weight': 'normal',
        'size': 12}
plt.rc('font', **font)
plt.rcParams["figure.figsize"] = [8, 6]
import ast


def graph():
    df = pd.read_csv('top_per_df_minmax_scaler.csv')
    df = df[(df['r2_score'] > 0.5) & (df['r2_score'] < 1)].reset_index(drop=True)
    print(df)
    for index, row in df.iterrows():
        print(row['metabolites'].replace(" ", "_").replace("/", "_"))


def Main():
    df = MasterData().unscaled()
    df.rename(columns={'Metabolites_labels': 'Metabolites_type_labels'}, inplace=True)
    df['Metabolites_labels'] = LabelEncoder().fit_transform(df['Metabolites'])
    df_ = df.drop(columns=['Metabolites_labels']).transpose().reset_index()
    df_.columns = ['soil_type'] + df['Metabolites'].to_list()
    master_df = df_.drop(index=1).reset_index(drop=True)

    df = pd.read_csv('top_per_df_minmax_scaler.csv')
    df = df[(df['r2_score'] > 0.5) & (df['r2_score'] < 1)].reset_index(drop=True)
    print(df)
    for index, row in df.iterrows():
        y = master_df[row['metabolites']].to_numpy()
        y = MinMaxScaler().fit_transform(y.reshape(-1, 1))

        X = master_df[ast.literal_eval(row['features_used'])]
        X = MinMaxScaler().fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LoLoRF()
        model.fit(X_train, y_train)

        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

        feature_names = master_df[ast.literal_eval(row['features_used'])].columns.tolist()
        forest_importances = pd.DataFrame(data={'feature': feature_names, 'importances': result.importances_mean, 'std_dev': result.importances_std})

        os.makedirs(str(row['metabolites']).replace("/", "_"), exist_ok=True)
        forest_importances.to_csv(f"{str(row['metabolites']).replace('/', '_')}/feature_importance.csv")



if __name__ == "__main__":
    #Main()
    graph()
