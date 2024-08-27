# Author: Arjun S Kulathuvayal. Intellectual property. Copyright strictly restricted
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn import metrics
from lolopy.learners import RandomForestRegressor as LoLoRF
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
pd.set_option('display.max_columns', 9)
pd.set_option('display.width', 10000)
font = {'family': 'serif',
        'weight': 'normal',
        'size': 12}
plt.rc('font', **font)
plt.rcParams["figure.figsize"] = [6, 6]
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler


def FI_plot(fi_df, R2, MSE):
    fi_df = fi_df[fi_df['importances'] >= 0.06]

    fig, ax1 = plt.subplots()
    plt.bar(fi_df['feature'], fi_df['importances'], yerr=fi_df['std_dev'], capsize=5, alpha=0.7, color='skyblue')
    ax1.set_title(f"| R2={R2:.2f} | MSE={MSE:.4e}")
    ax1.set_ylabel("Mean accuracy decrease")
    ax1.tick_params(axis='x', labelrotation=0, labelsize=10)
    ax1.tick_params(axis='x', bottom=False, top=False, right=False, rotation=45)
    #features = [s.replace("_", " ").replace("Diff", "D").replace("DM", "DM's").replace("DE", "DE's").replace("minimum", "min.") for s in fi_df['feature'].tolist()]
    features = fi_df['feature'].tolist()
    ax1.set_xticklabels(features, rotation=90)
    ax1.tick_params(axis='y', direction='in')
    #ax1.set_ylim(0, 0.6)
    #ax2 = ax1.twinx()
    #ax2.plot(range(0, len(feature_names)), df.drop(['D'], axis=1).std().tolist(), '--o', color="#2E4068", label="Standard deviation")
    #ax2.set_ylabel("Standard deviation")
    #ax2.tick_params(axis='x', labelrotation=0, labelsize=10)
    # ax2.tick_params(direction='in')
    fig.tight_layout()
    plt.savefig(f'Feature_importances_full.png')
    #plt.show()


def Feature_importances_LoLo_RF(df, log=True):


    y = df['D'].to_numpy()
    X = df.drop(['D'], axis=1).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LoLoRF()
    model.fit(X_train, y_train)
    LoLo_y_pred, LoLo_y_uncer = model.predict(X_test, return_std=True)
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

    feature_names = df.drop(['D'], axis=1).columns.tolist()
    forest_importances = pd.DataFrame(data={'feature': feature_names, 'importances': result.importances_mean, 'std_dev': result.importances_std})
    with open(f'Results/{folder}/feature_importance.txt', "w") as file:
        for i, j in enumerate(feature_names):
            file.write(f"{i+1}) {j}\n")

    RFr2 = r2_score(y_test, LoLo_y_pred)
    RFmse = mean_squared_error(np.exp(-y_test), np.exp(-LoLo_y_pred))
    FI_plot(forest_importances, folder, RFr2, RFmse)
    return print("Task done")


def Feature_importances_MLPR(df):
    df = df.drop('Metabolites', axis=1)
    y = df['Metabolites_labels'].to_numpy()
    X = df.drop(['Metabolites_labels'], axis=1).to_numpy()
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MLPRegressor(activation='relu', alpha=0.0001, hidden_layer_sizes=(250, 200, 150, 50), learning_rate='adaptive', learning_rate_init=0.001, max_iter=10000, solver='sgd')
    model.fit(X_train, y_train)
    MLPR_y_pred = model.predict(X_test)
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

    MLPR_mse = mean_squared_error(y_test,MLPR_y_pred)
    MLPR_r2 = r2_score(y_test, MLPR_y_pred)
    print(f"R2 : {MLPR_r2}, MSE : {MLPR_mse}")

    feature_names = df.drop(['Metabolites_labels'], axis=1).columns.tolist()
    forest_importances = pd.DataFrame(data={'feature': feature_names, 'importances': result.importances_mean, 'std_dev': result.importances_std})
    forest_importances.to_csv(f'feature_importance.csv')


    FI_plot(forest_importances, MLPR_r2, MLPR_mse)
    return print("Task done")


if __name__ == '__main__':
    scaled_data = pd.read_csv('scaled.csv')
    master_data = pd.read_csv('master_data.csv')
    master_data_ = master_data.drop(columns=['Metabolites_labels']).replace(0, np.nan).dropna().reset_index()
    met_lables = []
    for indices in master_data_['index']:
        m = master_data.iloc[indices, :].to_frame().transpose()
        met_lables.append(int(m['Metabolites_labels'].to_list()[0]))

    new_data = scaled_data.transpose().reset_index()
    new_data.insert(loc=0, column='Metabolites_labels', value=met_lables)
    new_data.columns = master_data.columns
    Feature_importances_MLPR(new_data)

