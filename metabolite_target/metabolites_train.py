import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from data import MasterData
from lolopy.learners import RandomForestRegressor as LoLoRF
from sklearn.ensemble import RandomForestRegressor as SKRF
from sklearn.preprocessing import MinMaxScaler, StandardScaler
font = {'family': 'serif',
        'weight': 'normal',
        'size': 12}
plt.rc('font', **font)
plt.rcParams["figure.figsize"] = [12, 6]


def col_filter(df):
    correlation_matrix = df.corr()
    similar_columns = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.50:
                colname = correlation_matrix.columns[i]
                similar_columns.add(colname)
    filtered_df = df.drop(columns=similar_columns)
    return filtered_df



def metabolite_group_training__(df):
    df.rename(columns={'Metabolites_labels': 'Metabolites_type_labels'}, inplace=True)
    df['Metabolites_labels'] = LabelEncoder().fit_transform(df['Metabolites'])
    df_ = df.drop(columns=['Metabolites_labels']).transpose().reset_index()
    df_.columns = ['soil_type'] + df['Metabolites'].to_list()
    df = df_.drop(index=1).reset_index(drop=True)

    r2_, mse_, metabolite_, train_set, features_used = [], [], [], [], []
    for i, eachMetaType in enumerate(metanolites_cat):
        print(eachMetaType)
        metabolite_df = df[df.columns[df.iloc[0] == i]].iloc[1:, :]

        for eachMeta in metabolite_df.columns:
            y = metabolite_df[eachMeta].to_numpy()
            y = MinMaxScaler().fit_transform(y.reshape(-1, 1))
            X = col_filter(metabolite_df.drop(eachMeta, axis=1)).to_numpy()
            X = MinMaxScaler().fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LoLoRF()
            model.fit(X_train, y_train)
            LoLo_y_pred, LoLo_y_uncer = model.predict(X_test, return_std=True)  # , range(14)
            LoLo_r2 = r2_score(y_test, LoLo_y_pred)
            y_resid = LoLo_y_pred - y_test
            LoLo_mse = mean_squared_error(y_test, LoLo_y_pred)
            LoLo_mae = mean_absolute_error(y_test, LoLo_y_pred)
            #print(f"For metabolite type {eachMetaType} with target {eachMeta} | R2: {LoLo_r2}, MAE: {LoLo_mae}, MSE {LoLo_mse}")
            r2_.append(LoLo_r2)
            mse_.append(LoLo_mse)
            metabolite_.append(eachMeta)
            train_set.append(str(X.shape))
            features_used.append(col_filter(metabolite_df.drop(eachMeta, axis=1)).columns.to_list())

    top_per_df = pd.DataFrame(dict(zip(['metabolites', 'r2_score', 'features_used'], [metabolite_, r2_, features_used])))
    top_per_df.to_csv("top_per_df_minmax_scaler.csv", index=False)
    t = top_per_df[(top_per_df['r2_score'] > 0) & (top_per_df['r2_score'] < 1)]
    print(f"Number of optimally perfoming models {len(t)}")

    plt.bar(metabolite_, r2_)
    for i in range(len(train_set)):
        plt.text(i-0.25, r2_[i]-1, train_set[i], rotation=90, fontsize=8)
    plt.xlabel("Metabolites")
    plt.ylabel("R2 score")
    plt.xlim(left=-1, right=67)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("rf_unscaled_corr_applied.png", dpi=300)
    plt.show()


def metabolite_group_training(df):
    for i, eachMetaType in enumerate(metanolites_cat):
        #print(eachMetaType)
        metabolite_df = df[df.columns[df.iloc[0] == i]].iloc[1:, :]
        print(metabolite_df)
        for eachMeta in metabolite_df.columns:
            y = metabolite_df[eachMeta].to_numpy()
            X = metabolite_df.drop(eachMeta, axis=1).to_numpy()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LoLoRF()
            model.fit(X_train, y_train)
            LoLo_y_pred, LoLo_y_uncer = model.predict(X_test, return_std=True)  # , range(14)
            LoLo_r2 = r2_score(y_test, LoLo_y_pred)
            y_resid = LoLo_y_pred - y_test
            LoLo_mse = mean_squared_error(y_test, LoLo_y_pred)
            LoLo_mae = mean_absolute_error(y_test, LoLo_y_pred)
            print(f"For metabolite type {eachMetaType} with target {eachMeta} | R2: {LoLo_r2}, MAE: {LoLo_mae}, MSE {LoLo_mse}")
        exit()

def Main():
    data = MasterData().main_data(drop_labels=False)
    unscaled_data = MasterData().unscaled()
    data.rename(columns={'Metabolites_labels': 'Metabolites_type_labels'}, inplace=True)
    data['Metabolites_labels'] = LabelEncoder().fit_transform(data['Metabolites'])
    samples = data.drop(['Metabolites', 'Metabolites_labels', 'Metabolites_type_labels'], axis=1).columns.to_list()
    data_ = data.drop(columns=['Metabolites_labels']).transpose().reset_index()
    data_.columns = ['soil_type'] + data['Metabolites'].to_list()
    train_data = data_.drop(index=1).reset_index(drop=True)
    #metabolite_group_training(train_data)
    metabolite_group_training__(unscaled_data)




if __name__ == "__main__":
    metanolites_cat = ['Amino_acid', 'Antioxidant', 'Fatty_acids', 'Nucleobase_or_side_or_tide', 'Organic_acid_or_phenolics', 'Vitamin_B', 'Sugar_or_Sugar_alcohol']
    samples = ['Soil_alone', 'Soil_nanoplatic', 'soil_biochar', 'Soil_biochar_nanoplastic', 'Soil_Fe-biochar', 'Soil_Fe-biochar_nanoplastic']
    Main()
