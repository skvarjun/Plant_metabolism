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



def train_feature_importance_dnn(df):
    y = df['Metabolites_labels'].to_numpy()
    X = df.drop('Metabolites_labels', axis=1).to_numpy()
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'mlpregressor__hidden_layer_sizes': [(65, 50, 50, 40, 30), (250, 200, 150, 50)],
        'mlpregressor__activation': ['relu', 'tanh'],
        'mlpregressor__solver': ['adam', 'sgd'],
        'mlpregressor__alpha': [0.0001, 0.001, 0.01],
        'mlpregressor__learning_rate': ['constant', 'adaptive'],
        'mlpregressor__learning_rate_init': [0.001, 0.01, 0.1],
        'mlpregressor__max_iter': [10000]
    }
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlpregressor', MLPRegressor())
    ])
    model = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

    #model = MLPRegressor()
    model.fit(X_train, y_train)

    MLP_y_pred = model.predict(X_test)
    MLP_r2 = r2_score(y_test, MLP_y_pred)
    MLP_mse = mean_squared_error(y_test, MLP_y_pred)
    MLP_mae = mean_absolute_error(y_test, MLP_y_pred)
    print(f"R2: {MLP_r2}, MAE: {MLP_mae}, MSE {MLP_mse}")

    plt.errorbar(y_test, MLP_y_pred)
    plt.plot(y_test, y_test, ':', lw=1, color='black')
    plt.show()


def weighted_labels(df):
    weights = df['Metabolites_type_labels'].value_counts(normalize=True).to_dict()
    weighted_label = []
    for index, row in df.iterrows():
        for key, value in weights.items():
            if row['Metabolites_type_labels'] == key:
                x = row['Metabolites_labels'] * value
                weighted_label.append(x)
    df.insert(loc=0, column='weighted_label', value=weighted_label)
    return df

def train_feature_importance(df, mainData):
    mainData = mainData.drop(columns=df.columns.to_list())
    df = df.drop(['Metabolites', 'Metabolites_labels', 'Metabolites_type_labels'], axis=1)
    result_df = pd.DataFrame(columns=['Target_sample', 'Trial_1', 'Trial_2', 'Trial_3', 'Trial_4', 'Trial_5'])
    for eachSample in mainData.columns:
        y = mainData[eachSample].to_numpy()
        X = df.to_numpy()

        r2s = [eachSample]
        for i in range(5):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=200)
            model = LoLoRF()
            model.fit(X_train, y_train)
            LoLo_y_pred, LoLo_y_uncer = model.predict(X_test, return_std=True)  # , range(14)
            LoLo_r2 = r2_score(y_test, LoLo_y_pred)
            y_resid = LoLo_y_pred - y_test
            LoLo_mse = mean_squared_error(y_test, LoLo_y_pred)
            LoLo_mae = mean_absolute_error(y_test, LoLo_y_pred)
            #print(f"For trial {i} | R2: {LoLo_r2}, MAE: {LoLo_mae}, MSE {LoLo_mse}")
            r2s.append(LoLo_r2)
        result_df.loc[len(result_df)] = r2s

        # plt.errorbar(y_test, LoLo_y_pred, LoLo_y_uncer, fmt='o', ms=2.5, ecolor='grey')
        # plt.xlabel("Actual weight of metabolites (scaled)")
        # plt.ylabel("Predicted weight of metabolites (scaled)")
        # plt.title(f"Sample: {eachSample}\nR2: {LoLo_r2:.2f}, MAE: {LoLo_mae:.4f} MSE: {LoLo_mse:.4f}")
        # plt.plot(y_test, y_test, ':', lw=1, color='black')
        # plt.show()
        # plt.close()



    result_df['Average_r2'] = result_df[['Trial_1', 'Trial_2', 'Trial_3', 'Trial_4', 'Trial_5']].mean(axis=1)
    print(result_df)
    x = result_df['Average_r2'].to_numpy()
    reshaped_arr = x.reshape(5, 4)
    reshaped_arr = np.transpose(reshaped_arr)

    num_groups = reshaped_arr.shape[0]
    x = np.arange(reshaped_arr.shape[1])

    plt.rcParams["figure.figsize"] = [8, 6]
    fig, ax = plt.subplots()
    bar_width = 0.15
    opacity = 0.8

    for i in range(num_groups):
        ax.bar(x + i * bar_width, reshaped_arr[i], bar_width,
               alpha=opacity,
               label=f'Sample {i + 1}')

    ax.set_xlabel('Five types of diverse soil samples (each has 4 samples)')
    ax.set_ylabel('R2 score')
    ax.set_title("Random forest performance of models when each nanopastic\nsoil sample is taken as target for 'pure soil' data")
    ax.set_xticks(x + bar_width * (num_groups - 1) / 2)
    ax.set_xticklabels(['Soil nanoplatic', 'Soil biochar', 'Soil biochar nanoplastic', 'Soil Fe-biochar', 'Soil Fe-biochar nanoplastic'], rotation=90)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("RF_performaance_t_w_pureSoil.png", dpi=300)
    plt.show()


def Main():
    data = MasterData().main_data(drop_labels=False)
    data = MasterData().unscaled()
    data.rename(columns={'Metabolites_labels': 'Metabolites_type_labels'}, inplace=True)
    data['Metabolites_labels'] = LabelEncoder().fit_transform(data['Metabolites'])


    samples = data.drop(['Metabolites', 'Metabolites_labels', 'Metabolites_type_labels'], axis=1).columns.to_list()
    for i in range(0, len(samples), 4):
        chunk = samples[i:i + 4]
        df = data[chunk+['Metabolites_labels', 'Metabolites_type_labels', 'Metabolites']]
        train_feature_importance(df, data)
        #train_feature_importance_dnn(df)
        exit()



if __name__ == "__main__":
    metanolites_cat = ['Amino_acid', 'Antioxidant', 'Fatty_acids', 'Nucleobase_or_side_or_tide', 'Organic_acid_or_phenolics', 'Vitamin_B', 'Sugar_or_Sugar_alcohol']
    samples = ['Soil_alone', 'Soil_nanoplatic', 'soil_biochar', 'Soil_biochar_nanoplastic', 'Soil_Fe-biochar', 'Soil_Fe-biochar_nanoplastic']
    Main()
    #influence_of_metabolites()
