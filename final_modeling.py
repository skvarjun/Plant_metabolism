import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from data import MasterData
from lolopy.learners import RandomForestRegressor as LoLoRF
from sklearn.ensemble import RandomForestRegressor as SKRF
font = {'family': 'serif',
        'weight': 'normal',
        'size': 12}
plt.rc('font', **font)
plt.rcParams["figure.figsize"] = [9, 7]
from sklearn.preprocessing import OneHotEncoder
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 10000)


def nanoplasticContent():
    df_ = pd.read_csv('nano_.csv')
    df = df_.dropna()
    df = df.transpose()
    df = df.iloc[1:, :]
    names = df_.dropna()
    df.columns = names['nanopastic_info'].to_list()
    df = df.reset_index(drop=True)
    return df


def onehot_encode_column(df, column_name):
    encoder = OneHotEncoder(sparse_output=False)  # drop='first' to avoid multicollinearity
    onehot_encoded = encoder.fit_transform(df[[column_name]])
    onehot_encoded_df = pd.DataFrame(onehot_encoded, columns=encoder.get_feature_names_out([column_name]))
    df = pd.concat([df, onehot_encoded_df], axis=1)
    df.drop(column_name, axis=1, inplace=True)
    return df


def met_data(df):
    t_data = df.transpose()
    t_data.columns = df['Metabolites'].to_list()
    t_data = t_data.reset_index(drop=True)
    t_data = t_data.drop(1).reset_index(drop=True)
    return t_data


def Main():
    data, plant_ftrs = MasterData().unscaled()
    t_data = met_data(data)
    t_data.insert(0, 'Soil_type', [''] + data.drop(['Metabolites_labels', 'Metabolites'], axis=1).columns.to_list())
    without_type_of_meta = t_data.drop(0).reset_index(drop=True)
    final_df = onehot_encode_column(without_type_of_meta, column_name='Soil_type')

    soil_types = ['soil_alone', 'soil_nanoplatic', 'soil_biochar', 'soil_biochar_nanoplastic', 'soil_Fe-biochar', 'soil_Fe-biochar_nanoplastic']
    st = [[i]*4 for i in soil_types]
    sl = [[i]*4 for i in range(6)]
    soil_NP = [1 if str(i).endswith('soil_nanoplatic') else 0 for i in [item for sublist in st for item in sublist]]
    soil_biochar_NP = [1 if 'soil_biochar_nanoplastic' in str(i) else 0 for i in [item for sublist in st for item in sublist]]
    soil_Fe_biochar_NP = [1 if 'Fe-biochar_nanoplastic' in str(i) else 0 for i in [item for sublist in st for item in sublist]]
    group_df = pd.DataFrame({'soil_with_NP': soil_NP, 'soil_biochar_with_NP': soil_biochar_NP, 'soil_Fe_biochar_with_NP': soil_Fe_biochar_NP})

    #I and YaSu grouping
    np_gp = [1 if 'tic' in str(i) else 0 for i in[item for sublist in st for item in sublist]]
    biochar_gp = [1 if 'biochar' in str(i) else 0 for i in [item for sublist in st for item in sublist]]
    biochar_np_gp = [1 if 'biochar_nano' in str(i) else 0 for i in [item for sublist in st for item in sublist]]
    Fe_biochar_gp = [1 if 'Fe-biochar' in str(i) else 0 for i in [item for sublist in st for item in sublist]]
    Fe_biochar_nano_gp = [1 if 'Fe-biochar_nano' in str(i) else 0 for i in [item for sublist in st for item in sublist]]
    I_Ya_Su_df = pd.DataFrame({'np_gp': np_gp, 'biochar_gp': biochar_gp, 'biochar_np_gp': biochar_np_gp, 'Fe_biochar_gp':Fe_biochar_gp, 'Fe_biochar_nano_gp':Fe_biochar_nano_gp})

    nano_wt_df = nanoplasticContent()


    for eachGroup in I_Ya_Su_df.columns:
        os.makedirs(f"test_final_modeling/{eachGroup}", exist_ok=True)
        for eachMeta in data['Metabolites']:
            s_type = pd.DataFrame({'soil_type': [item for sublist in sl for item in sublist],
                                   'soil_type_': [item for sublist in st for item in sublist]})
            new_df = pd.DataFrame({'np_conc_in_lettuce': nano_wt_df['total nanoplastics in one lettuce (ng)'] / plant_ftrs['leaf_wt']})
            target_df = final_df[[eachMeta]].astype(int).to_numpy()
            target_df = MinMaxScaler().fit_transform(target_df)
            target_df = pd.DataFrame({f"{eachMeta}": target_df.flatten()})
            train_df = pd.concat([target_df, plant_ftrs, I_Ya_Su_df[eachGroup], ], axis=1) #group_df[eachGroup]
            print(train_df)

            y = train_df[eachMeta].to_numpy()
            X = train_df.drop(eachMeta, axis=1).to_numpy()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LoLoRF()
            model.fit(X_train, y_train)
            LoLo_y_pred, LoLo_y_uncer = model.predict(X_test, return_std=True)
            RFr2 = r2_score(y_test, LoLo_y_pred)
            RFmse = mean_squared_error(y_test, LoLo_y_pred)
            RFmae = mean_absolute_error(y_test, LoLo_y_pred)

            result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=200)
            feature_names = train_df.drop(eachMeta, axis=1).columns.tolist()
            fi = pd.DataFrame(data={'feature': feature_names, 'importances': result.importances_mean, 'std_dev': result.importances_std})
            result_info_df = pd.DataFrame({'r2': [RFr2], 'mse': [RFmse], 'mae': [RFmae]})

            plt.rcParams["figure.figsize"] = [10, 7.5]
            fig, axs = plt.subplots(1, 2)
            axs[0].errorbar(y_test, LoLo_y_pred, LoLo_y_uncer, fmt='o', ms=2.5, ecolor='grey')
            axs[0].set_xlabel("Actual weight of metabolites (ng/g)")
            axs[0].set_ylabel("Predicted weight of metabolites (ng/g)")
            axs[0].set_title("Prediction performance")
            axs[0].plot(y_test, y_test, ':', lw=1, color='black')
            axs[0].legend([f'Test size : {len(y_test)}', f'Train size : {len(y_train)}'], markerscale=0, handlelength=0,
                          loc='upper left')

            axs[1].bar(fi['feature'], fi['importances'], yerr=fi['std_dev'], capsize=5, alpha=0.7, color='blue')
            axs[1].set_ylabel("Mean accuracy decrease")
            axs[1].set_xlabel("Soil samples")
            axs[1].set_title("Feature importance")
            axs[1].tick_params(axis='x', labelrotation=90, labelsize=8)
            fig.suptitle(f"Sample: {eachMeta}\nR2: {RFr2:.2f} | MSE: {RFmse:.4e} | MAE: {RFmae:.4e}")
            fig.subplots_adjust(top=0.9)
            fig.tight_layout()
            os.makedirs(f"test_final_modeling/{eachGroup}/{eachMeta}", exist_ok=True)
            fi.to_csv(f'test_final_modeling/{eachGroup}/{eachMeta}/fi.csv', index=False)
            result_info_df.to_csv(f'test_final_modeling/{eachGroup}/{eachMeta}/res.csv', index=False)
            plt.savefig(f'test_final_modeling/{eachGroup}/{eachMeta}/results.png', dpi=300)
            plt.show()
            plt.close()
            print(f"Task done for group {eachGroup} and metabolite {eachMeta} with R2 {RFr2}")
            exit()
            




def Classification():
    data, plant_ftrs = MasterData().unscaled()
    t_data = met_data(data)
    t_data.insert(0, 'Soil_type', [''] + data.drop(['Metabolites_labels', 'Metabolites'], axis=1).columns.to_list())
    without_type_of_meta = t_data.drop(0).reset_index(drop=True)
    final_df = onehot_encode_column(without_type_of_meta, column_name='Soil_type')

    soil_types = ['soil_alone', 'soil_nanoplatic', 'soil_biochar', 'soil_biochar_nanoplastic', 'soil_Fe-biochar',
                  'soil_Fe-biochar_nanoplastic']

    for each_type in soil_types:
        for eachMeta in ['Isoleucine']:
            s_type = pd.DataFrame({'soil_type': [0]*4 + [1]*4 + [2]*4 + [3]*4 + [4]*4 + [5]*4})

            train_df = final_df[[eachMeta]].astype(int)
            #train_df.loc[:, f'tw_{eachMeta}'] = train_df[eachMeta] * plant_ftrs['fdw']
            train_df = pd.concat([train_df, plant_ftrs, s_type], axis=1)

            #train_df = train_df[['soil_type', 'leaf_wt', 'root_wt']]

            # print(train_df)
            # exit()
            target = 'soil_type'
            y = train_df[target].to_numpy()
            X = train_df.drop(target, axis=1).to_numpy()

            X_train, X_test, y_train, y_test = train_test_split(X, y.astype(int), test_size=0.8, random_state=200)
            X_tr, X_te, y_tr, y_te = train_test_split(X, y.astype(int), train_size=0.90, random_state=200)
            model = KNeighborsClassifier(n_neighbors=3, weights='distance')
            model.fit(X_tr, y_tr)
            y_score = model.predict_proba(X_test)

            # mesh_size = 0.015
            # margin = 0.0
            # x_min, x_max = X_test[:, 0].min() - margin, X_test[:, 0].max() + margin
            # y_min, y_max = X_test[:, 1].min() - margin, X_test[:, 1].max() + margin
            # xrange = np.arange(x_min, x_max, mesh_size)
            # yrange = np.arange(y_min, y_max, mesh_size)
            # xx, yy = np.meshgrid(xrange, yrange)
            # Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            # Z = Z.reshape(xx.shape)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            print("Accuracy:", accuracy)

            soil_labels = {'soil_alone': 0, 'soil_nanoplatic': 1, 'soil_biochar': 2, 'soil_biochar_nanoplastic': 3, 'soil_Fe-biochar': 4, 'soil_Fe-biochar_nanoplastic': 5}
            probability = []
            soil_type = []
            for row in y_score:
                max_index = np.argmax(row)
                probability.append(max(row))
                soil_type.append(list(soil_labels.keys())[max_index])
            print(soil_type)
            print(probability)
            # print([soil_labels[item] for item in soil_type])
            # X_flat = xx.flatten()
            # Y_flat = yy.flatten()
            # Z_flat = Z.flatten()
            # plt.scatter(X_flat, Y_flat, c=Z_flat, cmap='viridis')

            sc = plt.scatter(X_test[:, 0], X_test[:, 4], c=[soil_labels[item] for item in soil_type])
            plt.xlabel(f"Weight of {eachMeta}")
            plt.ylabel(f"Root weight")
            plt.legend(handles=sc.legend_elements()[0], labels=soil_labels.keys())
            plt.title(f"Model: KNeighborsClassifier | {eachMeta} | Accuracy :{accuracy:.2f}")
            os.makedirs(f"test_final_modeling/classification/{eachMeta}", exist_ok=True)
            plt.savefig(f"test_final_modeling/classification/{eachMeta}/root_weight.png", dpi=300)
            plt.show()
            exit()


def meta_soil_type():
    data, plant_ftrs = MasterData().unscaled()
    for eachGroup in ['np_gp', 'biochar_gp', 'biochar_np_gp', 'Fe_biochar_gp', 'Fe_biochar_nano_gp']: #['soil_with_NP', 'soil_biochar_with_NP', 'soil_Fe_biochar_with_NP']
        metabolites, soil_impo = [], []
        for eachMeta in data['Metabolites']:
            df = pd.read_csv(f'test_final_modeling/{eachGroup}/{eachMeta}/fi.csv')
            metabolites.append(eachMeta)
            x = df.iloc[2, 1:2].to_dict()
            
            for key, value in x.items():
                soil_impo.append(value)

        threshold = 0.020
        colors = ['red' if value > threshold else 'blue' for value in soil_impo]
        bars = plt.bar(metabolites, soil_impo, color=colors)
        above_threshold_patch = plt.Line2D([0], [0], color='red', lw=4, label=f'Importance score > {threshold}')
        below_threshold_patch = plt.Line2D([0], [0], color='blue', lw=4, label=f'Importance score <= {threshold}')
        plt.legend(handles=[above_threshold_patch, below_threshold_patch])

        #plt.bar(metabolites, soil_impo)
        plt.xlim(left=-1, right=len(metabolites))
        plt.xticks(rotation=90, fontsize=8)
        plt.xlabel("Metabolites")
        plt.ylabel("Importance score")
        plt.title(f"Importance of {eachGroup} when each metabolite becomes target")
        plt.tight_layout()
        plt.savefig(f'test_final_modeling/{eachGroup}/importance_.png', dpi=300)
        plt.show()
        plt.close()
        # exit()
        print(f"Task done for group {eachGroup}")


if __name__ == "__main__":
    Main()
    #Classification()
    #meta_soil_type()
