import matplotlib.pyplot as plt
import numpy as np
import os

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from data import MasterData
from lolopy.learners import RandomForestRegressor as LoLoRF
from sklearn.ensemble import RandomForestRegressor as SKRF


font = {'family': 'serif',
        'weight': 'normal',
        'size': 12}
plt.rc('font', **font)
plt.rcParams["figure.figsize"] = [8, 6]


def influence_of_metabolites():
    data = MasterData().unscaled().drop(['Metabolites', 'Metabolites_labels'], axis=1)
    samples = data.columns.to_list()
    for i in range(0, len(samples), 4):
        chunk = samples[i:i + 4]
        for eachSoilSample in chunk:
            # print(data[eachSoilSample])
            # print(data.drop(columns=chunk, axis=1))
            # exit()
            y = data[eachSoilSample].to_numpy()
            X = data.drop(columns=chunk, axis=1).to_numpy()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LoLoRF()
            model.fit(X_train, y_train)
            LoLo_y_pred, LoLo_y_uncer = model.predict(X_test, return_std=True)
            RFr2 = r2_score(y_test, LoLo_y_pred)
            RFmse = mean_squared_error(y_test, LoLo_y_pred)
            RFmae = mean_absolute_error(y_test, LoLo_y_pred)

            result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=200)
            feature_names = data.drop(columns=chunk, axis=1).columns.tolist()
            fi = pd.DataFrame(data={'feature': feature_names, 'importances': result.importances_mean, 'std_dev': result.importances_std})

            plt.rcParams["figure.figsize"] = [8, 6.5]
            fig, axs = plt.subplots(1, 2)
            axs[0].errorbar(y_test, LoLo_y_pred, LoLo_y_uncer, fmt='o', ms=2.5, ecolor='grey')
            axs[0].set_xlabel("Actual weight of metabolites (ng/g)")
            axs[0].set_ylabel("Predicted weight of metabolites (ng/g)")
            axs[0].set_title("Prediction performance")
            axs[0].plot(y_test, y_test, ':', lw=1, color='black')
            axs[0].legend([f'Test size : {len(y_test)}', f'Train size : {len(y_train)}'], markerscale=0, handlelength=0, loc='upper left')

            colors = ['red', 'blue', 'green', 'purple', 'orange']
            bar_colors = []
            for i in range(len(fi['feature'])):
                bar_colors.append(colors[i // 4])

            axs[1].bar(fi['feature'], fi['importances'], yerr=fi['std_dev'], capsize=5, alpha=0.7, color=bar_colors)
            axs[1].set_ylabel("Mean accuracy decrease")
            axs[1].set_xlabel("Soil samples")
            axs[1].set_title("Feature importance")
            axs[1].tick_params(axis='x', labelrotation=90, labelsize=8)
            fig.suptitle(f"Sample: {eachSoilSample}\nR2: {RFr2:.2f} | MSE: {RFmse:.4e} | MAE: {RFmae:.4e}")
            fig.subplots_adjust(top=0.9)
            fig.tight_layout()
            os.makedirs(f"soil_target/{eachSoilSample}", exist_ok=True)
            plt.savefig(f'soil_target/{eachSoilSample}/results.png', dpi=300)
            plt.show()
            plt.close()
            exit()






if __name__ == "__main__":
    influence_of_metabolites()
