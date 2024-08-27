# Author: Arjun S Kulathuvayal. Intellectual property. Copyright strictly restricted
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, re
from matplotlib.lines import Line2D
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 10000)
font = {'family': 'serif',
        'weight': 'normal',
        'size': 12}
plt.rc('font', **font)
plt.rcParams["figure.figsize"] = [18, 14]


def CorrelationHeatmap(data):
    data = data.drop(columns=['Metabolites', 'Metabolites_labels'])
    scaled_df = pd.DataFrame(MinMaxScaler().fit_transform(data), columns=data.columns)

    scaled_correlation_matrix = scaled_df.corr()
    print(scaled_correlation_matrix)
    plt.figure(figsize=(10, 8))
    sns.heatmap(scaled_correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.1)
    plt.title('Pearson Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f'Corr_Heatmap.png', dpi=300)
    plt.show()

def CovarianceHeatmap(data):
    T_data = data.transpose().reset_index()
    T_data.columns = ['Metabolites'] + data.Metabolites.to_list()
    T_data = T_data.drop(1, axis=0).reset_index(drop=True)
    T_data_content = T_data.iloc[1:, 1:]

    X = pd.DataFrame(StandardScaler().fit_transform(T_data_content), columns=T_data_content.columns)

    covariance_estimator = EmpiricalCovariance()
    covariance_estimator.fit(X)
    covariance_matrix = covariance_estimator.covariance_
    print(covariance_matrix)
    cor_mat = np.array(covariance_matrix)

    sns.heatmap(cor_mat, annot=False, cmap='coolwarm', vmin=-1, vmax=1, xticklabels=X.columns, yticklabels=X.columns)
    plt.title(f'Covariance Heatmap', fontsize=16)
    plt.tick_params(direction="in")
    plt.tight_layout()

    #plt.savefig(f'Corr_Heatmap.pdf')
    #plt.savefig(f'Cov_Heatmap.png', dpi=300)
    plt.show()
    #plt.close()


def skv(data):
    T_data = data.transpose().reset_index()
    T_data.columns = ['Metabolites']+ data.Metabolites.to_list()
    T_data = T_data.drop(1, axis=0).reset_index(drop=True)
    T_data_content = T_data.iloc[1:, 1:]
    scaled_df = pd.DataFrame(MinMaxScaler().fit_transform(T_data_content), columns=T_data_content.columns)
    scaled_df.to_csv('scaled.csv', index=False)

    plt.figure(figsize=(16, 6))
    sns.heatmap(scaled_df.to_numpy(), cmap='OrRd')
    label_dict = dict(zip(scaled_df.columns.to_list(), T_data.iloc[0, 1:].to_list()))
    plt.xticks(ticks=np.arange(0.5, 67.5, 1), labels=list(label_dict.keys()), rotation=90, fontsize=8, ha='center')

    i = 0
    for key, value in label_dict.items():
        if value == 0:
            plt.gca().get_xticklabels()[i].set_color("red")
        elif value == 1:
            plt.gca().get_xticklabels()[i].set_color("blue")
        elif value == 2:
            plt.gca().get_xticklabels()[i].set_color("green")
        elif value == 3:
            plt.gca().get_xticklabels()[i].set_color("orange")
        elif value == 4:
            plt.gca().get_xticklabels()[i].set_color("brown")
        elif value == 5:
            plt.gca().get_xticklabels()[i].set_color("black")
        elif value == 6:
            plt.gca().get_xticklabels()[i].set_color("cyan")
        i += 1

    plt.yticks(ticks=np.arange(0.5, 24.5), labels=T_data.drop(0, axis=0).Metabolites.to_list(), rotation=0, fontsize=8, ha='right')
    plt.title('Comparison relative abundance between type of soil and metabolites', fontsize=16)
    plt.xlabel('Metabolites')
    plt.ylabel('Soil sample')

    custom_lines = [Line2D([0], [0], color='r', marker='s', linestyle='', label='Amino Acid'),
                    Line2D([0], [0], color='b', marker='s', linestyle='', label='Antioxidant'),
                    Line2D([0], [0], color='g', marker='s', linestyle='', label='Fatty acids'),
                    Line2D([0], [0], color='orange', marker='s', linestyle='', label='Nucleobase/side/tide'),
                    Line2D([0], [0], color='brown', marker='s', linestyle='', label='Organic acid/phenolics'),
                    Line2D([0], [0], color='black', marker='s', linestyle='', label='Vitamin B'),
                    Line2D([0], [0], color='cyan', marker='s', linestyle='', label='Sugar/Sugar alcohol'),
                    ]
    plt.legend(handles=custom_lines, loc='upper center', bbox_to_anchor=(1.23, 0), ncol=1, fontsize='small')
    plt.tight_layout()
    plt.savefig(f'Comparison_soil_metabolites_Heatmap.png', dpi=300)
    plt.show()



def Main():
    main_df = pd.read_csv('master_data.csv')
    #df = main_df.drop()
    df = main_df.drop(columns=['Metabolites_labels']).replace(0, np.nan)
    df = pd.concat([main_df.Metabolites_labels, df], axis=1)
    df = df.dropna(ignore_index=True)
    #print(df)
    CovarianceHeatmap(df)
    # CorrelationHeatmap(df)
    skv(df)




if __name__ == "__main__":
    Main()
