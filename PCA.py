# Author: Arjun S Kulathuvayal. Intellectual property. Copyright strictly restricted
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, re
pd.set_option('display.max_columns', 9)
pd.set_option('display.width', 10000)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
font = {'family': 'serif',
        'weight': 'normal',
        'size': 12}
plt.rc('font', **font)
plt.rcParams["figure.figsize"] = [8, 6]
pd.set_option('display.max_columns', 6)
pd.set_option('display.width', 10000)
from sklearn.decomposition import PCA

def Main(data):
    T_data = data.transpose().reset_index()
    T_data.columns = ['Metabolites'] + data.Metabolites.to_list()
    T_data = T_data.drop(1, axis=0).reset_index(drop=True)
    T_data_content = T_data.iloc[1:, 1:]
    scaled_df = pd.DataFrame(MinMaxScaler().fit_transform(T_data_content), columns=T_data_content.columns)

    #print(scaled_df.transpose())
    new_df = scaled_df.transpose().reset_index()
    new_df.columns = data.drop(columns=['Metabolites_labels'], axis=1).columns.to_list()
    new_df['Metabolites_labels'] = data.Metabolites_labels.to_list()
    scaled_correlation_matrix = new_df.drop(columns=['Metabolites'], axis=1)

    #print(scaled_correlation_matrix)
    comparion_df = pd.DataFrame()
    for i, col in enumerate(scaled_correlation_matrix.drop(columns=['Metabolites_labels']).columns):
        s_number = re.findall(r'\d+', col)[0]
        if s_number.isdigit() and int(s_number) % 4 == 0:
            print(re.sub(r'\d', '', col[4:]))

            features = scaled_correlation_matrix.columns.to_list()
            X = scaled_correlation_matrix.iloc[:, i-3:i+1]
            X = StandardScaler().fit_transform(X)

            pca = PCA(n_components=4)
            sample_set_score = pca.fit_transform(X)
            PC_vars = [i * 100 for i in pca.explained_variance_ratio_][0:3]
            print(f"Explained variance in % - first PC : {PC_vars}")

            score = sample_set_score[:, 0:2]
            xs = score[:, 0]
            ys = score[:, 1]
            coeff = np.transpose(pca.components_[0:2, :])
            n = coeff.shape[0]
            scalex = 1.0 / (xs.max() - xs.min())
            scaley = 1.0 / (ys.max() - ys.min())
            plt.scatter(xs * scalex, ys * scaley, color="c", alpha=0.75, label="Scaled metabolites (ng/g)")
            plt.legend(loc="lower left")
            plt.xlabel(f"First Principal Component ({PC_vars[0]:.2f}%)")
            plt.ylabel(f"Second Principal Component ({PC_vars[1]:.2f}%)")
            plt.title(f"PC space includes {PC_vars[0]+PC_vars[1]:.2f}% variance of four samples of\n '{col[4:].replace('_', ' ')}'", pad=15)
            plt.ylim(-1, 1)
            plt.xlim(-1, 1)
            for i in range(n):
                plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='b', alpha=0.5, head_width=0.03, head_length=0.04, label='Sample')
                plt.text(coeff[i, 0] + 0.3, coeff[i, 1], f"{i + 1}) {features[i]}", color='black', ha='center',
                         va='center', fontsize=12)
            plt.tight_layout()
            plt.savefig(f"PCA/{col[4:].replace('_', '')}.png")
            plt.close()

            PC_info = pd.Series(sample_set_score[:, 0:1].flatten())

            comparion_df = pd.concat([comparion_df, PC_info.to_frame(name=re.sub(r'\d', '', col[4:]))], axis=1)

    print(comparion_df.corr())

    sns.heatmap(comparion_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, xticklabels=comparion_df.columns, yticklabels=comparion_df.columns)
    plt.title(f'Correlation heatmap \n first PC of each sample has considered', fontsize=16, pad=20)
    plt.tick_params(direction="in")
    plt.tight_layout()

    #plt.savefig(f'Corr_Heatmap.pdf')
    plt.savefig(f'Person_corr_coeff_bw_samples.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main_df = pd.read_csv('master_data.csv')
    df = main_df.drop(columns=['Metabolites_labels']).replace(0, np.nan)
    df = pd.concat([main_df.Metabolites_labels, df], axis=1)
    df = df.dropna()
    Main(df)
