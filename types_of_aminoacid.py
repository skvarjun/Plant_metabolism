# Author: Arjun S Kulathuvayal. Intellectual property. Copyright strictly restricted
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lolopy.learners import RandomForestRegressor as LoLoRF
import seaborn as sns
plt.rcParams["figure.figsize"] = [7, 6]

from data import MasterData

def Classi_Met_cat(df):
    df = df.drop('Metabolites', axis=1)

    y = df['Metabolites_labels'].to_numpy()
    X = df.drop('Metabolites_labels', axis=1).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LoLoRF()
    model.fit(X_train, y_train)


    # cv_prediction = cross_val_predict(model, X_test, y_test, cv=KFold(5, shuffle=True))

    LoLo_y_pred, LoLo_y_uncer = model.predict(X_test, return_std=True)
    LoLo_r2 = r2_score(y_test, LoLo_y_pred)
    y_resid = LoLo_y_pred - y_test

    LoLo_mse = mean_squared_error(y_test, -LoLo_y_pred)
    LoLo_mae = mean_absolute_error(y_test, LoLo_y_pred)





    fig, axs = plt.subplots(1, 2)
    # axs[0].hist2d(pd.to_numeric(y_test), cv_prediction, norm=LogNorm(), bins=64, cmap='Blues', alpha=0.9)
    # axs[0].scatter(pd.to_numeric(y_test), LoLo_y_pred, s=5)
    axs[0].errorbar(y_test, LoLo_y_pred, LoLo_y_uncer, fmt='o', ms=2.5, ecolor='gray')
    axs[0].set_xlabel(r'Actual diffusion coefficient $(m^2/s)$')
    axs[0].set_ylabel(r'Predicted diffusion coefficient $(m^2/s)$')
    axs[0].plot(y_test, y_test, '--', color='gray')
    axs[0].set_title('RF Performance : R2 = {:.2g}, MSE = {:.3g}'.format(LoLo_r2, LoLo_mse),
                     fontsize=10, pad=15)
    axs[0].legend([f'Test size : {len(y_test)}', f'Train size : {len(y_train)}'], markerscale=0, handlelength=0,
                  loc='upper left')

    x = np.linspace(-6, 6, 50)
    conv_resid = np.divide(y_resid, np.sqrt(np.power(y_resid, 2).mean()))
    axs[1].hist(conv_resid, x,
                density=True)  # conventional cross validation error => constant error (MSE) estimation for all points
    axs[1].plot(x, norm.pdf(x), 'k--',
                lw=1.00)  # probability density function (pdf) of the normal distribution for data points x
    axs[1].set_title('Uncertainty', fontsize=10, pad=15)
    axs[1].set_ylabel('Probability Density')
    axs[1].set_xlabel('Normalized residual')
    fig.suptitle(f'Training output')
    fig.subplots_adjust(top=0.9)
    fig.tight_layout()
    plt.savefig(f'rf_train.png')
    plt.close()
    return LoLo_r2




def Corre_Met_cat(df):
    df = df.drop('Metabolites', axis=1)
    metanolites_cat = ['Amino_acid', 'Antioxidant', 'Fatty_acids',
                       'Nucleobase_or_side_or_tide', 'Organic_acid_or_phenolics', 'Vitamin_B', 'Sugar_or_Sugar_alcohol']
    samples = ['Soil_alone', 'Soil_nanoplatic', 'soil_biochar', 'Soil_biochar_nanoplastic', 'Soil_Fe-biochar',
               'Soil_Fe-biochar_nanoplastic']
    for cat, eachCat in enumerate(metanolites_cat):
        ddd = 6
        df = df.loc[(df[['Metabolites_labels']] == ddd).any(axis=1)]
        comparison_df = pd.DataFrame()
        for i, j in enumerate(np.arange(1, 24, 4)):
            sample_data = df.iloc[:, j:j + 4]
            X = StandardScaler().fit_transform(sample_data)
            pca = PCA(n_components=4)
            sample_set_score = pca.fit_transform(X)  # data in thePC space
            PC_vars = [k * 100 for k in pca.explained_variance_ratio_][0:4]
            print(f"Explained variance in % - first PC for  : {PC_vars}")

            PC_info = pd.Series(sample_set_score[:, 0:1].flatten())
            comparison_df = pd.concat([comparison_df, PC_info.to_frame(name=samples[i])], axis=1)

        print(comparison_df.corr())

        sns.heatmap(comparison_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                    xticklabels=comparison_df.columns,
                    yticklabels=comparison_df.columns)
        plt.title(f"Correlation heatmap \n'{metanolites_cat[ddd].replace('_', ' ')}' as metabolites", fontsize=16, pad=20)
        plt.tick_params(direction="in")
        plt.tight_layout()
        plt.savefig(f'pca/eachAminoacid/{metanolites_cat[ddd]}.png', dpi=300)
        plt.show()
        exit()






def pc_analysis(df):
    df = df.drop('Metabolites', axis=1)

    samples = ['Soil_alone', 'Soil_nanoplatic', 'soil_biochar', 'Soil_biochar_nanoplastic', 'Soil_Fe-biochar', 'Soil_Fe-biochar_nanoplastic']
    comparison_df = pd.DataFrame()
    for j, i in enumerate(np.arange(1, 24, 4)):
        sample_data = df.iloc[:, i:i+4]
        X = StandardScaler().fit_transform(sample_data)
        #X = sample_data
        pca = PCA(n_components=4)
        sample_set_score = pca.fit_transform(X) #data in thePC space
        PC_vars = [i * 100 for i in pca.explained_variance_ratio_][0:4]
        print(f"Explained variance in % - first PC for  : {PC_vars}")

        # coeff = np.transpose(pca.components_[0:2, :])
        # pc1_x, pc2_y = coeff[3, 0], coeff[3, 1] #the third feature has respective component in PC1 AND PC2


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
        plt.title(f"PC space includes {PC_vars[0] + PC_vars[1]:.2f}% variance of four samples of\n '{samples[j].replace('_', ' ')}'", pad=15)
        plt.ylim(-1, 1)
        plt.xlim(-1, 1)
        for i in range(n):
            plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='b', alpha=0.5, head_width=0.03, head_length=0.04, label='Sample')
            plt.text(coeff[i, 0] + 0.3, coeff[i, 1], f"{i + 1}) {sample_data.columns.to_list()[i]}", color='black', ha='center', va='center', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"pca/{samples[j]}.png")
        plt.close()



        PC_info = pd.Series(sample_set_score[:, 0:1].flatten())
        comparison_df = pd.concat([comparison_df, PC_info.to_frame(name=samples[j])], axis=1)

    print(comparison_df)
    # plt.scatter(comparison_df.Soil_alone, comparison_df['Soil_Fe-biochar_nanoplastic'], c='r')
    # plt.scatter(comparison_df.Soil_alone, comparison_df['Soil_nanoplatic'], c='b')
    # plt.scatter(comparison_df.Soil_alone, comparison_df['Soil_Fe-biochar'], c='g')

    x = [2, 4, 6, 8, 10, 12]
    y = [0.3, 2.1, 5.1, 7.6, 9.3, 11.7]
    #y = [11.7, 9.3, 7.6, 5.1, 2.1, 0.3]
    plt.scatter(x, y)

    print(np.corrcoef(comparison_df.Soil_alone, comparison_df['Soil_Fe-biochar_nanoplastic']))
    plt.show()
    exit()


    ax = sns.heatmap(comparison_df.corr(), annot=True, mask=np.triu(np.ones_like(comparison_df.corr(), dtype=bool)), cmap='coolwarm', vmin=-1, vmax=1, xticklabels=comparison_df.columns.to_list()[0:5],
                yticklabels=comparison_df.columns)
    new_yticks = np.delete(ax.get_yticks(), 0)
    new_yticklabels = np.delete(ax.get_yticklabels(), 0)
    ax.set_yticks(new_yticks)
    ax.set_yticklabels(new_yticklabels)
    plt.title(f'Correlation heatmap \n first PC of each sample has considered', fontsize=16, pad=20)
    plt.tick_params(direction="in")
    plt.tight_layout()
    plt.savefig(f'pca/Person_corr_coeff_bw_samples.png', dpi=300)


    plt.show()


if __name__ == "__main__":
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
    pc_analysis(new_data)
    # classi(new_data)
    # Classi_Met_cat(new_data)


    # x = MasterData().main_data(drop_labels=False)
    # num_unique_elements = x['Metabolites_labels'].value_counts()
    # print(num_unique_elements)
