# Author: Arjun S Kulathuvayal. Intellectual property. Copyright strictly restricted
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.api as sm
from data import MasterData
from tools import make_transpose
font = {'family': 'serif',
        'weight': 'normal',
        'size': 12}
plt.rc('font', **font)
plt.rcParams["figure.figsize"] = [8, 12]
from scipy import stats
import itertools
import seaborn as sns


def do_LSD_Test(df, metabolite):
    reshaped_data = df[metabolite].values.reshape(-1, 4)
    data = pd.DataFrame(reshaped_data.T, columns=soil_types)
    #print(data)

    data_new = pd.melt(data.reset_index(), id_vars=['index'], value_vars=soil_types)
    data_new.columns = ['index', 'treatment', 'value']
    data_new['value'] = pd.to_numeric(data_new['value'], errors='coerce')
    #print(data_new)

    model = ols('value ~ C(treatment)', data=data_new).fit()
    anova_table = sm.stats.anova_lm(model, typ=1)
    #print(anova_table)

    #Fisher’s least significant difference (LSD) method
    MSE = anova_table.iloc[1, 2]
    DF = anova_table.iloc[1, 0]
    alpha = 0.05
    t = 1 - alpha/2
    crit_val = stats.t.ppf(t, DF)
    LSD = crit_val*np.sqrt(2*MSE/4)

    rest = [item for item in soil_types if item != "soil_alone"]
    unique_pairs = [("soil_alone", item) for item in rest]

    soil_groups, status = [], []
    for eachGrp in unique_pairs:
        abs_avg_soil_alone = data['soil_alone'].mean()
        soil_sample = data[eachGrp[1]].mean()
        abs_of_diff_of_avg = np.abs(abs_avg_soil_alone - soil_sample)
        if abs_of_diff_of_avg > LSD:
            soil_groups.append(eachGrp)
            status.append(0)
            print(eachGrp, "Reject" )
        else:
            soil_groups.append(eachGrp)
            status.append(1)
            print(eachGrp, "Accept, similar" )

    return metabolite, status


def Main():
    metabolites_df, plant_df, nanoplastic_df = MasterData().unscaled()
    nano_df = make_transpose(nanoplastic_df, header_col_index=0, first_col="sample")
    meta_df = make_transpose(metabolites_df.drop(columns=['Metabolites_labels']), header_col_index=0, first_col="soil_types")

    metabolites, statuses = [], []
    for eachMeta in meta_df.iloc[:, 1:]:
        print(eachMeta)
        os.makedirs(f"analysis/{eachMeta}", exist_ok=True)
        data = pd.concat([meta_df.iloc[:, 0], meta_df[eachMeta], nano_df['total_nanoplastic_in_leaf_(ng)']], axis=1)
        metabolite, status = do_LSD_Test(data, eachMeta)
        metabolites.append(metabolite)
        statuses.append(status)

        chunks = [data.iloc[i:i + 4] for i in range(0, len(data), 4)]
        ana_df = pd.DataFrame(columns=['count',
                                       'sum_nano_content',
                                       'avg_nano_content',
                                       'variance_nano_content',
                                       'sum_meta_weight',
                                       'avg_meta_weight',
                                       'variance_meta_weight',
                                       'soil_type',
                                       'log_of_diff_length',
                                       'normalized_value'])
        for i, chunk in enumerate(chunks):
            df = chunk.reset_index(drop=True)
            if "soil_alone" in str(df['soil_types'][0]):
                magic_num = np.log(np.abs(np.average(df[eachMeta].to_list()))) - np.log(np.average(df['total_nanoplastic_in_leaf_(ng)'].to_list()))
            subtraction = np.log(np.abs(np.average(df[eachMeta].to_list()))) - np.log(np.average(df['total_nanoplastic_in_leaf_(ng)'].to_list()))
            normalization = subtraction / magic_num

            ana_df.loc[len(ana_df)] = [len(df),
                                       df['total_nanoplastic_in_leaf_(ng)'].sum(),
                                       df['total_nanoplastic_in_leaf_(ng)'].mean(),
                                       df['total_nanoplastic_in_leaf_(ng)'].var(),
                                       df[eachMeta].sum(),
                                       df[eachMeta].mean(),
                                       df[eachMeta].var(),
                                       soil_types[i],
                                       subtraction,
                                       normalization]
        #print(ana_df)
        ana_df.to_csv(f"analysis/{eachMeta}/analysis.csv", index=False)

    df = pd.DataFrame(statuses, index=metabolites, columns=['Soil alone &\nSoil + nanoplastic',
                                                            'Soil alone &\nSoil + biochar',
                                                            'Soil alone &\nSoil + biochar + nanoplastic',
                                                            'Soil alone &\nSoil + Fe-biochar',
                                                            'Soil alone &\nSoil + Fe-biochar + nanoplastic'
                                                            ])
    plt.figure(figsize=(10, 16))
    sns.heatmap(df, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90, fontsize=16)
    plt.tight_layout()
    plt.savefig("LSD_heatmap.pdf")
    plt.show()


if __name__ == "__main__":
    soil_types = ['soil_alone', 'soil_nanoplatic', 'soil_biochar', 'soil_biochar_nanoplastic', 'soil_Fe-biochar', 'soil_Fe-biochar_nanoplastic']
    metanolites_cat = ['Amino_acid', 'Antioxidant', 'Fatty_acids', 'Nucleobase_or_side_or_tide', 'Organic_acid_or_phenolics', 'Vitamin_B', 'Sugar_or_Sugar_alcohol']
    Main()
