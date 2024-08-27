import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as mpatches
import pandas as pd
from data import MasterData
font = {'family': 'serif',
        'weight': 'normal',
        'size': 18}
plt.rc('font', **font)
plt.rcParams["figure.figsize"] = [9, 6]
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 10000)
import seaborn as sns




def nanoplasticContent():
    df_ = pd.read_csv('../nano_.csv')
    df = df_.dropna()
    df = df.transpose()
    df = df.iloc[1:, :]
    names = df_.dropna()
    df.columns = names['nanopastic_info'].to_list()
    df = df.reset_index(drop=True)
    return df


def func(a, b, c, x_list):
    y = []
    for eachX in x_list:
        y_val = a*eachX**2 + b*eachX + c
        y.append(y_val)
    return y


def met_data(df):
    t_data = df.transpose()
    t_data.columns = df['Metabolites'].to_list()
    t_data = t_data.reset_index(drop=True)
    t_data = t_data.drop(1).reset_index(drop=True)
    return t_data



def new_fig(df, nano, metabolite):
    x = df.to_frame().reset_index()
    x['meta_Group'] = x.index // 4
    x.columns = ['soil_type', 'wt_metabolite', 'meta_Group']
    meta_avg_df = x.groupby('meta_Group')['wt_metabolite'].mean().reset_index()
    meta_std_df = x.groupby('meta_Group')['wt_metabolite'].std().reset_index()
    meta_df = pd.concat([meta_avg_df, meta_std_df], axis=1)



    nano_df = pd.DataFrame({'np_in_leaf':nano})
    nano_df['np_Group'] = nano_df.index // 4
    nano_avg_df = nano_df.groupby('np_Group')['np_in_leaf'].mean().reset_index()
    nano_std_df = nano_df.groupby('np_Group')['np_in_leaf'].std().reset_index()
    nano_df = pd.concat([nano_avg_df, nano_std_df], axis=1)

    comb_df = pd.concat([meta_df, nano_df], axis=1)
    comb_df.columns = ['meta_Group',  'avg_wt_metabolite',  'meta_Group_',  'std_wt_metabolite',  'np_Group',  'avg_np_in_leaf',  'np_Group_',  'std_np_in_leaf']
    #print(comb_df)


    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twiny()
    y_pos = np.arange(len(soil_types))
    ax1.barh(y_pos, comb_df['avg_wt_metabolite'], xerr=comb_df['std_wt_metabolite'], color='skyblue', edgecolor='black', height=0.4, align='center')
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax2.barh(y_pos, comb_df['avg_np_in_leaf'], xerr=comb_df['std_np_in_leaf'], color='red', height=0.2, align='center')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([str(i).replace("_", " ") for i in soil_types])
    ax1.set_ylabel('Soil types')
    ax1.set_xlabel(f'Weight of metabolite {metabolite} (ng/g)')
    ax2.set_xlabel('Weight of nanoplastic content in leaf (ng)')
    ax1.set_title(f'{metabolite}', color='brown')
    red_patch = mpatches.Patch(color='skyblue', label='Metabolite weight')
    blue_patch = mpatches.Patch(color='red', label='Nanoplastic content')
    plt.legend(handles=[red_patch, blue_patch], loc='center right', ncol=1, )  # bbox_to_anchor=(1., 0.5)

    plt.tight_layout()
    plt.savefig(f"Analysis/{metabolite}/avg_std_comparison.png", dpi=300)
    plt.savefig(f"Analysis/{metabolite}/avg_std_comparison.pdf")
    #plt.show()
    plt.close()


def Main():
    data, plant_ftrs = MasterData().unscaled()
    t_data = met_data(data)
    t_data.insert(0, 'Soil_type', [''] + data.drop(['Metabolites_labels', 'Metabolites'], axis=1).columns.to_list())
    main_df = t_data.drop(0).reset_index(drop=True)
    nano_content = nanoplasticContent()
    main_df.insert(1, 'nanoplastic_in_leaf', nano_content['total_nanoplastic_in_leaf_(ng)'])

    new_df = main_df.transpose()
    new_df.columns = new_df.iloc[0]
    new_df = new_df.iloc[1:, 0:].reset_index()
    new_df = pd.concat([new_df['index'], new_df.iloc[:, 1:]], axis=1)
    # print(new_df)

    for index, row in new_df.iloc[1:, :].iterrows():
        metabolite = row.iloc[0]
        #os.makedirs(f"Analysis/{metabolite}", exist_ok=True)
        data = row.iloc[1:, ].to_frame().reset_index()
        data.columns = ['soil_type', metabolite]
        data = pd.concat([data, main_df['nanoplastic_in_leaf']], axis=1)
        #data[metabolite] = MinMaxScaler().fit_transform(data[[metabolite]])
        # xs = [0, 1, 2, 3]
        # ys = [1.1, 3.9, 11.2, 21.5]
        # f = np.poly1d(np.polyfit(xs, ys, deg=3))
        # print(f[0], f[1], f[2])
        # print(f)
        #print(data)
        # exit()

        chunks = [data.iloc[i:i + 4] for i in range(0, len(data), 4)]
        ana_df = pd.DataFrame(columns=['avg_nano_content', f'avg_{metabolite}_weight', 'soil_type', 'log_of_diff_length', 'normalized_value'])
        for i, chunk in enumerate(chunks):
            df = chunk.reset_index(drop=True)

            if "soil_alone" in str(df['soil_type'][0]):

                magic_num = np.log(np.abs(np.average(df[metabolite].to_list()))) - np.log(np.average(df['nanoplastic_in_leaf'].to_list()))
            subtraction = np.log(np.abs(np.average(df[metabolite].to_list()))) - np.log(np.average(df['nanoplastic_in_leaf'].to_list()))
            normalization = subtraction/magic_num

            ana_df.loc[len(ana_df)] = [np.average(df['nanoplastic_in_leaf'].to_list()), np.average(df[metabolite].to_list()), soil_types[i], subtraction,  normalization]
        print(ana_df)
        exit()
        ana_df.to_csv(f"Analysis/{metabolite}/analysis.csv", index=False)

        x = row.iloc[1:]
        y_values = x.index.to_list()
        x_values = row.iloc[1:].to_list()
        z_values = main_df['nanoplastic_in_leaf'].to_list()
        new_fig(x, z_values, metabolite)


        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax2 = ax1.twiny()
        y_pos = np.arange(len(y_values))
        ax1.barh(y_pos, x_values, color='skyblue', edgecolor='black', height=0.4, align='center')
        ax1.set_ylim(-0.5, 23.5)
        ax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax2.barh(y_pos, z_values, color='red', height=0.2, align='center')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([str(i).replace("_", " ") for i in y_values])
        ax1.set_ylabel('Soil types')
        ax1.set_xlabel(f'Weight of metabolite {metabolite} (ng/g)')
        ax2.set_xlabel('Weight of nanoplastic content in leaf (ng)')
        ax1.set_title(f'{metabolite}', color='brown')
        red_patch = mpatches.Patch(color='skyblue', label='Metabolite weight')
        blue_patch = mpatches.Patch(color='red', label='Nanoplastic content')
        plt.legend(handles=[red_patch, blue_patch], loc='center right', ncol=1, ) #bbox_to_anchor=(1., 0.5)
        plt.tight_layout()
        plt.savefig(f"Analysis/{metabolite}/comparison.png", dpi=300)
        plt.savefig(f"Analysis/{metabolite}/comparison.pdf")
        plt.show()
        #plt.close()
        print(f"Task done for {metabolite}")
        exit()



def Analysis():
    for j, eachCat in enumerate(metanolites_cat):
        data, plant_ftrs = MasterData().unscaled()
        data = data[data['Metabolites_labels'] == j]
        Df = pd.DataFrame()
        for i, eachMeta in enumerate(data['Metabolites'].to_list()):
            da = pd.read_csv(f"Analysis/{eachMeta}/analysis.csv")
            Df.insert(loc=i, column=eachMeta, value=da['normalized_value'])

        plt.figure(figsize=(12, 6))
        sns.heatmap(Df, cmap='viridis', annot=False, fmt='.2f', linewidths=.5)
        plt.title(f'Metabolite category {str(eachCat.replace("_", " "))}', color='r')
        plt.xlabel('Metabolites')
        plt.yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], [str(i).replace("_", " ") for i in soil_types], rotation=0)
        plt.ylabel('Normalized with pure soil')
        plt.tight_layout()
        plt.savefig(f"{eachCat}_result.png", dpi=300)
        plt.close()





if __name__ == "__main__":
    soil_types = ['soil_alone', 'soil_nanoplatic', 'soil_biochar', 'soil_biochar_nanoplastic', 'soil_Fe-biochar',
                  'soil_Fe-biochar_nanoplastic']
    metanolites_cat = ['Amino_acid', 'Antioxidant', 'Fatty_acids',
                       'Nucleobase_or_side_or_tide', 'Organic_acid_or_phenolics', 'Vitamin_B', 'Sugar_or_Sugar_alcohol']
    Main()
    #Analysis()
