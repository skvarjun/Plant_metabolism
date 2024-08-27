# Author: Arjun S Kulathuvayal. Intellectual property. Copyright strictly restricted
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
pd.set_option('display.max_columns', 9)
pd.set_option('display.width', 10000)


class MasterData:
    def __init__(self):
        self.scaled_data = pd.read_csv('data/scaled.csv')
        self.master_data = pd.read_csv('data/master_data.csv')
        self.np_data = pd.read_csv('data/nano_.csv')
        self.fdw = [2.46, 4.12, 1.97, 2.33, 2.23, 4.13, 3.13, 3.82, 3.41, 4.71, 4.38, 4.49, 4.25, 3.96, 2.17, 4.62, 3.35, 4.93, 3.13, 2.44, 4.19, 3.80, 3.30, 2.35]
        self.leaf_nos = [12.00, 12.00, 12.00, 17.00, 10.00, 13.00, 14.00, 12.00, 11.00, 17.00, 13.00, 16.00, 16.00, 11.00, 8.00, 15.00, 12.00, 14.00, 12.00, 10.00, 12.00, 12.00, 11.00, 10.00 ]
        self.leaf_wt = [2.90, 4.39, 2.27, 2.66, 2.57, 4.47, 3.41, 4.19, 4.00, 5.08, 4.73, 4.82, 4.64, 4.26, 2.43, 5.12, 3.65, 5.20, 3.54, 2.86, 4.55, 4.22, 3.58, 2.57]
        self.root_wt = [0.53, 1.15, 0.57, 0.70, 0.23, 1.15, 0.68, 3.02, 0.49, 0.87, 1.04, 0.74, 1.25, 0.94, 0.676, 0.72, 0.48, 1.39, 0.69, 0.61, 0.59, 1.00, 0.76, 0.29]



    def main_data(self, drop_labels=False):
        master_data_ = self.master_data.drop(columns=['Metabolites_labels']).replace(0, np.nan).dropna().reset_index()
        met_lables = []
        for indices in master_data_['index']:
            m = self.master_data.iloc[indices, :].to_frame().transpose()
            met_lables.append(int(m['Metabolites_labels'].to_list()[0]))

        new_data = self.scaled_data.transpose().reset_index()
        if not drop_labels:
            new_data.insert(loc=0, column='Metabolites_labels', value=met_lables)
            new_data.columns = self.master_data.columns
            return new_data
        else:
            return new_data


    def scale(self, data):
        T_data = data.transpose().reset_index()
        T_data.columns = ['Metabolites'] + data.Metabolites.to_list()
        T_data = T_data.drop(1, axis=0).reset_index(drop=True)
        T_data_content = T_data.iloc[1:, 1:]
        scaled_df = pd.DataFrame(MinMaxScaler().fit_transform(T_data_content), columns=T_data_content.columns)
        return scaled_df


    def OHE(self, df):
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        encoded = one_hot_encoder.fit_transform(df[['Metabolites']])
        encoded_df = pd.DataFrame(encoded, columns=one_hot_encoder.get_feature_names_out(['Metabolites']))
        df = pd.concat([df, encoded_df], axis=1)
        df.to_csv('test.csv', index=False)
        return df


    def unscaled(self, ):
        df_ = self.master_data.iloc[:, 1:].replace(0, np.nan)
        df_ = pd.concat([self.master_data['Metabolites_labels'], df_], axis=1).dropna().reset_index(drop=True)
        fdw_ = pd.DataFrame({'fdw': self.fdw, 'leaf_nos': self.leaf_nos, 'leaf_wt': self.leaf_wt, 'root_wt': self.root_wt})
        np_ = self.np_data
        return df_, fdw_, np_

    def make_transpose(self, df):
        pass








if __name__ == '__main__':
    pass
