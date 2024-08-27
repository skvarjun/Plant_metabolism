# Author: Arjun S Kulathuvayal. Intellectual property. Copyright strictly restricted
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, re


def make_transpose(df, header_col_index=0, first_col="no_name"):
    T_df = df.transpose().reset_index()
    T_df.columns = [first_col]+df.iloc[0:, header_col_index].to_list()
    T_df = T_df.drop(0).reset_index(drop=True)
    return T_df

