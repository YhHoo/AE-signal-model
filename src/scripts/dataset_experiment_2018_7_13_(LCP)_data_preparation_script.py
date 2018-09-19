import numpy as np
import pandas as pd
from os import listdir
# self lib
from src.utils.helpers import read_single_tdms, direct_to_dir, plot_multiple_timeseries, lollipop_plot

# all file name
tdms_dir = 'F:/Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/1 bar/Leak/'
all_tdms_filename = [f.split(sep='.')[0] for f in listdir(tdms_dir) if f.endswith('.tdms')]


lcp_dir = direct_to_dir(where='result') + 'lcp_1bar_near_segmentation_2.csv'
lcp_df = pd.read_csv(lcp_dir, index_col=0)

lcp_confident_cf = lcp_df.loc[lcp_df['confident LCP'] == 1]
lcp_confident_cf = lcp_confident_cf.drop(['contain other source'], axis=1)

