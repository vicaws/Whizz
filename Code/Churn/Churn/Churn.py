import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
import utility.iolib
import utility.plotlib
import utility.df
from utility.feature import Feature
from utility.feature import FeatureCM
from model.dataEngine import DataEngine


def test():
    # Setup configuration
    cfg = config.ResearchConfig
    time_format = cfg.CSV_TIME_FORMAT
    date_format = cfg.CSV_DATE_FORMAT
    cutoff_date = pd.to_datetime(cfg.CUTOFF_DATE, format=cfg.CSV_DATE_FORMAT)

    # Retrieve data
    df_subspt, df_lesson, df_incomp, df_crclum, df_pupils = utility.iolib.retrieve_data(cfg)
    print("Complete loading data for subscription and lesson history!")

    # Filter data
    cutoff_date = pd.to_datetime(cfg.CUTOFF_DATE, format=cfg.CSV_DATE_FORMAT)
    first_date_impFromData = df_subspt.subscription_start_date.min()

    pupils_toBeRemoved = utility.df.filter_subspt_data(
        df_subspt, first_date_impFromData, cutoff_date, remove_annual_subspt=cfg.MONTHLY_ONLY)
    df_lesson1 = df_lesson[~df_lesson['pupilId'].isin(pupils_toBeRemoved)]
    df_incomp1 = df_incomp[~df_incomp['pupilId'].isin(pupils_toBeRemoved)]
    df_subspt1 = df_subspt[~df_subspt['pupilId'].isin(pupils_toBeRemoved)]

    df_subspt1 = utility.df.compute_customer_month(df_subspt1, cfg)

    # Construct dates frame
    df_datesFrame = utility.df.construct_dates_frame(df_subspt1, df_lesson1, df_incomp1, cfg)
    df_datesFrame.fillna(0, inplace=True)

    data_engine = DataEngine(df_subspt1, df_datesFrame, df_lesson1, df_incomp1, df_pupils, cfg)
    data_engine.aggregate_features([42])

    ftr_list = ['usage', 'usage_complete', 'usage_incomplete', 'rate_incomplete_usage',
                'last_access', 
                'age', 'math_age',
                'rate_assess', 'sum_help', 
                'progress', 'progress_delta',
                'num_attempt',  'num_complete', 'num_incomplete', 'rate_incomplete_num', 
                'num_pass', 'num_replay', 'num_fail',
                'holiday',
                'age_diff'
               ]
    ftr_list = ['usage', 'rate_incomplete_usage',
                'last_access', 
                'math_age',
                'rate_assess', 'sum_help', 
                'progress', 'progress_delta',
                'num_attempt',  'rate_incomplete_num', 
                'num_pass', 'num_replay', 'num_fail',
                'age_diff'
               ]
    data_engine.select_features(ftr_list)

    X = data_engine.data_
    y = data_engine.target_
    df_whizz1 = data_engine.df_whizz_

if __name__ == "__main__":
    test()