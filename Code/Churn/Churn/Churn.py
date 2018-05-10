import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
import utility.iolib
import utility.plotlib
import utility.df


def test_dates_frame():
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

    pupils_toBeRemoved = utility.df.filter_subspt_data(df_subspt, first_date_impFromData, cutoff_date, remove_annual_subspt=False)
    df_lesson1 = df_lesson[~df_lesson['pupilId'].isin(pupils_toBeRemoved)]
    df_incomp1 = df_incomp[~df_incomp['pupilId'].isin(pupils_toBeRemoved)]
    df_subspt1 = df_subspt[~df_subspt['pupilId'].isin(pupils_toBeRemoved)]

    df_datesFrame = utility.df.construct_dates_frame(df_lesson1, df_incomp1)
    df_subspt1 = utility.df.compute_customer_month(df_subspt1, cfg)
    df_datesFrame = utility.df.assign_customer_month(df_subspt1, df_datesFrame, cfg)
    pass

if __name__ == "__main__":
    test_dates_frame()