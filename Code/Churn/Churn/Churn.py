import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
import utility.iolib
import utility.plotlib
import utility.df

if __name__ == "__main__":

    # Setup configuration
    cfg = config.ResearchConfig
    time_format = cfg.CSV_TIME_FORMAT
    date_format = cfg.CSV_DATE_FORMAT

    # Retrieve data
    df_subspt, df_lesson, df_incomp, df_crclum = utility.iolib.retrieve_data(cfg)
    print("Complete loading data for subscription and lesson history!")


    start_date = pd.to_datetime('2016-02-01')
    end_date = pd.to_datetime('2016-03-01')
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    subspt_type = 'Monthly'
    p_term = 24
    p_unit = 'M'

    survival_counts = np.zeros([len(dates), p_term])

    i = 0
    for date in dates:
        survival = utility.df.subspt_survival(df_subspt, subspt_type, date, p_term, p_unit)
        survival_counts[i,:] = np.array(survival.survival_count)
        i = i + 1

    utility.plotlib.survival(survival_counts, cfg)
    # Distribution of subscription length per pupil
    #utility.plotlib.subspt_dist_cancelled(df_subspt, cfg)

    #print("Start preparing the time-series data for subscription.")
    #df_subspt_timeseries = utility.df.subspt_timeseries(df_subspt, cfg)

    # Active subscriptions over time
    # Average residual subscription length per pupil
    #utility.plotlib.active_subspt(df_subspt_timeseries, cfg)


    