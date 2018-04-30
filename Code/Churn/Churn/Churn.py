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
    df_subspt, df_lesson = utility.iolib.retrieve_data(cfg)
    print("Complete loading data for subscription and lesson history!")

    # Distribution of subscription length per pupil
    utility.plotlib.subspt_dist_cancelled(df_subspt, cfg)

    print("Start preparing the time-series data for subscription.")
    df_subspt_timeseries = utility.df.subspt_timeseries(df_subspt, cfg)

    # Active subscriptions over time
    # Average residual subscription length per pupil
    utility.plotlib.active_subspt(df_subspt_timeseries, cfg)


    