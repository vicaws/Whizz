import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
import utility.iolib
import utility.plotlib

if __name__ == "__main__":

    # Setup configuration
    cfg = config.ResearchConfig
    time_format = cfg.CSV_TIME_FORMAT
    date_format = cfg.CSV_DATE_FORMAT

    # Retrieve data
    df_subspt, df_lesson = utility.iolib.retrieve_data(cfg)
    print("Complete loading data for subscription and lesson history!")

    # Distribution of subscription length per pupil
    utility.plotlib.subscpt_dist(df_subspt, cfg)