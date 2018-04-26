import numpy as np
import pandas as pd

def subspt_timeseries(df_subspt, configuration):
    
    # Identify the dates range for study
    last_date = df_subspt.subscription_end_date.max()
    first_date = df_subspt.subscription_start_date.min()
    dates = list(pd.date_range(start=first_date, end=last_date, freq='D'))
    
    num_subspt = np.zeros(len(dates))
    num_subspt_monthly = np.zeros(len(dates))
    num_subspt_annual = np.zeros(len(dates))
    res_subspt_length = np.zeros(len(dates))
    
    temp = df_subspt.groupby(['subscription_start_date', 'subscription_length']).count()
    for row in temp.itertuples(index=True):
        start_date, length = row.Index
        i_start = dates.index(start_date)
    
        # Number of subscriptions
        if length > 31:
            num_subspt_annual[i_start : i_start+length+1] = \
            num_subspt_annual[i_start : i_start+length+1] + 1*row.subscription_type
        else:
            num_subspt_monthly[i_start : i_start+length+1] = \
            num_subspt_monthly[i_start : i_start+length+1] + 1*row.subscription_type
        # Residual subscrption length
        res_subspt_length[i_start : i_start+length+1] = \
        res_subspt_length[i_start : i_start+length+1] + np.arange(length+1,0,-1)*row.subscription_type

    num_subspt = num_subspt_monthly + num_subspt_annual
    res_subspt_length = res_subspt_length / num_subspt
    df_subspt_timeseries = pd.DataFrame({'num_subscriptions': num_subspt,
                                    'num_subscriptions_monthly': num_subspt_monthly,
                                    'num_subscriptions_annual': num_subspt_annual,
                                    'res_subscriptions_length': res_subspt_length}, index=dates)

    return df_subspt_timeseries