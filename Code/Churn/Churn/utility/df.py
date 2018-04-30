import numpy as np
import pandas as pd

def _remove_unsteady_data(df_subspt_timeseries):

    first_date = df_subspt_timeseries.index.min()
    threshold_date_mon = first_date + pd.to_timedelta(31, unit='D')
    threshold_date_ann = first_date + pd.to_timedelta(1, unit='Y')

    df_subspt_timeseries['num_ccl_mon'] = df_subspt_timeseries['num_ccl_mon'].mask(df_subspt_timeseries.index < threshold_date_mon, np.nan)
    df_subspt_timeseries['num_rnl_m2m'] = df_subspt_timeseries['num_rnl_m2m'].mask(df_subspt_timeseries.index < threshold_date_mon, np.nan)
    df_subspt_timeseries['num_rnl_m2a'] = df_subspt_timeseries['num_rnl_m2a'].mask(df_subspt_timeseries.index < threshold_date_mon, np.nan)
    df_subspt_timeseries['num_rtn_m2m'] = df_subspt_timeseries['num_rtn_m2m'].mask(df_subspt_timeseries.index < threshold_date_mon, np.nan)
    df_subspt_timeseries['num_rtn_m2a'] = df_subspt_timeseries['num_rtn_m2a'].mask(df_subspt_timeseries.index < threshold_date_mon, np.nan)
    #df_subspt_timeseries['num_new_mon'] = df_subspt_timeseries['num_new_mon'].mask(df_subspt_timeseries.index < threshold_date_mon, np.nan)

    df_subspt_timeseries['num_ccl_ann'] = df_subspt_timeseries['num_ccl_ann'].mask(df_subspt_timeseries.index < threshold_date_ann, np.nan)
    df_subspt_timeseries['num_rnl_a2a'] = df_subspt_timeseries['num_rnl_a2a'].mask(df_subspt_timeseries.index < threshold_date_ann, np.nan)
    df_subspt_timeseries['num_rnl_a2m'] = df_subspt_timeseries['num_rnl_a2m'].mask(df_subspt_timeseries.index < threshold_date_ann, np.nan)
    df_subspt_timeseries['num_rtn_a2a'] = df_subspt_timeseries['num_rtn_a2a'].mask(df_subspt_timeseries.index < threshold_date_ann, np.nan)
    df_subspt_timeseries['num_rtn_a2m'] = df_subspt_timeseries['num_rtn_a2m'].mask(df_subspt_timeseries.index < threshold_date_ann, np.nan)   
    #df_subspt_timeseries['num_new_ann'] = df_subspt_timeseries['num_new_ann'].mask(df_subspt_timeseries.index < threshold_date_ann, np.nan)

    df_subspt_timeseries['num_subscriptions_monthly'] = df_subspt_timeseries['num_subscriptions_monthly'].mask(df_subspt_timeseries.index < threshold_date_mon, np.nan)
    df_subspt_timeseries['num_subscriptions_annual'] = df_subspt_timeseries['num_subscriptions_annual'].mask(df_subspt_timeseries.index < threshold_date_ann, np.nan)
    df_subspt_timeseries['num_subscriptions'] = df_subspt_timeseries['num_subscriptions'].mask(df_subspt_timeseries.index < threshold_date_ann, np.nan)
    df_subspt_timeseries['res_subscriptions_length_mon'] = df_subspt_timeseries['res_subscriptions_length_mon'].mask(df_subspt_timeseries.index < threshold_date_mon, np.nan)
    df_subspt_timeseries['res_subscriptions_length_ann'] = df_subspt_timeseries['res_subscriptions_length_ann'].mask(df_subspt_timeseries.index < threshold_date_ann, np.nan)
    df_subspt_timeseries['res_subscriptions_length'] = df_subspt_timeseries['res_subscriptions_length'].mask(df_subspt_timeseries.index < threshold_date_ann, np.nan)

    print("Due to configuration, data as of unsteady period have been removed!")

    return df_subspt_timeseries

def subspt_timeseries(df_subspt, configuration): 
    
    # Identify the dates range for study
    last_date = df_subspt.subscription_end_date.max()
    first_date = df_subspt.subscription_start_date.min()
    dates = list(pd.date_range(start=first_date, end=last_date, freq='D'))
    
    ## Part-I: ACTIVE SUBSCRIPTIONS
    # Initialisation
    num_subspt_monthly = np.zeros(len(dates))   # number of active monthly subscriptions
    num_subspt_annual = np.zeros(len(dates))    # number of active annual subscriptions 
    num_subspt = np.zeros(len(dates))
    res_len_monthly = np.zeros(len(dates))      # average residual length of active monthly subscriptions
    res_len_annual = np.zeros(len(dates))       # average residual length of active annual subscriptions
    res_subspt_length = np.zeros(len(dates))    # average residual length of active subscriptions
    
    temp = df_subspt.groupby(['subscription_start_date', 'subscription_length']).count()
    for row in temp.itertuples(index=True):
        start_date, length = row.Index
        i_start = dates.index(start_date)
        
        if length > 31:
            # Number of subscriptions
            num_subspt_annual[i_start : i_start+length+1] = \
            num_subspt_annual[i_start : i_start+length+1] + 1*row.subscription_type
            # Residual subscrption length
            res_len_annual[i_start : i_start+length+1] = \
            res_len_annual[i_start : i_start+length+1] + np.arange(length+1,0,-1)*row.subscription_type
        else:
            # Number of subscriptions
            num_subspt_monthly[i_start : i_start+length+1] = \
            num_subspt_monthly[i_start : i_start+length+1] + 1*row.subscription_type
            # Residual subscrption length
            res_len_monthly[i_start : i_start+length+1] = \
            res_len_monthly[i_start : i_start+length+1] + np.arange(length+1,0,-1)*row.subscription_type
        
    num_subspt = num_subspt_monthly + num_subspt_annual
    res_subspt_length = res_len_monthly + res_len_annual
    # Handel zeros in data
    num_subspt_monthly = num_subspt_monthly.astype('float')
    num_subspt_monthly[num_subspt_monthly==0] = np.nan
    res_len_monthly = res_len_monthly / num_subspt_monthly
    res_len_annual = res_len_annual / num_subspt_annual
    res_subspt_length = res_subspt_length / num_subspt
    df_subspt_timeseries = pd.DataFrame({'num_subscriptions': num_subspt,
                                    'num_subscriptions_monthly': num_subspt_monthly,
                                    'num_subscriptions_annual': num_subspt_annual,
                                    'res_subscriptions_length_mon': res_len_monthly,
                                    'res_subscriptions_length_ann': res_len_annual,
                                    'res_subscriptions_length': res_subspt_length}, index=dates)

    # Part-II: COUNT OF CANCELLATIONS, RETURNS, RENEWALS AND NEW-SUBSCRIPTIONS
    # Initialise
    num_ccl_mon = np.zeros(len(dates)) # Number of cancellations from existing monthly subscription
    num_ccl_ann = np.zeros(len(dates)) # Number of cancellations from existing annually subscription
    num_rnl_m2m = np.zeros(len(dates)) # Number of renewals from monthly to monthly subscription 
    num_rnl_m2a = np.zeros(len(dates)) # Number of renewals from monthly to annually subscription
    num_rnl_a2m = np.zeros(len(dates)) # Number of renewals from annually to monthly subscription
    num_rnl_a2a = np.zeros(len(dates)) # Number of renewals from annually to annually subscription
    num_rtn_m2m = np.zeros(len(dates)) # Number of returns from monthly to monthly subscription 
    num_rtn_m2a = np.zeros(len(dates)) # Number of returns from monthly to annually subscription
    num_rtn_a2m = np.zeros(len(dates)) # Number of returns from annually to monthly subscription
    num_rtn_a2a = np.zeros(len(dates)) # Number of returns from annually to annually subscription
    num_new_mon = np.zeros(len(dates)) # Number of new monthly subscriptions
    num_new_ann = np.zeros(len(dates)) # Number of new annually subscription

    return_gap = pd.to_timedelta(configuration.RETURN_GAP, unit=configuration.RETURN_GAP_UNIT)

    pupil_list = df_subspt.pupilId.unique()
    for pupil in pupil_list:
        df_pupil = df_subspt[df_subspt.pupilId==pupil]
    
        # Sort by subscription start date
        df_pupil.set_index('subscription_start_date', inplace=True)
        df_pupil.sort_index(inplace=True)
        num_records = df_pupil.shape[0]
    
        # Count new subscriptions
        i_update_new = dates.index(df_pupil.index[0])
        if df_pupil.iloc[0].loc['subscription_type'] == configuration.TYPE_MONTHLY:
            num_new_mon[i_update_new] = num_new_mon[i_update_new] + 1
        else:
            num_new_ann[i_update_new] = num_new_ann[i_update_new] + 1

        # Count renewals and returns
        if num_records == 1:
            # if there is only 1 record found, then there is no renewal, but 1 cancellation            
            i_update_ccl = dates.index(df_pupil.iloc[0].loc['subscription_end_date'])
            if df_pupil.iloc[0].loc['subscription_type'] == configuration.TYPE_MONTHLY:
                num_ccl_mon[i_update_ccl] = num_ccl_mon[i_update_ccl] + 1
            else:
                num_ccl_ann[i_update_ccl] = num_ccl_ann[i_update_ccl] + 1
        else:
            for irow in range(1, num_records): # the first row is only used to count new subscriptions
                prev_type = df_pupil.iloc[irow-1].loc['subscription_type']
                curr_type = df_pupil.iloc[irow].loc['subscription_type']
                prev_end_date = df_pupil.iloc[irow-1].loc['subscription_end_date']
                curr_start_date = df_pupil.index[irow]
                curr_end_date = df_pupil.iloc[irow].loc['subscription_end_date']
            
                i_update = dates.index(curr_start_date) # update the date of renewal or re-subscritpion,
                                                        # which is the start date of a new subscription
            
                # Determine if it is a renewal or a return
                if prev_end_date + return_gap >= curr_start_date: # renewal
                    if prev_type == configuration.TYPE_MONTHLY and curr_type == configuration.TYPE_MONTHLY:
                        num_rnl_m2m[i_update] = num_rnl_m2m[i_update] + 1
                    elif prev_type == configuration.TYPE_MONTHLY and curr_type == configuration.TYPE_ANNUAL:
                        num_rnl_m2a[i_update] = num_rnl_m2a[i_update] + 1
                    elif prev_type == configuration.TYPE_ANNUAL and curr_type == configuration.TYPE_MONTHLY:
                        num_rnl_a2m[i_update] = num_rnl_a2m[i_update] + 1
                    elif prev_type == configuration.TYPE_ANNUAL and curr_type == configuration.TYPE_ANNUAL:
                        num_rnl_a2a[i_update] = num_rnl_a2a[i_update] + 1
                else: # return
                    # if re-subscribe, then by definition there is a cancellation happening at the end of last subscription
                    i_update_ccl = dates.index(prev_end_date)
                
                    if prev_type == configuration.TYPE_MONTHLY and curr_type == configuration.TYPE_MONTHLY:
                        num_rtn_m2m[i_update] = num_rtn_m2m[i_update] + 1
                        num_ccl_mon[i_update_ccl] = num_ccl_mon[i_update_ccl] + 1
                    elif prev_type == configuration.TYPE_MONTHLY and curr_type == configuration.TYPE_ANNUAL:
                        num_rtn_m2a[i_update] = num_rtn_m2a[i_update] + 1
                        num_ccl_mon[i_update_ccl] = num_ccl_mon[i_update_ccl] + 1
                    elif prev_type == configuration.TYPE_ANNUAL and curr_type == configuration.TYPE_MONTHLY:
                        num_rtn_a2m[i_update] = num_rtn_a2m[i_update] + 1
                        num_ccl_ann[i_update_ccl] = num_ccl_ann[i_update_ccl] + 1
                    elif prev_type == configuration.TYPE_ANNUAL and curr_type == configuration.TYPE_ANNUAL:
                        num_rtn_a2a[i_update] = num_rtn_a2a[i_update] + 1
                        num_ccl_ann[i_update_ccl] = num_ccl_ann[i_update_ccl] + 1
            
                # Count cancellations
                if irow == num_records-1:
                    i_update = dates.index(curr_end_date)
                    if curr_type == configuration.TYPE_MONTHLY:
                        num_ccl_mon[i_update] = num_ccl_mon[i_update] + 1
                    else:
                        num_ccl_ann[i_update] = num_ccl_ann[i_update] + 1

    df_subspt_timeseries['num_ccl_mon'] = num_ccl_mon
    df_subspt_timeseries['num_ccl_ann'] = num_ccl_ann
    df_subspt_timeseries['num_rnl_a2a'] = num_rnl_a2a
    df_subspt_timeseries['num_rnl_a2m'] = num_rnl_a2m
    df_subspt_timeseries['num_rnl_m2a'] = num_rnl_m2a
    df_subspt_timeseries['num_rnl_m2m'] = num_rnl_m2m
    df_subspt_timeseries['num_rtn_a2a'] = num_rtn_a2a
    df_subspt_timeseries['num_rtn_a2m'] = num_rtn_a2m
    df_subspt_timeseries['num_rtn_m2a'] = num_rtn_m2a
    df_subspt_timeseries['num_rtn_m2m'] = num_rtn_m2m
    df_subspt_timeseries['num_new_mon'] = num_new_mon
    df_subspt_timeseries['num_new_ann'] = num_new_ann

    if configuration.IGNORE_UNSTEADY_PERIOD:
        df_subspt_timeseries = _remove_unsteady_data(df_subspt_timeseries)

    return df_subspt_timeseries

def subspt_survival(df_subspt, start_date, period_term, period_unit):
    first_date = df_subspt.subscription_start_date.min()
    if start_date < first_date:
        raise IndexError(f'The input start date {start_date} is earlier than the first observed date in the data!')
    pass