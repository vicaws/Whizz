import warnings
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

#region Utility Functions

def _remove_unsteady_data(df_subspt_timeseries):
    '''Replace values by numpy.NaN for unsteady period in the subscription descriptive time series
    data. Unstead period refers to the first month for monthly subscription data, or the first year
    for annual subscription data.

    Parameters:
    -----------
    df_subspt_timeseries: pandas.DataFrame
        Subscription descriptive time series data frame.

    Returns:
    --------
    df_subspt_timeseries: pandas.DataFrame
        Updated subscription descriptive time series data frame with data points during unsteady
        period being removed.
    '''

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

#endregion

#region Subscription Pre-process

def filter_subspt_data(df_subspt, start_date, end_date, remove_annual_subspt):
    '''Filter the subsciption table data by given start and end dates of study.

    Parameters
    ----------
    df_subspt: pandas.DataFrame
        Subscription table.

    start_date: pandas.TimeStamp
        Start date of study period of interest.

    end_date: pandas.TimeStamp
        End date of study period of interest.

    remove_annual_subspt: boolean
        If True, then remove all subscribers who have had any annual subscriptions.

    Returns
    -------
    pupils_toBeRemoved: array-like
        1D array of pupils' ID. 
    '''
    # remove customers who have not already cancelled
    pupils_notcancelled = df_subspt[df_subspt['subscription_end_date']>end_date]\
        ['pupilId'].unique()
    print('By the cutoff date {}, there are {} active subscriptions.'.\
        format(end_date.date(), len(pupils_notcancelled)))
    print('These subscribers shall be removed from the analysis because we have '
          'no evidence to know the lifetime of their subscriptions. \n')

    # remove existing customer in the front month 
    start_date1 = start_date + pd.to_timedelta(1, 'M')
    pupils_firstperiod = df_subspt[df_subspt['subscription_start_date']<start_date1]\
        .pupilId.unique()
    print('In the first month of dataset starting from {}, there are {} renewal '
          'or new subscriptions.'.format(start_date.date(), len(pupils_firstperiod)))
    print('These subscribers shall be removed from the analysis because we have '
          'no evidence to show if they renewed or newly joined. \n')

    # remove annual subscrobers
    pupils_annual = []
    if remove_annual_subspt:
        pupils_annual = df_subspt[df_subspt['subscription_type']=='Annual']\
            ['pupilId'].unique()
        print('We also choose to remove {} annual subscribers. \n'.\
            format(len(pupils_annual)))
    
    # aggregate
    pupils_toBeRemoved = np.unique(np.concatenate((pupils_notcancelled, \
        pupils_annual, pupils_firstperiod), axis=0))
    print('In summary, there are {}/{} subscribers being removed from the '
          'dataset in the analysis. \n'.\
        format(len(pupils_toBeRemoved), df_subspt.pupilId.unique().shape[0]))
    
    return pupils_toBeRemoved

def compute_customer_month(df_subspt, configuration):
    '''Compute customer month for each subscription record.

    Parameters
    ----------
    df_subspt: pandas.DataFrame
        Subscription table.

    configuration: Config object
        Configuration settings.

    Returns
    -------
    df_subspt: pandas.DataFrame
        Updated subscription table which includes a new column of assigned
        customer month.
    '''
    
    print("Calculate customer month in the subscription table.")
    df_subspt = df_subspt.assign(num_months=1)
    df_subspt.iloc[:].loc[
        df_subspt['subscription_type']==configuration.TYPE_ANNUAL, 
        'num_months'] = 12
    
    df_subspt['customer_month'] = 0
    def calc_customer_month(df_pupil):
        # Sort the date in ascending order
        df_pupil.sort_values('subscription_start_date', inplace=True)  
    
        df_pupil['customer_month'] = df_pupil['num_months'].cumsum()
    
        return df_pupil
    g = df_subspt.groupby('pupilId')
    
    tqdm.pandas()
    df_subspt = g.progress_apply(calc_customer_month).reset_index(level=0, drop=True)
    
    return df_subspt

#endregion

#region Dates Frame

def _combine_dates_frame(df_lesson, df_incomp):
    '''Construct the union dates frame from all dates available in both complete 
    and incomplete lesson history tables. The date frame also contains the number
    of records within a day in each of the lesson history tables.

    Parameters
    ----------
    df_lesson: pandas.DataFrame
        Complete lesson history table.

    df_incomp: pandas.DataFrame
        Incomplete lesson history table.

    Returns
    -------
    df_datesFrame: pandas.DataFrame
        Dates frame that contain all dates available in both complete and 
        incomplete lesson history tables.
        Index level 0 = pupilId;
        Index level 1 = date.
    '''
    
    # Find out all dates in the complete lesson table    
    dates_lesson = df_lesson.groupby(['pupilId', df_lesson['date']])['pupilId'].\
        count()
    dates_lesson.rename('count_complete', inplace=True)

    # Find out all dates in the incomplete lesson table
    dates_incomp = df_incomp.groupby(['pupilId', df_incomp['date']])['pupilId'].\
        count()
    dates_incomp.rename('count_incomplete', inplace=True)

    # Union two dates lists
    df_datesFrame = pd.concat(\
        [pd.DataFrame(dates_lesson), pd.DataFrame(dates_incomp)], axis=1)
    df_datesFrame.fillna(0, inplace=True)

    return df_datesFrame

def _assign_customer_month(df_subspt, df_datesFrame, configuration):
    '''Assign customer month to each date in the common dates frame.

    Parameters
    ----------
    df_subspt: pandas.DataFrame
        Subscription table. The customer month shall have already been computed
        and assigned to each subscription record.

    df_datesFrame: pandas.DataFrame
        Dates frame that contain all dates available in both complete and incomplete
        lesson history tables.
        Index level 0 = pupilId;
        Index level 1 = date.

    configuration: Config object
        Configuration settings.

    Returns
    -------
    df_datesFrame: pandas.DataFrame
        Updated dates frame that includes an additional column of customer month.
    '''
    
    df_datesFrame.reset_index(inplace=True) # remove the index hierarchy for the
    # following calculation
    df_datesFrame['customer_month'] = 0     # initialisation
    num_records = df_subspt.shape[0]

    warnings.filterwarnings('ignore')
    for i_record, row in zip(tqdm(range(num_records)), df_subspt.itertuples()):
        criterion1 = df_datesFrame.pupilId == row.pupilId
        criterion21 = df_datesFrame.date >= row.subscription_start_date
        criterion22 = df_datesFrame.date <= row.subscription_end_date
        
        if row.subscription_type == configuration.TYPE_MONTHLY:
            df_datesFrame.loc[criterion1 & criterion21 & criterion22, 
                              'customer_month'] = row.customer_month
        elif row.subscription_type == configuration.TYPE_ANNUAL:
            cmonth = row.customer_month - 12 + \
                np.ceil(( df_datesFrame.date-row.subscription_start_date)\
                .dt.days/30)
            df_datesFrame.loc[criterion1 & criterion21 & criterion22, 
                              'customer_month'] = cmonth
        
        # Assign subscription end date
        df_datesFrame.loc[criterion1 & criterion21 & criterion22, 
                              'subscription_end_date'] = \
                                  row.subscription_end_date
    warnings.filterwarnings('default')

    df_datesFrame.set_index(['pupilId', 'date']) # recover the index hierarchy

    return df_datesFrame

def _generate_dates_frame(df_subspt, df_lesson, df_incomp, configuration):
    
    # Construct the union dates frame
    df_datesFrame = _combine_dates_frame(df_lesson, df_incomp)
    
    # Assign customer month to the dates frame
    df_datesFrame = _assign_customer_month(df_subspt, df_datesFrame, 
                                           configuration)

    # Save into CSV file
    fpath = configuration.DATA_FOLDER_PATH + configuration.FILE_INTERMEDIATE
    fname = fpath + configuration.DATA_DATES

    if not os.path.exists(fpath):
        os.makedirs(fpath)

    return df_datesFrame

def _load_dates_frame(configuration):
    
    fname = configuration.DATA_FOLDER_PATH + configuration.FILE_INTERMEDIATE + \
        configuration.DATA_DATES
    df_datesFrame = pd.read_csv(fname, delimiter=',', index_col=0)
    
    # Convert date string into object
    date_format = configuration.CSV_DATE_FORMAT
    df_datesFrame['date'] = pd.to_datetime(df_datesFrame['date'], 
                                           format=date_format)
    df_datesFrame['subscription_end_date'] = pd.to_datetime(
        df_datesFrame['subscription_end_date'], format=date_format)

    # Set index hierarchy
    df_datesFrame.set_index(['pupilId', 'date'], inplace=True)

    print('The dates frame has already been assigned customer month and saved',
          ' in a file. The file has been loaded!')

    return df_datesFrame

def construct_dates_frame(df_subspt, df_lesson, df_incomp, configuration):
    '''Construct the union dates frame from all dates available in both complete 
    and incomplete lesson history tables. The date frame also contains: 
    (1) the number of records within a day in each of the lesson history tables;
    (2) the corresponding customer month.

    Parameters
    ----------
    df_subspt: pandas.DataFrame
        Subscription table. The customer month shall have already been computed
        and assigned to each subscription record.
        
    df_lesson: pandas.DataFrame
        Complete lesson history table.

    df_incomp: pandas.DataFrame
        Incomplete lesson history table.

    configuration: Config object
        Configuration settings.

    Returns
    -------
    df_datesFrame: pandas.DataFrame
        Dates frame that contain all dates available in both complete and 
        incomplete lesson history tables.
        Index level 0 = pupilId;
        Index level 1 = date.
    '''
    
    print('Construct data-driven dates frame.')
    fname = configuration.DATA_FOLDER_PATH + configuration.FILE_INTERMEDIATE + \
        configuration.DATA_DATES
    if os.path.isfile(fname):
        return _load_dates_frame(configuration)
    else:
        return _generate_dates_frame(\
            df_subspt, df_lesson, df_incomp, configuration)
    
    return df_datesFrame

#endregion Date Frame

def _generate_subspt_timeseries(df_subspt, configuration): 
    
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
    print("Prepare data for active subscriptions.")
    warnings.filterwarnings('ignore')
    for irow, row in zip(tqdm(range(temp.shape[0])), temp.itertuples(index=True)):
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

    print("Prepare data for counts of cancellations, returns, renewals and new subscriptions.")
    pupil_list = df_subspt.pupilId.unique()
    for pupil in tqdm(pupil_list):
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

    # Save into CSV file
    fpath = configuration.DATA_FOLDER_PATH + configuration.FILE_INTERMEDIATE
    fname = fpath + configuration.DATA_DESCR

    if not os.path.exists(fpath):
        os.makedirs(fpath)
    df_subspt_timeseries.to_csv(fname, index=True) 

    warnings.filterwarnings('default')
    return df_subspt_timeseries

def _load_subspt_timeseries(configuration):
    fname = configuration.DATA_FOLDER_PATH + configuration.FILE_INTERMEDIATE + configuration.DATA_DESCR
    df_subspt_timeseries = pd.read_csv(fname, delimiter=',', index_col=0)
    
    time_format = configuration.CSV_TIME_FORMAT
    date_format = configuration.CSV_DATE_FORMAT
    df_subspt_timeseries.index = pd.to_datetime(df_subspt_timeseries.index, format=date_format+" "+time_format)

    print('The descriptive stats have already been computed and saved in a file. The file has been loaded!')

    return df_subspt_timeseries

def subspt_timeseries(df_subspt, configuration):
    fname = configuration.DATA_FOLDER_PATH + configuration.FILE_INTERMEDIATE + configuration.DATA_DESCR
    if os.path.isfile(fname):
        return _load_subspt_timeseries(configuration)
    else:
        return _generate_subspt_timeseries(df_subspt, configuration)

def subspt_survival(df_subspt, subscription_type, start_date, period_term, period_unit):
    
    # Filter data
    df_subspt = df_subspt[df_subspt['subscription_type']==subscription_type]

    first_date = df_subspt.subscription_start_date.min()
    last_date = df_subspt.subscription_end_date.max()
    
    # Errors
    if start_date < first_date:
        raise IndexError(f'The input start date {start_date} is earlier than the first observed date {first_date} in the data!')
    elif start_date > last_date:
        raise IndexError(f'The input start date {start_date} is later than the last observed date {last_date} in the data!')

    end_date = start_date + pd.to_timedelta(period_term, unit=period_unit)
    if end_date > last_date:
        raise IndexError(f'The indicated end date {end_date} is later than the last observed date {last_date} in the data!')

    # Warnings
    if subscription_type == 'Monthly':
        threshold_date = first_date + pd.to_timedelta(1, unit='M')
    else:
        threshold_date = first_date + pd.to_timedelta(1, unit='Y')
    if start_date < threshold_date:
        warnings.warn('The input start date is suggested to be at least 1-month or 1-year later than the first observed date in the data!')

    # Identify exisiting subscribers in the first period
    first_end_date = start_date + pd.to_timedelta(1, unit=period_unit)
    criterion1 = (df_subspt['subscription_start_date'] <= start_date) & (df_subspt['subscription_end_date'] > start_date)
    criterion2 = (df_subspt['subscription_start_date'] > start_date) & (df_subspt['subscription_start_date'] < first_end_date)
    initial_users = df_subspt[criterion1|criterion2]['pupilId'].unique()

    survival_count = np.zeros(period_term)
    dates = []
    survival_count[0] = len(initial_users)
    dates.append(first_end_date)

    # Calculate survivals
    for iperiod in range(1, period_term):
        p_start_date = start_date + pd.to_timedelta(iperiod, unit=period_unit)
        p_end_date = p_start_date + pd.to_timedelta(1, unit=period_unit)
        criterion1 = (df_subspt['subscription_start_date'] <= p_start_date) & (df_subspt['subscription_end_date'] > p_start_date)
        criterion2 = (df_subspt['subscription_start_date'] > p_start_date) & (df_subspt['subscription_start_date'] < p_end_date)
        active_users = df_subspt[criterion1|criterion2]['pupilId'].unique()
        
        survival_count[iperiod] = len(np.intersect1d(active_users, initial_users))
        dates.append(p_end_date)

    return pd.DataFrame({'survival_count': survival_count}, index=dates)