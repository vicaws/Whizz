import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

def subspt_dist(df_subspt, configuration):
    fig = plt.figure(figsize=(10,3))

    sub_length = np.array(df_subspt[df_subspt.subscription_type==\
                                    configuration.TYPE_MONTHLY].groupby('pupilId')['subscription_length'].sum())
    ax = fig.add_subplot(121)
    ax.hist(sub_length, bins=round(df_subspt.pupilId.unique().shape[0]/80), alpha=0.6)
    ax.set_xlabel("Length of Subscription (days)")
    ax.set_ylabel("Number of Pupils")
    ax.set_title("Histogram of Subsrciption Length (Monthly)")
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.title.set_fontsize(12)
    ax.title.set_fontweight('bold')
    ax.grid(True)
    
    sub_length = np.array(df_subspt[df_subspt.subscription_type==\
                                    configuration.TYPE_ANNUAL].groupby('pupilId')['subscription_length'].sum())
    ax = fig.add_subplot(122)
    ax.hist(sub_length, bins=round(df_subspt.pupilId.unique().shape[0]/80), alpha=0.6)
    ax.set_xlabel("Length of Subscription (days)")
    ax.set_ylabel("Number of Pupils")
    ax.set_title("Histogram of Subsrciption Length (Annual)")
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.title.set_fontsize(12)
    ax.title.set_fontweight('bold')    
    ax.grid(True)
    
    fname = configuration.PLOT_FOLDER + configuration.PLOT_SUBSPT_DIST
    plt.tight_layout()
    plt.savefig(fname)

def subspt_dist_cancelled(df_subspt, configuration):
    
    cutoff_date = pd.to_datetime(configuration.CUTOFF_DATE, format=configuration.CSV_DATE_FORMAT)
    temp = df_subspt[df_subspt['subscription_end_date'] > cutoff_date]
    not_cancelled_pupilId = temp['pupilId'].unique()

    df_subspt_cancelled = df_subspt[~df_subspt['pupilId'].isin(not_cancelled_pupilId)]

    subspt_dist(df_subspt_cancelled, configuration)

def active_subspt(df_subspt_timeseries, configuration):
    last_date = df_subspt_timeseries.index.max()
    cutoff_date = pd.to_datetime(configuration.CUTOFF_DATE, format=configuration.CSV_DATE_FORMAT)
    
    # Active subscriptions over time
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(211)
    ax.plot(df_subspt_timeseries.index, df_subspt_timeseries.num_subscriptions, label='Total')
    ax.plot(df_subspt_timeseries.index, df_subspt_timeseries.num_subscriptions_monthly, '--', label='Monthly')
    ax.plot(df_subspt_timeseries.index, df_subspt_timeseries.num_subscriptions_annual, '--', label='Annually')
    ax.axvspan(cutoff_date, last_date, color='red', alpha=0.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Subscriptions')
    ax.set_title('Active Subscriptions Over Time')
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.title.set_fontsize(12)
    ax.title.set_fontweight('bold')
    ax.legend()

    # Average remaining subscription length
    ax = fig.add_subplot(212)
    ax.plot(df_subspt_timeseries.index, df_subspt_timeseries.res_subscriptions_length, label='Total')
    ax.plot(df_subspt_timeseries.index, df_subspt_timeseries.res_subscriptions_length_mon, '--', label='Monthly')
    ax.plot(df_subspt_timeseries.index, df_subspt_timeseries.res_subscriptions_length_ann, '--', label='Annual')
    ax.axvspan(cutoff_date, last_date, color='red', alpha=0.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Days')
    ax.set_title('Average Residual Subscription Length')
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.title.set_fontsize(12)
    ax.title.set_fontweight('bold')
    ax.legend()

    fname = configuration.PLOT_FOLDER + configuration.PLOT_ACTIVE_SUBSPT
    plt.tight_layout()
    plt.savefig(fname)

def performance_count(df_subspt_timeseries, configuration):
    ''' Plot of descriptive time series  
    (1) Cancellations over time
    (2) Returns over time
    (3) Renewals over time
    (4) New Subscriptions overtime
    '''
    cutoff_date = pd.to_datetime(configuration.CUTOFF_DATE, format=configuration.CSV_DATE_FORMAT)
    df_agg = df_subspt_timeseries[df_subspt_timeseries.index < cutoff_date].\
        resample(configuration.PLOT_PERM_UNIT).apply(lambda x: x.values.sum())
    
    fig = plt.figure(figsize=(16,6))

    ax = fig.add_subplot(221)
    ax.plot(df_agg.index, df_agg[['num_ccl_ann', 'num_ccl_mon']].sum(axis=1, skipna=True, min_count=1), label='Total')
    ax.plot(df_agg.index, df_agg.num_ccl_mon, '--', label='Monthly')
    ax.plot(df_agg.index, df_agg.num_ccl_ann, '--', label='Annual')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Cancellations')
    ax.set_title('Cancellations Over Time')
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12) 
    ax.title.set_fontsize(12)
    ax.title.set_fontweight('bold')
    ax.legend()

    ax = fig.add_subplot(222)
    ax.plot(df_agg.index, df_agg[['num_rtn_a2a',
            'num_rtn_a2m', 'num_rtn_m2a', 'num_rtn_m2m']].sum(axis=1, skipna=True, min_count=1))
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Returns')
    ax.set_title('Returns Over Time')
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.title.set_fontsize(12)
    ax.title.set_fontweight('bold')

    ax = fig.add_subplot(223)
    ax.plot(df_agg.index, df_agg[['num_rnl_m2m', 'num_rnl_a2a', \
           'num_rnl_m2a', 'num_rnl_a2m']].sum(axis=1, skipna=True, min_count=1), label='Total')
    ax.plot(df_agg.index, df_agg[['num_rnl_m2m', 'num_rnl_m2a']].sum(axis=1, skipna=True, min_count=1), '--', label='Monthly')
    ax.plot(df_agg.index, df_agg[['num_rnl_a2a', 'num_rnl_a2m']].sum(axis=1, skipna=True, min_count=1), '--', label='Annual')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Renewals')
    ax.set_title('Renewals Over Time')
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.title.set_fontsize(12)
    ax.title.set_fontweight('bold')
    ax.legend()

    ax = fig.add_subplot(224)
    ax.plot(df_agg.index, df_agg[['num_new_ann','num_new_mon']].sum(axis=1, skipna=True, min_count=1), label='Total')
    ax.plot(df_agg.index, df_agg.num_new_mon, '--', label='Monthly')
    ax.plot(df_agg.index, df_agg.num_new_ann, '--', label='Annual')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of New Subscriptions')
    ax.set_title('New Subscriptions Over Time')
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.title.set_fontsize(12)
    ax.title.set_fontweight('bold')
    ax.legend()

    fname = configuration.PLOT_FOLDER + configuration.PLOT_PERM_COUNT
    plt.tight_layout()
    plt.savefig(fname)

def performance_ratio(df_subspt_timeseries, configuration):
    ''' Plot of descriptive time series  
    (1) Retention rate over time
    (2) New subscription rate over time
    '''
    cutoff_date = pd.to_datetime(configuration.CUTOFF_DATE, format=configuration.CSV_DATE_FORMAT)
    df_agg = df_subspt_timeseries[df_subspt_timeseries.index < cutoff_date].\
        resample(configuration.PLOT_PERM_UNIT).apply(lambda x: x.values.sum())

    fig = plt.figure(figsize=(8,6))

    ax = fig.add_subplot(211)
    num_rnl_mon = df_agg[['num_rnl_m2a', 'num_rnl_m2m']].sum(axis=1, skipna=True, min_count=1)
    num_rnl_ann = df_agg[['num_rnl_a2m', 'num_rnl_a2a']].sum(axis=1, skipna=True, min_count=1)
    num_opt_mon = pd.concat((df_agg.num_ccl_mon, num_rnl_mon), axis=1).sum(axis=1, skipna=True, min_count=1)
    num_opt_ann = pd.concat((df_agg.num_ccl_ann, num_rnl_ann), axis=1).sum(axis=1, skipna=True, min_count=1)
    ax.plot(df_agg.index, pd.concat((num_rnl_mon,num_rnl_ann), axis=1).sum(axis=1, skipna=True, min_count=1)/\
            pd.concat((num_opt_mon,num_opt_ann), axis=1).sum(axis=1, skipna=True, min_count=1), label='Total')
    ax.plot(df_agg.index, num_rnl_mon/num_opt_mon, '--', label='Monthly')
    ax.plot(df_agg.index, num_rnl_ann/num_opt_ann, '--', label='Annual')
    ax.set_xlabel('Date')
    ax.set_ylabel('Retention Rate')
    ax.set_title('Retention Rate Over Time')
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12) 
    ax.title.set_fontsize(12)
    ax.title.set_fontweight('bold')
    ax.legend()

    ax = fig.add_subplot(212)
    ax.plot(df_agg.index, df_agg[['num_new_mon','num_new_ann']].sum(axis=1, skipna=True, min_count=1)/\
            df_agg[['num_subscriptions_monthly','num_subscriptions_annual']].sum(axis=1, skipna=True, min_count=1), label='Total')
    ax.plot(df_agg.index, df_agg.num_new_mon/df_agg.num_subscriptions_monthly, '--', label='Monthly')
    ax.plot(df_agg.index, df_agg.num_new_ann/df_agg.num_subscriptions_annual, '--', label='Annual')
    ax.set_xlabel('Date')
    ax.set_ylabel('New Subscription Rate')
    ax.set_title('New Subscription Rate Over Time')
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.title.set_fontsize(12)
    ax.title.set_fontweight('bold')
    ax.legend()

    fname = configuration.PLOT_FOLDER + configuration.PLOT_PERM_RATIO
    plt.tight_layout()
    plt.savefig(fname)

def survival(survival_counts, configuration):
    
    num_trials = survival_counts.shape[0]

    fig = plt.figure(figsize=(12,6))

    ax = fig.add_subplot(221)
    for i in range(0, num_trials):
        ax.plot(survival_counts[i,:])
    ax.set_title('Survival Count')
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of subscriptions')

    ax = fig.add_subplot(222)
    for i in range(0, num_trials):
        ax.plot(survival_counts[i,:]/np.max(survival_counts[i,:]))
    ax.set_title('Survival rate')
    ax.set_xlabel('Time')
    ax.set_ylabel('Percent of remaining subscriptions')
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals]);

    ax = fig.add_subplot(223)
    ax.plot(np.mean(survival_counts/np.max(survival_counts, axis=1).reshape(num_trials,1), axis=0), '-ko')   
    survival_count = pd.DataFrame(np.mean(survival_counts/np.max(survival_counts, axis=1).reshape(num_trials,1), axis=0))
    gmean_survival_rate = stats.gmean(1.0 + np.array(survival_count.pct_change())[1:])
    survival_theory = np.power(gmean_survival_rate, range(0, survival_counts.shape[1]))
    ax.plot(survival_theory, '--')
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals]);
    ax.set_xlabel('Time')
    ax.set_ylabel('Percent of remaining subscriptions')
    ax.set_yscale('log')
    ax.set_title('Sample-Average Survival Rate')

    ax = fig.add_subplot(224)
    ax.plot(survival_count.pct_change(), '-ko')
    ax.axhline(y=gmean_survival_rate-1.0, linestyle='--')
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals]);
    ax.set_xlabel('Time')
    ax.set_ylabel('Cancellation Rate')
    ax.set_title('Sample-Average Cancellation Rate')

    fname = configuration.PLOT_FOLDER + configuration.PLOT_SURVIVAL
    plt.tight_layout()
    plt.savefig(fname)

def survival_customer_month(df_subspt, configuration):
    survival_population = df_subspt.groupby('customer_month')['pupilId'].count()
    
    fig = plt.figure(figsize=(12,3))

    ax = fig.add_subplot(121)
    gmean_survival_rate =  (survival_population.values[-1] / survival_population.values[0])**\
        (1.0 / (survival_population.shape[0]-1))
    survival_theory = np.power(gmean_survival_rate, range(0, survival_population.shape[0])) * survival_population.values[0]
    ax.plot(survival_population, '-ko')
    ax.plot(survival_population.index, survival_theory, '--')
    ax.set_title('Population Survival Count')
    ax.set_xlabel('Customer Month')
    ax.set_ylabel('Number of subscribers')
    ax.set_yscale('log')

    ax = fig.add_subplot(122)
    ax.plot(survival_population.pct_change(), '-ko')
    ax.axhline(y=gmean_survival_rate-1.0, linestyle='--')
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals]);
    ax.set_xlabel('Customer Month')
    ax.set_ylabel('Cancellation Rate')
    ax.set_title('Population Cancellation Rate')

    fname = configuration.PLOT_FOLDER + configuration.PLOT_SURVIVAL_CM
    plt.tight_layout()
    plt.savefig(fname)