import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    ax.plot(df_subspt_timeseries.index, df_subspt_timeseries.res_subscriptions_length, 'o', markersize=1, alpha=0.5)
    ax.axvspan(cutoff_date, last_date, color='red', alpha=0.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Days')
    ax.set_title('Average Residual Subscription Length')
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.title.set_fontsize(12)
    ax.title.set_fontweight('bold')

    fname = configuration.PLOT_FOLDER + configuration.PLOT_ACTIVE_SUBSPT
    plt.tight_layout()
    plt.savefig(fname)