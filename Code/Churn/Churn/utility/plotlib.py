import matplotlib.pyplot as plt
import numpy as np

def subscpt_dist(df_subspt, configuration):
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
    
    plt.tight_layout()
    plt.show() 