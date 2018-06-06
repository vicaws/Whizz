import warnings

import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy import stats

from utility.feature import Feature
from utility.feature import FeatureCM

class DataEngine(object):
    """Class of preparing data for learning models

    Attributes
    ----------
    config: Config object
        Configuration.

    max_customer_month_: numpy.int64
        The maximal customer month number that can be inferred from the
        subscription table (df_subspt data frame).

    target_: array-like, shape=(n_sample, 1)
        The sampled values of dependent variable y. To be specific, it is an
        array with binary values 0 or 1 where 1 indicates churn and 0 otherwise.

    data_: array-like, shape=(n_sample, n_feature)
        The sample values of all features X.

    Methods
    -------
    aggregate_features:
        Aggregate features data over specified customer months. If the list of
        customer months is not specified, then all available customer months
        inferred from the subscription table will be included.

    select_features:
        Prepare data for selected features.
    """

    def __init__(self, 
                 df_subspt, df_datesFrame, df_lesson, df_incomp, df_pupils, 
                 configuration):
        self._df_subspt = df_subspt
        self._df_datesFrame = df_datesFrame
        self._df_lesson = df_lesson
        self._df_incomp = df_incomp
        self._df_pupils = df_pupils

        self.config = configuration  
        self.max_customer_month_ = df_subspt['customer_month'].max()
        self.target_ = []
        self.data_ = []
        self.data_bc_ = []
        self.feature_list_ = [] 

        self._initialise()

    def _initialise(self):

        # Construct utility.Feature object
        print("Construct feature object.")
        feature = Feature(df_features=self._df_datesFrame)
        feature.add_usageTime(self._df_lesson, self._df_incomp)
        feature.add_progressions(self._df_lesson)
        feature.add_age(self._df_pupils)
        feature.add_mathAge(self._df_lesson, self._df_incomp)
        feature.add_outcome(self._df_lesson)
        feature.add_hardship(self._df_lesson, self._df_incomp)
        feature.add_mark(self._df_lesson, self._df_incomp)

        self.feature_ = feature

    def aggregate_features(self, 
        customer_month_list=None):

        customer_month_list = customer_month_list \
            if customer_month_list is not None \
            else range(1, self.max_customer_month_+1)

        # Construct features and target data frame
        df_whizz = pd.DataFrame()

        warnings.filterwarnings('ignore')
        # Aggregate over specified customer months
        print("Aggregate data over customer months.")
        for cmonth in tqdm(customer_month_list):
            ftrCM = FeatureCM(self.feature_, cmonth, 
                              self._df_subspt, self._df_pupils, 
                              self.config,
                              verbose=False)

            if ftrCM.df_whizz_.empty:
                continue

            ftrCM.add_usageTime()
            ftrCM.add_progress()
            ftrCM.add_age()
            ftrCM.add_outcome()
            ftrCM.add_hardship()
            ftrCM.add_mark()
            ftrCM.add_mathAge()
    
            df = ftrCM.df_whizz_
            df_whizz = pd.concat([df_whizz, df], axis=0)
        warnings.filterwarnings('default')

        # Set index hierarchy
        df_whizz1 = df_whizz.reset_index()
        df_whizz1.set_index(['customer_month', 'pupilId'], inplace=True)

        self.df_whizz_ = df_whizz1

    def select_features(self, feature_list):
        df_whizz1 = self.df_whizz_[feature_list]

        self.feature_list_ = feature_list
        self.data_ = np.array(df_whizz1)
        self.target_ = self.df_whizz_['churn'].values

    def transform_boxCox(self):
        df_whizz1 = self.df_whizz_[self.feature_list_]
        Xt = []

        for ftr in self.feature_list_:
            x = df_whizz1[ftr].values + 1   # make 0s posiitve
            if ftr == 'age_diff':           # make data positive, age_diff 
                                            # contains negative points
                x += 10
            xt, _ = stats.boxcox(x)
            Xt.append(xt)

        self.data_bc_ = np.array(Xt).transpose()