import warnings

import numpy as np
import pandas as pd

from tqdm import tqdm
from utility.feature import Feature
from utility.feature import FeatureCM

class DataEngine(object):
    """Class of preparing data for learning models

    Attributes
    ----------

    Methods
    -------
    """

    def __init__(self, df_subspt, df_datesFrame, df_lesson, df_incomp, df_pupils, 
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
        feature.add_score(self._df_lesson, self._df_incomp)
        feature.add_hardship(self._df_lesson)

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
            ftrCM = FeatureCM(self.feature_, cmonth, self._df_subspt, 
                              self.config)

            ftrCM.add_usageTime()
            ftrCM.add_progress()
            ftrCM.add_age()
            ftrCM.add_outcome()
            ftrCM.add_score()
            ftrCM.add_hardship()
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

        self.data_ = np.array(df_whizz1)
        self.target_ = self.df_whizz_['churn'].values
