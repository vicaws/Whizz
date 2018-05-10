import numpy as np
import pandas as pd

class Feature(object):
    """Class of constructing, updating, managing and using features

    Attributes
    ----------
    df_features: pandas.DataFrame
        Features.
        Level 0 index = pupilId;
        Level 1 index = date.

    Methods
    -------
    add_usageTime:
        Add usage time to features data frame. 2 columns are added:
        (1) time_taken_complete
        (2) time_taken_incomplete

    add_progressions:
        Add progressions to features data frame. 2 columns are added:
        (1) progressions - max progression during the day
        (2) progressions_delta - progression change during the day
    """

    def __init__(self, df_features):
        self.df_features_ = df_features

    def add_usageTime(self, df_lesson, df_incomp):
        '''Add usage time to features data frame.
        '''
        
        # Sum time taken within a day
        df_lesson_daily = pd.DataFrame(\
            df_lesson.groupby(['pupilId', df_lesson['date']])\
            ['time_taken_complete'].sum(), columns=['time_taken_complete'])
        df_incomp_daily = pd.DataFrame(\
            df_incomp.groupby(['pupilId', df_incomp['date']])\
            ['time_taken_incomplete'].sum(), columns=['time_taken_incomplete'])
    
        # Append to feature data frame
        df_features1 = pd.concat(\
            [self.df_features_, df_lesson_daily, df_incomp_daily], \
            axis=1)

        # Fill NaN
        df_features1.replace(np.nan, 0.0, inplace=True)
        
        print('Add feature: usage time.')
        self.df_features_ = df_features1

    def add_progressions(self, df_lesson):
        '''Add progressions to features data frame.
        '''

        # Remove 'replay' lesson type - progressions will always be recorded as 0
        df_lesson1 = df_lesson[df_lesson['lesson_type']!='replay']

        # Taken the largest progression number for a day
        max_progress = df_lesson1.groupby(['pupilId', df_lesson1['date']])\
            ['progressions'].max()

        # Compute the daily delta progression
        delta_progress = max_progress.groupby(level=0).diff()
        df_lesson_progress = pd.DataFrame({'progressions': max_progress, \
                                           'progressions_delta': delta_progress})
        def set_first_value(df):
            df['progressions_delta'].iloc[0] = df['progressions'].iloc[0]
            return df

        df_lesson_progress = df_lesson_progress.groupby(level=0).\
            apply(set_first_value)
        #df_lesson_progress.reset_index(level=1, inplace=True)

        # Append to feature data frame
        df_features1 = pd.concat([self.df_features_, df_lesson_progress], \
            axis=1)

        # Fill NaN for 'progressions' field 
        df_features1['progressions'] = df_features1.groupby(level=0)\
            ['progressions'].apply(lambda x: x.ffill())
        df_features1['progressions'].fillna(0, inplace=True)

        # Fill NaN for 'progressions_delta' field
        df_features1['progressions_delta'].fillna(0, inplace=True)

        print('Add feature: progressions.')
        self.df_features_ = df_features1