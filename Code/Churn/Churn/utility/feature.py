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
        (1) time_taken_complete;
        (2) time_taken_incomplete.

    add_progressions:
        Add progressions to features data frame. 2 columns are added:
        (1) progressions - max progression during the day;
        (2) progressions_delta - progression change during the day.

    add_age:
        Add pupils' age to features data frame. 1 column is added:
        (1) age - precise float representation where rounding may be necessary.

    add_outcome:
        Add count of different lesson outcomes to feature data frame.
        The outcomes have been categorised into 6 groups according to 2 fields
        in the complete lesson table: outcome and run_mode. 8 columns are added:
        (1) num_fwrd;
        (2) num_pass;
        (3) num_stat;
        (4) num_fail;
        (5) num_back;
        (6) num_repl;
        (7) num_assess - sum of (1)-(5);
        (8) num_attempt - sum of (1)-(6) and count_incomplete from the dates 
        frame.
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
        
        print('+ Add feature: usage time.')
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

        print('+ Add feature: progressions.')
        self.df_features_ = df_features1

    def add_age(self, df_pupils):
        '''Add pupils' age to features data frame.
        '''

        map_dob = dict(zip(df_pupils.index.values, df_pupils['dob'].values))

        df_features = self.df_features_
        df_features['age'] = 1.0
        def calc_age(df):
            dob = map_dob[df.name]
            df['age'] = (df.index.get_level_values(level=1) - dob).days / 365.25
            return df

        print('+ Add feature: pupils\' age.')
        self.df_features_ = df_features.groupby(level=0).apply(calc_age)

    def add_outcome(self, df_lesson):
        '''Add count of different lesson outcomes to feature data frame.
        The outcomes have been categorised into 6 groups according to 2 fields
        in the complete lesson table: outcome and run_mode.
        '''

        mask_fwrd = (df_lesson['outcome']=='p') & (df_lesson['run_mode']=='j')
        num_fwrd = df_lesson[mask_fwrd].\
            groupby(['pupilId', 'date'])['outcome'].count()

        mask_pass = ((df_lesson['outcome']=='p')&(df_lesson['run_mode']!='j')) |\
            ((df_lesson['outcome']=='s') & (df_lesson['run_mode']=='j'))
        num_pass = df_lesson[mask_pass].\
            groupby(['pupilId', 'date'])['outcome'].count()

        mask_stat = (df_lesson['outcome']=='s') & (df_lesson['run_mode']!='j')
        num_stat = df_lesson[mask_stat].\
            groupby(['pupilId', 'date'])['outcome'].count()

        mask_fail = (df_lesson['outcome']=='f') & (df_lesson['run_mode']=='x')
        num_fail = df_lesson[mask_fail].\
            groupby(['pupilId', 'date'])['outcome'].count()

        mask_back = (df_lesson['outcome']=='f') & (df_lesson['run_mode']=='b')
        num_back = df_lesson[mask_back].\
            groupby(['pupilId', 'date'])['outcome'].count()

        mask_repl = (df_lesson['outcome']=='0') & (df_lesson['run_mode']=='r')
        num_repl = df_lesson[mask_repl].\
            groupby(['pupilId', 'date'])['outcome'].count()

        num_fwrd.rename('num_fwrd', inplace=True)
        num_pass.rename('num_pass', inplace=True)
        num_stat.rename('num_stat', inplace=True)
        num_fail.rename('num_fail', inplace=True)
        num_back.rename('num_back', inplace=True)
        num_repl.rename('num_repl', inplace=True)

        df_outcome = pd.concat(
            [num_fwrd, num_pass, num_stat, num_fail, num_back, num_repl], 
            axis=1)
        df_outcome.fillna(0, inplace=True)

        df_outcome['num_attempt'] = df_outcome.sum(axis=1)
        df_outcome['num_assess'] = df_outcome['num_attempt'] - \
            df_outcome['num_repl']

        # Append to feature data frame
        df_features1 = pd.concat([self.df_features_, df_outcome], axis=1)
        
        # Fill NaN
        df_features1.replace(np.nan, 0.0, inplace=True)

        # The number of attempts should also include those in incomplete table
        df_features1.iloc[:].loc[:,'num_attempt'] = \
            df_features1['num_attempt'] + df_features1['count_incomplete']

        print('+ Add feature: outcome.')
        self.df_features_ = df_features1
