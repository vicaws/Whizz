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
    
    add_hardship:
        Add measures of hardship to feature data frame. Two measures are
        included: stack depth and number of helps. 3 columns are added:
        (1) ave_stackDepth
        (2) ave_help
        (3) sum_help
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

    def add_score(self, df_lesson, df_incomp):
        '''Add scores achieved and the total number of questions to feature data
        frame. Note that each question has a score of 1, so that the number of 
        questions equal full marks.
        '''

        df_lesson_daily = df_lesson.groupby(['pupilId', 'date'])\
            [['score','totalQuestions']].sum()
        df_lesson_daily.rename(
            columns={'score':'score_complete', 
                     'totalQuestions':'question_complete'}, inplace=True)
        
        # For incomplete lesson history records, each row records one attempt
        # and all attempts on one lesson share the same ID. Therefore, we should 
        # only consider the last attempt (where score and the number of questions
        # are highest).
        df_incomp_combineAttempts = df_incomp.\
            groupby('incomplete_lesson_log_id')\
            [['pupilId', 'date', 'score','total_questions']].max()
        df_incomp_combineAttempts.reset_index(inplace=True)
        df_incomp_daily = df_incomp_combineAttempts.groupby(['pupilId', 'date'])\
            [['score','total_questions']].sum()
        df_incomp_daily.rename(
            columns={'score':'score_incomplete', 
                     'total_questions':'question_incomplete'}, inplace=True)

        df = pd.concat([df_lesson_daily, df_incomp_daily], axis=1)
        
        # Fill NaN
        df.fillna(0.0, inplace=True)
        
        df['score'] = df.score_complete + df.score_incomplete
        df['num_questions'] = df.question_complete + df.question_incomplete
        df.drop(columns=['score_complete', 'score_incomplete', 
                         'question_complete', 'question_incomplete'], 
                inplace=True)

        # Append to feature data frame
        df_features1 = pd.concat([self.df_features_, df], axis=1)

        print('+ Add feature: score.')
        self.df_features_ = df_features1

    def add_hardship(self, df_lesson):
        '''Add measures of hardship to feature data frame. Two measures are
        included: stack depth and number of helps.
        '''
        
        # Stack Depth: 'replay' has stackDepth = 0
        df_lesson_sd = df_lesson[df_lesson['lesson_type']!='replay']
        sum_stackDepth = df_lesson_sd.groupby(['pupilId', 'date'])\
            ['stackDepth'].sum()
        sum_stackDepth.rename('sum_stackDepth', inplace=True)
        count_stackDepth = df_lesson_sd.groupby(['pupilId', 'date'])\
            ['stackDepth'].count()
        count_stackDepth.rename('count_stackDepth', inplace=True)

        # Help: 'tutor_pb' has total_help=0
        df_lesson_hp = df_lesson[df_lesson['lesson_type']!='tutor_pb']
        sum_help = df_lesson_hp.groupby(['pupilId', 'date'])['total_help'].sum()
        sum_help.rename('sum_help', inplace=True)
        count_help = df_lesson_hp.groupby(['pupilId', 'date'])\
            ['stackDepth'].count()
        count_help.rename('count_help', inplace=True)

        df_hard = pd.concat(
            [sum_stackDepth, count_stackDepth, sum_help, count_help], axis=1)
        
        # Append to feature data frame
        df_features1 = pd.concat([self.df_features_, df_hard], axis=1)

        # Fill NaN
        df_features1.fillna(0.0, inplace=True)

        print('+ Add feature: hardship.')
        self.df_features_ = df_features1

    def add_mathAge(self, df_lesson, df_incomp):
        '''Add Whizz math age to feature data frame.
        Note: current calculation only gives a very rough estimate of math age,
        namely, the average math age over all attempts (which can be different 
        topics) within a day.
        '''

        math_age1 = df_lesson.groupby(['pupilId', 'date'])['age'].mean() / 100.0
        math_age2 = df_incomp.groupby(['pupilId', 'date'])['age'].mean() / 100.0

        math_age = pd.concat([math_age1, math_age2], axis=1).mean(axis=1)
        math_age.rename('math_age', inplace=True)

        # Append to feature data frame
        df_features1 = pd.concat([self.df_features_, math_age], axis=1)

        print('+ Add feature: math age.')
        self.df_features_ = df_features1


class FeatureCM(object):
    """Class of constructing, updating, managing and using features aggregated 
    in a specific customer month.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, feature, customer_month, df_subspt, configuration):
        self.feature = feature
        self.cmonth = customer_month
        self.df_subspt = df_subspt
        self.config = configuration

        self._df_features, self.df_whizz_ = self._initialise()
    
    def _initialise(self):
        
        # Abstract information of the studying customer month only
        df_features = self.feature.df_features_
        df_features1 = df_features[df_features['customer_month']==self.cmonth]

        # Keep only pupils who do have the option on cancellation in next month
        last_access_date = df_features1.groupby(level=0).apply(
            lambda df: df.index.get_level_values(level=1).max())
        subspt_end_date = df_features1.groupby(level=0)\
            ['subscription_end_date'].last()
        time_last_access = (subspt_end_date - last_access_date).dt.days

        time_last_access = time_last_access[time_last_access<=31] # remove data
        pupils_withOption = time_last_access.index.values

        df_features1 = df_features1[
            df_features1.index.get_level_values(level=0).\
                isin(pupils_withOption)]

        # Construct the basic data frame
        num_complete = df_features1.groupby(level=0)['count_complete'].sum()
        num_incomplete = df_features1.groupby(level=0)['count_incomplete'].sum()

        df_whizz = pd.DataFrame({'num_complete': num_complete,
                                 'num_incomplete': num_incomplete,
                                 'num_attempt': num_complete+num_incomplete,
                                 'rate_incomplete_num': num_incomplete/\
                                     (num_complete+num_incomplete)
                                 })

        # Identify churner and non-churner
        pupils_nextMonth = self.df_subspt[
            self.df_subspt['customer_month']==(self.cmonth+1)]\
                ['pupilId'].unique()
        pupils_nextAnnual = self.df_subspt[
            self.df_subspt['customer_month']==(self.cmonth+12)]\
                ['pupilId'].unique()
        pupils_nextMonthRenew = np.union1d(pupils_nextMonth, pupils_nextAnnual)

        pupils_thisMonth = self.df_subspt[
            self.df_subspt['customer_month']==self.cmonth]['pupilId'].unique()
        pupils_nextMonthCancel = np.setdiff1d(pupils_thisMonth, 
                                              pupils_nextMonthRenew)

        df_whizz = df_whizz.assign(churn= 0)
        df_whizz.loc[df_whizz.index.isin(pupils_nextMonthCancel), 'churn'] = 1

        # Add customer month label
        df_whizz = df_whizz.assign(customer_month=self.cmonth)

        # Add calendar month label
        cal_month = df_features1.groupby(level=0).\
            apply(lambda df: 
                  df.index.get_level_values(level=1).max().month)
        
        df_whizz = df_whizz.assign(calendar_month=cal_month)

        # Add holiday/term label
        df_whizz = df_whizz.assign(holiday=0)
        df_whizz.loc[
            df_whizz['calendar_month'].isin(self.config.HOLIDAY_MONTH), 
            'holiday'] = 1

        # Add time since last access
        df_whizz = df_whizz.assign(
            last_access=time_last_access)

        return df_features1, df_whizz
    
    def add_usageTime(self):
        df_features1 = self._df_features
        time_taken = df_features1.time_taken_complete + \
            df_features1.time_taken_incomplete
        df_features1 = df_features1.assign(time_taken=time_taken)
        
        usage_complete = df_features1.groupby(level=0)\
            ['time_taken_complete'].sum()
        usage_incomplete = df_features1.groupby(level=0)\
            ['time_taken_incomplete'].sum()
        usage = df_features1.groupby(level=0)['time_taken'].sum()

        self.df_whizz_ = self.df_whizz_.assign(usage_complete=usage_complete,
                                               usage_incomplete=usage_incomplete,
                                               usage=usage,
                                               rate_incomplete_usage=\
                                                   usage_incomplete/usage)

    def add_progress(self):
        df_features1 = self._df_features
        progress_delta = df_features1.groupby(level=0)['progressions_delta'].sum()
        progress = df_features1.groupby(level=0)['progressions'].max()

        self.df_whizz_ = self.df_whizz_.assign(progress=progress,
                                               progress_delta=progress_delta)
        
    def add_age(self):
        df_features1 = self._df_features
        age = df_features1.groupby(level=0)['age'].mean()

        self.df_whizz_ = self.df_whizz_.assign(age=age)

    def add_outcome(self):
        df_features1 = self._df_features
        
        num_assess = df_features1.groupby(level=0)['num_assess'].sum()
        num_attempt = df_features1.groupby(level=0)['num_attempt'].sum()
        num_replay = num_attempt-num_assess
        num_fwrd = df_features1.groupby(level=0)['num_fwrd'].sum()
        num_back = df_features1.groupby(level=0)['num_back'].sum()
        num_pass = df_features1.groupby(level=0)['num_pass'].sum()
        num_fail = df_features1.groupby(level=0)['num_fwrd'].sum()

        rate_assess = num_assess / num_attempt
        rate_pass = (num_pass+num_fwrd) / num_assess
        rate_fail = (num_fail+num_back) / num_assess
        rate_fwrd = num_fwrd / num_assess
        rate_back = num_back / num_assess

        self.df_whizz_ = self.df_whizz_.assign(num_pass=num_pass,
                                               num_fail=num_fail,
                                               num_replay=num_replay,
                                               rate_assess=rate_assess,
                                               rate_pass=rate_pass,
                                               rate_fail=rate_fail,
                                               rate_fwrd=rate_fwrd,
                                               rate_back=rate_back)

    def add_score(self):
        df_features1 = self._df_features

        score = df_features1.groupby(level=0)['score'].sum()
        num_questions = df_features1.groupby(level=0)['num_questions'].sum()

        self.df_whizz_ = self.df_whizz_.assign(score=score/num_questions)

    def add_hardship(self):
        df_features1 = self._df_features

        df_hard = df_features1.groupby(level=0)\
            [['sum_stackDepth', 'count_stackDepth', 'sum_help', 'count_help']].\
            sum()

        ave_stackDepth = df_hard['sum_stackDepth'] / df_hard['count_stackDepth']
        ave_help = df_hard['sum_help'] / df_hard['count_help']
        sum_help = df_hard['sum_help']

        self.df_whizz_ = self.df_whizz_.assign(ave_stackDepth=ave_stackDepth,
                                               ave_help=ave_help,
                                               sum_help=sum_help)

    def add_mathAge(self):
        df_features1 = self._df_features
        
        math_age = df_features1.groupby(level=0)['math_age'].mean()
        age = self.df_whizz_['age']
        age_diff = age - math_age

        self.df_whizz_ = self.df_whizz_.assign(math_age=math_age,
                                               age_diff=age_diff)