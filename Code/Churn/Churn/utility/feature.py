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
        Add measures of hardship to feature data frame. The hardship is measured
        in terms of attempts, helps and time used in different stack depths. The 
        stack depth is binned to 3 levels: 0(stackDepth=0), 1(stackDepth=1) and 
        2(stackDepth>1). 9 columns are added:
        (1)-(3) num_sd0, num_sd1, num_sd2: attempts in different stack depths;
        (4)-(6) help_sd0, help_sd1, help_sd2: helps in different stack depths;
        (7)-(9) usage_sd0, usage_sd1, usage_sd2: time in different stack depths

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

    def add_mark(self, df_lesson, df_incomp):
        '''Add marks achieved and the total number of questions to feature data
        frame.
        '''
        
        mark_lesson = df_lesson.groupby(['pupilId', 'date'])['mark'].mean()
        mark_lesson.rename('mark_complete', inplace=True)
        
        # For incomplete lesson history records, each row records one attempt
        # and all attempts on one lesson share the same ID. Therefore, we should 
        # only consider the last attempt (where mark is highest).
        df_incomp_combineAttempts = df_incomp.\
            groupby('incomplete_lesson_log_id')\
            [['pupilId', 'date', 'mark']].max()
        df_incomp_combineAttempts.reset_index(inplace=True)
        mark_incomp = df_incomp_combineAttempts.groupby(['pupilId', 'date'])\
            ['mark'].mean()
        mark_incomp.rename('mark_incomplete', inplace=True)

        df_mark = pd.concat([mark_lesson, mark_incomp], axis=1)

        # Append to feature data frame
        df_features1 = pd.concat([self.df_features_, df_mark], axis=1)

        print('+ Add feature: mark.')
        self.df_features_ = df_features1

    def add_hardship(self, df_lesson, df_incomp):
        '''Add measures of hardship to feature data frame. The hardship is 
        measured in terms of attempts, helps and time used in different stack 
        depths. The stack depth is binned to 3 levels: 0(stackDepth=0), 
        1(stackDepth=1) and 2(stackDepth>1).
        '''
        
        def bin_stackDepth(df, incomplete_table):
            '''One-hot-encoding on the stackDepth
            '''
            s0 = df['stackDepth']==0
            s1 = df['stackDepth']==1
            s2 = df['stackDepth']>1
            s0.replace({True:1, False:0}, inplace=True)
            s1.replace({True:1, False:0}, inplace=True)
            s2.replace({True:1, False:0}, inplace=True)
            if not incomplete_table:
                num_sd0 = s0
                num_sd1 = s1
                num_sd2 = s2
            else:
                num_sd0 = s0 * df['count']
                num_sd1 = s1 * df['count']
                num_sd2 = s2 * df['count']
    
            help_sd0 = s0*df['total_help']
            help_sd1 = s1*df['total_help']
            help_sd2 = s2*df['total_help']
    
            col_usage = 'time_taken_incomplete' if incomplete_table \
                else 'time_taken_complete'
            usage_sd0 = s0*df[col_usage]
            usage_sd1 = s1*df[col_usage]
            usage_sd2 = s2*df[col_usage]

            df = df.assign(num_sd0=num_sd0, num_sd1=num_sd1, num_sd2=num_sd2,
                           help_sd0=help_sd0, help_sd1=help_sd1, 
                           help_sd2=help_sd2, usage_sd0=usage_sd0, 
                           usage_sd1=usage_sd1, usage_sd2=usage_sd2)

            return df.groupby(['pupilId','date'])\
                [['num_sd0', 'num_sd1', 'num_sd2',
                  'help_sd0', 'help_sd1', 'help_sd2',
                  'usage_sd0', 'usage_sd1', 'usage_sd2']].sum()
        
        # Construct the binned stackDepth data frame from the complete 
        # lesson table
        print('Start binning stackDepth for complete lesson table.')
        df_lesson_stackDepth = bin_stackDepth(df_lesson, incomplete_table=False)

        # Construct the binned stackDepth data frame from the complete 
        # lesson table
        print('Start binning stackDepth for incomplete lesson table.')
        df_incomp_stackDepth = self._infer_stackDepth_incomp(df_lesson, 
                                                             df_incomp)
        df_incomp_stackDepth = bin_stackDepth(df_incomp_stackDepth, 
                                              incomplete_table=True)

        # Combine the binned stackDepth data frames
        df_stackDepth = df_lesson_stackDepth.add(df_incomp_stackDepth, 
                                                 fill_value=0)

        # Append to feature data frame
        df_features1 = pd.concat([self.df_features_, df_stackDepth], axis=1)

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

    def _infer_stackDepth_incomp(self, df_lesson, df_incomp):
        
        # Group stackDepth values 2 and 3
        temp = df_lesson.replace({'stackDepth': {3:2}})
        # Get from complete lesson table the last valid day of the same 
        # stackDepth
        df_sd_lastDate = temp.groupby(['pupilId', 'lesson_key', 'stackDepth'])\
            ['date'].max()
        df_sd_lastDate = df_sd_lastDate.reset_index(level='stackDepth')
        # Add a new column date_temp with values 0 for following calculations
        df_sd_lastDate = df_sd_lastDate.assign(date_temp=pd.to_timedelta(0))

        # Add a new column count with value 1 for following calculations of 
        # count of attempts
        temp = df_incomp.assign(count=1)
        # Aggregate incomplete lesson records at daily level
        df_incomp_dailySum = temp.groupby(['pupilId', 'lesson_key', 'date'])\
            [['time_taken_incomplete', 'total_help', 'count']].sum()
        df_incomp_dailySum.reset_index(level='date', inplace=True)
        # Add a new column date_temp with values 0 for following calculations
        df_incomp_dailySum = df_incomp_dailySum.assign(
            date_temp=df_incomp_dailySum.date, stackDepth=0)

        # Calculate indicator data frame
        # In the indicater data frame, the indicater is the column 'date'. 
        # Because for one attempt in the incomplete lesson table, 'date' is 
        # calculated as the date of that attempt minus the last valid date of a 
        # stackDepth value of the corresponding attempt in the complete lesson 
        # table. Hence, if 'date' > 0, then the stackDepth of that incomplete 
        # attempt should be 'stackDepth'+1 
        df1 = df_sd_lastDate[['date', 'date_temp', 'stackDepth']]
        df2 = df_incomp_dailySum[['date', 'date_temp', 'stackDepth']]
        df_indicator = df2-df1 

        # Keep only those that not NaN. NaN appears when the lesson key is not 
        # present in both complete and incomplete lesson tables
        df_indicator = df_indicator.dropna()
        df_indicator.reset_index(inplace=True)
        df_indicator.loc[:, 'stackDepth'] = -df_indicator['stackDepth']

        # Keep only records with 'date' <= 0
        df_indicator = df_indicator[df_indicator['date']<=pd.to_timedelta(0)]

        # For the case where there are multiple 'date' values for the same 
        # lesson and date of attempt, take the maximum. For example, choose -29 
        # if we have -29 and -143. This means to keep the one whose date is 
        # closest to last valid date of a stackDepth value of the corresponding 
        # attempt in the complete lesson table.
        df_indicator = df_indicator.loc[
            df_indicator.groupby(['pupilId','lesson_key','date_temp'])['date'].\
                idxmax()]
        df_indicator.drop(columns='date', inplace=True)
        df_indicator.rename(columns={'date_temp':'date'}, inplace=True)
        df_indicator.set_index(['pupilId', 'lesson_key', 'date'], inplace=True)

        # Remove temporary columns which were created for previous calculations
        df_incomp_dailySum.drop(columns=['date_temp', 'stackDepth'], 
                                inplace=True)
        df_incomp_dailySum.set_index(['date'], append=True, inplace=True)

        df_incomp_stackDepth = pd.concat([df_incomp_dailySum, df_indicator], 
                                         axis=1)
        # Fill NaN for the stackDepth column by 0
        # The stackDepth is NaN only when there is no corresponding attempt in 
        # the complete lesson table.
        df_incomp_stackDepth.loc[:, 'stackDepth'] = \
            df_incomp_stackDepth['stackDepth'].fillna(0)
        df_incomp_stackDepth.reset_index(inplace=True)

        return df_incomp_stackDepth


class FeatureCM(object):
    """Class of constructing, updating, managing and using features aggregated 
    in a specific customer month.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, feature, customer_month, 
                 df_subspt, df_pupils,
                 configuration, verbose=True):
        self.feature = feature
        self.cmonth = customer_month
        self.df_subspt = df_subspt
        self.df_pupils = df_pupils
        self.config = configuration

        self._df_features, self.df_whizz_ = self._initialise(verbose)
    
    def _initialise(self, verbose):
        
        df_features = self.feature.df_features_
        df_features = df_features[df_features['customer_month']>0]
        self.feature.df_features_ = df_features
        
        # Identify and categorise pupils
        self._identify_pupils(verbose)
        
        # Abstract information of the studying customer month only
        df_features1 = df_features[df_features['customer_month']==self.cmonth]
        
        # Keep only pupils who do have the option on churn in next month
        # Note: this may result an empty data frame, which bascially indicates 
        # that there is no active pupils who have churn option
        df_features1 = df_features1[
            df_features1.index.get_level_values(level=0).\
                isin(self.pupils_churnOption_)]
        
        # Construct the basic data frame
        num_complete = df_features1.groupby(level=0)['count_complete'].sum()
        num_incomplete = df_features1.groupby(level=0)['count_incomplete'].sum()
        df_whizz = pd.DataFrame({'num_complete': num_complete,
                                 'num_incomplete': num_incomplete,
                                 'num_attempt': num_complete+num_incomplete
                                 })
        
        # Append inactive pupils - all fields are filled by 0
        sr_pupils_inactive = pd.Series(self.pupils_inactive_churnOption_, 
                                       name='pupilId')
        df_whizz_inactive = pd.DataFrame(0, index=sr_pupils_inactive, 
                                         columns=df_whizz.columns)
        df_whizz = df_whizz.append(df_whizz_inactive)

        # Add percentage of incomplete attempts - can result in NaN
        df_whizz = df_whizz.assign(rate_incomplete_num=num_incomplete/\
                                     (num_complete+num_incomplete))

        # Add customer month label
        df_whizz = df_whizz.assign(customer_month=self.cmonth)

        # Identify churner and non-churner
        pupils_renew = self.df_subspt[
            self.df_subspt['customer_month']>self.cmonth]['pupilId'].unique()
        pupils_churn = np.setdiff1d(self.pupils_subspt_, pupils_renew)
        df_whizz = df_whizz.assign(churn=0)
        df_whizz.loc[df_whizz.index.isin(pupils_churn), 'churn'] = 1

        # Calculate time since last access
        time_last_access, subspt_end_date, subspt_end_date_inactive = \
            self._calc_last_access(df_features, df_features1)
        df_whizz = df_whizz.assign(last_access=time_last_access)
        
        # Add active/inactive label
        sr_active = pd.Series(1, index=self.pupils_active_churnOption_)
        sr_inactive = pd.Series(0, index=self.pupils_inactive_churnOption_)
        df_whizz = df_whizz.assign(active=sr_active.append(sr_inactive))
        
        # Add calendar month label
        cal_month = subspt_end_date.dt.month \
            if not df_features1.empty else pd.Series()
        cal_month_inactive = subspt_end_date_inactive.dt.month \
            if self.pupils_inactive_churnOption_.size!=0 else pd.Series()
        df_whizz = df_whizz.assign(
            calendar_month=cal_month.append(cal_month_inactive))
    
        # Add holiday/term label
        df_whizz = df_whizz.assign(holiday=0)
        df_whizz.loc[
            df_whizz['calendar_month'].isin(self.config.HOLIDAY_MONTH), 
            'holiday'] = 1

        return df_features1, df_whizz
    
    def _identify_pupils(self, verbose):
        
        # Identify pupils who are now subscribing
        pupils_subspt = self.df_subspt[
            self.df_subspt['customer_month']>=self.cmonth]\
                ['pupilId'].unique()
        pupils_subspt_ann = self.df_subspt[
            (self.df_subspt['customer_month']>=self.cmonth)&\
                (self.df_subspt['subscription_type']==self.config.TYPE_ANNUAL)]\
                ['pupilId'].unique()

        # Identify subscribed pupils who are active and inactive in the customer
        # month. 'Active' means the pupils have records kept in lesson history.
        df_features = self.feature.df_features_
        df_features1 = df_features[df_features['customer_month']==self.cmonth]
        pupils_active = df_features1.index.get_level_values(level=0).unique()
        pupils_inactive = np.setdiff1d(pupils_subspt, pupils_active)

        # Identify pupils with churn option
        pupils_churnOption = self.df_subspt[
            self.df_subspt['customer_month']==self.cmonth]\
                ['pupilId'].unique()
        pupils_active_churnOption = np.intersect1d(pupils_active, 
                                                   pupils_churnOption)
        pupils_inactive_churnOption = np.intersect1d(pupils_inactive,
                                                     pupils_churnOption)

        if verbose:
            print("In customer month {}:".format(self.cmonth))
            print("Number of subscribed pupils (Annual)        = {} ({})".\
                format(pupils_subspt.shape[0], pupils_subspt_ann.shape[0]))
            print("Number of active pupils                     = {}".\
                format(pupils_active.shape[0]))
            print("Number of inactive pupils                   = {}".\
                format(pupils_inactive.shape[0]))
            print("Number of pupils with churn option          = {}".\
                format(pupils_churnOption.shape[0]))
            print("Number of active pupils with churn option   = {}".\
                format(pupils_active_churnOption.shape[0]))
            print("Number of inactive pupils with churn option = {}".\
                format(pupils_inactive_churnOption.shape[0]))
        
        self.pupils_subspt_ = pupils_subspt
        self.pupils_churnOption_ = pupils_churnOption
        self.pupils_active_churnOption_ = pupils_active_churnOption
        self.pupils_inactive_churnOption_ = pupils_inactive_churnOption

    def _calc_last_access(self, df_features, df_features1):
        # For active pupils, we can find time since last access from the lesson 
        # history records in the current customer month
        if df_features1.empty:
            time_last_access = pd.Series()
            subspt_end_date = pd.Series()
        else:
            last_access_date = df_features1.groupby(level=0).apply(
                lambda df: df.index.get_level_values(level=1).max())
            subspt_end_date = df_features1.groupby(level=0)\
                ['subscription_end_date'].last()
            time_last_access = (subspt_end_date-last_access_date).dt.days

        # For inactive pupils, we need to search lesson history records in all
        # past customer months
        if self.pupils_inactive_churnOption_.size==0:
            pupils_activePast = pd.Series()
            pupils_neverActive = pd.Series()
            subspt_start_date_inactive = pd.Series()
            subspt_end_date_inactive = pd.Series()
            last_access_date_inactive = pd.Series()
            time_last_access_inactive = pd.Series()
        else:
            df_sd = self.df_subspt[
                (self.df_subspt['customer_month']==1)&\
                    (self.df_subspt['pupilId'].isin(
                        self.pupils_inactive_churnOption_))]\
                            [['pupilId','subscription_start_date']]
            df_sd.set_index(['pupilId'], inplace=True)
            subspt_start_date_inactive = df_sd['subscription_start_date']

            df_ed = self.df_subspt[
                (self.df_subspt['customer_month']==self.cmonth)&\
                    (self.df_subspt['pupilId'].isin(
                        self.pupils_inactive_churnOption_))]\
                            [['pupilId','subscription_end_date']]
            df_ed.set_index(['pupilId'], inplace=True)
            subspt_end_date_inactive = df_ed['subscription_end_date']

            if self.cmonth==1:
                last_access_date_inactive = subspt_start_date_inactive
                pupils_activePast = np.array([])
                pupils_neverActive = self.pupils_inactive_churnOption_
            else:
                # if the pupil has been active in the past, then find the latest 
                # active date in history
                def find_lastDate(df):
                    pupilId = df.name
                    spt_end_date = subspt_end_date_inactive.loc[pupilId]
                    mask = df.index.get_level_values(level=1)<spt_end_date
                    last_date = df[mask].index.get_level_values(level=1).max()
                    return last_date
            
                mask_inactive = df_features.index.get_level_values(level=0).\
                    isin(self.pupils_inactive_churnOption_)
                mask_cm = df_features['customer_month']<self.cmonth

                last_access_date_inactive = df_features[mask_inactive & mask_cm].\
                    groupby(level=0).apply(find_lastDate)

                # If the pupil has never been active in the past, then assign the 
                # subscription start date of the first ever subscription
                last_access_date_inactive.loc[
                    last_access_date_inactive.isna()] = \
                    subspt_start_date_inactive
                pupils_activePast = last_access_date_inactive.\
                    index.unique().values
                pupils_neverActive = np.setdiff1d(
                    self.pupils_inactive_churnOption_, pupils_activePast)
                last_access_date_neverActive = \
                    subspt_start_date_inactive[pupils_neverActive]
                last_access_date_inactive = last_access_date_inactive.\
                    append(last_access_date_neverActive)
        
            time_last_access_inactive = \
                (subspt_end_date_inactive-last_access_date_inactive).dt.days
        
        # Add class members
        self.subspt_start_date_inactive_ = subspt_start_date_inactive
        self.subspt_end_date_inactive_ = subspt_end_date_inactive
        self.last_access_date_inactive_ = last_access_date_inactive
        self.pupils_activePast_ = pupils_activePast
        self.pupils_neverActive_ = pupils_neverActive
        self.subspt_end_date_active_ = subspt_end_date

        return time_last_access.append(time_last_access_inactive), \
            subspt_end_date, subspt_end_date_inactive

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
                                               usage=usage)
        
        # Fill 0s for inactive subscribers
        self.df_whizz_.fillna(0.0, inplace=True)

        # Note: leave NaN for inactive subscribers as the rate is not defined
        # for them
        self.df_whizz_ = self.df_whizz_.assign(rate_incomplete_usage=\
                                                   usage_incomplete/usage)

    def add_progress(self):
        df_features = self.feature.df_features_
        df_features1 = self._df_features
        progress_delta = df_features1.groupby(level=0)\
            ['progressions_delta'].sum()
        progress = df_features1.groupby(level=0)['progressions'].max()

        self.df_whizz_ = self.df_whizz_.assign(progress=progress,
                                               progress_delta=progress_delta)

        # Fill progress_delta for inactive subscribers
        self.df_whizz_.loc[:, 'progress_delta'] = \
            self.df_whizz_['progress_delta'].fillna(0.0)

        # Fill progress for inactive subscribers. We need to search lesson 
        # history records in all past customer month
        if self.cmonth == 1:
            self.df_whizz_.loc[:,'progress'] = \
                self.df_whizz_['progress'].fillna(0.0)
        else:
            # Fill for never active pupils 
            self.df_whizz_.loc[
                self.df_whizz_.index.isin(self.pupils_neverActive_), 
                'progress'] = 0

            # Fill for pupils who have been active in the past
            def find_lastProgress(df):
                pupilId = df.name
                last_date = self.last_access_date_inactive_.loc[pupilId]
                progress = df.iloc[:].loc[
                    df.index.get_level_values(level=1)==last_date, 
                    'progressions']
                return progress.values[0]

            mask = df_features.index.get_level_values(level=0).\
                isin(self.pupils_activePast_)
            progress_activePast = df_features[mask].groupby(level=0).\
                apply(find_lastProgress)

            self.df_whizz_.loc[
                self.df_whizz_.index.isin(self.pupils_activePast_), 
                'progress'] = progress_activePast
        
        # Add effective progress
        self.df_whizz_ = self.df_whizz_.assign(
            effective_progress=self.df_whizz_['progress_delta']/\
                self.df_whizz_['usage']*3600.)

    def add_age(self):
        subspt_end_date = self.subspt_end_date_active_.\
            append(self.subspt_end_date_inactive_)
        pupils = self.pupils_churnOption_
        dob = self.df_pupils[self.df_pupils.index.isin(pupils)]['dob']
        age = (subspt_end_date - dob).dt.days / 365.

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
        num_stat = df_features1.groupby(level=0)['num_stat'].sum()

        assess = num_assess>0
        assess.replace({True:1, False:0}, inplace=True)

        self.df_whizz_ = self.df_whizz_.assign(
            num_pass=num_pass+num_fwrd,
            num_fail=num_fail+num_stat+num_back,
            assess=assess,
            num_assess = num_assess,
            num_replay=num_replay)

        # Fill for inactive subscribers
        if self.config.FILL_NAN_BY_ZERO:
            self.df_whizz_.loc[:, ['num_pass','num_fail','num_replay',
                                   'num_assess','assess']] = \
                self.df_whizz_[['num_pass','num_fail','num_replay',
                                'num_assess','assess']].fillna(0)
        else:
            # num_pass, num_fail is NA if there is no assessment at all
            # so leave them as NaN
            self.df_whizz_.loc[:, ['num_replay','num_assess','assess']] = \
                self.df_whizz_[['num_replay','num_assess','assess']].fillna(0)

        rate_assess = num_assess / num_attempt
        rate_pass = (num_pass+num_fwrd) / num_assess
        rate_fail = (num_fail+num_back+num_stat) / num_assess
        rate_fwrd = num_fwrd / num_assess
        rate_back = num_back / num_assess
        
        # The rate calculation will lead to NaN in the data frame due to 0 
        # denominator
        self.df_whizz_ = self.df_whizz_.assign(rate_assess=rate_assess,
                                               rate_pass=rate_pass,
                                               rate_fail=rate_fail,
                                               rate_fwrd=rate_fwrd,
                                               rate_back=rate_back)

        # NaN check - if there is no assessment at all, then all outcome shall 
        # be NaN
        if not self.config.FILL_NAN_BY_ZERO:
            self.df_whizz_.loc[self.df_whizz_['assess']==0, 
                               ['num_pass', 'num_fail']] = np.nan

    def add_mark(self):
        df_features1 = self._df_features

        df_mark = df_features1.groupby(level=0)\
            [['mark_complete', 'mark_incomplete']].mean()
        
        self.df_whizz_ = pd.concat([self.df_whizz_, df_mark], axis=1)

        # Fill NaN for inactive subscribers
        if self.config.FILL_NAN_BY_ZERO:
            self.df_whizz_.loc[:, ['mark_complete', 'mark_incomplete']] = \
                self.df_whizz_[['mark_complete', 'mark_incomplete']].fillna(0)

        # NaN check - if there is no assessment at all, then all outcome shall 
        # be NaN
        if not self.config.FILL_NAN_BY_ZERO:
            self.df_whizz_.loc[self.df_whizz_['assess']==0, 
                               ['mark_complete', 'mark_incomplete']] = np.nan
            self.df_whizz_.loc[self.df_whizz_['num_complete']==0, 
                               ['mark_complete']] = np.nan
            self.df_whizz_.loc[self.df_whizz_['num_incomplete']==0, 
                               ['mark_incomplete']] = np.nan

    def add_hardship(self):
        df_features1 = self._df_features

        df_hard = df_features1.groupby(level=0)\
            [['num_sd0','num_sd1','num_sd2',
              'help_sd0', 'help_sd1', 'help_sd2',
              'usage_sd0', 'usage_sd1', 'usage_sd2']].sum()

        # Fills 0s for inactive subscribers
        df_hard_inactive = pd.DataFrame(0, 
                                        index=self.pupils_inactive_churnOption_, 
                                        columns=df_hard.columns)
        if not self.config.FILL_NAN_BY_ZERO:
            # number of helps used by inactive subscribers are NA because they 
            # do not take any lesson
            df_hard_inactive.loc[:, ['help_sd0', 'help_sd1', 'help_sd2']] = \
                np.nan

        df_hard = df_hard.append(df_hard_inactive)

        # Add more features
        # Note: leave NaN for inactive subscribers as the rate is not defined
        # for them
        df_hard = df_hard.assign(
            sum_help=df_hard.help_sd0+df_hard.help_sd1+df_hard.help_sd2,
            rate_stackDepth23_num=df_hard.num_sd2/\
                (df_hard.num_sd0+df_hard.num_sd1+df_hard.num_sd2),
            rate_stackDepth23_usage=df_hard.usage_sd2/\
                (df_hard.usage_sd0+df_hard.usage_sd1+df_hard.usage_sd2))
        
        self.df_whizz_ = pd.concat([self.df_whizz_, df_hard], axis=1)

    def add_mathAge(self):
        df_features1 = self._df_features
        
        math_age = df_features1.groupby(level=0)['math_age'].mean()
        age = self.df_whizz_['age']
        age_diff = age - math_age

        self.df_whizz_ = self.df_whizz_.assign(math_age=math_age,
                                               age_diff=age_diff)