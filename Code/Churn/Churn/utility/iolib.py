import pandas as pd

def _generate_lesson_key(df):
    
    num2str = lambda x: str(x) if x>=1000 else '0'+str(x)
    lsntype = lambda x: x[-2:]

    df['lesson_key'] = df['topicId'] + df['age'].apply(num2str) + df['lesson_type'].apply(lsntype) + df['exerciseId'].apply(num2str)
    
    return df

def retrieve_data(configuration):
    '''
    Load data from CSV files and pre-process data
    '''
    
    # Load Data
    fpath = configuration.DATA_FOLDER_PATH
    fpathchar = configuration.FILE_SUBFOLDER

    fname_lesson = fpath + fpathchar + configuration.FILE_LH_COMPLETE
    fname_incomp = fpath + fpathchar + configuration.FILE_LH_INCOMPLETE
    fname_crclum = fpath + fpathchar + configuration.FILE_CURRICULUM
    fname_subspt = fpath + fpathchar + configuration.FILE_SUBSCRIPTION

    df_lesson = pd.read_csv(fname_lesson, delimiter=',')
    df_incomp = pd.read_csv(fname_incomp, delimiter=',')
    df_crclum = pd.read_csv(fname_crclum, delimiter=',')
    df_subspt = pd.read_csv(fname_subspt, delimiter=',')

    time_format = configuration.CSV_TIME_FORMAT
    date_format = configuration.CSV_DATE_FORMAT
    root_datetime = pd.to_datetime("00:00:00", format=time_format)

    ## Pre-process fname_lesson
    # Renew DataFrame indices
    df_lesson.set_index('lesson_history_id', inplace=True)
    # Covert duration string to numerics (in seconds)
    df_lesson.timeTaken = pd.to_datetime(df_lesson.timeTaken, format=time_format)
    df_lesson.timeTaken = df_lesson.timeTaken - root_datetime
    df_lesson.timeTaken = df_lesson.timeTaken.dt.seconds
    # Convert datetime string to object
    df_lesson.marked = pd.to_datetime(df_lesson.marked, format=date_format+" "+time_format)
    # Re-fill 'NA' in the field topicId
    df_lesson['topicId'] = df_lesson['topicId'].fillna('NA')
    # Add lesson keys
    df_lesson = _generate_lesson_key(df_lesson)
    
    ## Pre-process fname_incomp
    # Renew DataFrame indices
    df_incomp.set_index('incomplete_lesson_log_id', inplace=True)
    # Covert duration string to numerics (in seconds)
    df_incomp['time_taken'] = pd.to_datetime(df_incomp['time_taken'], format=time_format)
    df_incomp['time_taken'] = df_incomp['time_taken'] - root_datetime
    df_incomp['time_taken'] = df_incomp['time_taken'].dt.seconds
    # Convert datetime string to object
    df_incomp['created'] = pd.to_datetime(df_incomp['created'], format=date_format+" "+time_format)
    # Re-fill 'NA' in the field topicId
    df_incomp['topicId'] = df_incomp['topicId'].fillna('NA')
    # Add lesson keys
    df_incomp = _generate_lesson_key(df_incomp)

    ## Pre-process fname_subspt
    # Convert datetime string to object
    df_subspt.subscription_start_date = pd.to_datetime(df_subspt.subscription_start_date, format=date_format)
    df_subspt.subscription_end_date = pd.to_datetime(df_subspt.subscription_end_date, format=date_format)
    # Add a new column measuring the subscription lenth
    df_subspt['subscription_length'] = (df_subspt.subscription_end_date-df_subspt.subscription_start_date).dt.days
    
    ## Pre-process fname_crclum
    # NOTE: in the topicId field, "NA" entries will be read as nan type, need to convert back to string type
    df_crclum['topicId'] = df_crclum['topicId'].fillna('NA')

    return df_subspt, df_lesson, df_incomp, df_crclum