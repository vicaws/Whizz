import pandas as pd

def retrieve_data(configuration):
    '''
    Load data from CSV files and pre-process data
    '''
    
    # Load Data
    fpath = configuration.DATA_FOLDER_PATH
    fpathchar = configuration.FILE_SUBFOLDER

    fname_lesson = fpath + fpathchar + configuration.FILE_LH_COMPLETE
    fname_subspt = fpath + fpathchar + configuration.FILE_SUBSCRIPTION

    df_lesson = pd.read_csv(fname_lesson, delimiter=',')
    df_subspt = pd.read_csv(fname_subspt, delimiter=',')

    # Renew DataFrame indices
    df_lesson.set_index('lesson_history_id', inplace=True)

    time_format = configuration.CSV_TIME_FORMAT
    date_format = configuration.CSV_DATE_FORMAT

    # Covert duration string to numerics (in seconds)
    df_lesson.timeTaken = pd.to_datetime(df_lesson.timeTaken, format=time_format)
    root_datetime = pd.to_datetime("00:00:00", format=time_format)
    df_lesson.timeTaken = df_lesson.timeTaken - root_datetime
    df_lesson.timeTaken = df_lesson.timeTaken.dt.seconds

    # Convert datetime string to object
    df_lesson.marked = pd.to_datetime(df_lesson.marked, format=date_format+" "+time_format)
    
    df_subspt.subscription_start_date = pd.to_datetime(df_subspt.subscription_start_date, format=date_format)
    df_subspt.subscription_end_date = pd.to_datetime(df_subspt.subscription_end_date, format=date_format)

    df_subspt['subscription_length'] = (df_subspt.subscription_end_date-df_subspt.subscription_start_date).dt.days
    
    return df_subspt, df_lesson