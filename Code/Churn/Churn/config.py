class Config:
    # Data
    DATA_FOLDER_PATH = "../../../Data/"
    CSV_TIME_FORMAT = "%H:%M:%S"
    CSV_DATE_FORMAT = "%Y-%m-%d"
    
    FILE_SUBSCRIPTION = "subscriptions_table.csv"
    FILE_LH_COMPLETE = "homesubs_lh.csv"
    FILE_LH_INCOMPLETE = "homesubs_incomplete_lh.csv"
    FILE_CURRICULUM = "curriculum_number_of_lessons.csv"
    
    TYPE_MONTHLY = "Monthly"
    TYPE_ANNUAL = "Annual"

class ResearchConfig(Config):
    
    FILE_SUBFOLDER = "20180426/"
    
    CUTOFF_DATE = "2018-04-20"
    RENEWAL_GAP = 5
    
class TestConfig(Config):
    pass