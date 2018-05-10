################################################################################
# RULE of Length of Line = 80 chars

class Config:
    # Data
    DATA_FOLDER_PATH = "../../../Data/"
    CSV_TIME_FORMAT = "%H:%M:%S"
    CSV_DATE_FORMAT = "%Y-%m-%d"
    
    FILE_SUBSCRIPTION = "subscriptions_table.csv"
    FILE_LH_COMPLETE = "homesubs_lh.csv"
    FILE_LH_INCOMPLETE = "homesubs_incomplete_lh.csv"
    FILE_CURRICULUM = "curriculum_number_of_lessons.csv"
    FILE_PUPILS = "pupils.csv"
    
    TYPE_MONTHLY = "Monthly"
    TYPE_ANNUAL = "Annual"

    IGNORE_UNSTEADY_PERIOD = False

    # Output
    PLOT_FOLDER = "../../../Result/Plot/"

class ResearchConfig(Config):  
    # Data
    FILE_SUBFOLDER = "20180503/"
    FILE_INTERMEDIATE = FILE_SUBFOLDER + "Intermediate/"

    # Study
    CUTOFF_DATE = "2018-04-20"
    RETURN_GAP = 5
    RETURN_GAP_UNIT = 'D'

    IGNORE_UNSTEADY_PERIOD = True

    # Output
    DATA_DESCR = "descriptive_stats.csv"
    DATA_DATES = "dates_frame.csv"
    DATA_USAGE = "usage.csv"

    PLOT_SUBSPT_DIST = "subspt_dist.png"
    PLOT_ACTIVE_SUBSPT = "active_subspt.png"
    
    PLOT_PERM_UNIT = 'M'
    PLOT_PERM_COUNT = "performance_count.png"
    PLOT_PERM_RATIO = "performance_ratio.png"

    PLOT_SURVIVAL = "survival_analysis.png"
    PLOT_SURVIVAL_CM = "survival_analysis_CM.png"
    
class TestConfig(Config):
    pass