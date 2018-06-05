from pomegranate import *

class FeatureAttribute(object):
    """Struct-like class for configuring data transformation and distributional 
    modelling for different features. 
    """
    name = ''
    
    multiplierPre = 1.
    shiftPre = 0.
    bcTransform = False
    multiplierPost = 100-1.
    shiftPost = 1.
    distributionList = []

class FeatureConfig_G1(object):
    """Feature configuration for Group 1.
    """

    def __init__(self):
        self.ftr_list_independent = []

        self.ftr_list_multivariate = ['last_access',
                                      'progress',
                                      'age',
                                      'calendar_month']
        
        self.ftr_list = self.ftr_list_independent + self.ftr_list_multivariate

        # Multivariate Components
        self.multivariate_shift = 1.

class FeatureConfig_G2(object):
    """Feature configuration for Group 2.
    """

    def __init__(self):
        self.ftr_list_independent = ['rate_incomplete_num']

        self.ftr_list_multivariate = ['num_attempt',
                                      'last_access', 
                                      'usage',
                                      'sum_help',
                                      'age',
                                      'progress',
                                      'calendar_month']
        
        self.ftr_list = self.ftr_list_independent + self.ftr_list_multivariate
        
        # Multivariate Components
        self.multivariate_shift = 1.

        # Independent Components
        self.rate_incomplete_num = FeatureAttribute()
        self.rate_incomplete_num.name = 'rate_incomplete_num'
        self.rate_incomplete_num.shiftPre = 1.
        self.rate_incomplete_num.bcTransform = True
        self.rate_incomplete_num.distributionList = [UniformDistribution(0,10),
                                                     UniformDistribution(90, 100),
                                                     NormalDistribution(50, 20)]

class FeatureConfig_G3(object):
    """Feature configuration for Group 3.
    """

    def __init__(self):
        self.ftr_list_independent = ['rate_incomplete_num',
                                     'rate_fail',
                                     'rate_stackDepth23_num',
                                     'rate_assess']

        self.ftr_list_multivariate = ['num_attempt',
                                      'last_access', 
                                      'usage',
                                      'sum_help',
                                      'age',
                                      'mark_complete',
                                      'progress', 'progress_delta',
                                      'calendar_month',
                                      'effective_progress']

        self.ftr_list = self.ftr_list_independent + self.ftr_list_multivariate

        # Multivariate Components
        self.multivariate_shift = 1.

        # Independent Components
        self.rate_incomplete_num = FeatureAttribute()
        self.rate_incomplete_num.name = 'rate_incomplete_num'
        self.rate_incomplete_num.shiftPre = 1.
        self.rate_incomplete_num.bcTransform = True
        self.rate_incomplete_num.distributionList = [UniformDistribution(0,4),
                                                     NormalDistribution(20, 9),
                                                     NormalDistribution(55, 10),
                                                     NormalDistribution(85, 6)]
        
        self.rate_incomplete_usage = FeatureAttribute()
        self.rate_incomplete_usage.name = 'rate_incomplete_usage'
        self.rate_incomplete_usage.shiftPre = 1.
        self.rate_incomplete_usage.bcTransform = True
        self.rate_incomplete_usage.distributionList = [UniformDistribution(0,4),
                                                     NormalDistribution(20, 9),
                                                     NormalDistribution(55, 10),
                                                     NormalDistribution(85, 6)]

        self.rate_fail = FeatureAttribute()
        self.rate_fail.name = 'rate_fail'
        self.rate_fail.shiftPre = 1.
        self.rate_fail.bcTransform = True
        self.rate_fail.distributionList = [UniformDistribution(0,4),
                                           UniformDistribution(94, 100),
                                           NormalDistribution(25, 10),
                                           NormalDistribution(50, 15)]
        
        self.rate_stackDepth23_num = FeatureAttribute()
        self.rate_stackDepth23_num.name = 'rate_stackDepth23_num'
        self.rate_stackDepth23_num.shiftPre = 2.
        self.rate_stackDepth23_num.multiplierPre = -1.
        self.rate_stackDepth23_num.bcTransform = True
        self.rate_stackDepth23_num.distributionList = [ExponentialDistribution(0.25),
                                                       NormalDistribution(25, 10),
                                                       NormalDistribution(55, 15),
                                                       UniformDistribution(91, 100)]
        
        self.rate_assess = FeatureAttribute()
        self.rate_assess.name = 'rate_assess'
        self.rate_assess.shiftPre = 2.
        self.rate_assess.multiplierPre = -1.
        self.rate_assess.bcTransform = True
        self.rate_assess.distributionList = [UniformDistribution(0, 4), 
                                             NormalDistribution(14, 6),
                                             NormalDistribution(50, 20),
                                             NormalDistribution(90, 8)]

