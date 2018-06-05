import numpy as np
import pandas as pd

import utility.plotlib

from pomegranate import *
from model.featureConfig import \
    FeatureConfig_G1, FeatureConfig_G2, FeatureConfig_G3

class ModelParam(object):
    """Struct-like class for holding mixture modelling parameters.
    
    Attributes
    ----------
    n_components: int
        The (maximal) number of components of multivariate distributions for 
        (Baysian) Gaussian mixture model.

    n_trials: int
        The number of initialisations/trials for running the EM algorithms to 
        estimate weights and distributional parameters for the multivariate 
        Gaussian mixture model.
        
    n_multi: int
        We can include more than one fitted multivariate Gaussian mixture model 
        (GMM) to identify clustering within each trial. This parameter indicates
        the number of fitted GMMs we would like to use.

    baysian: boolean
        True if we want to use Baysian GMM, False if we want to use GMM. Baysian
        GMM infers the number of components from the data set, but the inferred 
        number if not more than the specified n_components.

    """

    def __init__(self, n_components, n_trials, n_multi, baysian):
        self.n_components_ = n_components
        self.n_trials_ = n_trials
        self.n_multi_ = n_multi
        self.baysian_ = baysian

class GroupConfig(object):
    """Struct-like class for configuring mixture modelling parameters for 
    different groups.
    """

    def __init__(self, data_engine, 
                 modelParam_G1, modelParam_G2, modelParam_G3):
        # Import data frame from DataEngine object
        self.df_whizz_ = data_engine.df_whizz_

        # Define mask for grouping
        mask_inactive = self.df_whizz_['active']==0
        mask_noassess = (self.df_whizz_['active']==1) & \
            (self.df_whizz_['assess']==0)
        mask_fine = ~mask_inactive & ~mask_noassess

        # Define data frames of groups
        self.df_whizz_G1_ = self.df_whizz_[mask_inactive]
        self.df_whizz_G2_ = self.df_whizz_[mask_noassess]
        self.df_whizz_G3_ = self.df_whizz_[mask_fine]

        self.ftrConfig_G1_ = FeatureConfig_G1()
        self.ftrConfig_G2_ = FeatureConfig_G2()
        self.ftrConfig_G3_ = FeatureConfig_G3()

        # Define GMM parameters
        self.modelParam_G1_ = modelParam_G1
        self.modelParam_G2_ = modelParam_G2
        self.modelParam_G3_ = modelParam_G3

class MixtureModel(object):
    """Class of fitting and evaluating mixture model
    """

    def __init__(self, group_config):
        self.group_config_ = group_config
        self.map_group_model_ = {}

    def fit(self):
        # Fit for Group 1
        print('Start fitting mixture model for G1.')
        df_grouping_list1, map_group_pupilId_list1, map_group_cmonth_list1 = \
            self.fit_mixtureModel(
                self.group_config_.df_whizz_G1_,
                self.group_config_.ftrConfig_G1_, 
                self.group_config_.modelParam_G1_.n_components_, 
                self.group_config_.modelParam_G1_.n_trials_, 
                self.group_config_.modelParam_G1_.n_multi_, 
                self.group_config_.modelParam_G1_.baysian_)
        
        # Fit for Group 2
        print('Start fitting mixture model for G2.')
        df_grouping_list2, map_group_pupilId_list2, map_group_cmonth_list2 = \
            self.fit_mixtureModel(
                self.group_config_.df_whizz_G2_, 
                self.group_config_.ftrConfig_G2_, 
                self.group_config_.modelParam_G2_.n_components_,
                self.group_config_.modelParam_G2_.n_trials_, 
                self.group_config_.modelParam_G2_.n_multi_, 
                self.group_config_.modelParam_G2_.baysian_)

        # Fit for Group 3
        print('Start fitting mixture model for G3.')
        df_grouping_list3, map_group_pupilId_list3, map_group_cmonth_list3 = \
            self.fit_mixtureModel(
                self.group_config_.df_whizz_G3_, 
                self.group_config_.ftrConfig_G3_, 
                self.group_config_.modelParam_G3_.n_components_, 
                self.group_config_.modelParam_G3_.n_trials_, 
                self.group_config_.modelParam_G3_.n_multi_, 
                self.group_config_.modelParam_G3_.baysian_)
        
        map_group_model = {'G1': [df_grouping_list1, 
                                  map_group_pupilId_list1, 
                                  map_group_cmonth_list1],
                           'G2': [df_grouping_list2, 
                                  map_group_pupilId_list2, 
                                  map_group_cmonth_list2],
                           'G3': [df_grouping_list3, 
                                  map_group_pupilId_list3, 
                                  map_group_cmonth_list3]}

        self.map_group_model_ = map_group_model

    def fit_independentComponent(self, df_whizz, feature_config, feature_name, 
                                 plot=0, hist_bin=35):
        '''Fit mixed distributions for features defined as independent component.
    
        Parameters
        ----------
        df_whizz: pandas.DataFrame
            Data frame storing feature and target data for specific group.
    
        feature_config: FeatureConfig object
            Feature configuration for specific group.
    
        feature_name: string
            Feature name.
    
        plot: int
            If plot=0, then no plot will be shown;
            If plot=1, then the density plot of fitted mixture model along with 
            histogram will be shown;
            If plot>1, then the Gaussian KDE and histogram plots of raw and 
            transformed data will be shown.
    
        hist_bin: int
            The number of bins in the density and histogram plot (when plot=1).
    
        Returns
        -------
        gmm: GeneralMixtureModel object
            The trained GeneralMixtureModel object which can be used to predict 
            the labels and probabilities.
    
        x: array_like, shape=(n_sample, 1)
            Feature data.
        '''
        from scipy import stats
        from sklearn.preprocessing import MinMaxScaler
    
        ftr_config = eval('feature_config.'+feature_name)

        if plot>1:
            utility.plotlib.compare_transformed_singleValueRemoved(
                df_whizz, feature_name, 'min')
        
        # Define the data to be fitted
        x_raw = df_whizz[feature_name].values
        # Linear transformation
        x = x_raw * ftr_config.multiplierPre
        x += ftr_config.shiftPre
        # Box-cox transformation
        if ftr_config.bcTransform:
            xt, bc_param = stats.boxcox(x)
            x = xt
        x = x.reshape(-1,1)
        # Standardisation
        scaler = MinMaxScaler()
        scaler.fit(x)
        x = scaler.transform(x)
        x *= ftr_config.multiplierPost
        x += ftr_config.shiftPost

        # Define the list of distributions to be mixed
        distribution_list = ftr_config.distributionList

        gmm = GeneralMixtureModel(distribution_list)
        print('Improvement = {}'.format(gmm.fit(x, verbose=False)))
        print('Weights = {}'.format(np.exp(gmm.weights)))

        if plot==1:
            utility.plotlib.density_mixtureModel(
                feature_name, x, gmm, hist_bin=hist_bin)
    
        return gmm, x

    def getData_multivariateComponent(self, df_whizz, feature_config):
        '''Prepare the feature and target data for multivariate compoments.
    
        Parameters
        ----------
        df_whizz: pandas.DataFrame
            Data frame storing feature and target data for specific group.
        
        feature_config: FeatureConfig object
            Feature configuration for specific group.
    
        Returns
        -------
        X: array_like, shape=(n_sample, n_features)
            Feature data.
        
        y: array_like, shape=(n_sample, 1)
            Target data (1 for churn, 0 otherwise).
        '''
    
        from scipy import stats
        from sklearn.preprocessing import MinMaxScaler
    
        ftr_list_multivariate = feature_config.ftr_list_multivariate

        # Linear transformation
        X = np.array(df_whizz[ftr_list_multivariate]+\
            feature_config.multivariate_shift)
        # Box-Cox transformation
        Xt = X
        for i in range(0, X.shape[1]):
            xt, _ = stats.boxcox(X[:,i])
            Xt[:, i] = xt
        # Standardisation
        Xt_scaled = MinMaxScaler().fit(Xt).transform(Xt)

        y = df_whizz.churn.values
    
        return Xt_scaled, y

    def construct_df_grouping(self, group, group_name, df_whizz):
        # Identify unique groups/labels and frequency
        label = np.array(group).transpose()
        unq_rows, count = np.unique(label, axis=0, return_counts=True)
        map_group_count = {tuple(i):j for i,j in zip(unq_rows,count)}

        # Compute churn rate within each group
        map_group_churn = {}
        map_group_pupilId = {}
        map_group_cmonth = {}
        pupilId = df_whizz.reset_index()['pupilId'].values
        customer_month = df_whizz.reset_index()['customer_month'].values
        y = df_whizz.churn.values

        for k in range(0, len(unq_rows)):
            indices = [i for i, x in enumerate(label.tolist()) if 
                       x==unq_rows[k].tolist()]
            l = y[indices]
            map_group_churn[tuple(unq_rows[k])] = l.sum()*1. / len(l)
            map_group_pupilId[k] = pupilId[indices]
            map_group_cmonth[k] = customer_month[indices]
        
        # Construct the grouping data frame
        df_grouping = pd.DataFrame(unq_rows, columns=group_name)
        df_grouping['groupId'] = range(len(df_grouping))
        df_grouping = df_grouping.assign(count=count,
                                         churn=list(map_group_churn.values()))
        df_grouping.sort_values(by='churn', ascending=False, inplace=True)
        df_grouping['cumcount'] = df_grouping['count'].cumsum()
    
    
        return df_grouping, map_group_pupilId, map_group_cmonth

    def compute_groupingScore(self, df_grouping_list, expectation, 
                              criterion='deviation'):
        score_list = []
        y = expectation
        for df_grouping in df_grouping_list:
            base_churn = y.sum()*1. / len(y)
            group_churn = df_grouping['churn'].values
            group_count = df_grouping['count'].values
            if criterion=='deviation':
                score = np.sum(abs(base_churn-group_churn)* group_count)
            elif criterion=='variance':
                score = np.sum((base_churn-group_churn)**2 * group_count**2)
            elif criterion=='max_number':
                score = max((group_churn-base_churn)* group_count)
            elif criterion=='min_number':
                score = -min((group_churn-base_churn)* group_count)
            score_list.append(score)
    
        return score_list

    def fit_mixtureModel(self, df_whizz, ftrConfig, 
                         n_components, n_trials, n_multi, baysian):
        from sklearn import mixture
        X, y = self.getData_multivariateComponent(df_whizz, ftrConfig)

        ftr_list_multivariate = ftrConfig.ftr_list_multivariate
        ftr_list_independent = ftrConfig.ftr_list_independent

        # Mixture model for independent components
        group_ic = []
        group_ic_name = []
        n_var_indep = len(ftr_list_independent)
        for i_var in range(n_var_indep):
            ftr_str = ftr_list_independent[i_var]
            gmm, x = self.fit_independentComponent(df_whizz, ftrConfig, ftr_str,
                                                   plot=0, hist_bin=35)
            l = gmm.predict(x)
            group_ic.append(l)
            group_ic_name.append('indep'+str(i_var))
    
        # Mixture model for multivariate components
        n_features = len(ftr_list_multivariate)
        df_grouping_list = []
        map_group_pupilId_list = []
        map_group_cmonth_list = []
        print('Start fitting multivariate mixture models.')
        for trial in range(0, n_trials):
            print('Trial NO. = {}/{}'.format(trial+1, n_trials))
    
            group = group_ic[:]
            group_name = group_ic_name[:]
    
            for i_multi in range(0, n_multi):
                if baysian:
                    dpgmm = mixture.BayesianGaussianMixture(
                        n_components=n_components,
                        covariance_type='full',
                        covariance_prior=1e0 * np.eye(n_features),
                        weight_concentration_prior_type='dirichlet_process',
                        init_params="random", tol=1e-6, max_iter=10000, n_init=1, 
                        verbose=0, verbose_interval=100).fit(X)
                    l = dpgmm.predict(X)
                else:
                    gmm = mixture.GaussianMixture(
                        n_components=n_components, 
                        covariance_type='full',
                        init_params='random',
                        verbose=0, verbose_interval=100,
                        n_init=1, tol=1e-6, max_iter=1000).fit(X)
                    l = gmm.predict(X)
        
                group.append(l)
                group_name.append('multi'+str(i_multi))
    
            df_grouping, map_group_pupilId, map_group_cmonth = \
                self.construct_df_grouping(group, group_name, df_whizz)
            df_grouping_list.append(df_grouping)
            map_group_pupilId_list.append(map_group_pupilId)
            map_group_cmonth_list.append(map_group_cmonth)
        
        return  df_grouping_list, map_group_pupilId_list, map_group_cmonth_list

    def select_bestMixtureModel(self, criterion='deviation'):
        ''' Evaluate different groupings and select the best
        '''
        map_group_model_best = {}
        map_group_expectaion = {}

        # G1
        y = self.group_config_.df_whizz_G1_['churn'].values
        map_group_expectaion['G1'] = y
        df_grouping_list = self.map_group_model_['G1'][0]
        map_group_pupilId_list = self.map_group_model_['G1'][1]
        map_group_cmonth_list = self.map_group_model_['G1'][2]

        score_list = self.compute_groupingScore(df_grouping_list, y, criterion)
        idx_maxScore = np.argmax(score_list)
        map_group_model_best['G1'] = [df_grouping_list[idx_maxScore],
                                      map_group_pupilId_list[idx_maxScore],
                                      map_group_cmonth_list[idx_maxScore]]
        # G2
        y = self.group_config_.df_whizz_G2_['churn'].values
        map_group_expectaion['G2'] = y
        df_grouping_list = self.map_group_model_['G2'][0]
        map_group_pupilId_list = self.map_group_model_['G2'][1]
        map_group_cmonth_list = self.map_group_model_['G2'][2]

        score_list = self.compute_groupingScore(df_grouping_list, y, criterion)
        idx_maxScore = np.argmax(score_list)
        map_group_model_best['G2'] = [df_grouping_list[idx_maxScore],
                                      map_group_pupilId_list[idx_maxScore],
                                      map_group_cmonth_list[idx_maxScore]]
        # G3
        y = self.group_config_.df_whizz_G3_['churn'].values
        map_group_expectaion['G3'] = y
        df_grouping_list = self.map_group_model_['G3'][0]
        map_group_pupilId_list = self.map_group_model_['G3'][1]
        map_group_cmonth_list = self.map_group_model_['G3'][2]

        score_list = self.compute_groupingScore(df_grouping_list, y, criterion)
        idx_maxScore = np.argmax(score_list)
        map_group_model_best['G3'] = [df_grouping_list[idx_maxScore],
                                      map_group_pupilId_list[idx_maxScore],
                                      map_group_cmonth_list[idx_maxScore]]
    
        self.map_group_expectation_ = map_group_expectaion
        return map_group_model_best