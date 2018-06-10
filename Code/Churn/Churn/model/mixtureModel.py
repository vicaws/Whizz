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

    Attributes
    ----------
    map_group_model_: dictionary,
            key=string, group ID - 'G1', 'G2' or 'G3';
            value=list of 3 elements.
        The value is a list of 3 elements:
        (1) df_grouping_list: list of df_grouping data frame of different 
            trials;
        (2) map_group_pupilId_list: list if map_group_pupilId dictionary of 
            different trials;
        (3) map_group_cmonth_list: list if map_group_cmonth dictionary of 
            different trials;

    """

    def __init__(self, group_config):
        self.group_config_ = group_config
        self.map_group_model_ = {}
        self.map_group_model_best_ = {}
        self.map_group_expectation_ = {}
        self.map_group_transform_ = {}

        self.df_cluster_ = pd.DataFrame()

    def fit(self):
        # Fit for Group 1
        print('Start fitting mixture model for G1.')
        df_grouping_list1, map_group_pupilId_list1, map_group_cmonth_list1, \
            map_group_mModel_list1, map_var_ftr1, map_ftr_transform1 = \
            self.fit_mixtureModel(
                self.group_config_.df_whizz_G1_,
                self.group_config_.ftrConfig_G1_, 
                self.group_config_.modelParam_G1_.n_components_, 
                self.group_config_.modelParam_G1_.n_trials_, 
                self.group_config_.modelParam_G1_.n_multi_, 
                self.group_config_.modelParam_G1_.baysian_)
        
        # Fit for Group 2
        print('Start fitting mixture model for G2.')
        df_grouping_list2, map_group_pupilId_list2, map_group_cmonth_list2, \
           map_group_mModel_list2, map_var_ftr2, map_ftr_transform2 = \
            self.fit_mixtureModel(
                self.group_config_.df_whizz_G2_, 
                self.group_config_.ftrConfig_G2_, 
                self.group_config_.modelParam_G2_.n_components_,
                self.group_config_.modelParam_G2_.n_trials_, 
                self.group_config_.modelParam_G2_.n_multi_, 
                self.group_config_.modelParam_G2_.baysian_)

        # Fit for Group 3
        print('Start fitting mixture model for G3.')
        df_grouping_list3, map_group_pupilId_list3, map_group_cmonth_list3, \
           map_group_mModel_list3, map_var_ftr3, map_ftr_transform3 = \
            self.fit_mixtureModel(
                self.group_config_.df_whizz_G3_, 
                self.group_config_.ftrConfig_G3_, 
                self.group_config_.modelParam_G3_.n_components_, 
                self.group_config_.modelParam_G3_.n_trials_, 
                self.group_config_.modelParam_G3_.n_multi_, 
                self.group_config_.modelParam_G3_.baysian_)
        
        map_group_model = {'G1': [df_grouping_list1, 
                                  map_group_pupilId_list1, 
                                  map_group_cmonth_list1,
                                  map_group_mModel_list1,
                                  map_var_ftr1,
                                  map_ftr_transform1],
                           'G2': [df_grouping_list2, 
                                  map_group_pupilId_list2, 
                                  map_group_cmonth_list2,
                                  map_group_mModel_list2,
                                  map_var_ftr2,
                                  map_ftr_transform2],
                           'G3': [df_grouping_list3, 
                                  map_group_pupilId_list3, 
                                  map_group_cmonth_list3,
                                  map_group_mModel_list3,
                                  map_var_ftr3,
                                  map_ftr_transform3]}

        self.map_group_model_ = map_group_model

    def cluster(self, map_cluster_anchors):
        '''Cluster subgroups in each group accroding to the pre-defined anchor 
        points.
        '''
        list_group_name = ['G1','G2','G3']
        list_df_grouping_cluster = []
        list_map_cluster_pupilId = []
        list_map_cluster_cmonth = []

        for g in list_group_name:
            # Get data
            df_grouping = self.map_group_model_best_[g][0]
            map_group_pupilId = self.map_group_model_best_[g][1]
            map_group_cmonth = self.map_group_model_best_[g][2]
            y = self.map_group_expectation_[g]
            cluster_anchors = map_cluster_anchors[g]
            
            # Define place-holder
            num_clusters = len(cluster_anchors) - 1
            cluster_churn = []
            cluster_count = []
            cluster_Id = []
            map_cluster_pupilId = {}
            map_cluster_cmonth = {}

            l = np.arange(0, len(y)) + 1
            for i in range(len(cluster_anchors)):
                if i==0:
                    continue
                else:
                    lower = np.percentile(l, cluster_anchors[i-1])
                    upper = np.percentile(l, cluster_anchors[i])

                # Define sub-dataframe: [...) [...) ... [...) [...]
                if i==len(cluster_anchors)-1:
                    mask = (df_grouping['cumcount']>=lower) & \
                        (df_grouping['cumcount']<=upper)
                else:
                    mask = (df_grouping['cumcount']>=lower) & \
                        (df_grouping['cumcount']<upper)
                
                df = df_grouping[mask]

                # Get information of pupilId and customer_month
                groupId = df['groupId'].values
                pupilId = np.hstack([map_group_pupilId[key] for key in groupId])
                cmonth = np.hstack([map_group_cmonth[key] for key in groupId])

                temp = df['churn'] * df['count']
                num_churn = temp.sum()
                cluster_count.append(df['count'].sum())
                cluster_churn.append(num_churn*1./cluster_count[i-1])
                cluster_Id.append(g+str(i))
                map_cluster_pupilId[g+str(i)] = pupilId
                map_cluster_cmonth[g+str(i)] = cmonth

            # Construct df_grouping_cluster
            data = {'count': cluster_count,
                    'churn': cluster_churn,
                    'clusterId': cluster_Id}
            df_grouping_cluster = pd.DataFrame(data)
            df_grouping_cluster['cumcount'] = \
                df_grouping_cluster['count'].cumsum()

            # Update the list place-holder
            list_df_grouping_cluster.append(df_grouping_cluster)
            list_map_cluster_pupilId.append(map_cluster_pupilId)
            list_map_cluster_cmonth.append(map_cluster_cmonth)
        
        # Aggregate over groups G1, G2 and G3
        df_cluster = pd.DataFrame()
        map_cluster_pupilId = {}
        map_cluster_cmonth = {}
        for i in range(len(list_group_name)):
            df_cluster = pd.concat([df_cluster, list_df_grouping_cluster[i]], 
                                   axis=0)
            map_cluster_pupilId.update(list_map_cluster_pupilId[i])
            map_cluster_cmonth.update(list_map_cluster_cmonth[i])

        df_cluster.reset_index(inplace=True)

        # Output
        self.df_cluster_ = df_cluster

        # Construct df_cluster_details which holds information of pupilId and 
        # customer_month
        df_cluster_details = pd.DataFrame()
        for key in map_cluster_pupilId.keys():
            df = pd.DataFrame()
            df = df.assign(pupilId = map_cluster_pupilId[key],
                           cmonth = map_cluster_cmonth[key])
            df['clusterId'] = key
            df_cluster_details = pd.concat([df_cluster_details, df], axis=0)

        df_cluster_details.reset_index(inplace=True)

        # Output
        self.df_cluster_details_ = df_cluster_details

    def transform(self, feature_name, feature_data):
        
        pass

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
        else:
            bc_param = -10000
        x = x.reshape(-1,1)
        # Standardisation
        scaler = MinMaxScaler()
        scaler.fit(x)
        x = scaler.transform(x)
        x *= ftr_config.multiplierPost
        x += ftr_config.shiftPost

        # Record the transform
        map_transform_coeff = {'multiplierPre': ftr_config.multiplierPre,
                               'shiftPre': ftr_config.shiftPre,
                               'multiplierPost': ftr_config.multiplierPost,
                               'shiftPost': ftr_config.shiftPost,
                               'scaler': scaler,
                               'bc': bc_param}

        # Define the list of distributions to be mixed
        distribution_list = ftr_config.distributionList

        gmm = GeneralMixtureModel(distribution_list)
        print('Improvement = {}'.format(gmm.fit(x, verbose=False)))
        print('Weights = {}'.format(np.exp(gmm.weights)))

        if plot==1:
            utility.plotlib.density_mixtureModel(
                feature_name, x, gmm, hist_bin=hist_bin)
    
        return gmm, x, map_transform_coeff

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
        map_ftr_transform = {}

        # Linear transformation
        X = np.array(df_whizz[ftr_list_multivariate]+\
            feature_config.multivariate_shift)
        # Box-Cox transformation
        Xt = X
        for i in range(0, X.shape[1]):
            xt, bc_param = stats.boxcox(X[:,i])
            Xt[:, i] = xt
            # Record the transform
            map_transform_coeff = {'bc': bc_param}
            map_ftr_transform[ftr_list_multivariate[i]] = map_transform_coeff
        # Standardisation
        scaler = MinMaxScaler().fit(Xt)
        Xt_scaled = scaler.transform(Xt)
        
        map_ftr_transform['multivariate_shift'] = \
            feature_config.multivariate_shift
        map_ftr_transform['multivariate_scaler'] = scaler

        y = df_whizz.churn.values
    
        return Xt_scaled, y, map_ftr_transform

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
            elif criterion=='max_rate':
                score = max(group_churn)
            elif criterion=='min_number':
                score = -min((group_churn-base_churn)* group_count)
            elif criterion=='min_rate':
                score = -min(group_churn)
            score_list.append(score)
    
        return score_list

    def fit_mixtureModel(self, df_whizz, ftrConfig, 
                         n_components, n_trials, n_multi, baysian):
        from sklearn import mixture
        X, y, map_ftr_transform =\
           self.getData_multivariateComponent(df_whizz, ftrConfig)

        ftr_list_multivariate = ftrConfig.ftr_list_multivariate
        ftr_list_independent = ftrConfig.ftr_list_independent

        map_var_ftr = {}
        
        # Mixture model for independent components
        group_ic = []
        group_ic_name = []
        map_group_mModel_ic = {}
        n_var_indep = len(ftr_list_independent)
        for i_var in range(n_var_indep):
            ftr_str = ftr_list_independent[i_var]
            gmm, x, map_transform_coeff = \
                self.fit_independentComponent(df_whizz, ftrConfig, ftr_str,
                                              plot=0, hist_bin=35)
            l = gmm.predict(x)
            group_ic.append(l)
            group_ic_name.append('indep'+str(i_var))
            map_group_mModel_ic['indep'+str(i_var)] = gmm
            map_var_ftr['indep'+str(i_var)] = ftr_str
            
            # Update the data transformation map
            map_ftr_transform[ftr_str] = map_transform_coeff

        # Mixture model for multivariate components
        n_features = len(ftr_list_multivariate)
        df_grouping_list = []
        map_group_pupilId_list = []
        map_group_cmonth_list = []
        map_group_mModel_list = []
        print('Start fitting multivariate mixture models.')
        for trial in range(0, n_trials):
            print('Trial NO. = {}/{}'.format(trial+1, n_trials))
            
            group = group_ic[:] # Pass by value
            group_name = group_ic_name[:] # Pass by value
            map_group_mModel = map_group_mModel_ic.copy() # Pass by value
    
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
                    map_group_mModel['multi'+str(i_multi)] = dpgmm
                else:
                    gmm = mixture.GaussianMixture(
                        n_components=n_components, 
                        covariance_type='full',
                        init_params='random',
                        verbose=0, verbose_interval=100,
                        n_init=1, tol=1e-6, max_iter=1000).fit(X)
                    l = gmm.predict(X)
                    map_group_mModel['multi'+str(i_multi)] = gmm
        
                group.append(l)
                group_name.append('multi'+str(i_multi))
    
            df_grouping, map_group_pupilId, map_group_cmonth = \
                self.construct_df_grouping(group, group_name, df_whizz)
            df_grouping_list.append(df_grouping)
            map_group_pupilId_list.append(map_group_pupilId)
            map_group_cmonth_list.append(map_group_cmonth)
            map_group_mModel_list.append(map_group_mModel)
        
        return  df_grouping_list, map_group_pupilId_list, \
            map_group_cmonth_list, map_group_mModel_list, \
            map_var_ftr, map_ftr_transform

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
        map_group_mMonth_list = self.map_group_model_['G1'][3]
        map_var_ftr = self.map_group_model_['G1'][4]

        score_list = self.compute_groupingScore(df_grouping_list, y, criterion)
        idx_maxScore = np.argmax(score_list)
        map_group_model_best['G1'] = [df_grouping_list[idx_maxScore],
                                      map_group_pupilId_list[idx_maxScore],
                                      map_group_cmonth_list[idx_maxScore],
                                      map_group_mMonth_list[idx_maxScore],
                                      map_var_ftr]
        # G2
        y = self.group_config_.df_whizz_G2_['churn'].values
        map_group_expectaion['G2'] = y
        df_grouping_list = self.map_group_model_['G2'][0]
        map_group_pupilId_list = self.map_group_model_['G2'][1]
        map_group_cmonth_list = self.map_group_model_['G2'][2]
        map_group_mMonth_list = self.map_group_model_['G2'][3]
        map_var_ftr = self.map_group_model_['G2'][4]

        score_list = self.compute_groupingScore(df_grouping_list, y, criterion)
        idx_maxScore = np.argmax(score_list)
        map_group_model_best['G2'] = [df_grouping_list[idx_maxScore],
                                      map_group_pupilId_list[idx_maxScore],
                                      map_group_cmonth_list[idx_maxScore],
                                      map_group_mMonth_list[idx_maxScore],
                                      map_var_ftr]
        # G3
        y = self.group_config_.df_whizz_G3_['churn'].values
        map_group_expectaion['G3'] = y
        df_grouping_list = self.map_group_model_['G3'][0]
        map_group_pupilId_list = self.map_group_model_['G3'][1]
        map_group_cmonth_list = self.map_group_model_['G3'][2]
        map_group_mMonth_list = self.map_group_model_['G3'][3]
        map_var_ftr = self.map_group_model_['G3'][4]

        score_list = self.compute_groupingScore(df_grouping_list, y, criterion)
        idx_maxScore = np.argmax(score_list)
        map_group_model_best['G3'] = [df_grouping_list[idx_maxScore],
                                      map_group_pupilId_list[idx_maxScore],
                                      map_group_cmonth_list[idx_maxScore],
                                      map_group_mMonth_list[idx_maxScore],
                                      map_var_ftr]
    
        self.map_group_expectation_ = map_group_expectaion
        self.map_group_model_best_ = map_group_model_best

        return map_group_model_best