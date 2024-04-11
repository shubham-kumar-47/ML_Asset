"""
    **Project**
    -----------
    Exploratory Data Analysis 
    
    **Description**
    ---------------
    This class contains functions for : 
    
    - Data Train/Test split(Random, Stratified, Out of time) 
    - Univariate Analysis 
    - Missing Value imputation(mean, median, knn, mice method)
    - Outlier Treatment(Simple capping and flooring, Kmeans, Z-score method) 
    - Correlation plot generation
    - Bivariate Analysis 
    
        
    ###**Data Split**
    Division of data into training and validation is generally the very first step of any model building pipeline. 
    The overall data is divided into two parts- training data which is used for Model Development and test data which is used for Validation of the model.
    
    *data_split* function can be used for three types of split -
    
        1. Simple Random Sampling- Randomly dividing the data into two parts in the ratio of 80:20(this can be tweaked).
        2. Stratified Sampling- Randomly dividing the data in two parts for different strata, for example, let’s say Y variable is used for stratification. The data is randomly divided in two parts for event data(train_1,test_1) and similarly the data is divided in two parts for the non-event data(train_0,test_0) and then these four datasets are combined to get two datasets(train_0+train_1 and test_0+test_1). This is how stratified sampling works. 
        3. Out Of Time Sampling- Dividing the data based on a date to get training and out of time data. For example, let’s say the data is from 2014-2019 and it is decided to validate the data on recent past for example 2019 so the data can be divided into 2014-2018 for training and 2019 for validation of the model. 
    
    ###**Univariate Analysis** 
    It is the analysis of variables wherein the data is described and analysed to understand the pattern of one variable at a time (‘uni’ means one). 
    Multiple variable statistics like- Mean, Median, P1, P95, Standard Deviation, etc are calculated as part of univariate analysis. 
    In the model building pipeline, *univariate analysis* plays a pivotal role in removing the unimportant variables. 
    Variables with less variation or high missing percentage can be dropped as they generally do not have any significant information.
    
    *univariate_analysis* function can be used to generate the univariate analysis files.
    
    ###**Imputation** 
    There are multiple ways of dealing with the missing data. 
    *Imputation* is one such way wherein data is imputed or filled with a representative value for example, mean or mode of the variable. 
    
    *impute* function provides multiple imputation strategies-
    
        1. Mean/Median/Mode Imputation- Filling the missing values with mean of the respective variables, for example, if for few customers, age is missing and the mean age of the data is 28 then missing values of age will be imputed by 28 as it is the representative of the population.
        2. KNN Imputation- Filling the missing values with the mean of k nearest neighbours, let’s say for the previous example of missing value in age, we have information of income and height. Nearest neighbours based on height and income can be found (for example low income and moderate height person will generally be a college going teenager kid) and then mean of k (can be 5) nearest neighbours can be taken to fill the missing value.
        3. MICE imputation- It is also a multivariate imputation strategy wherein non missing values are considered as train data and missing values are considered as test data. Model is built on the train data and it can then be used to predict the test data and thereby fill all the missing values. This strategy can be repeated for each variable considering it as the Y.
    
    ###**Outlier Treatment**
    *Outliers* are extreme observations that deviate a lot from the original data. 
    These observations can cause serious harm to the model so it’s important to treat the outliers. 
    
    *outlier_treatment* function provides multiple ways of carrying out the same -
    
        1. Z-Scores- In a standard normal distribution, points with greater than 3 or -3 values can be considered as   extreme values or outliers. Each variable can be standardised (z score of each variable) and depending on the z score of each variable, outlier points or customers can be detected.
        2. K means- K means is an unsupervised algorithm which clusters the points based on the distance between the points (by point we mean customer id etc). Points which have very high distance from the centre of the data can be treated as outliers. This asset provides user the flexibility to choose the percentage of outliers the user wants to remove.
        3. Capping and flooring- There are multiple ways to deal with the outliers. One of the ways can be to cap the outliers to 95th percentile of the variable. For example, let’s say there are outliers in a variable ‘age’. So the 95th percentile of the data is 83 year but the 99th percentile is 200. This shows that data has outliers. Also, 83 can actually be a genuine age while 200 is certainly an outlier so what one can do is- he can cap the data at 83 year age so any age which is greater than 83, will be considered as 83 only. In a similar way we can restrict the lower possible value for a variable and that is called as flooring.
    
    ###**Correlation**
    *Correlation* is a measure of strength of linear relationship between two variables. 
    This can be used to check if two predictors are highly related. 
    In that case, one can pick the variable which has strong relationship with the response and remove the other predictor. 
    To check the strength of relationship between the predictor and response one can use the information value of the predictor which can be obtained using bivariate analysis.
    
    ###**Bivariate Analysis**
    As the name suggests, it is meant to find the *bivariate relationship* of response and the predictor. 
    Information Value shows the strength of relationship. It can be used to find the important predictors for prediction.
    
    *bivariate_analysis* function can be used to generate the bivariate analysis files.
    
    
    Please refer to each class and the corresponding methods to understand their functionality in detail.
    
    Dependencies
    ------------
        Import the following python libraries as shown : 

        scipy==1.4.1
        scikit-learn==0.23.1
        impyute==0.0.8
        packaging==20.4
        
    **Additional Files**
    --------------------
        
        1. excel_formatting.py 
    
    Please ensure that the above files are also present in the working directory and packages mentioned in the dependencies section are installed.
    
    **How to run?**
    ---------------
        Sample Codes:    
    
        import eda as eda    
        eda_obj = eda.model_EDA()
        
        # Sample variable Names
        target = "dpd_24mob0P_6b"
        ID = "appl_id"
        
        train, test = eda_obj.data_split(dataset = data.copy(), method = 'stratify', test_size = 0.25, strata = [target]) 
        
        categorical, continous, var_list = eda_obj.univariate_analysis(data = train.copy(),ID=ID, missing_cutoff = 10, n_levels = 20, perc_cutoff = [5,95]) 
        
        train_imputed, test_imputed = eda_obj.impute(df_train = train, target=target, ID=ID ,df_test = test, method = 'mean', strata = 50) 
        
        train_o, test_o, train_ex, test_ex = eda_obj.outlier_treatment(train_data = train_imputed,test_data = test_imputed, method = 'cap_floor', strata = [0.05,0.95])
            
    **Author**
    ----------
    Created on Fri Jun  12 18:28:27 2020 
    
    @author: Kumar, Shubham
"""


import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
from impyute.imputation.cs import fast_knn
from impyute.imputation.cs import mice
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from datetime import datetime 
from .excel_formatting import excel_formatting as ef
import packaging.version
import os
import warnings
warnings.filterwarnings("ignore")



class model_EDA:
    """
    Description :
    ----------
    This class contains functions for : 

    -Data Train/Test split(Random, Stratified, Out of time) 
    
    -Univariate Analysis  
    
    -Missing Value imputation(mean, median, knn, mice method) 
    
    -outlier treatment(Simple capping and flooring, Kmeans, Z-score method) 
    
    -Correlation plot between variables 
    
    -Bivariate Analysis 
       
    """
    
    __save_directory = 'results/EDA'
    
    if os.path.isdir("results")==False:
        os.mkdir("results")
    
    if os.path.isdir(__save_directory)==False:
        os.mkdir(__save_directory)

    
    # Formats - xlsxwriter
    __round0 = ef.formatting_xlsxwriter(format_name='round0')
    __round1 = ef.formatting_xlsxwriter(format_name='round1')
    __round2 = ef.formatting_xlsxwriter(format_name='round2')
    __round3 = ef.formatting_xlsxwriter(format_name='round3')
    __percsign = ef.formatting_xlsxwriter(format_name='percsign')
    __percsignmultipled = ef.formatting_xlsxwriter(format_name='percsignmultipled')
    __cell_highlight1 = ef.formatting_xlsxwriter(format_name='cell_highlight1')
    __cell_highlight2 = ef.formatting_xlsxwriter(format_name='cell_highlight2')
    __cell_highlight3 = ef.formatting_xlsxwriter(format_name='cell_highlight3')
    __cell_highlight4 = ef.formatting_xlsxwriter(format_name='cell_highlight4')
    __cell_highlight5 = ef.formatting_xlsxwriter(format_name='cell_highlight5')
    __cell_highlight6 = ef.formatting_xlsxwriter(format_name='cell_highlight6')
    __cell_anomaly_bad = ef.formatting_xlsxwriter(format_name='cell_anomaly_bad')
    __cell_anomaly_good = ef.formatting_xlsxwriter(format_name='cell_anomaly_good')
    __table_header= ef.formatting_xlsxwriter(format_name='table_header')
    __table_title = ef.formatting_xlsxwriter(format_name='table_title')
    __cell_even = ef.formatting_xlsxwriter(format_name='cell_even')
    __cell_odd = ef.formatting_xlsxwriter(format_name='cell_odd')
    __align_centre = ef.formatting_xlsxwriter(format_name='align_centre')
    __border1 = ef.formatting_xlsxwriter(format_name='border1')
    __border2 = ef.formatting_xlsxwriter(format_name='border2')
    
    def __init__(self):
        pass
    
    def final_var_selection(self,var_selection_Data,variable_col_name='Variable',Var_Select_Name='Var_Select',Variable_Selection_Label='Y', fileName = "Categorical"):
        """Takes the input dataframe and returns the selected variables based on the flag chosen"""
        var_list = var_selection_Data.loc[var_selection_Data[Var_Select_Name]==Variable_Selection_Label,variable_col_name].to_list()
        pd.DataFrame(var_list,columns=["Variable"]).to_csv(os.path.join(os.getcwd(),model_EDA.__save_directory,fileName+".csv"))
        return var_list

    @classmethod
    def __reset_header_default(self):
        version = packaging.version.parse(pd.__version__)
        if version < packaging.version.parse('0.18'):
            pd.pandas.core.format = None
        elif version < packaging.version.parse('0.20'):
            pd.pandas.formats.format = None
        else:
            pd.pandas.io.formats.excel = None
            
    @classmethod
    def __random_split(self,df,s,random_state=0):
        '''
        This function splits the data into Train and Test randomly, with 75% of data on the training set 
        and 25% data on the testing set
        
        Parameters
        ----------
        df : Dataframe that needs to be split into train and test
        
        Returns
        ----------
        train : training dataframe
        test : testing dataframe
        
        '''
        
        train, test = train_test_split(df, test_size=s,random_state=random_state)

        return train, test

    @classmethod
    def __stratified_split(self,df,s,a,random_state=0):
        '''
        This function splits the data into Train and Test, stratifying it on the variables mentions, 
        with 75% of data on the training set and 25% data on the testing set.
        
        Parameters
        ----------
        df : Dataframe that needs to be split into train and test
        a  : List of variables on which the dataframe needs to be stratified
        
        Returns
        ----------
        train : training dataframe
        test : testing dataframe        
        
        '''
        
        train, test = train_test_split(df,stratify=df[a],test_size=s,random_state=random_state)

        return train, test
    
    @classmethod
    def __out_of_time_split(self,df,strata):
        
        col_name = str(strata[0])
        date_entry =str(strata[1])
        year, month, day = map(int, date_entry.split('-'))
        date1 = datetime.date(year, month, day)
        df[col_name] = pd.to_datetime(df[col_name],errors = 'coerce')
        df[col_name] = pd.DatetimeIndex(df[col_name]).date
        df_train = df[df[col_name]<date1]
        df_test = df[df[col_name]>=date1]

        return df_train, df_test

    @classmethod
    def data_split(self,dataset,method,test_size,strata,random_state=0):
        
        '''
        This function calls the Random or Stratified or Out of time split function based on the user input, 
        and returns the Train and Test dataset
        
        Parameters
        ----------
        dataset : Dataframe that needs to be split into train and test
        method : The splitting strategy to be followed (random, stratify, time)
        test_size = The fraction of data you want in your test set, the remaining fraction becomes your training set 
        strata = [] (when, method = 'random') 
        strata = ['variable1','variable2'....] (i.e. The names of variables you want to stratify by, when method = 'stratify') 
        strata = ['Time variable name','Splitting date in YYYY-MM-DD format'] (when method = 'time') 
        random_state : integer value. Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.

        
        Returns
        ----------
        train : training dataframe 
        test : testing dataframe  
        
        Sample calling command :
        ----------
        
        train, test = eda_obj.data_split(dataset = data.copy(), method = 'stratify', test_size = 0.25, strata = [target])         
        
        '''
        #dataset.drop_duplicates(inplace = True)
        dataset = dataset.replace(np.Inf, -888888)
        dataset = dataset.replace(-np.Inf, -888888)
        method = str(method).upper()      
        test_size = float(test_size)
        if method == 'RANDOM':
            train, test = self.__random_split(dataset,test_size)
        elif method == 'STRATIFY':
            train, test = self.__stratified_split(dataset,test_size,strata)
        elif method == 'TIME':
            train, test = self.__out_of_time_split(dataset,strata)
      
         
        
        filename1 = 'Train_' + method.lower()
        filename1 = ef.create_version(base_filename=filename1, path = os.path.join(os.getcwd(),model_EDA.__save_directory))
        filename1 += '.csv'
        train.to_csv(os.path.join(os.getcwd(),model_EDA.__save_directory,filename1),index=False)
        
        filename2 = 'Test_' + method.lower()
        filename2 = ef.create_version(base_filename=filename2, path = os.path.join(os.getcwd(),model_EDA.__save_directory))
        filename2 += '.csv'
        test.to_csv(os.path.join(os.getcwd(),model_EDA.__save_directory,filename2),index=False)
    
        print("Train, test tables stored at: \n"+os.path.join(os.getcwd(),model_EDA.__save_directory))
        print("Filenames : \n"+filename1,", ",filename2)
                
        
        return train,test
        
        
           
    @classmethod
    def __univariate_continuous(self,data,perc_inp_1,perc_inp_2,cut_off_continuos_var):
        
        
        ''' 
        In univariate analysis we are looking for the different characteristic of the data. 
        By characteristic we mean the range,inter quartile range, skewness, kurtosis, mean,
        median,standard deviation. Outliers in the data detected by using the inter-quartile
        range.The function also calculates the number of missing values and the percentage of 
        missing values for different variable.It returns a dataframe which contains the results 
        of the operations done here.
         
        Parameters
        ----------
        data : Dataframe on which univariate analysis is to be done
        
        Returns
        ----------
        unistat : Univariate analysis table for the continuous variables
         
         
        '''

        x = [x for x in data.columns if data[x].dtypes != 'object']
        
        if len(x) == 0:
            return 0

        #x contains only the numeric variables present in the data.  

        df = data[x]

        #Range is the difference between the maximum and minimum value in the data for a single variable

        def Range(y):
            mi = np.min(y)

            ma = np.max(y)

            rng = ma-mi

            return rng

        Rnge = df.apply(Range)

         #Inter-quartlie range is the difference between the 75th percentile and 25th percentile

        def iqr(y):

            x25 = np.nanpercentile(y,25)

            x75 = np.nanpercentile(y,75)

            Iqr = x75 - x25

            return Iqr

        iQr = df.apply(iqr)


        #Skewness is asymmetry in a statistical distribution, in which the curve appears distorted or skewed either to the left or 
    # to the right. Skewness can be quantified to define the extent to which a distribution differs from a normal distribution

        def Skew(y):

            d = skew(y)

            return d

        skewness = df.apply(Skew)


        def level(y):
            no_of_levels = len(y.unique())

            return no_of_levels

        nlevels = df.apply(level)


        # Like skewness, kurtosis is a statistical measure that is used to describe the distribution. 
        # Distributions with low kurtosis exhibit tail data that are generally less extreme than the tails of the normal distribution

        def Kurt(y):

            d = kurtosis(y)

            return d

        Kurtosis = df.apply(Kurt)

        mean = df.apply(np.mean)

        stdv = df.apply(np.std)

        median = df.apply(np.nanmedian,)

        missingno = df.isnull().sum()

         #An observation is said to be an outlier if it lies outside 
        #(q3+ 1.5 * inter-quartile range) and (q1- 1.5 * inter-quartile range) 

        def outlier(y):
            q1 = np.percentile(y,25)
            # q1 is the 25th percetile'''
            q3 = np.percentile(y,75)
            # q3 is the 75th percentile'''
            iqr = q3-q1
            #Inter-quartile range is the difference between 75th percentile and the 25th percentile'''
            outlier = ((y > (q3 +1.5*iqr)) |(y <(q1-1.5*iqr ))).sum()

            return outlier

        Q1 = df.quantile(.25)
        p1 = df.quantile(0.01)
        p10 = df.quantile(0.10)
        p20 = df.quantile(0.20)
        p30 = df.quantile(0.30)
        p40 = df.quantile(0.40)
        p50 = df.quantile(0.50)
        p60 = df.quantile(0.60)
        p70 = df.quantile(0.70)
        p80 = df.quantile(0.80)
        p90 = df.quantile(0.90)
        p3 = df.quantile(0.99)
        Q3 = df.quantile(.75)
        
        
        px = df.quantile(float(perc_inp_1)/100)
        py = df.quantile(float(perc_inp_2)/100)
        
        pname1 = str('P_'+str(perc_inp_1))
        pname2 = str('P_'+str(perc_inp_2))

        out = df.apply(outlier)
        count = df[x].count()

        missingper=(missingno/df[x].shape[0])*100

        Frame = [Rnge,Q1,Q3,p1,p10,p20,p30,p40,p50,p60,p70,p80,p90,p3,px,py,iQr,skewness,nlevels,Kurtosis,mean,stdv,median,missingno,out,missingper,count]
        #creating a dataframe it will contain all the results we have derived here
        unistat = pd.concat(Frame,axis =1)

        unistat.columns = ['Range','Q1','Q3','p1','p10','p20','p30','p40','p50','p60','p70','p80','p90','p99',pname1,pname2,'IQR','Skewness','nlevels','Kurtosis','mean','stdv','median',
                           'MissingValue','outlier','Missing_perc','# of observations']

        ## Calculate the missing percentage of the variables
        l=[]

        for i in missingper:
            if i>99:
                l.append(1)
            else:
                l.append(0)

        unistat['Missing_Criteria']=l

        unistat=unistat[['MissingValue','Missing_perc','mean','median','stdv','p1','p10','p20','p30','p40','p50','p60','p70','p80','p90','p99',pname1,pname2,'Range','Q1','Q3','IQR',
                         'outlier','Skewness','nlevels','Kurtosis','# of observations']]
        unistat['var_name']=x

        #Setting the perc_flag as 1 for the variables with no variability, and 0 otherwise
        unistat['perc_flag'] = 0
        unistat['perc_flag'] = np.where( ( (unistat[pname1]==unistat[pname2]) & (unistat['nlevels']!=2)) , 1, 0)
        
        #Takes a cuttoff input from the user, and sets missing_flag as 1 if missing % is greater than the cuttoff
        unistat['missing_flag'] = 0
        unistat['missing_flag'] = np.where( ( (unistat['Missing_perc']>cut_off_continuos_var)) , 1, 0)
        
        # Sets var_select as 'Y', if perc_flag & missing_flag both are 0, else sets it as 'N'
        unistat['var_select'] = 'Y'
        unistat['var_select'] = np.where( ( (unistat['perc_flag']==1) | (unistat['missing_flag']==1)) , 'N', 'Y')
        

        return unistat
    
    
    
    @classmethod
    def __univariate_categorical(self,data,cut_off_categorical_var,no_levels):
        
        ''' 
        This function carries out the Univariate analysis of categorical variables present in the data.
        No of levels in a different variables, total number of missing values in each variable, 
        percentage of missing values are calculated here.This function returns a dataframe containing 
        all the records mentioned above.
        
        Parameters
        ----------
        data : Dataframe on which univariate analysis is to be done
        cut_off_categorical_var : Missing percentage cutoff for categorical variables
        no_levels : Cutoff for maximum number of levels in categorical variables
        
        Returns
        ----------
        dff : Univariate analysis table for the categorical variables
        
        
        
        '''

        z = [x for x in data.columns if data[x].dtypes == 'object']

        #Filtering the variables which are categorical 

        data_cat=data[z]
        dff= pd.DataFrame(columns=['No_of_levels', 'Total_missing', 'Missing_perc'])
        no_of_levels=[]
        total_missing=[]
        missing_perc=[]
        for k in z:
            no_of_levels.append(len(data_cat[k].unique()))
            total_missing.append(data_cat[k].isnull().sum())
            missing_perc.append(data_cat[k].isnull().sum()*100/len(data_cat[k]))
        dff['No_of_levels']=no_of_levels
        dff['Total_missing']=total_missing
        dff['Missing_perc']=missing_perc
        dff['var_name']=z
        dff=dff[['var_name','No_of_levels', 'Total_missing', 'Missing_perc' ]]
        
        #Setting the perc_flag as 1 for the variables with no variability, and 0 otherwise
        dff['perc_flag'] = 0
        dff['perc_flag'] = np.where( ( (dff['No_of_levels']<2) | (dff['No_of_levels']>no_levels)) , 1, 0)
        
        
        #Takes a cuttoff input from the user, and sets missing_flag as 1 if missing % is greater than the cuttoff
        dff['missing_flag'] = 0
        dff['missing_flag'] = np.where( ( (dff['Missing_perc']>cut_off_categorical_var)) , 1, 0)
        
        ## Sets var_select as 'Y', if both No_of_levels and Missing_perc are below the cutoff
        dff['var_select']=np.nan
        dff['var_select'] = np.where(((dff['perc_flag']==1) | (dff['missing_flag']==1)), 'N', 'Y')
        
        z= [x for x in data.columns if data[x].dtypes == 'object']
        var = data[z].columns.tolist()
        df_res = pd.DataFrame(columns=['Level','Count','Population %','Variable'])
        for i in var:
            level = data[i].value_counts().index.tolist()
            count = data[i].value_counts().tolist()
            df_new = pd.DataFrame(level,columns =['Level'])
            df_new['Count'] = count
            df_new['Population %'] = round(df_new['Count']*100/df_new['Count'].sum(),2)
            df_new['Variable'] = i
            df_res = pd.concat([df_res,df_new], axis=0)
        df_res = df_res[['Variable','Level','Count','Population %']]
        
        cat_df = [dff,df_res]
        
        return cat_df
    
    @classmethod
    def univariate_analysis(self,data,ID, missing_cutoff, n_levels, perc_cutoff):
        
        '''
        This function calls the univariate analysis functions for categorical and continuous variables, 
        and returns the Univariate analysis tabel for Continuous and Categorical variables, along with the 
        selected variable list(based on variability and cutoff conditions set by the user)
        
        
        Parameters
        ----------
        data : Dataframe on which univariate analysis is to be done
        ID : Application Id 
        missing_cutoff = Integer value of the maximum allowable missing percentage in a variable. 
        n_levels = Integer value of the maximum allowable number of levels in a categorical variable. 
        perc_cutoff = [Integer value for the lower percentile and upper percentile value for variability check seperated by ','] 
        
        
        Returns
        ----------
        categorical : Univariate analysis table for the categorical variables 
        continous : Univariate analysis table for the continuous variables
        rem_var : List of selected variables after univariate analysis
        
        Sample calling command :
        ----------
        
        categorical, continous, var_list = eda.model_EDA.univariate_analysis(data = train.copy(),ID=ID, missing_cutoff = 10, n_levels = 20, perc_cutoff = [5,95]) 
        

                
        Note : (Recommended)
        ----------
        The variables with 
        
               -Fill rates below the required cutoff 
               
               -Invariant variables 
               
               -Number of levels above the required cutoff
               
        have been flagged as N and others has been marked as Y in the var_select column. 
        
        In case the user wants to drop these variables from further analysis, the below code is to be executed.
        
        Code for dropping Variables flagged as N : 
        ----------
        train_dropped = train[rem_var] 
        test_dropped = test[rem_var] 
        ---------- 
        
        train_dropped : Training dataset with 'N' flagged variables dropped 
        
        test_dropped : Testing dataset with 'N' flagged variables dropped 
        
        train : Training dataset with all variables 
        
        test : Testing datset with all variables 
        
        rem_var : List of variables flagged as Y returned by the univariate_analysis() function 
              
        
        
        
                
        '''
        
        date_columns = [x for x in data.columns if ('date' in str(data[x].dtypes)) or ('time' in str(data[x].dtypes))]
        if len(date_columns)>0:
            print("The date columns:", date_columns, "have been removed before analysis")
            data.drop(columns=date_columns,inplace=True)
        
        
        dataset = data[data.columns.tolist()]
        #data_id = dataset[[ID]]
        dataset.drop(columns=[ID],inplace=True)
        #dataset.drop_duplicates(inplace = True)
        dataset = dataset.replace(np.Inf, -888888)
        dataset = dataset.replace(-np.Inf, -888888)
        # Takes input from user for the missing % and number of allowable levels cuttoff for the categorical variables
        cut_off_categorical_var= int(missing_cutoff)
        no_levels= int(n_levels)
        perc_inp_1 = int(perc_cutoff[0])
        perc_inp_2 = int(perc_cutoff[1])
        cut_off_continuos_var= int(missing_cutoff)
        
        categorical= self.__univariate_categorical(dataset,cut_off_categorical_var,no_levels)  
    
        continous= self.__univariate_continuous(dataset,perc_inp_1,perc_inp_2,cut_off_continuos_var)
        
        if str(type(continous)) != "<class 'int'>":
            continous = continous.round(2)
        categorical[0] = categorical[0].round(2)
        
        
        #Dropping variables with low fill rate, high number of levels and low variability
        categorical_filtered=categorical[0][categorical[0]["var_select"]=='Y']
        if str(type(continous)) != "<class 'int'>":
            continous_filtered=continous[continous['var_select']=='Y']
        
        categorical[0]=categorical[0].set_index('var_name')
        if str(type(continous)) != "<class 'int'>":
            continous=continous.set_index('var_name')
        
        if str(type(continous)) != "<class 'int'>":  
            continous.rename(columns={"MissingValue":"Missing Counts",
                                  "Missing_perc":"Missing %",
                                  "mean":"Mean",
                                  "median":"Median",
                                  "stdv":"Std Dev",
                                  "outlier":"Outlier",
                                  "# of observations":"# Of Observations",
                                  "perc_flag":"Perc Flag",
                                  "missing_flag":"Missing Flag"},inplace=True)
        categorical[0].rename(columns={"No_of_levels":"nlevels",
                                       "Total_missing":"Missing Counts",
                                       "Missing_perc":"Missing %",
                                       "perc_flag":"Perc Flag",
                                       "missing_flag":"Missing Flag"},inplace=True)
        
        
        
        categorical1 = categorical[0].copy()
        categorical2 = categorical[1].copy()
        
        ef.reset_header_default()
        filename = 'Univariate'
        filename = ef.create_version(base_filename=filename, path = os.path.join(os.getcwd(),model_EDA.__save_directory))
        filename += '.xlsx'
        writer = pd.ExcelWriter(os.path.join(os.getcwd(),model_EDA.__save_directory,filename), engine='xlsxwriter')
        workbook  = writer.book
        
        # FORMATS--------------------------------------------------------------------------
        
        
        
        format_round0 = workbook.add_format(model_EDA.__round0)
#        format_round1 = workbook.add_format(model_EDA.__round1)
        format_round2 = workbook.add_format(model_EDA.__round2)
#        format_round3 = workbook.add_format(model_EDA.__round3)
        format_percsign = workbook.add_format(model_EDA.__percsign)
#        format_percsignmultipled = workbook.add_format(model_EDA.__percsignmultipled)
        format_cell_highlight1 = workbook.add_format(model_EDA.__cell_highlight1)
#        format_cell_highlight2 = workbook.add_format(model_EDA.__cell_highlight2)
#        format_cell_highlight3 = workbook.add_format(model_EDA.__cell_highlight3)
#        format_cell_highlight4 = workbook.add_format(model_EDA.__cell_highlight4)
#        format_cell_highlight5 = workbook.add_format(model_EDA.__cell_highlight5)
#        format_cell_highlight6 = workbook.add_format(model_EDA.__cell_highlight6)
        format_table_header= workbook.add_format(model_EDA.__table_header)
#        format_table_title = workbook.add_format(model_EDA.__table_title)
        format_cell_even = workbook.add_format(model_EDA.__cell_even)
        format_cell_odd = workbook.add_format(model_EDA.__cell_odd)
#        format_align_centre = workbook.add_format(model_EDA.__align_centre)
        format_cell_anomaly_bad = workbook.add_format(model_EDA.__cell_anomaly_bad)
        format_cell_anomaly_good = workbook.add_format(model_EDA.__cell_anomaly_good)
#        format_border1 = workbook.add_format(model_EDA.__border1)
#        format_border2 = workbook.add_format(model_EDA.__border2)
        
        # SHEET: Continuous--------------------------------------------------------------------------
        if str(type(continous)) != "<class 'int'>":
            continous.to_excel(writer, sheet_name='Continuous',index_label='Variable',startrow=4,startcol=1) 

            worksheet = writer.sheets['Continuous']
            worksheet.hide_gridlines(2)
            
            # applying formatting
               
            # table header
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number=5,column_number=2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number=5,column_number=len(continous.columns)+2,fix_row=True),
                                          {'type': 'no_blanks','format': format_table_header})
            
            # logo
            
            worksheet.insert_image(ef.generate_excel_cell_name(row_number=2,column_number=2), os.path.join(os.getcwd(),'src','ks_logo.png'))
    
           
            
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number=6,column_number=len(continous.columns)+2)+':'+ef.generate_excel_cell_name(row_number=len(continous)+6,column_number=len(continous.columns)+2), 
            {'type':'formula', 
            'criteria': '='+ef.generate_excel_cell_name(row_number=6,column_number=len(continous.columns)+2)+'="N"',  
            'format': format_cell_anomaly_bad
            })
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number=6,column_number=len(continous.columns)+2)+':'+ef.generate_excel_cell_name(row_number=len(continous)+6,column_number=len(continous.columns)+2), 
            {'type':'formula', 
            'criteria': '='+ef.generate_excel_cell_name(row_number=6,column_number=len(continous.columns)+2)+'="Y"',  
            'format': format_cell_anomaly_good
            })
    
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number=6,column_number=2)+':'+ef.generate_excel_cell_name(row_number=len(continous)+6,column_number=2), 
                                         {'type': 'no_blanks','format': format_cell_highlight1})
            
            
            # table cells                       
            rows = list(range(5,len(continous)+6))          
            
            for row in rows:
                worksheet.conditional_format(ef.generate_excel_cell_name(row_number=row,column_number=3,fix_row=True)+':'+ef.generate_excel_cell_name(row_number=row,column_number=len(continous.columns)+2,fix_row=True),
                                             {'type': 'no_blanks','format': (format_cell_even if row%2==0 else format_cell_odd)})
            
            
        
            max_column_width = max([len(x) + 2 for x in continous.index])
            worksheet.set_column('B:B', max_column_width)
            
            worksheet.set_column('C:C', 16, format_round0)
            worksheet.set_column('D:D', 12, format_percsign)
            worksheet.set_column('E:X', 8,format_round2)
            worksheet.set_column('Z:Z', 8,format_round2)
            worksheet.set_column('AB:AB', 9, format_round2)
            worksheet.set_column('AC:AC', 16)
    
    
            
            
    
        # SHEET: Categorical--------------------------------------------------------------------------
        
        if len(categorical1)>0:  
            categorical1.to_excel(writer, sheet_name='Categorical',index_label='Variable',startrow=4,startcol=1) 
            worksheet = writer.sheets['Categorical']
            worksheet.hide_gridlines(2)
        
            
            # applying formatting
                
            # table header
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number=5,column_number=2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number=5,column_number=len(categorical1.columns)+2,fix_row=True),
                                          {'type': 'no_blanks','format': format_table_header})
            
            # logo
            
            worksheet.insert_image(ef.generate_excel_cell_name(row_number=2,column_number=2), os.path.join(os.getcwd(),'src','ks_logo.png'))
           
            
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number=6,column_number=len(categorical1.columns)+2)+':'+ef.generate_excel_cell_name(row_number=len(categorical1)+6,column_number=len(categorical1.columns)+2), 
            {'type':'formula', 
            'criteria': '='+ef.generate_excel_cell_name(row_number=6,column_number=len(categorical1.columns)+2)+'="N"',  
            'format': format_cell_anomaly_bad
            })
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number=6,column_number=len(categorical1.columns)+2)+':'+ef.generate_excel_cell_name(row_number=len(categorical1)+6,column_number=len(categorical1.columns)+2), 
            {'type':'formula', 
            'criteria': '='+ef.generate_excel_cell_name(row_number=6,column_number=len(categorical1.columns)+2)+'="Y"',  
            'format': format_cell_anomaly_good
            })
    
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number=6,column_number=2)+':'+ef.generate_excel_cell_name(row_number=len(categorical1)+6,column_number=2), 
                                         {'type': 'no_blanks','format': format_cell_highlight1})
            
            
            # table cells                       
            rows = list(range(5,len(categorical1)+6))          
            
            for row in rows:
                worksheet.conditional_format(ef.generate_excel_cell_name(row_number=row,column_number=3,fix_row=True)+':'+ef.generate_excel_cell_name(row_number=row,column_number=len(categorical1.columns)+2,fix_row=True),
                                             {'type': 'no_blanks','format': (format_cell_even if row%2==0 else format_cell_odd)})
            
            
            max_column_width = max([len(x) + 2 for x in categorical1.index])
            worksheet.set_column('B:B', max_column_width)
            
            worksheet.set_column('C:C', 8)
            worksheet.set_column('D:D', 16)
            worksheet.set_column('E:E', 12,format_percsign)
    
    
        # SHEET: Categorical - Level wise--------------------------------------------------------------------------
    
        if len(categorical2)>0: 
    
            
            categorical2.to_excel(writer, sheet_name='Categorical - Level wise',index=False,startrow=4,startcol=1) 
            worksheet = writer.sheets['Categorical - Level wise']
            worksheet.hide_gridlines(2)
            
            
            # applying formatting
                
            # table header
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number=5,column_number=2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number=5,column_number=len(categorical2.columns)+2,fix_row=True),
                                          {'type': 'no_blanks','format': format_table_header})
            
            # logo
            
            worksheet.insert_image(ef.generate_excel_cell_name(row_number=2,column_number=2), os.path.join(os.getcwd(),'src','ks_logo.png'))
            
            
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number=6,column_number=2)+':'+ef.generate_excel_cell_name(row_number=len(categorical2)+6,column_number=2), 
                                         {'type': 'no_blanks','format': format_cell_highlight1})
            
            
            # table cells                       
            rows = list(range(5,len(categorical2)+6))          
            
            for row in rows:
                worksheet.conditional_format(ef.generate_excel_cell_name(row_number=row,column_number=3,fix_row=True)+':'+ef.generate_excel_cell_name(row_number=row,column_number=len(categorical2.columns)+2,fix_row=True),
                                             {'type': 'no_blanks','format': (format_cell_even if row%2==0 else format_cell_odd)})
            
            
            max_column_width = max([len(x) + 2 for x in categorical2.Variable])
            worksheet.set_column('B:B', max_column_width)
           
            max_column_width = max([len(str(x)) + 2 for x in categorical2.Level])
            worksheet.set_column('C:C', max_column_width)
            worksheet.set_column('D:D', 11)
            worksheet.set_column('E:E', 12,format_percsign)
    
        
        writer.save()
        writer.close()
        #Compiling the Continuous and Categorical variables to form the filtered dataset 
        if str(type(continous)) != "<class 'int'>":
            rem_var=list(categorical_filtered['var_name'])+list(continous_filtered['var_name'])
        else :
            rem_var=list(categorical_filtered['var_name'])
        rem_var.append(ID)
            
        return categorical, continous, rem_var
    
    
    
    
    @classmethod
    def impute(self,df_train,df_test,method,target,ID,strata):
        
        '''
        Before going into modeling missing value imputation is necessary. There will be two types of variable in the data.
        One is continuous and the other is categorical. Methods for imputation will be different for different data type.
        For impuatation of a continuous varaible mean median, KNN, and MICE strategies are used. 
        For imputation of a categorical variable mode of that variable is used.
        This function will take a dataframe as input, and it will provide a dataframe with no missing value
        
        Parameters
        ----------
        df_train : Training dataframe on which imputation is to be done 
        
        df_test : Testing dataframe on which imputation is to be done 
        
        method : Imputation startegy (mean, median, knn, mice) 
        
        target : Target variable 
        
        ID : Application ID 
        
        strata : Integer value for number of neighbours (when method = 'knn') 
        
        strata : '' (when method is other than 'knn') 
        
        
        
        Returns
        ----------
        result_train : Imputed Training dataframe
        result_test : Imputed Testing dataframe 
        
        Sample calling command :
        ----------
        
        train_imputed, test_imputed = eda.model_EDA.impute(df_train = train, target=target, ID=ID ,df_test = test, method = 'mean', strata = 50)
        
        '''
        train_data = df_train[df_train.columns.tolist()]
        test_data = df_test[df_test.columns.tolist()]
        train_data.drop_duplicates(inplace = True)
        test_data.drop_duplicates(inplace = True)
        
        date_columns1 = [x for x in train_data.columns if ('date' in str(train_data[x].dtypes)) or ('time' in str(train_data[x].dtypes))]
        date_columns2 = [x for x in train_data.columns if ('date' in str(train_data[x].dtypes)) or ('time' in str(train_data[x].dtypes))]
        
        target_train = train_data[[target,ID]+date_columns1]
        target_test = test_data[[target,ID]+date_columns2]
        
        train_data.drop(columns=[target,ID]+date_columns1,inplace=True)
        test_data.drop(columns=[target,ID]+date_columns2,inplace=True)
        
        train_data = train_data.replace(np.Inf, -888888)
        train_data = train_data.replace(-np.Inf, -888888)
        test_data = test_data.replace(np.Inf, -888888)
        test_data = test_data.replace(-np.Inf, -888888)

        method = str(method).lower()
        kn_n = int(strata)
        
        #train_na_list = (train_data[(train_data.isnull().values).any(axis=1)]).index.tolist()
        #test_na_list = (test_data[(test_data.isnull().values).any(axis=1)]).index.tolist()
        

        #Imputation for Continuous variables
        x= [x for x in train_data.columns if train_data[x].dtypes != 'object']


        if train_data[x].isnull().values.any():
            
            if method == 'all':
                
                #mean
                imputed_train_mean = train_data[x].fillna(train_data[x].mean())
                imputed_train_mean.columns = [str(col) + '_mean' for col in imputed_train_mean.columns]
                imputed_test_mean = test_data[x].fillna(test_data[x].mean())
                imputed_test_mean.columns = [str(col) + '_mean' for col in imputed_test_mean.columns]
                
                #median
                imputed_train_median = train_data[x].fillna(train_data[x].median())
                imputed_train_median.columns = [str(col) + '_median' for col in imputed_train_median.columns]
                imputed_test_median = test_data[x].fillna(test_data[x].median())
                imputed_test_median.columns = [str(col) + '_median' for col in imputed_test_median.columns]
                
                #knn
                imputed_train_df_knn=fast_knn(train_data[x].values, k=kn_n)
                imputed_test_df_knn=fast_knn(test_data[x].values, k=kn_n)            
                suf_res_knn = [sub + '_knn' for sub in train_data[x].columns.tolist()]
                imputed_train_knn = pd.DataFrame(data=imputed_train_df_knn,columns=suf_res_knn)
                imputed_test_knn = pd.DataFrame(data=imputed_test_df_knn,columns=suf_res_knn)
                
                #mice
                imputed_train_df_mice = mice(train_data[x].values)
                imputed_test_df_mice = mice(test_data[x].values)
                suf_res_mice = [sub + '_mice' for sub in train_data[x].columns.tolist()]
                imputed_train_mice = pd.DataFrame(data=imputed_train_df_mice,columns=suf_res_mice)
                imputed_test_mice = pd.DataFrame(data=imputed_test_df_mice,columns=suf_res_mice)
                
                imputed_train_mean.reset_index(drop=True, inplace = True)
                imputed_train_median.reset_index(drop=True, inplace = True)
                imputed_train_knn.reset_index(drop=True, inplace = True)
                imputed_train_mice.reset_index(drop=True, inplace = True)
                
                imputed_test_mean.reset_index(drop=True, inplace = True)
                imputed_test_median.reset_index(drop=True, inplace = True)
                imputed_test_knn.reset_index(drop=True, inplace = True)
                imputed_test_mice.reset_index(drop=True, inplace = True)
                
                imputed_train = pd.concat([imputed_train_mean,imputed_train_median,imputed_train_knn,imputed_train_mice], axis=1)
                imputed_test = pd.concat([imputed_test_mean,imputed_test_median,imputed_test_knn,imputed_test_mice], axis=1)
                
                
                

            # Imputation by mean
            elif method == 'mean':
                imputed_train = train_data[x].fillna(train_data[x].mean())
                imputed_train.columns = [str(col) + '_mean' for col in imputed_train.columns]
                imputed_test = test_data[x].fillna(test_data[x].mean())
                imputed_test.columns = [str(col) + '_mean' for col in imputed_test.columns]
                

            # Imputation by median
            elif method == 'median':
                imputed_train = train_data[x].fillna(train_data[x].median())
                imputed_train.columns = [str(col) + '_median' for col in imputed_train.columns]
                imputed_test = test_data[x].fillna(test_data[x].median())
                imputed_test.columns = [str(col) + '_median' for col in imputed_test.columns]
                
                
            # Imputation by knn
            elif (method == 'knn'):
                imputed_train_df=fast_knn(train_data[x].values, k=kn_n)
                imputed_test_df=fast_knn(test_data[x].values, k=kn_n)            
                suf_res = [sub + '_knn' for sub in train_data[x].columns.tolist()]
                imputed_train = pd.DataFrame(data=imputed_train_df,columns=suf_res)
                imputed_test = pd.DataFrame(data=imputed_test_df,columns=suf_res)

            # Imputation by mice
            elif method == 'mice':
                imputed_train_df = mice(train_data[x].values)
                imputed_test_df = mice(test_data[x].values)
                suf_res = [sub + '_mice' for sub in train_data[x].columns.tolist()]
                imputed_train = pd.DataFrame(data=imputed_train_df,columns=suf_res)
                imputed_test = pd.DataFrame(data=imputed_test_df,columns=suf_res)

        # In case no nulls are present the original dataframe is passed
        else :
            imputed_train = train_data[x].copy()
            imputed_test = test_data[x].copy()



        # Imputation of categorical variables by their mode
        y= [y for y in train_data.columns if train_data[y].dtypes == 'object']
        cat_imp_train = train_data[y].copy()
        cat_imp_test = test_data[y].copy()

        for j in y:
            a = train_data[j].mode()
            cat_imp_train[j]=train_data[j].fillna(a.tolist()[0])
            cat_imp_test[j]=test_data[j].fillna(a.tolist()[0])
            
        df_con = pd.DataFrame(train_data[x].columns.tolist(),columns=['variable'])
        df_con['mean'] = train_data[x].mean().tolist()
        df_con['median'] = train_data[x].median().tolist()
        df_con = df_con.round(2)

        df_cat = train_data[y].mode().T
        df_cat.reset_index(inplace = True)
        df_cat.rename(columns = {'index':'Variable', 0:'Mode'}, inplace = True)

        #Resetting Index
        cat_imp_train.reset_index(drop=True, inplace = True)
        #df_train_o = train_data[x].reset_index(drop=True)
        imputed_train.reset_index(drop=True, inplace = True)

        cat_imp_test.reset_index(drop=True, inplace = True)
        #df_test_o = test_data[x].reset_index(drop=True)
        imputed_test.reset_index(drop=True, inplace = True)



        #Compiling the Imputed Continuous and Categorical Datasets
        result_train = pd.concat([cat_imp_train,imputed_train], axis=1)
        result_test = pd.concat([cat_imp_test,imputed_test], axis=1)
        
        result_train.reset_index(drop = True,inplace= True)
        result_test.reset_index(drop = True,inplace= True)
        target_train.reset_index(drop = True,inplace= True)
        target_test.reset_index(drop = True,inplace= True)
        
        result_train[target] = target_train[target]
        result_test[target] = target_test[target]
        result_train[ID] = target_train[ID]
        result_test[ID] = target_test[ID]
        result_train[date_columns1] = target_train[date_columns1]
        result_test[date_columns2] = target_test[date_columns2]
        
        
        
        #impu_train = result_train[result_train.index.isin(train_na_list)]
        #impu_test = result_test[result_test.index.isin(test_na_list)]
        
        ef.reset_header_default()
        filename = 'Train_imputed'
        filename = ef.create_version(base_filename=filename, path = os.path.join(os.getcwd(),model_EDA.__save_directory))
        filename += '.csv'
        result_train.to_csv(os.path.join(os.getcwd(),model_EDA.__save_directory,filename),index=False)
    
        filename = 'Test_imputed'
        filename = ef.create_version(base_filename=filename, path = os.path.join(os.getcwd(),model_EDA.__save_directory))
        filename += '.csv'
        result_test.to_csv(os.path.join(os.getcwd(),model_EDA.__save_directory,filename),index=False)
        
        
        filename = 'Imputation Summary'
        filename = ef.create_version(base_filename=filename, path = os.path.join(os.getcwd(),model_EDA.__save_directory))
        filename += '.xlsx'
        writer = pd.ExcelWriter(os.path.join(os.getcwd(),model_EDA.__save_directory,filename), engine='xlsxwriter')
        workbook  = writer.book
    
        # FORMATS--------------------------------------------------------------------------
    
        
#        format_round0 = workbook.add_format(model_EDA.__round0)
#        format_round1 = workbook.add_format(model_EDA.__round1)
#        format_round2 = workbook.add_format(model_EDA.__round2)
#        format_round3 = workbook.add_format(model_EDA.__round3)
#        format_percsign = workbook.add_format(model_EDA.__percsign)
#        format_percsignmultipled = workbook.add_format(model_EDA.__percsignmultipled)
#        format_cell_highlight1 = workbook.add_format(model_EDA.__cell_highlight1)
#        format_cell_highlight2 = workbook.add_format(model_EDA.__cell_highlight2)
#        format_cell_highlight3 = workbook.add_format(model_EDA.__cell_highlight3)
#        format_cell_highlight4 = workbook.add_format(model_EDA.__cell_highlight4)
#        format_cell_highlight5 = workbook.add_format(model_EDA.__cell_highlight5)
#        format_cell_highlight6 = workbook.add_format(model_EDA.__cell_highlight6)
        format_table_header= workbook.add_format(model_EDA.__table_header)
#        format_table_title = workbook.add_format(model_EDA.__table_title)
        format_cell_even = workbook.add_format(model_EDA.__cell_even)
        format_cell_odd = workbook.add_format(model_EDA.__cell_odd)
        format_align_centre = workbook.add_format(model_EDA.__align_centre)
#        format_cell_anomaly_bad = workbook.add_format(model_EDA.__cell_anomaly_bad)
#        format_cell_anomaly_good = workbook.add_format(model_EDA.__cell_anomaly_good)
#        format_border1 = workbook.add_format(model_EDA.__border1)
#        format_border2 = workbook.add_format(model_EDA.__border2)
        

        
        # SHEET: Continuous--------------------------------------------------------------------------
        if len(df_con)>0: 
            df_con.to_excel(writer, sheet_name='Continuous',index_label='Index',startrow=4,startcol=1) 
            worksheet = writer.sheets['Continuous']
            worksheet.hide_gridlines(2)
           
            
            # applying formatting
                
            # table header
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number=5,column_number=2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number=5,column_number=len(df_con.columns)+2,fix_row=True),
                                          {'type': 'no_blanks','format': format_table_header})
            
            # logo
            worksheet.insert_image(ef.generate_excel_cell_name(row_number=2,column_number=2), os.path.join(os.getcwd(),'src','ks_logo.png'))
            
    
             
            
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number=6,column_number=2)+':'+ef.generate_excel_cell_name(row_number=len(df_con)+6,column_number=2), 
                                         {'type': 'no_blanks','format': format_align_centre})
            
            
            # table cells                       
            rows = list(range(5,len(df_con)+6))          
            
            for row in rows:
                worksheet.conditional_format(ef.generate_excel_cell_name(row_number=row,column_number=3,fix_row=True)+':'+ef.generate_excel_cell_name(row_number=row,column_number=len(df_con.columns)+2,fix_row=True),
                                             {'type': 'no_blanks','format': (format_cell_even if row%2==0 else format_cell_odd)})
            
            
            max_column_width = max([len(x) + 2 for x in df_con.variable])
            worksheet.set_column('C:C', max_column_width)
            
        
        
        # SHEET: Test--------------------------------------------------------------------------
        if len(df_cat)>0:
            df_cat.to_excel(writer, sheet_name='Categorical',index_label='Index',startrow=4,startcol=1) 
            worksheet = writer.sheets['Categorical']
            worksheet.hide_gridlines(2)
            
            
            # applying formatting
                
            # table header
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number=5,column_number=2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number=5,column_number=len(df_cat.columns)+2,fix_row=True),
                                          {'type': 'no_blanks','format': format_table_header})
            
            # logo
            worksheet.insert_image(ef.generate_excel_cell_name(row_number=2,column_number=2), os.path.join(os.getcwd(),'src','ks_logo.png'))
        

          
            
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number=6,column_number=2)+':'+ef.generate_excel_cell_name(row_number=len(df_con)+6,column_number=2), 
                                         {'type': 'no_blanks','format': format_align_centre})
            
            
            # table cells                       
            rows = list(range(5,len(df_cat)+6))          
            
            for row in rows:
                worksheet.conditional_format(ef.generate_excel_cell_name(row_number=row,column_number=3,fix_row=True)+':'+ef.generate_excel_cell_name(row_number=row,column_number=len(df_cat.columns)+2,fix_row=True),
                                             {'type': 'no_blanks','format': (format_cell_even if row%2==0 else format_cell_odd)})
            
            
            max_column_width = max([len(x) + 2 for x in df_cat.Variable])
            worksheet.set_column('C:C', max_column_width)
            

        writer.save()
        writer.close()

        return result_train, result_test

    
    
    @classmethod
    def __capping_flooring(self,df_train, df_test,c,f):
        '''
        It is necessary to cap the continous variables present in the data. 
        Outier capping is usually done by setting the higher values to the 
        99th percentile and the lower values to the 1st percentile.This function 
        take a dataframe as input, and it returns a dataframe where the continuous 
        values get capped.
        
        Parameters
        ----------
        df_train : Training dataframe on which outlier treatment is to be done
        df_test : Testing dataframe on which outlier treatment is to be done
        
        Returns
        ----------
        df_train : Outlier treated Training dataframe
        df_test : Outlier treated Testing dataframe
        
        
        '''

        c=float(c)
        f=float(f)
        y=[f,c]
        
        train_floor = []
        train_cap = []
        test_floor = []
        test_cap = []

        x= [x for x in df_train.columns if df_train[x].dtypes != 'object']
        '''Filter the continuous variables from the dataframe'''
        train_ex = pd.DataFrame(x, columns =['variable'])
        test_ex = pd.DataFrame(x, columns =['variable'])

        for col in x:
            percentiles = df_train[col].quantile(y).values

            train_floor.append(percentiles[0])
            train_cap.append(percentiles[1])

            percentiles_test = df_test[col].quantile(y).values
            test_floor.append(percentiles_test[0])
            test_cap.append(percentiles_test[1])

            '''calculating the 1st and 99th percentile values'''

            df_train[col][df_train[col] <= percentiles[0]] = percentiles[0]

            '''flooring the values less than 1st percentile by the 1st percentile value'''
            df_train[col][df_train[col] >= percentiles[1]] = percentiles[1]



            '''capping the values greater than 99th percentile by the 99th percentile value'''

            df_test[col][df_test[col] <= percentiles[0]] = percentiles[0]

            '''flooring the values less than 1st percentile by the 1st percentile value'''
            df_test[col][df_test[col] >= percentiles[1]] = percentiles[1]



            '''capping the values greater than 99th percentile by the 99th percentile value''' 

        train_ex['floor'] = train_floor
        train_ex['cap'] = train_cap

        test_ex['floor'] = test_floor
        test_ex['cap'] = test_cap
        test_ex['floored_value'] = [train_floor[i] if train_floor[i]>=test_floor[i] else test_floor[i] for i in range(len(train_floor))]
        test_ex['capped_value'] = [train_cap[i] if train_cap[i]<=test_cap[i] else test_cap[i] for i in range(len(train_cap))]

        return df_train, df_test,train_ex, test_ex
    
    @classmethod
    def __kmean_outlier(self,df_train, df_test, cutoff):
        
        '''
        Using KMeans to find outliers in a cluster of points.Finding outliers means 
        finding the centroids and then looking for elements by their distance from the centroids.
        
        Parameters
        ----------
        df_train : Training dataframe on which outlier treatment is to be done
        df_test : Testing dataframe on which outlier treatment is to be done
        
        Returns
        ----------
        train_o : Outlier treated Training dataframe
        test_o : Outlier treated Testing dataframe
        train_ex : Dataframe of the outliers in training dataset
        test_ex : Dataframe of the outliers in testing dataset
        
        
        '''
        
        cut_off = float(cutoff)
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        newdf_train = df_train.select_dtypes(include=numerics)
        newdf_test = df_test.select_dtypes(include=numerics)
        newdf_train = newdf_train.fillna(newdf_train.mean())
        newdf_test = newdf_test.fillna(newdf_test.mean())
        std=MinMaxScaler()
        std_train = std.fit_transform(newdf_train)
        std_test = std.fit_transform(newdf_test)
        kmeans = KMeans(n_clusters=1)
        kmeans.fit(std_train)
        distances_train = kmeans.transform(std_train)
        distances_test = kmeans.transform(std_test)
        # argsort returns an array of indexes which will sort the array
        # in ascending order. Reverse it with [::-1]
        
        c_train = int(round(newdf_train.shape[0]*(cut_off/100))) 
        c_test = int(round(newdf_test.shape[0]*(cut_off/100))) 
        sorted_train = np.argsort(distances_train.ravel())[::-1][:c_train]
        sorted_test = np.argsort(distances_test.ravel())[::-1][:c_test]
        new_train = np.delete(newdf_train.index, sorted_train, axis=0)
        new_test = np.delete(newdf_test.index, sorted_test, axis=0)
        train_o = df_train[df_train.index.isin(new_train.tolist())]
        test_o = df_test[df_test.index.isin(new_test.tolist())]
        train_ex = df_train[df_train.index.isin(sorted_train.tolist())]
        test_ex = df_test[df_test.index.isin(sorted_test.tolist())]

        return train_o,test_o,train_ex,test_ex
    
    
    @classmethod
    def __z_score_outlier(self,df_train, df_test, cut_z):
        
        '''
        The Z-score is the signed number of standard deviations by which the value of an observation 
        or data point is above the mean value of what is being observed or measured. While calculating the 
        Z-score we re-scale and center the data and look for data points which are too far from zero. 
        These data points which are way too far from zero will be treated as the outliers.
        
        Parameters
        ----------
        df_train : Training dataframe on which outlier treatment is to be done
        df_test : Testing dataframe on which outlier treatment is to be done
        
        Returns
        ----------
        newdf_train_z : Outlier treated Training dataframe
        newdf_test_z : Outlier treated Testing dataframe
        newdf_train_ex : Outlier training dataset
        newdf_test_ex : outlier testing dataset
        
        '''
        cut_z = int(cut_z)
        
        vex = df_train.nunique().reset_index()
        vlist = [vex['index'][i] for i in range(vex.shape[0]) if (vex[0][i]>1)]
        df_train = df_train[vlist]
        df_test = df_test[vlist]
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        newdf_train = df_train.select_dtypes(include=numerics)
        newdf_test = df_test.select_dtypes(include=numerics)
        newdf_train = newdf_train.fillna(newdf_train.mean())
        newdf_test = newdf_test.fillna(newdf_test.mean())
        mean_train = newdf_train.values.mean(axis=0)
        std_train = newdf_train.values.std(axis=0)
        z_train = ((newdf_train-mean_train)/std_train).values
        z_test = ((newdf_test-mean_train)/std_train).values
        
        newdf_train_z = df_train[(z_train < cut_z).all(axis=1)]
        newdf_test_z = df_test[(z_test < cut_z).all(axis=1)]
        newdf_train_ex = df_train[df_train.index.isin(list(set(df_train.index.tolist())-set(newdf_train_z.index.tolist())))]
        newdf_test_ex = df_test[df_test.index.isin(list(set(df_test.index.tolist())-set(newdf_test_z.index.tolist())))]


        return newdf_train_z, newdf_test_z,newdf_train_ex,newdf_test_ex
    
    @classmethod
    def outlier_treatment(self,train_data, test_data, method, strata):
        
        '''
        This function calls the outlier treatment functions, as per the users choice.
        
        Parameters
        ----------
        train_data : Training dataframe on which outlier treatment is to be done
        test_data : Testing dataframe on which outlier treatment is to be done
        method = The method by which outlier treatment is to be done (eg. 'cap_floor', 'kmeans', 'zscore') 
        strata = [flooring percentile in fraction, capping pecentile in fraction] (eg. [0.05,0.95], for 5th and 95th percentile; when method = 'cap_floor') 
        strata = Float value for percentage rows cutoff (when, method = 'kmeans') 
        strata = Integer value for the z score cutoff (when, method = 'zscore') 
        
        Returns
        ----------
        df_train : Outlier treated Training dataframe
        df_test : Outlier treated Testing dataframe
        train_ex : Outlier dataframe from the training dataframe
        test_ex : Outlier dataframe from the testing dataframe
        
        Sample calling command :
        ----------
             
        train_o, test_o, train_ex, test_ex = eda.model_EDA.outlier_treatment(train_data = train_imputed,test_data = test_imputed, method = 'cap_floor', strata = [0.05,0.95])
        ---------- 
        train_o : Outlier treated Training dataframe 
        
        test_o : Outlier treated Testing dataframe 
        
        train_ex : Outlier dataframe from the training dataframe
        
        test_ex : Outlier dataframe from the testing dataframe 
        
        
        '''
        train_data.drop_duplicates(inplace = True)
        test_data.drop_duplicates(inplace = True)
        train_data = train_data.replace(np.Inf, -888888)
        train_data = train_data.replace(-np.Inf, -888888)
        test_data = test_data.replace(np.Inf, -888888)
        test_data = test_data.replace(-np.Inf, -888888)
    
        method = str(method).upper()

        if method == 'CAP_FLOOR':
            train_o,test_o,train_ex,test_ex =  self.__capping_flooring(train_data, test_data,strata[1],strata[0])
            filename = 'Outlier_summary'
            filename = ef.create_version(base_filename=filename, path = os.path.join(os.getcwd(),model_EDA.__save_directory))
            filename += '.xlsx'
            
            writer = pd.ExcelWriter(os.path.join(os.getcwd(),model_EDA.__save_directory,filename), engine='xlsxwriter')
            workbook  = writer.book
        
            # FORMATS--------------------------------------------------------------------------
#            format_round0 = workbook.add_format(model_EDA.__round0)
#            format_round1 = workbook.add_format(model_EDA.__round1)
#            format_round2 = workbook.add_format(model_EDA.__round2)
#            format_round3 = workbook.add_format(model_EDA.__round3)
#            format_percsign = workbook.add_format(model_EDA.__percsign)
#            format_percsignmultipled = workbook.add_format(model_EDA.__percsignmultipled)
            format_cell_highlight1 = workbook.add_format(model_EDA.__cell_highlight1)
#            format_cell_highlight2 = workbook.add_format(model_EDA.__cell_highlight2)
#            format_cell_highlight3 = workbook.add_format(model_EDA.__cell_highlight3)
#            format_cell_highlight4 = workbook.add_format(model_EDA.__cell_highlight4)
#            format_cell_highlight5 = workbook.add_format(model_EDA.__cell_highlight5)
#            format_cell_highlight6 = workbook.add_format(model_EDA.__cell_highlight6)
            format_table_header= workbook.add_format(model_EDA.__table_header)
#            format_table_title = workbook.add_format(model_EDA.__table_title)
            format_cell_even = workbook.add_format(model_EDA.__cell_even)
            format_cell_odd = workbook.add_format(model_EDA.__cell_odd)
#            format_align_centre = workbook.add_format(model_EDA.__align_centre)
#            format_cell_anomaly_bad = workbook.add_format(model_EDA.__cell_anomaly_bad)
#            format_cell_anomaly_good = workbook.add_format(model_EDA.__cell_anomaly_good)
#            format_border1 = workbook.add_format(model_EDA.__border1)
#            format_border2 = workbook.add_format(model_EDA.__border2)
                
            
            if len(train_ex)>0:      
                train_ex.to_excel(writer, sheet_name='Train',index=False,startrow=4,startcol=1) 
        
                worksheet = writer.sheets['Train']
                worksheet.hide_gridlines(2)
                
                # applying formatting
                # table header
                worksheet.conditional_format(ef.generate_excel_cell_name(row_number=5,column_number=2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number=5,column_number=len(train_ex.columns)+1,fix_row=True),
                                              {'type': 'no_blanks','format': format_table_header})
                
                # logo
                worksheet.insert_image(ef.generate_excel_cell_name(row_number=2,column_number=2), os.path.join(os.getcwd(),'src','ks_logo.png'))
                
                
                
                # table cells                       
                rows = list(range(5,len(train_ex)+6))          
                
                for row in rows:
                    worksheet.conditional_format(ef.generate_excel_cell_name(row_number=row,column_number=3,fix_row=True)+':'+ef.generate_excel_cell_name(row_number=row,column_number=len(train_ex.columns)+1,fix_row=True),
                                                 {'type': 'no_blanks','format': (format_cell_even if row%2==0 else format_cell_odd)})
                
                max_column_width = max([len(x) + 2 for x in train_ex.variable])
                worksheet.set_column('B:B', max_column_width)
#                
#                worksheet.set_column('C:C', 15, format_percsign)
#                worksheet.set_column('D:E', 12)
#                worksheet.set_column('H:J', 9,format_percsign)
#                worksheet.set_column('K:N', 8,format_round2)
#            
#                
                worksheet.conditional_format('B6:B'+str(len(train_ex)+5), {'type': 'no_blanks','format': format_cell_highlight1})
                
            if len(test_ex)>0:      
                test_ex.to_excel(writer, sheet_name='Test',index=False,startrow=4,startcol=1) 
        
                worksheet = writer.sheets['Test']
                worksheet.hide_gridlines(2)
                
                # applying formatting
                # table header
                worksheet.conditional_format(ef.generate_excel_cell_name(row_number=5,column_number=2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number=5,column_number=len(test_ex.columns)+1,fix_row=True),
                                              {'type': 'no_blanks','format': format_table_header})
                
                # logo
                worksheet.insert_image(ef.generate_excel_cell_name(row_number=2,column_number=2), os.path.join(os.getcwd(),'src','ks_logo.png'))
                
                
                
                # table cells                       
                rows = list(range(5,len(test_ex)+6))          
                
                for row in rows:
                    worksheet.conditional_format(ef.generate_excel_cell_name(row_number=row,column_number=3,fix_row=True)+':'+ef.generate_excel_cell_name(row_number=row,column_number=len(test_ex.columns)+1,fix_row=True),
                                                 {'type': 'no_blanks','format': (format_cell_even if row%2==0 else format_cell_odd)})
                
                max_column_width = max([len(x) + 2 for x in test_ex.variable])
                worksheet.set_column('B:B', max_column_width)
#                
                worksheet.set_column('E:F', 13)
#                worksheet.set_column('D:E', 12)
#                worksheet.set_column('H:J', 9,format_percsign)
#                worksheet.set_column('K:N', 8,format_round2)
#            
#                
                worksheet.conditional_format('B6:B'+str(len(test_ex)+5), {'type': 'no_blanks','format': format_cell_highlight1})
            
            writer.save()
            writer.close()
            
            
        elif method == 'KMEANS':
            train_o,test_o,train_ex,test_ex = self.__kmean_outlier(train_data, test_data, strata)
        elif method == 'ZSCORE':
            train_o,test_o,train_ex,test_ex = self.__z_score_outlier(train_data, test_data, strata)
        
      
    
        ef.reset_header_default()
        filename = 'Train_outlier_treated'
        filename = ef.create_version(base_filename=filename, path = os.path.join(os.getcwd(),model_EDA.__save_directory))
        filename += '.csv'
        train_o.to_csv(os.path.join(os.getcwd(),model_EDA.__save_directory,filename),index=False)
        
        filename = 'Test_outlier_treated'
        filename = ef.create_version(base_filename=filename, path = os.path.join(os.getcwd(),model_EDA.__save_directory))
        filename += '.csv'
        test_o.to_csv(os.path.join(os.getcwd(),model_EDA.__save_directory,filename),index=False)
        
            
            
        return train_o,test_o,train_ex,test_ex
    
    
    @classmethod
    def __WOE(self,data, varList, type0='Con', target_id='y'):
        """This Function performs bivariate on numeric as well as categorical variables
        
            Attributes
            -------
            data: pandas DataFrame, mostly refer to ABT(Analysis Basics Table)
            varList: variable list
            type0: Continuous or Discontinuous(Category), 'con' is the required input for Continuous
            target_id: y flag when gen the train data
            resfile: download path of the result file of WOE and IV

            Returns
            -------       
            pandas DataFrame, result of woe and iv value according y flag
            
        """
        data=data.astype({target_id: 'int32'})
        result = pd.DataFrame()
        for var in varList:
            print(var)
            try:
                if type0.upper() == "CON".upper():
                    data.replace( np.nan,-999999,inplace=True) 
                    if len(list(data[var].unique()))>10:
                        df, retbins = pd.qcut(data[var], q=10, retbins=True, duplicates="drop")
                        tmp = pd.crosstab(df, data[target_id])
                        tmp2 = pd.crosstab(df, data[target_id]).apply(lambda x: x / sum(x), axis=0)
                    else:
                        df = data[var]
                        tmp = pd.crosstab(data[var], data[target_id])
                        tmp2 = pd.crosstab(data[var], data[target_id]).apply(lambda x: x / sum(x),axis=0)
                else:
                    data.replace( np.nan,-999999,inplace=True) 
                    df = data[var]
                    tmp = pd.crosstab(data[var], data[target_id])
                    tmp2 = pd.crosstab(data[var], data[target_id]).apply(lambda x: x / sum(x),axis=0)


                res = tmp.merge(tmp2, how="left", left_index=True, right_index=True)

                res['Population_perc'] = np.round((res['0_x']+res['1_x'])* 100/data.shape[0],2)
                res['Good_perc'] = np.round((res['0_x']) / sum(res['0_x']),2)  # Adjusting Woe +0.5
                res['Bad_perc'] = np.round((res['1_x']) / sum(res['1_x']),2)  # Adjusting Woe +0.5
                res['WOE'] = np.round(np.log(res['Bad_perc'] / res['Good_perc']),2)
                res['dif'] = np.round(res['Bad_perc'] - res['Good_perc'],2)
                res['IV'] = np.round(res['WOE'] * res['dif'],2)
                res['name'] = var
                res.index.name = ""


                ###handling inf
                res['IV_check']=res['IV']
                res.loc[res['IV']==np.inf,'IV_check']=np.nan
                res['IV_sum'] = res['IV_check'].sum()

                #res['IV_sum'] = res['IV'].sum()
                del res['0_y']

                if type0.upper() == "CON".upper():
                    if len(list(data[var].unique()))>10:
                        res['low'] = retbins[:-1]
                        res['high'] = retbins[1:]
                        res.index = range(len(retbins) - 1)
                    else:
                        res['low'] = res.index
                        res['high'] = res.index
                        res.reset_index
                else:
                    res['low'] = res.index
                    res['high'] = res.index
                    res.reset_index
                res = res[
                    ['name','Population_perc', 'low', 'high', '0_x', '1_x', '1_y', 'Bad_perc', 'Good_perc', 'WOE',
                     'dif',
                     'IV', 'IV_sum']]
                result = result.append(res)

            except Exception as e:
                print(e, var)
        result.rename(columns = {'0_x':'Goods','1_x':'Bads','1_y':'Badrate'}, inplace = True)
        result['count'] = result['Goods']+result['Bads']
        #result.to_excel(resfile)
        result.round()
        #result.IV.replace([np.inf, -np.inf], np.nan,inplace=True)
        #result.WOE.replace([np.inf, -np.inf], np.nan,inplace=True)

        result['Badrate']=np.round(result['Badrate'],2)
        return result

    @classmethod
    def bivariate_analysis(self,data,ID,target_id='y'):
        """This Function performs bivariate analysis on the given data 
        
                Attributes
                ----------
                data: A Pandas Dataframe
                ID: Application Id
                target_id: Name of the target variable
                
                Returns
                -------
                A list of two dataframes: Numeric Bivariate and Categorical Bivariate
                
                Code
                -------
                
                rvc=eda_obj.bivariate_analysis(train,ID=ID,target_id=target_id)
        """
        
        date_columns = [x for x in data.columns if ('date' in str(data[x].dtypes)) or ('time' in str(data[x].dtypes))]
        if len(date_columns)>0:
            print("The date columns:", date_columns, "have been removed before analysis")
            data.drop(columns=date_columns,inplace=True)
        
        
        data.drop_duplicates(inplace = True)
        data = data.replace(np.Inf, -888888)
        data = data.replace(-np.Inf, -888888)
                
        X = [x for x in data.columns if data[x].dtypes != 'object' and x not in [target_id,ID]]  

        woe_num=self.__WOE(data, X, type0='Con', target_id=target_id)             

        X = [x for x in data.columns if data[x].dtypes == 'object' and x not in [target_id,ID]]  

        woe_cat=self.__WOE(data, X, type0='Cat', target_id=target_id)         
        woe_cat.reset_index(inplace=True,drop=True)

       # woe_num.to_excel(numeric_bivariate_output_file_name+".xlsx")
        #woe_cat.to_excel(categorical_bivariate_output_file_name+".xlsx")
        
        
        woe_num.rename(columns={"name":"Variable",
                                "Population_perc":"Population %",
                                "low":"Low",
                                "high":"High",
                                "Bad_perc":"Bad %",
                                "Good_perc":"Good %",
                                "dif":"Diff",
                                "IV_sum":"IV Sum",
                                "count":"Count"},inplace=True)
    
        woe_cat.rename(columns={"name":"Variable",
                                "Population_perc":"Population %",
                                "low":"Low",
                                "high":"High",
                                "Bad_perc":"Bad %",
                                "Good_perc":"Good %",
                                "dif":"Diff",
                                "IV_sum":"IV Sum",
                                "count":"Count"},inplace=True)
        
              
        
        filename = 'Bivariate'
        filename = ef.create_version(base_filename=filename, path = os.path.join(os.getcwd(),model_EDA.__save_directory))
        filename += '.xlsx'
        
        writer = pd.ExcelWriter(os.path.join(os.getcwd(),model_EDA.__save_directory,filename), engine='xlsxwriter')
        workbook  = writer.book
        
        
        # FORMATS--------------------------------------------------------------------------
        
        
        
#        format_round0 = workbook.add_format(model_EDA.__round0)
#        format_round1 = workbook.add_format(model_EDA.__round1)
        format_round2 = workbook.add_format(model_EDA.__round2)
#        format_round3 = workbook.add_format(model_EDA.__round3)
        format_percsign = workbook.add_format(model_EDA.__percsign)
#        format_percsignmultipled = workbook.add_format(model_EDA.__percsignmultipled)
        format_cell_highlight1 = workbook.add_format(model_EDA.__cell_highlight1)
#        format_cell_highlight2 = workbook.add_format(model_EDA.__cell_highlight2)
#        format_cell_highlight3 = workbook.add_format(model_EDA.__cell_highlight3)
#        format_cell_highlight4 = workbook.add_format(model_EDA.__cell_highlight4)
#        format_cell_highlight5 = workbook.add_format(model_EDA.__cell_highlight5)
#        format_cell_highlight6 = workbook.add_format(model_EDA.__cell_highlight6)
        format_table_header= workbook.add_format(model_EDA.__table_header)
#        format_table_title = workbook.add_format(model_EDA.__table_title)
        format_cell_even = workbook.add_format(model_EDA.__cell_even)
        format_cell_odd = workbook.add_format(model_EDA.__cell_odd)
#        format_align_centre = workbook.add_format(model_EDA.__align_centre)
#        format_cell_anomaly_bad = workbook.add_format(model_EDA.__cell_anomaly_bad)
#        format_cell_anomaly_good = workbook.add_format(model_EDA.__cell_anomaly_good)
#        format_border1 = workbook.add_format(model_EDA.__border1)
#        format_border2 = workbook.add_format(model_EDA.__border2)

        
        # SHEET: Continuous--------------------------------------------------------------------------
        
        
        if len(woe_num)>0:      
            
            woe_num.to_excel(writer, sheet_name='Continuous',index=False,startrow=4,startcol=1) 
    
            worksheet = writer.sheets['Continuous']
            worksheet.hide_gridlines(2)
            
            # applying formatting
            # table header
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number=5,column_number=2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number=5,column_number=len(woe_num.columns)+1,fix_row=True),
                                          {'type': 'no_blanks','format': format_table_header})
            
            # logo
            worksheet.insert_image(ef.generate_excel_cell_name(row_number=2,column_number=2), os.path.join(os.getcwd(),'src','ks_logo.png'))
            
            
            
            # table cells                       
            rows = list(range(5,len(woe_num)+6))          
            
            for row in rows:
                worksheet.conditional_format(ef.generate_excel_cell_name(row_number=row,column_number=3,fix_row=True)+':'+ef.generate_excel_cell_name(row_number=row,column_number=len(woe_num.columns)+1,fix_row=True),
                                             {'type': 'no_blanks','format': (format_cell_even if row%2==0 else format_cell_odd)})
            
            max_column_width = max([len(x) + 2 for x in woe_num.Variable])
            worksheet.set_column('B:B', max_column_width)
            
            worksheet.set_column('C:C', 15, format_percsign)
            worksheet.set_column('D:E', 12)
            worksheet.set_column('H:J', 9,format_percsign)
            worksheet.set_column('K:N', 8,format_round2)
        
            
            worksheet.conditional_format('B6:B'+str(len(woe_num)+5), {'type': 'no_blanks','format': format_cell_highlight1})
            

        # SHEET: Categorical--------------------------------------------------------------------------
        if len(woe_cat)>0:
        
            woe_cat.to_excel(writer, sheet_name='Categorical',index=False,startrow=4,startcol=1) 
            worksheet = writer.sheets['Categorical']
            worksheet.hide_gridlines(2)
            
            # applying formatting
            # table header
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number=5,column_number=2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number=5,column_number=len(woe_cat.columns)+1,fix_row=True),
                                          {'type': 'no_blanks','format': format_table_header})
            
            # logo
            worksheet.insert_image(ef.generate_excel_cell_name(row_number=2,column_number=2), os.path.join(os.getcwd(),'src','ks_logo.png'))
            
            
                
            # table cells                       
            rows = list(range(5,len(woe_cat)+6))          
            
            for row in rows:
                worksheet.conditional_format(ef.generate_excel_cell_name(row_number=row,column_number=3,fix_row=True)+':'+ef.generate_excel_cell_name(row_number=row,column_number=len(woe_cat.columns)+1,fix_row=True),
                                             {'type': 'no_blanks','format': (format_cell_even if row%2==0 else format_cell_odd)})
            
            max_column_width = max([len(x) + 2 for x in woe_cat.Variable])
            worksheet.set_column('B:B', max_column_width)
            
            worksheet.set_column('C:C', 15, format_percsign)
            worksheet.set_column('D:E', 12)
            worksheet.set_column('H:J', 9,format_percsign)
            worksheet.set_column('K:N', 8,format_round2)
        
            
            worksheet.conditional_format('B6:B'+str(len(woe_cat)+5), {'type': 'no_blanks','format': format_cell_highlight1})
    
    

        writer.save()
        writer.close()
        
        return ([woe_num,woe_cat])
    
    @classmethod
    def correlation(self,dataset):
        
        '''
        This functions generates the correlation table showing the positive or negative
        correlations amongst the variables in the dataset
        
        Parameters
        ----------
        dataset : The dataframe on which the correlation table has to be genrated
        
        Returns
        ----------
        corr : Object showing the correlation amongst the variables
        
        Sample calling command :
        -----------
        
        corr = eda_obj.correlation(dataset = train_o) 
        
               
        
        '''
        dataset.drop_duplicates(inplace = True)
        dataset = dataset.replace(np.Inf, -888888)
        dataset = dataset.replace(-np.Inf, -888888)
        
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        num_var = dataset.select_dtypes(include=numerics)
        x = num_var.columns.tolist()
        df_contiuous=dataset[x]
        corr = df_contiuous.corr()
        corr = corr.round(2)
        corr_result = corr.style.background_gradient(cmap='coolwarm')
        

        filename = 'Correlation_Table'
        filename = ef.create_version(base_filename=filename, path = os.path.join(os.getcwd(),model_EDA.__save_directory))
        filename += '.xlsx'
        
        corr_result.to_excel(os.path.join(os.getcwd(),model_EDA.__save_directory,filename))
        
        return corr_result





