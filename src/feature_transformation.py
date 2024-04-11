# -*- coding: utf-8 -*-
"""
<b>Dependencies</b>
-------------------
    python>=3.6
    pandas==0.24.2
    numpy==1.18.4
    sklearn==0.20.3

**Additional Files**
--------------------
        
    1. excel_formatting.py 
    
Please ensure that the above files are also present in the working directory and packages mentioned in the dependencies section are installed.

<b>How to import the module and run the code?</b>
-------------------------------------------------
    Sample Codes:    
        
    # Import the Feature_Transformation module
    import Feature_Transformation as ft
    
    # Sample code to perform transformation on continuous variables
    # selected_num_cols: The following functionality is for only numerical variables.Please put the names of the continuous variables in place of 'selected_num_cols'

    trans = ft.Variable_Transformation(train[selected_num_cols], test[selected_num_cols])
    test_trans = trans.variable_transformation()

<b>Description</b>
------------------
This module has various methods of Feature Transformations to perform on the independent variables while building a model.
From simple transformations like Logarithmic, One Hot Encoding, Percentile Binning to Automated and Optimized WOE Binning

The following functionalities are available.

    1. Variable_Transformation
    2. Encoding
    3. Binning_Perc
    4. WoE_Binning_Automated
    5. WoE_Binning_Optimized
    6. WoE_Transformation
    7. WoE_Transformation_Manual
    
###<b>Variable Transformation</b>
The independent variables are transfomed into their square or log or a different function of the variable before passing them to the model. This is done to change the shape of the distribution or relationship with the Target variable

For this the following functionality is provided to transform continuous variables. The following options are avialable.

    1. "Standard Scaler"
    2. "MinMax Scaler"
    3. "Square Root"
    4. "Square"
    5. 'Log'
    6. 'Inverse'
    7. 'Inverse Square Root'
    8. 'Rank'

    Sample Codes:

    import Feature_Transformation as ft
    
    # selected_num_cols: The following functionality is for only numerical variables. Please put the names of the continuous variables in place of 'selected_num_cols'
    trans = ft.Variable_Transformation(train[selected_num_cols], test[selected_num_cols], transform_var="Standard Scaler")
    trans.variable_transformation()


###<b>Encoding</b>
This is a process by which categorical variables are converted into a binary vector form that could be provided to ML algorithms as an input.

For this the following functionality is provided for Encoding (Dummification) the categorical variables (One Hot Encoding)

    Sample Codes:

    import Feature_Transformation as ft
    encode = ft.Encoding()
    
    # selected_cat_cols: The following functionality is for only categorical variables. Please put the names of the categorical variables in place of 'selected_cat_cols'
    encode.fit(train[selected_cat_cols])
    encode.transform(test[selected_cat_cols])

###<b>Binning</b>
####<b>Binning with Percentiles</b>
This is a process by which continuous variables are grouped to discover set of patterns which are difficult observe otherwise. Although bins are easy to analyze, this might lead to loss of information and power of data.

For this the following functionality is provided for Percentile Binning of continuous variables

    Sample Codes:

    import Feature_Transformation as ft
    bin = ft.Binning_Perc()

    # selected_num_cols: The following functionality is for only numerical variables. Please put the names of the continuous variables in place of 'selected_num_cols'
    bin.fit(train[selected_num_cols])
    bin.transform(test[selected_num_cols])

####<b>WOE Binning with Percentiles</b>
WOE binning generates a supervised fine and coarse classing of numeric variables and factors with respect to a dichotomous target variable. Its parameters provide flexibility in finding a binning that fits specific data characteristics and practical needs.

For this the following functionality is provided for WoE bucketing/binning of continuous and discrete variables. WOE Binning is done based on the number of buckets/percentiles passed. For each bucket or group the WOE and IV values calculated for each variable.

    Sample Codes:

    import Feature_Transformation as ft

    # For Numerical Variables
    woe = ft.WoE_Binning_Automated(v_type='c')
    
    # selected_num_cols: The following functionality is for only numerical variables as v_type is passes as "c" in the above class. Please put the names of the continuous variables in place of 'selected_num_cols'
    woe.woe_bins(train_f[selected_num_cols+[target]], target=target)
    
####<b>WOE Optimized Binning</b>
The WOE buckets/bins are optmized i.e., chosen using the DecisionTree splits between independent variable and target variable.

For this the following functionality is provided for WoE bucketing/binning of continuous variables. For each bucket or group the WOE and IV values calculated for each variable.

    Sample Codes:

    import Feature_Transformation as ft        
    woe = ft.WoE_Binning_Optimization(v_type='c')
    
    # selected_num_cols: The following functionality is for only numerical variables as v_type is passes as "c" in the above class. Please put the names of the continuous variables in place of 'selected_num_cols'
    woe.woe_bins_optimized(train_f[selected_num_cols+[target]], target=target)

####<b>WOE Automated Binning</b>
In the automated binning, the adjacent bins are regrouped or merged to make them exhibit monotonic behaviour wrt the target variable.

For this the following functionality is provided for WoE bucketing/binning of continuous and discrete variables. Pass monotonic=True to the woe_bins method For each bucket or group the WOE and IV values calculated for each variable.

    Sample Codes:

    import Feature_Transformation as ft
    woe = ft.WoE_Binning_Automated(v_type='c')
    
    # selected_num_cols: The following functionality is for only numerical variables as v_type is passes as "c" in the above class. Please put the names of the continuous variables in place of 'selected_num_cols'
    binned_df, mon_binned_df = woe.woe_bins(train_f[selected_num_cols+[target]], target=target, monotonic=True)
    
####<b>WOE Transformation</b>
For this the following functionality is provided to transform the variables to their WOE values using either Percentile bins or Automated bins or Optimized bins. 

To use Automated bins, pass transformation_type="Automated" in the class initialization and monotonic=True in the 'woe_transformation' method.

To use Optimized bins, pass transformation_type="Optimize".

    Sample Codes:

    import Feature_Transformation as ft
    woe = ft.WoE_Transformation(transformation_type='Automated',v_type="c")
    
    # selected_num_cols: The following functionality is for only numerical variables as v_type is passes as "c" in the above class. Please put the names of the continuous variables in place of 'selected_num_cols'
    woe_transform_test_num = woe.woe_transformation(train_o[selected_num_cols+[target]], test_o[selected_num_cols+[target]], monotonic=True, target=target)

####<b>WOE Transformation Manual</b>
Functionality to transform the independent variables into woe by using User Manual Bins.

    Sample Codes:

    import Feature_Transformation as ft
    
    # To run the following codes you need to have an excel with the WOE binned data with the following columns
    # ["Variable","Labels","Min","Max","Count","Bads","Goods","Population %","Bads %","Goods %","WOE","IV","Bad Rate","bins"]
    
    woe_data = pd.read_excel("WOE_Manual_Binning.xlsx")
    
    # The variables in the test data whould match with the varibales in the woe_data
    woe_manual_transformed = ft.WoE_Transformation_Manual().woe_data(target,test_data,woedata=woe_data)

<b>Please refer each class and corresponding methods to understand their functionality</b>
    
<b>Author</b>
------
Created on Fri Jun  12 15:40:27 2020

@author: Kumar, Shubham
"""

import pandas as pd
#from openpyxl import load_workbook
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
#from feature_engine import categorical_encoders as ce
import operator
import numpy as np
import re
#import matplotlib.pyplot as plt
from sklearn import tree
#from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
cwd = os.getcwd()
#cwd = r"C:\Users\sangaraju.madhuri\Desktop\ML Asset Library\python modules\Feature Transformation"
from sklearn.model_selection import cross_val_score
from .excel_formatting import excel_formatting as ef


class Variable_Transformation:
    """
    Basic Functionality to transform continuous variables. The following options are avialable.
    ["Standard Scaler","MinMax Scaler","Square Root","Square",'Log','Inverse','Inverse Square Root','Rank']
    
        Sample Codes:
    
        import Feature_Transformation as ft
        
        # selected_num_cols: The following functionality is for only numerical variables. Please put the names of the continuous variables in place of 'selected_num_cols'
        trans = ft.Variable_Transformation(train[selected_num_cols], test[selected_num_cols], transform_var="Standard Scaler")
        trans.variable_transformation()
    """
    
    save_directory = 'results/Feature Transformation'
    if os.path.isdir("results")==False:
        os.mkdir("results")
    
    if os.path.isdir(save_directory)==False:
        os.mkdir(save_directory)
    
    round0 = ef.formatting_xlsxwriter(format_name='round0')
    round2 = ef.formatting_xlsxwriter(format_name='round2')
    round3 = ef.formatting_xlsxwriter(format_name='round3')
    percsign = ef.formatting_xlsxwriter(format_name='percsign')
    percsignmultipled = ef.formatting_xlsxwriter(format_name='percsignmultipled')
    cell_highlight1 = ef.formatting_xlsxwriter(format_name='cell_highlight1')
    cell_highlight2 = ef.formatting_xlsxwriter(format_name='cell_highlight2')
    cell_anomaly_bad = ef.formatting_xlsxwriter(format_name='cell_anomaly_bad')
    table_header= ef.formatting_xlsxwriter(format_name='table_header')
    table_title = ef.formatting_xlsxwriter(format_name='table_title')
    cell_even = ef.formatting_xlsxwriter(format_name='cell_even')
    cell_odd = ef.formatting_xlsxwriter(format_name='cell_odd')
    align_centre = ef.formatting_xlsxwriter(format_name='align_centre')
    border1 = ef.formatting_xlsxwriter(format_name='border1')
    border2 = ef.formatting_xlsxwriter(format_name='border2')

    def __init__(self, train, test=None, ID=None, out_file="VariableTransformation_Output", transform_var=None, save=False):
        """ 
        Input
        -----
        df: Data to be encoded, Multiple variables can be passed
        
        transfrom: The transformations available are
        ["Standard Scaler","MinMax Scaler","Square Root","Square",'Log','Inverse','Inverse Square Root','Rank']

        Choose the type of transformation to be applied. If None, transforms all available transformations are applied. If a single type of transformation is passed, all the columns in the dataframe are transformed according to the transfrom variable. 
        You can pass it as dictionary with the column names as keys and type of transformation to be applied as the corresponding values.
        """
        self.save = save
        if ID is None:
            train.reset_index(drop=True,inplace=True)
        if ID is not None:
            train.set_index(ID, inplace=True)
            
        self.train = train.copy()
        if test is None:
            test = train

        if ID is None:
            test.reset_index(drop=True,inplace=True)
        if ID is not None:
            test.set_index(ID, inplace=True)
        self.test = test.copy()
        
        rem_cols = []
        for col in self.train.columns:
            try:
                self.train[col] = self.train[col].astype(float)
            except:
                rem_cols.append(col)
        if len(rem_cols)>0:
            print("The following variables are not continious in train data. Ignoring these variables for further calculations\n\n",rem_cols)
        self.train = self.train[list(set(self.train.columns)-set(rem_cols))]
        impute_cols = []
        for col in self.train.columns:
            self.train[col] = self.train[col].replace([float("inf"), float("-inf")], np.nan)
#            print(col)
            if self.train[col].isna().sum()>0:
                impute_cols.append(col)
#                print(col)
                self.train.loc[self.train[col].isna(),col]=self.train[col].mean()
        if len(impute_cols)>0:
            print("The following continuous variables have null or inf values in train data. Imputed with mean of the respective variables\n\n",impute_cols)

        rem_cols = []
        for col in self.test.columns:
            try:
                self.test[col] = self.test[col].astype(float)
            except:
                rem_cols.append(col)
        if len(rem_cols)>0:
            print("The following variables are not continious in test data. Ignoring these variables for further calculations\n\n",rem_cols)
        self.test = self.test[list(set(self.test.columns)-set(rem_cols))]
        impute_cols = []
        for col in self.test.columns:
            self.test[col] = self.test[col].replace([float("inf"), float("-inf")], np.nan)
#            print(col)
            if self.test[col].isna().sum()>0:
                impute_cols.append(col)
#                print(col)
                self.test.loc[self.test[col].isna(),col]=self.test[col].mean()
        if len(impute_cols)>0:
            print("The following continuous variables have null or inf values in test data. Imputed with mean of the respective variables\n\n",impute_cols)


        self.transform_var = transform_var
        self.out_file = out_file
        
    def variable_transformation(self):
        """
        Output
        ------
        Returns the dataframe with the transformed columns based on the type transformations passed to the 'transform' variable. Also writes the output to an excel file
        """
        transformations = ["Standard Scaler","MinMax Scaler","Square Root","Square",'Log','Inverse','Inverse Square Root','Rank']
        if self.transform_var is None:
            write_df = self.test.copy()
#            write_dict = [""]*len(transformations)
            for i,transformation in enumerate(transformations):
                self.transform_var = transformation
                if self.transform_var == "Standard Scaler":
                    tmp_df = self.standard_scaling(self.train,self.test)
    
                if self.transform_var == "MinMax Scaler":
                    tmp_df = self.min_max_scaling(self.train,self.test)
    
                elif self.transform_var not in ["Standard Scaler","MinMax Scaler"]:
                    tmp_df = self.transform_fn(self.test, type_trans=self.transform_var)
                
                write_df = write_df.merge(tmp_df, left_index=True, right_index=True)
            self.transform_var=None
                
        if str(type(self.transform_var))=="<class 'str'>":

            if self.transform_var == "Standard Scaler":
                write_df = self.standard_scaling(self.train,self.test)

            if self.transform_var == "MinMax Scaler":
                write_df = self.min_max_scaling(self.train,self.test)

            elif self.transform_var not in ["Standard Scaler","MinMax Scaler"]:
                write_df = self.transform_fn(self.test, type_trans=self.transform_var)
                
        if str(type(self.transform_var))=="<class 'dict'>":
            write_df = self.test.copy()

            for key in self.transform_var.keys():
                print(key,self.transform_var[key])
                if self.transform_var[key] == "Standard Scaler":
                    tmp_df = self.standard_scaling(self.train[[key]],self.test[[key]])

                if self.transform_var[key] == "MinMax Scaler":
                    tmp_df = self.min_max_scaling(self.train[[key]],self.test[[key]])

                elif self.transform_var[key] not in ["Standard Scaler","MinMax Scaler"]:
                    tmp_df = self.transform_fn(self.test[[key]], type_trans=self.transform_var[key])
                
                write_df = write_df.merge(tmp_df, left_index=True, right_index=True)

        write_df.sort_index(axis=1, inplace=True)
#        write_df = write_df.round(2)
        if self.save==True:
            filename = 'VariableTransformation'
            filename = ef.create_version(base_filename=filename, path = os.path.join(cwd,Variable_Transformation.save_directory))
            filename += ".csv"
            
            print("Saving the out put file in: \n",os.path.join(cwd,Variable_Transformation.save_directory))
            write_df.to_csv(os.path.join(cwd,Variable_Transformation.save_directory,filename),index=False)
            
        return write_df

    def min_max_scaling(self,train,test=None):
        """
        Input
        -----
        df: Dataframe with the variables to be transformed
        
        Output
        ------
        Returns the dataframe with the transformed columns
        """
        if test is None:
            test = train
        self.min_max_scaler = MinMaxScaler()
#        print(df.columns)
        self.min_max_scaler.fit(train.values)
#        print(df.columns)
        final_df = pd.DataFrame(self.min_max_scaler.transform(test.values))
        final_df.columns = [col+"_minmax_scaling" for col in test.columns]
#        ret_df = df.merge(final_df, left_index=True, right_index=True)
            
#        return ret_df
        return final_df
    
    def standard_scaling(self, train, test=None):
        """
        Input:
        -----    
        df: Dataframe with the variables to be transformed
        
        Output
        -----
        Returns the dataframe with the transformed columns
        """
        if test is None:
            test = train
            
        self.standard_scaler = StandardScaler()
        self.standard_scaler.fit(train.values)
        final_df = pd.DataFrame(self.standard_scaler.transform(test.values))
        final_df.columns = [col+"_std_scaling" for col in test.columns]
#        ret_df = df.merge(final_df, left_index=True, right_index=True)

#        return ret_df
        return final_df
    
    def transform_fn(self, df, type_trans):
        """
        Input
        -----
        df: Dataframe with the variables to be transformed
        
        Output
        -----
        Returns the dataframe with the transformed columns
        """
#        print(type_trans)
        ret_df = pd.DataFrame()
        for i in df.columns:
            if type_trans=="Square Root":
                ret_df[i+"_square root"]=df[i].apply(np.sqrt)
            elif type_trans=="Square":
                ret_df[i+"_square"]=df[i]**2
            elif type_trans=='Log':
                ret_df[i+'_log']=np.log10(df[i])
            elif type_trans=='Inverse':
                ret_df[i+'_inverse']=1/df[i]
            elif type_trans=='Inverse Square Root':
                ret_df[i+"_inverse square root"]=1/df[i].apply(np.sqrt)
            elif type_trans=='Rank':
                ret_df[i+"_rank"]=df[i].rank()
            else:
                raise Exception(type_trans, "The type of transformation chosen is not available")
                        
        return ret_df


class Encoding:
    """
    Basic functionality for Encoding (Dummification) the categorical variables (One Hot Encoding)
    
        Sample Codes:
    
        import Feature_Transformation as ft
        encode = ft.Encoding()
        encode.fit(train[selected_cat_cols])
        encode.transform(test[selected_cat_cols])

    """

    def __init__(self, out_file="OneHotEncoding_Output",save=False):
        """ 
        Description of parameters
        -------------------------
                    
        data: Categorical data to be encoded or dummified, Multiple variables/Columns can be passed. Should be a pandas dataframe
        
        cols: List of Categorical Column/Varibale names to be encoded. If not provided, considers all columns in the dataframe
        
        out_file: Name of the out put file
        """
        
        self.save = save
        self.data = None
        self.out_file = out_file

    def fit(self, data, ID=None, cols=None):
        """
        Input
        -----
        data: Data frame with the categorical variables
        
        cols: List of Columns/Variables in the dataframe to be encoded. If nothing passed, all the columns in the dataframe are considered
        
        Output
        ------
        Returns a class object
        """
        if ID is None:
            data.reset_index(drop=True,inplace=True)
        if ID is not None:
            data.set_index(ID, inplace=True)
        if cols is not None:
            self.data = data[cols].copy()
            
        else:
            self.data = data.copy()
        
        impute_cols = []
        for col in self.data.columns:
            self.data[col] = self.data[col].replace([float("inf"), float("-inf")], np.nan)
#            print(col)
            if self.data[col].isna().sum()>0:
                impute_cols.append(col)
#                print(col)
                self.data.loc[self.data[col].isna(),col]=self.data[col].mode()
        if len(impute_cols)>0:
            print("The following continuous variables have null or inf values. Imputed with mode of the respective variables\n\n",impute_cols)

        self.enc = OneHotEncoder(handle_unknown="ignore")
#        X = pd.DataFrame([['Male', 1], ['Female', 3], ['Female', 2]], columns = ['gender', 'group'])
#        self.data = self.data.append(pd.DataFrame([["OTHERS"]*len(self.data.columns)], columns = list(self.data.columns)))
        self.data = self.data.astype(str)
        
        self.enc.fit(self.data)
        return self

    def fit_transform(self, data, cols=None, ID=None):
        """
        Input
        -----
        data: Data frame with the categorical variables
        
        cols: List of Columns/Variables in the dataframe to be encoded. If nothing passed, all the columns in the dataframe are considered
        
        Output
        ------
        Returns encoded data. Also saves the data into an excel file
        """
        if ID is None:
            data.reset_index(drop=True,inplace=True)
        if ID is not None:
            data.set_index(ID, inplace=True)
            
        if cols is not None:
            self.data = data[cols].copy()
        else:
            self.data = data.copy()

        impute_cols = []
        for col in self.data.columns:
            self.data[col] = self.data[col].replace([float("inf"), float("-inf")], np.nan)
#            print(col)
            if self.data[col].isna().sum()>0:
                impute_cols.append(col)
#                print(col)
                self.data.loc[self.data[col].isna(),col]=self.data[col].mode()
        if len(impute_cols)>0:
            print("The following continuous variables have null or inf values. Imputed with mode of the respective variables\n\n",impute_cols)


        self.data = self.data.astype(str)            
        enc = OneHotEncoder(handle_unknown="ignore")
        
        data_encoded = pd.DataFrame(enc.fit_transform(self.data).toarray(),columns=list(enc.get_feature_names(list(self.data.columns))))
        data_encoded = self.data.merge(data_encoded, left_index=True, right_index=True)
        if self.save==True:
            filename = 'OneHotEncoded_Data'
            filename = ef.create_version(base_filename=filename, path = os.path.join(cwd,Variable_Transformation.save_directory))
            filename += ".csv"
            
            print("Saving the out put file in: \n",os.path.join(cwd,Variable_Transformation.save_directory))
            data_encoded.to_csv(os.path.join(cwd,Variable_Transformation.save_directory,filename),index=False)
            
        

        return data_encoded

    def transform(self, df, cols=None, ID=None):
        """
        Input
        -----
        df: Data frame with the categorical variables
        
        cols: List of Columns/Variables in the dataframe to be encoded. If nothing passed, all the columns in the dataframe are considered
        
        Output
        ------
        Returns encoded data (The new or unknown categories in all the variables are replaced with "OTHERS"). Also saves the data into an excel file
        """
        if ID is None:
            df.reset_index(drop=True,inplace=True)
        if ID is not None:
            df.set_index(ID, inplace=True)
            
        if cols is not None:
            df = df[cols].copy()
        else:
            df = df.copy()
            
        df = df.astype(str)

        impute_cols = []
        for col in df.columns:
            df[col] = df[col].replace([float("inf"), float("-inf")], np.nan)
#            print(col)
            if df[col].isna().sum()>0:
                impute_cols.append(col)
#                print(col)
                df.loc[df[col].isna(),col]=df[col].mode()
        if len(impute_cols)>0:
            print("The following continuous variables have null or inf values. Imputed with mode of the respective variables\n\n",impute_cols)

        if self.data is None:
            raise Exception("Please fit the train data first")
            
        if sorted(list(self.data.columns))!=sorted(list(df.columns)):
            raise Exception("There are different variables than in the train data")
            
#        for col in df.columns:
#            df[col].replace(list(set(df[col].unique())-set(self.data[col].unique())),"OTHERS",inplace=True)

        data_encoded = pd.DataFrame(self.enc.transform(df).toarray(),columns=list(self.enc.get_feature_names(list(self.data.columns))))
#        other_cols = [col for col in data_encoded if "_OTHERS" in col]
#        del_other_cols = []
#        for col in other_cols:
#            if int(data_encoded[col].sum())==0:
#                del_other_cols.append(col)

#        data_encoded.drop(del_other_cols,axis=1,inplace=True)
        data_encoded = df.merge(data_encoded, left_index=True, right_index=True)

        if self.save==True:
            filename = 'OneHotEncoded_Data'
            filename = ef.create_version(base_filename=filename, path = os.path.join(cwd,Variable_Transformation.save_directory))
            filename += ".csv"
            
            print("Saving the out put file in: \n",os.path.join(cwd,Variable_Transformation.save_directory))
            data_encoded.to_csv(os.path.join(cwd,Variable_Transformation.save_directory,filename),index=False)
               

        return data_encoded


class Binning_Perc:
    """
    Basic functionality for Percentile Binning of continuous variables
    
        Sample Codes:
    
        import Feature_Transformation as ft
        bin = ft.Binning_Perc()
        
        # selected_num_cols: The following functionality is for only numerical variables. Please put the names of the continuous variables in place of 'selected_num_cols'
        bin.fit(train[selected_num_cols])
        bin.transform(test[selected_num_cols])

    """

    def __init__(self, bin_missing_values_fill= float("-inf"), no_of_bins=10, out_file="Binning_Output",save=False):
        """ 
        Description of parameters
        -------------------------
        bin_missing_values_fill: Numeric value to fill the missing values for all variables, Default value -Inf

        no_of_bins: Number of bins to bin the variable

        out_file: Name of the out_put file
        """
        self.save = save
        self.bins_df = None
        self.bin_missing_values_fill = bin_missing_values_fill # Numeric value to fill the missing values
#        self.handle_duplicates = handle_duplicates  # arguement from pd.qcut for 'duplicates'
        self.no_of_bins = no_of_bins
        self.out_file = out_file
        
    def bins_list(self, col):
        """
        Input
        -----
        Takes the column/variable name as input
        
        Output
        ------
        Returns the list of bins for the input variable/column
        """
        
        #impute na's with -9999999999999
        self.data.loc[self.data[col].isna(), col] = self.bin_missing_values_fill
        labels,bins = pd.qcut(self.data[col],q=self.no_of_bins,retbins=True,duplicates="drop",labels=False)
        labels = sorted(list(set(labels)))
        labels.insert(0,float("-inf"))
        bins = list(bins)
        bins = [round(i) for i in bins]
        if (self.bin_missing_values_fill in bins) & (float("-inf") not in bins):
            bins.insert(0,self.bin_missing_values_fill-1)
        elif float("-inf") not in bins:
            bins.insert(0,self.bin_missing_values_fill)
#            bins.insert(0,self.bin_missing_values_fill-1)
        return labels, bins

    def fit(self, data, cols=None, ID=None):
        """
        Input
        -----
        data: Data to be binned, Multiple variables can be passed
        
        cols: List of Column/Varibale names to be binned. If not provided, considers all columns in the dataframe

        Output
        ------
        Returns class object
        """
        if ID is None:
            data.reset_index(drop=True, inplace=True)
        if ID is not None:
            data.set_index(ID, inplace=True)
        
        self.data = data.copy()
        try:
            self.data = self.data.astype(float)
        except Exception as e:
            print(e)
            
        if cols==None:
            self.cols = self.data.columns
        else:
            self.cols = cols

        self.bins_df = pd.DataFrame(columns = ['column_name', 'bin_values'])
        self.bins_df['column_name'] = self.cols
        self.bins_df['bin_values'] = self.bins_df['column_name'].apply(lambda x:self.bins_list(x)[1])
        self.bins_df['labels'] = self.bins_df['column_name'].apply(lambda x:self.bins_list(x)[0])
        self.bins_df.set_index('column_name', inplace=True)
        return self
        
    def fit_transform(self, data, cols=None, ID=None):
        """
        Input
        ------
        data: Data to be binned, Multiple variables can be passed
        
        cols: List of Column/Varibale names to be binned. If not provided, considers all columns in the dataframe

        Output
        ------
        Returns binned data in a Dataframe output and saves the same in an excel
        """
        if ID is None:
            data.reset_index(drop=True, inplace=True)
        if ID is not None:
            data.set_index(ID, inplace=True)
            
        self.fit(data, cols=cols)
        bins_df = self.bins_df
#        print(bins_df)
#        print(bins_df)
        self.data = self.data.apply(lambda x: pd.cut(x, bins=bins_df.loc[x.name,'bin_values'], include_lowest=True, labels=False, duplicates="drop"))
        self.data.columns = [col+"_bin" for col in self.data.columns]
        self.data = data.merge(self.data, left_index=True, right_index=True)
        
        if self.save==True:
            filename = self.out_file
            filename = ef.create_version(base_filename=filename, path = os.path.join(cwd,Variable_Transformation.save_directory))
            filename += ".xlsx"
            
            
            writer = pd.ExcelWriter(os.path.join(cwd,Variable_Transformation.save_directory,filename), engine = 'xlsxwriter')
            workbook  = writer.book
        
            # FORMATS--------------------------------------------------------------------------
            format_table_header = workbook.add_format(Variable_Transformation.table_header)
            format_cell_odd = workbook.add_format(Variable_Transformation.cell_odd)
            
            print("Saving the out put file in: \n",os.path.join(cwd,Variable_Transformation.save_directory))
            self.data.to_excel(writer, sheet_name='Binned Data',index=False)
            bins_df.reset_index().to_excel(writer, sheet_name='Bins List',index=False,startrow=4,startcol=1) 
            worksheet = writer.sheets['Bins List']
            worksheet.hide_gridlines(2)
            # applying formatting
            
            # table header
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 5,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = 5,column_number = len(bins_df.columns)+2,fix_row=True),
                                         {'type': 'no_blanks','format': format_table_header})
            
            # logo
            worksheet.insert_image(ef.generate_excel_cell_name(row_number=2,column_number=2), os.path.join(os.getcwd(),'src','ks_logo.png'))
            
            # table cells
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 6,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = len(bins_df)+6,column_number = len(bins_df.columns)+2,fix_row=True),
                                             {'type': 'no_blanks','format': format_cell_odd})
    
            max_column_width = max([len(x) + 2 for x in bins_df.index])
            worksheet.set_column(ef.generate_excel_cell_name(row_number = None,column_number = 2)+':'+ef.generate_excel_cell_name(row_number = None,column_number = 2),
                                 max_column_width)
            
            max_column_width = max([len(str(x)) + 2 for x in bins_df.bin_values])
            worksheet.set_column(ef.generate_excel_cell_name(row_number = None,column_number = 3)+':'+ef.generate_excel_cell_name(row_number = None,column_number = 3),
                                 max_column_width)
            
            max_column_width = max([len(str(x)) + 2 for x in bins_df.labels])
            worksheet.set_column(ef.generate_excel_cell_name(row_number = None,column_number = 4)+':'+ef.generate_excel_cell_name(row_number = None,column_number = 4),
                                 max_column_width)
        
        
            writer.save()
            writer.close()
            
        return self.data
 
    def transform(self, df, cols=None, ID=None, manual_bins_df=None):
        """
        Input
        -----
        df: Data to be binned, Multiple variables can be passed
        
        cols: List of Column/Varibale names to be binned. If not provided, considers all columns in the dataframe
        
        manual_bins_df: Dataframe with two columns ['column_name', 'bin_values', "labels"]. column_name has all the variables names and bin_values contain the list of bin values for the corresponding variables

        Output
        ------
        Returns binned data in a Dataframe output and saves the same in an excel

        """
        if ID is None:
            df.reset_index(drop=True, inplace=True)
        if ID is not None:
            df.set_index(ID, inplace=True)
            
        if cols is not None:
            df = df[cols]
        old_df = df.copy()
        df.fillna(self.bin_missing_values_fill,inplace=True)
        if manual_bins_df is not None:
            bins_df = manual_bins_df

            missing_cols = []
            for col in df.columns:
                if col not in bins_df['column_name'].to_list():
                    missing_cols.append(col)
            if len(missing_cols)>0:
                print(missing_cols)
                raise Exception("The list of bins for the above variables are missing in the manual bins data frame")    
    #        print(bins_df)
            df = df.apply(lambda x: pd.cut(x, bins=bins_df.loc[x.name,'bin_values'], include_lowest=True, labels=False, duplicates="drop"))
            
        else:
            if self.bins_df is None:
                raise Exception("Fit the train data first")
            else:
                bins_df = self.bins_df
                bins_df.reset_index(inplace=True)
        #        print(bins_df)
                missing_cols = []
                for col in df.columns:
                    if col not in bins_df['column_name'].to_list():
                        missing_cols.append(col)
                if len(missing_cols)>0:
                    print(missing_cols)
                    raise Exception("The above variables are missing in the train data")

                bins_df.set_index("column_name",inplace=True)
                df = df.apply(lambda x: pd.cut(x, bins=bins_df.loc[x.name,'bin_values'], include_lowest=True, labels=False, duplicates="drop"))
                # Fill the values that are greater than the max bin value with the max bin value
                df = df.fillna(df.max())        
        
        df.columns = [col+"_bin" for col in df.columns]
        df = old_df.merge(df, left_index=True, right_index=True)
#        df = df.round(2)
        if self.save==True:
            filename = self.out_file
            filename = ef.create_version(base_filename=filename, path = os.path.join(cwd,Variable_Transformation.save_directory))
            filename += ".xlsx"
            
            
            writer = pd.ExcelWriter(os.path.join(cwd,Variable_Transformation.save_directory,filename), engine = 'xlsxwriter')
            workbook  = writer.book
        
            # FORMATS--------------------------------------------------------------------------
            format_table_header = workbook.add_format(Variable_Transformation.table_header)
            format_cell_odd = workbook.add_format(Variable_Transformation.cell_odd)
            
            print("Saving the out put file in: \n",os.path.join(cwd,Variable_Transformation.save_directory))
            df.to_excel(writer, sheet_name='Binned Data',index=False)
            bins_df.reset_index().to_excel(writer, sheet_name='Bins List',index=False,startrow=4,startcol=1) 
            worksheet = writer.sheets['Bins List']
            worksheet.hide_gridlines(2)
            # applying formatting
            
            # table header
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 5,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = 5,column_number = len(bins_df.columns)+2,fix_row=True),
                                         {'type': 'no_blanks','format': format_table_header})
            
            # logo
            worksheet.insert_image(ef.generate_excel_cell_name(row_number=2,column_number=2), os.path.join(os.getcwd(),'src','ks_logo.png'))
            
            # table cells
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 6,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = len(bins_df)+6,column_number = len(bins_df.columns)+2,fix_row=True),
                                             {'type': 'no_blanks','format': format_cell_odd})
    
            max_column_width = max([len(x) + 2 for x in bins_df.index])
            worksheet.set_column(ef.generate_excel_cell_name(row_number = None,column_number = 2)+':'+ef.generate_excel_cell_name(row_number = None,column_number = 2),
                                 max_column_width)
            
            max_column_width = max([len(str(x)) + 2 for x in bins_df.bin_values])
            worksheet.set_column(ef.generate_excel_cell_name(row_number = None,column_number = 3)+':'+ef.generate_excel_cell_name(row_number = None,column_number = 3),
                                 max_column_width)
            
            max_column_width = max([len(str(x)) + 2 for x in bins_df.labels])
            worksheet.set_column(ef.generate_excel_cell_name(row_number = None,column_number = 4)+':'+ef.generate_excel_cell_name(row_number = None,column_number = 4),
                                 max_column_width)
        
        
            writer.save()
            writer.close()
            
        return df


class WoE_Binning_Automated:
    """
    Description
    -----------
    Basic functionality for WoE bucketing of continuous and discrete variables. WOE Binning is done based on the number of buckets passed. For each bucker or group the WOE and IV values calculated for each variable. If opted for automated binning (monotonic=True), the adjacent bins are regrouped or merged to exhibit monotonic behaviour wrt the target variable.

        Sample Codes:
    
        import Feature_Transformation as ft
        woe = ft.WoE_Binning_Automated(v_type='c')
        
        # selected_num_cols: The following functionality is for only numerical variables as v_type is passes as "c" in the above class. Please put the names of the continuous variables in place of 'selected_num_cols'
        binned_df, mon_binned_df = woe.woe_bins(train_f[selected_num_cols+[target]], target=target, monotonic=True)
    """

    def __init__(self, qnt_num=16, min_block_size=16, spec_values=[float("-inf"), float("inf")], v_type='c', bins=None, t_type='b', out_file="WOE_Binning_Automated"):
        """
        Description of Parameters
        -------------------------
        qnt_num: Number of buckets (quartiles) for continuous variable split
        
        min_block_size: min number of obs in bucket (continuous variables), incl. optimization restrictions
        
        spec_values: List or Dictionary {'label': value} of special values (frequent items etc.)
        
        v_type: 'c' for continuous variable, 'd' - for discrete
        
        bins: Predefined bucket borders for continuous variable split
        
        t_type: Binary 'b' or continous 'c' target variable
        
        return: initialized class
        """
        self.__qnt_num = qnt_num  # Num of buckets/quartiles
        self._predefined_bins = None if bins is None else np.array(bins)  # user bins for continuous variables
        self.v_type = v_type  # if 'c' variable should be continuous, if 'd' - discrete
        self._min_block_size = min_block_size  # Min num of observation in bucket
        self._gb_ratio = None  # Ratio of good and bad in the sample
        self.bins = None  # WoE Buckets (bins) and related statistics
        self.df = None  # Training sample DataFrame with initial data and assigned woe
        self.qnt_num = None  # Number of quartiles used for continuous part of variable binning
        self.t_type = t_type  # Type of target variable
        if type(spec_values) == dict:  # Parsing special values to dict for cont variables
            self.spec_values = {}
            for k, v in spec_values.items():
                if v.startswith('d_'):
                    self.spec_values[k] = v
                else:
                    self.spec_values[k] = 'd_' + v
        else:
            if spec_values is None:
                self.spec_values = {}
            else:
                self.spec_values = {i: 'd_' + str(i) for i in spec_values}
        self.out_file = out_file
        
    def woe_bins(self, df, target="y", ID=None, monotonic=False, hypothesis=0):
        """
        Input
        -----
        Fit WoE transformation
        
        df: Dataframe to be trained/fitted along with target variable
        
        target: Name of target variable
        
        ID: Name of the ID variable (string)
        
        monotonic: if True, forcfully monotonise the woe bins
        
        hypothesis: 0, if the independent variable have direct relationship with Target variable or 1 if the relationship is inverse
        
        Output
        ------
        WoE class
        """
        if ID is None:
            df.reset_index(drop=True, inplace=True)
        if ID is not None:
            df.set_index(ID, inplace=True)
            
#        df.replace([float("inf"), float("-inf")], np.nan,inplace=True)
        cols = list(df.columns)
        cols.remove(target)
#        df = df[cols]
        mon_bins = [""]*len(list(df[cols].columns))
        all_bins = [""]*len(list(df[cols].columns))
        filename = self.out_file
        filename = ef.create_version(base_filename=filename, path = os.path.join(cwd,Variable_Transformation.save_directory))
        filename += ".xlsx"
            
            
            
        with pd.ExcelWriter(os.path.join(cwd,Variable_Transformation.save_directory,filename), engine = 'xlsxwriter') as writer:
            for i,col in enumerate(list(df[cols].columns)):
                if str(df[col].dtypes)=="object":
                    df[col]=df[col].astype(str)
                if str(type(self.v_type))=="<class 'str'>":
                    self.v_type = self.v_type
                    
                if str(type(self.v_type))=="<class 'dict'>":
                    self.v_type = self.v_type[col]
    
    #            print(df[col])
                self.fit(df[col], df[target])
    #            print(type(df[col].name))
                all_bins[i] = self.out_bins
#                print(self.out_bins)
                all_bins[i]["Variable"]=col
    #            print(self.out_bins[["Labels","WOE"]])
                if monotonic==True:
                    hypothesis = 0
                    if self.v_type=="c":
                        bins_fit = self.out_bins[~self.out_bins.Labels.str.contains("d_")]
                        if len(bins_fit)>2:
                            coeff = np.polyfit(bins_fit.Labels.astype(float), list(bins_fit.WOE.values), deg=1)
                            
                            if coeff[-2]>=0:
                                hypothesis = 0
                            elif coeff[-2]<0:
                                hypothesis = 1
                        else:
                            hypothesis = 0

                    if self.v_type=="d":
                        print(col, "  : Discrete variable. No Monotonic behaviour. Returning same bins")
                    
                    woe = self.force_monotonic(x=df[col].name, hypothesis=hypothesis)
                    mon_bins[i] = woe.out_bins
                    mon_bins[i]["Variable"]=col
            all_bins_df = pd.concat(all_bins, sort=False)
            print("Saving the out put file in: \n",os.path.join(cwd,Variable_Transformation.save_directory))
            round_cols = ['Min', 'Max', 'Count', 'Bads', 'Goods','Population_Perc', 'Bads_Perc', 'Goods_Perc', 'WOE', 'IV', 'Bad_Rate','bins']
#            all_bins_df[round_cols] = all_bins_df[round_cols].round(2)
            all_bins_df.drop("bins",axis=1,inplace=True)
#            all_bins_df.to_excel(writer, sheet_name='WOE_Binning',index=False)
            
            workbook  = writer.book
            # FORMATS--------------------------------------------------------------------------
            
            
            format_highlight1 = workbook.add_format(Variable_Transformation.cell_highlight1) 
            format_border2 = workbook.add_format(Variable_Transformation.border2) 
            #format_align_centre = workbook.add_format(Variable_Transformation.align_centre)
            format_table_header = workbook.add_format(Variable_Transformation.table_header)       
            format_cell_odd = workbook.add_format(Variable_Transformation.cell_odd)
            format_cell_even = workbook.add_format(Variable_Transformation.cell_even)
            format_percsignmultipled = workbook.add_format(Variable_Transformation.percsignmultipled)
            
            # SHEET: WOE_Binning--------------------------------------------------------------------------
            all_bins_df.rename(columns={'Population_Perc':'Population %',
                                'Bads_Perc':'Bads %',
                                'Goods_Perc':'Goods %',
                                'Bad_Rate':'Bad Rate',},inplace=True)
            all_bins_df.to_excel(writer, sheet_name='WOE_Binning',index=False,startrow=4,startcol=1) 
        
            worksheet = writer.sheets['WOE_Binning']
            worksheet.hide_gridlines(2)
            
            # applying formatting
                
            # table header       
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 5,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = 5,column_number = len(all_bins_df.columns)+1,fix_row=True),
                                         {'type': 'no_blanks','format': format_table_header})
            
            # logo
            worksheet.insert_image(ef.generate_excel_cell_name(row_number=2,column_number=2), os.path.join(os.getcwd(),'src','ks_logo.png'))
        
            # table cells  
            all_bins_df.reset_index(drop=True,inplace=True)        
    
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 6,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = len(all_bins_df)+6,column_number = len(all_bins_df.columns)+1,fix_row=True),
                                     {'type': 'formula',
                                      'criteria': "=" + ef.generate_excel_cell_name(row_number = 7,column_number = 2,fix_column=True)+'<>'+ef.generate_excel_cell_name(row_number = 6,column_number = 2,fix_column=True),
                                      'format': format_border2})
            
            rows = list(range(6,len(all_bins_df)+6))          
            
            for row in rows:
                worksheet.conditional_format(ef.generate_excel_cell_name(row_number = row,column_number = 3,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = row,column_number = len(all_bins_df.columns)+1,fix_row=True),
                                             {'type': 'no_blanks','format':  (format_cell_even if row%2==0 else format_cell_odd)})
            
            max_column_width = max([len(x) + 2 for x in all_bins_df['Variable']])
            worksheet.set_column(ef.generate_excel_cell_name(row_number = None,column_number = 2)+':'+ef.generate_excel_cell_name(row_number = None,column_number = 2),
                                     max_column_width)
            
            worksheet.set_column(ef.generate_excel_cell_name(row_number = None,column_number = 2)+':'+ef.generate_excel_cell_name(row_number = None,column_number = 2),
                                     max_column_width)
            
            
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 6,column_number = 2)+':'+ef.generate_excel_cell_name(row_number = len(all_bins_df)+6,column_number = 2),
                                         {'type': 'no_blanks','format': format_highlight1})
            
            worksheet.set_column(ef.generate_excel_cell_name(row_number = 6,column_number = 3,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = len(all_bins_df)+6,column_number = 3,fix_row=True),
                                         15,format_percsignmultipled)
            
            worksheet.set_column(ef.generate_excel_cell_name(row_number = 6,column_number = 9,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = len(all_bins_df)+6,column_number = 11,fix_row=True),
                                        11,format_percsignmultipled)
                
            worksheet.set_column(ef.generate_excel_cell_name(row_number = 6,column_number = len(all_bins_df.columns)+1,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = len(all_bins_df)+6,column_number = len(all_bins_df.columns)+1,fix_row=True),
                                         12,format_percsignmultipled)
                        
                
                
            mon_bins_df = pd.DataFrame()
            if monotonic==True:
                mon_bins_df = pd.concat(mon_bins, sort=False)
#                print("Saving the out put file in: \n",cwd)
                
#                mon_bins_df[round_cols] = mon_bins_df[round_cols].round(2)
                mon_bins_df.drop("bins",axis=1,inplace=True)
                
                # SHEET: WOE_Binning--------------------------------------------------------------------------
                mon_bins_df.rename(columns={'Population_Perc':'Population %',
                                    'Bads_Perc':'Bads %',
                                    'Goods_Perc':'Goods %',
                                    'Bad_Rate':'Bad Rate',},inplace=True)
    
                mon_bins_df.to_excel(writer, sheet_name='WOE_Binning_Automated',index=False,startrow=4,startcol=1) 
            
                worksheet = writer.sheets['WOE_Binning_Automated']
                worksheet.hide_gridlines(2)
                
                # applying formatting
                    
                # table header       
                worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 5,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = 5,column_number = len(mon_bins_df.columns)+1,fix_row=True),
                                             {'type': 'no_blanks','format': format_table_header})
                
                # logo
                worksheet.insert_image(ef.generate_excel_cell_name(row_number=2,column_number=2), os.path.join(os.getcwd(),'src','ks_logo.png'))
            
                # table cells  
                mon_bins_df.reset_index(drop=True,inplace=True)        
        
                worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 6,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = len(mon_bins_df)+6,column_number = len(mon_bins_df.columns)+1,fix_row=True),
                                         {'type': 'formula',
                                          'criteria': "=" + ef.generate_excel_cell_name(row_number = 7,column_number = 2,fix_column=True)+'<>'+ef.generate_excel_cell_name(row_number = 6,column_number = 2,fix_column=True),
                                          'format': format_border2})
                
                rows = list(range(6,len(mon_bins_df)+6))          
                
                for row in rows:
                    worksheet.conditional_format(ef.generate_excel_cell_name(row_number = row,column_number = 3,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = row,column_number = len(mon_bins_df.columns)+1,fix_row=True),
                                             {'type': 'no_blanks','format':  (format_cell_even if row%2==0 else format_cell_odd)})

                
                max_column_width = max([len(x) + 2 for x in mon_bins_df['Variable']])
                worksheet.set_column(ef.generate_excel_cell_name(row_number = None,column_number = 2)+':'+ef.generate_excel_cell_name(row_number = None,column_number = 2),
                                         max_column_width)
                
                
                worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 6,column_number = 2)+':'+ef.generate_excel_cell_name(row_number = len(mon_bins_df)+6,column_number = 2),
                                         {'type': 'no_blanks','format': format_highlight1})
            
                worksheet.set_column(ef.generate_excel_cell_name(row_number = 6,column_number = 3,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = len(mon_bins_df)+6,column_number = 3,fix_row=True),
                                             15,format_percsignmultipled)
                
                worksheet.set_column(ef.generate_excel_cell_name(row_number = 6,column_number = 9,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = len(mon_bins_df)+6,column_number = 11,fix_row=True),
                                            11,format_percsignmultipled)
                    
                worksheet.set_column(ef.generate_excel_cell_name(row_number = 6,column_number = len(all_bins_df.columns)+1,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = len(mon_bins_df)+6,column_number = len(all_bins_df.columns)+1,fix_row=True),
                                             12,format_percsignmultipled)
                        
                
                writer.save()
                writer.close()

                

        writer.save()
        writer.close()
        return all_bins_df, mon_bins_df

    def fit(self, x, y):
        """
        Fit WoE transformation
        Input
        -----
        x: continuous or discrete predictor
        
        y: binary target variable
        
        Output
        ------
        WoE class
        """
        # Data quality checks
        if not isinstance(x, pd.Series):
            x = pd.Series(x)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        if not x.size == y.size:
            raise Exception("X size don't match Y size")
        # Calc total good bad ratio in the sample
        t_bad = np.sum(y)
        if t_bad == 0 or t_bad == y.size:
            raise ValueError("There should be BAD and GOOD observations in the sample")
        if np.max(y) > 1 or np.min(y) < 0:
            raise ValueError("Y range should be between 0 and 1")
        # setting discrete values as special values
        if self.v_type == 'd':
            sp_values = {i: 'd_' + str(i) for i in x.unique()}
            if len(sp_values) > 200:
                raise type("DiscreteVarOverFlowError", (Exception,),
                           {"args": ('Discrete variable with too many unique values (more than 100)',)})
            else:
                if self.spec_values:
                    sp_values.update(self.spec_values)
                self.spec_values = sp_values
        # Make data frame for calculations
        df = pd.DataFrame({"X": x, "Y": y, 'order': np.arange(x.size)})
        # Separating NaN and Special values
        df_sp_values, df_cont = self._split_sample(df)
        # # labeling data
        df_cont, c_bins = self._cont_labels(df_cont)
        df_sp_values, d_bins = self._disc_labels(df_sp_values)
        # getting continuous and discrete values together
        self.df = df_sp_values.append(df_cont)
        self.bins = d_bins.append(c_bins)
        # calculating woe and other statistics
        self._calc_stat()
        # sorting appropriately for further cutting in transform method
#        print(self.bins)
        self.bins.sort_values('bins', inplace=True)
        self.bins["Variable"] = x.name
#        print(self.bins)
        # returning to original observation order
        self.df.sort_values('order', inplace=True)
        self.df.set_index(x.index, inplace=True)
#        self.df.columns = [x.name+"_"+col for col in self.df.columns if (col=="X") or (col=="Y")]
        out_bins = self.bins.rename(columns={"labels":"Labels","bad":"Bads","good":"Goods","obs":"Count","woe":"WOE","iv":"IV","High":"Max","Low":"Min"})
        out_bins["Bad_Rate"] = out_bins["Bads"]/out_bins["Count"]
        out_bins["Population_Perc"] = out_bins["Count"]/out_bins["Count"].sum()
        out_bins["Bads_Perc"] = out_bins["Bads"]/out_bins["Bads"].sum()
        out_bins["Goods_Perc"] = out_bins["Goods"]/out_bins["Goods"].sum()
        self.out_bins = out_bins[["Variable", 'Labels', "Min", "Max", "Count", "Bads", "Goods", "Population_Perc", "Bads_Perc", "Goods_Perc", "WOE", "IV", "Bad_Rate","bins"]]

#        with pd.ExcelWriter(os.path.join(cwd, self.out_file+".xlsx"), engine = 'openpyxl') as writer:
#            print("Saving the out put file in: \n",cwd)
#            out_bins.to_excel(writer, sheet_name='WOE_Binning',index=False)        
        
        return self

    def _split_sample(self, df):
        """
        Helper Function
        """
        if self.v_type == 'd':
            return df, None
#        sp_values_flag = df['X'].isin(self.spec_values.keys()).values | df['X'].isnull().values | np.isfinite(df['X']).values
        sp_values_flag = df['X'].isin(self.spec_values.keys()).values | df['X'].isnull().values
        df_sp_values = df[sp_values_flag].copy()
        df_cont = df[np.logical_not(sp_values_flag)].copy()
        return df_sp_values, df_cont

    def _disc_labels(self, df):
        """
        Helper Function
        """
        df['labels'] = df['X'].apply(
            lambda x: self.spec_values[x] if x in self.spec_values.keys() else 'd_' + str(x))
        d_bins = pd.DataFrame({"bins": df['X'].unique()})
        d_bins['labels'] = d_bins['bins'].apply(
            lambda x: self.spec_values[x] if x in self.spec_values.keys() else 'd_' + str(x))
        d_bins['High'] = d_bins['labels']
        d_bins['Low'] = d_bins['labels']
        return df, d_bins

    def _cont_labels(self, df):
        """
        Helper Function
        """
        # check whether there is a continuous part
        if df is None:
            return None, None
        # Max buckets num calc
        self.qnt_num = int(np.minimum(df['X'].unique().size / self._min_block_size, self.__qnt_num)) + 1
        # cuts - label num for each observation, bins - quartile thresholds
        bins = None
        cuts = None
        if self._predefined_bins is None:
            try:
                cuts, bins = pd.qcut(df["X"], self.qnt_num, retbins=True, labels=False, duplicates='drop')
            except ValueError as ex:
                if ex.args[0].startswith('Bin edges must be unique'):
                    ex.args = ('Please reduce number of bins or encode frequent items as special values',) + ex.args
                    raise
#            bins = np.append((-float("inf"),), bins[1:-1])
            bins = np.append((-888888,), bins[1:-1])
        else:
            bins = self._predefined_bins
#            if bins[0] != float("-Inf"):
            if bins[0] != -888888:
#                bins = np.append((-float("inf"),), bins)
                bins = np.append((-888888,), bins)
#            cuts = pd.cut(df['X'], bins=np.append(bins, (float("inf"),)),labels=np.arange(len(bins)).astype(str))
            cuts = pd.cut(df['X'], bins=np.append(bins, (float("inf"),)),labels=np.arange(len(bins)).astype(str))

        df["labels"] = cuts.astype(str)
        c_bins = pd.DataFrame({"bins": bins, "labels": np.arange(len(bins)).astype(str)})
        c_bins["Low"] = c_bins["bins"]
        c_bins["High"] = c_bins["bins"].shift(-1, fill_value=df["X"].max())
        return df, c_bins

    def _calc_stat(self):
        """
        Helper Function
        """
        # calculating WoE
        col_names = {'count_nonzero': 'bad', 'size': 'obs'}
        stat = self.df.groupby("labels")['Y'].agg([np.mean, np.count_nonzero, np.size]).rename(columns=col_names).copy()
        if self.t_type != 'b':
            stat['bad'] = stat['mean'] * stat['obs']
        stat['good'] = stat['obs'] - stat['bad']
#        t_good = np.maximum(stat['good'].sum(), 0.5)
#        t_bad = np.maximum(stat['bad'].sum(), 0.5)
        t_good = stat['good'].sum()
        t_bad = stat['bad'].sum()
#        stat['woe'] = stat.apply(self._bucket_woe, axis=1) + np.log(t_bad / t_good)
#        stat['iv'] = (stat['good'] / t_good - stat['bad'] / t_bad) * stat['woe']
        stat['woe'] = np.log((stat["bad"]/t_bad) / (stat["good"]/t_good))
        stat['iv'] = ((stat['bad']/t_bad) - (stat['good'] / t_good)) * stat['woe']
        self.iv = stat['iv'].sum()
        # adding stat data to bins
        self.bins = pd.merge(stat, self.bins, left_index=True, right_on=['labels'])
        label_woe = self.bins[['woe', 'labels']].drop_duplicates()
        self.df = pd.merge(self.df, label_woe, left_on=['labels'], right_on=['labels'])

    def __get_cont_bins(self):
        """
        Helper function
        Output
        ------
        Return continous part of self.bins
        """
        return self.bins[self.bins['labels'].apply(lambda z: not z.startswith('d_'))]

    @staticmethod
    def _bucket_woe(x):
        """
        Helper Function
        """
        t_bad = x['bad']
        t_good = x['good']
        t_bad = 0.5 if t_bad == 0 else t_bad
        t_good = 0.5 if t_good == 0 else t_good
        return np.log(t_good / t_bad)
    
    def merge(self, x, label1, label2=None):
        """
        Merge of buckets with given labels.
        In case of discrete variable, both labels should be provided. As the result labels will be marget to one bucket.
        In case of continous variable, only label1 should be provided. It will be merged with the next label.

        Input
        -----
        x: The independent variable column
        
        label1: first label to merge
        
        label2: second label to merge
        
        Output
        ------
        Returns new WOE class oblect after merging the two labels
        """
        spec_values = self.spec_values.copy()
        c_bins = self.__get_cont_bins().copy()
        if label2 is None and not label1.startswith('d_'):  # removing bucket for continuous variable
            c_bins = c_bins[c_bins['labels'] != label1]
        else:
            if not (label1.startswith('d_') and label2.startswith('d_')):
                raise Exception('Labels should be discrete simultaneously')
            for i in self.bins[self.bins['labels'] == label1]['bins']:
                spec_values[i] = label1 + '_' + label2
            bin2 = self.bins[self.bins['labels'] == label2]['bins'].iloc[0]
            spec_values[bin2] = label1 + '_' + label2
        new_woe = WoE_Binning_Automated(self.__qnt_num, self._min_block_size, spec_values, self.v_type, c_bins['bins'], self.t_type)

        return new_woe.fit(self.df['X'], self.df['Y'])

    def force_monotonic(self, x, hypothesis=0):
        """
        Makes transformation monotonic if possible, given relationship hypothesis (otherwise - MonotonicConstraintError
        exception)
        
        Input
        -----
        x: The independent variable column
        
        hypothesis: direct (0) or inverse (1) hypothesis relationship between predictor and target variable
        
        Output
        ------
        New WoE object with monotonic transformation
        """
        if hypothesis == 0:
            op_func = operator.gt
        else:
            op_func = operator.lt
        cont_bins = self.__get_cont_bins()
        new_woe = self
        for i, w in enumerate(cont_bins[1:]['woe']):
            if op_func(cont_bins.iloc[i].loc['woe'], w):
                if cont_bins.shape[0] < 3:
#                    print(x,"  : Cannot make monotonic as there are only two bins")
                    return new_woe
                
                else:
                    new_woe = self.merge(x, cont_bins.iloc[i+1].loc['labels'])
                    new_woe = new_woe.force_monotonic(x, hypothesis)
                    return new_woe
        return new_woe

    
class WoE_Binning_Optimization:
    """
    Basic functionality for WoE bucketing of continuous and discrete variables. The WOE buckets are optmized i.e., chosen using the DecisionTree splits between independent variable and target variable.

        Sample Codes:
    
        import Feature_Transformation as ft        
        woe = ft.WoE_Binning_Optimization(v_type='c')
        
        # selected_num_cols: The following functionality is for only numerical variables as v_type is passes as "c" in the above class. Please put the names of the continuous variables in place of 'selected_num_cols'
        woe.woe_bins_optimized(train_f[selected_num_cols+[target]], target=target)
    """

    def __init__(self, qnt_num=16, min_block_size=16, spec_values=[float("-inf"), float("inf")], v_type='c', bins=None, t_type='b', out_file="WOE_Binning_Optimized"):
        """
        Description of Parameters
        -------------------------
        qnt_num: Number of buckets (quartiles) for continuous variable split
        
        min_block_size: min number of obs in bucket (continuous variables), incl. optimization restrictions
        
        spec_values: List or Dictionary {'label': value} of special values (frequent items etc.)
        
        v_type: 'c' for continuous variable, 'd' - for discrete
        
        bins: Predefined bucket borders for continuous variable split
        
        t_type: Binary 'b' or continous 'c' target variable
        """
        self.__qnt_num = qnt_num  # Num of buckets/quartiles
        self._predefined_bins = None if bins is None else np.array(bins)  # user bins for continuous variables
        self.v_type = v_type  # if 'c' variable should be continuous, if 'd' - discrete
        self._min_block_size = min_block_size  # Min num of observation in bucket
        self._gb_ratio = None  # Ratio of good and bad in the sample
        self.bins = None  # WoE Buckets (bins) and related statistics
        self.df = None  # Training sample DataFrame with initial data and assigned woe
        self.qnt_num = None  # Number of quartiles used for continuous part of variable binning
        self.t_type = t_type  # Type of target variable
        if type(spec_values) == dict:  # Parsing special values to dict for cont variables
            self.spec_values = {}
            for k, v in spec_values.items():
                if v.startswith('d_'):
                    self.spec_values[k] = v
                else:
                    self.spec_values[k] = 'd_' + v
        else:
            if spec_values is None:
                self.spec_values = {}
            else:
                self.spec_values = {i: 'd_' + str(i) for i in spec_values}
        self.out_file = out_file
        
    def woe_bins_optimized(self, df, target="y", ID=None, monotonic=False, hypothesis=0, criterion=None, fix_depth=2, max_depth=None, cv=3, scoring=None, min_samples_leaf=None):
        """
        Fit WoE transformation
        
        Input
        -----
        df: Dataframe to be trained/fitted along with target variable
        
        y: target variable
        
        criterion: binary tree split criteria
        
        fix_depth: use tree of a fixed depth (2^fix_depth buckets)
        
        max_depth: maximum tree depth for a optimum cross-validation search
        
        cv: number of cv buckets
        
        scoring: scorer for cross_val_score
        
        min_samples_leaf: minimum number of observations in each of optimized buckets
        
        Output
        ------
        WoE class
        """
        if ID is None:
            df.reset_index(drop=True, inplace=True)
        if ID is not None:
            df.set_index(ID, inplace=True)
#        df.replace([float("inf"), float("-inf")], np.nan,inplace=True)
        cols = list(df.columns)
        cols.remove(target)
#        df = df[cols]
        all_bins = [""]*len(list(df[cols].columns))
        mon_bins = [""]*len(list(df[cols].columns))
        filename = self.out_file
        filename = ef.create_version(base_filename=filename, path = os.path.join(cwd,Variable_Transformation.save_directory))
        filename += ".xlsx"
        
        with pd.ExcelWriter(os.path.join(cwd,Variable_Transformation.save_directory,filename), engine = 'xlsxwriter') as writer:
            for i,col in enumerate(list(df[cols].columns)):
                if str(df[col].dtypes)=="object":
                    df[col]=df[col].astype(str)
                    
                if str(type(self.v_type))=="<class 'str'>":
                    self.v_type = self.v_type
                    
                if str(type(self.v_type))=="<class 'dict'>":
                    self.v_type = self.v_type[col]
    
    #            print(df[col])
                woe = WoE_Binning_Automated(self.__qnt_num, self._min_block_size, self.spec_values, self.v_type, None, self.t_type)
                woe.fit(df[col], df[target])
    #            print("print---------------------", woe.df)
    
                self.df = woe.df
                self.optimize(x=df[col], criterion=criterion, fix_depth=fix_depth, max_depth=max_depth, cv=cv, scoring=scoring, min_samples_leaf=min_samples_leaf)
                all_bins[i] = self.new_woe.out_bins
                all_bins[i]["Variable"]=col
    
                if monotonic==True:
    #                print("Monotonic----------")
                    if self.v_type=="d":
                        print(col,"  : Discrete Variable. No Monotonic. Returning same bins")
                    hypothesis = 0
                    if self.v_type=="c":
                        bins_fit = self.new_woe.out_bins[~self.new_woe.out_bins.Labels.str.contains("d_")]
                        if len(bins_fit)>2:
                            coeff = np.polyfit(bins_fit.Labels.astype(float), list(bins_fit.WOE.values), deg=1)
                            
                            if coeff[-2]>=0:
                                hypothesis = 0
                            elif coeff[-2]<0:
                                hypothesis = 1
                        else:
                            hypothesis = 0
                    
                    mon_woe = self.new_woe.force_monotonic(x=df[col].name, hypothesis=hypothesis)
                    mon_bins[i] = mon_woe.out_bins
                    mon_bins[i]["Variable"]=col
    
            all_bins_df = pd.concat(all_bins, sort=False)
            all_bins_df["Variable"] = all_bins_df["Variable"].str.replace("_X","")
            round_cols = ['Min', 'Max', 'Count', 'Bads', 'Goods','Population_Perc', 'Bads_Perc', 'Goods_Perc', 'WOE', 'IV', 'Bad_Rate','bins']
#            all_bins_df[round_cols] = all_bins_df[round_cols].round(2)
            all_bins_df.drop("bins",axis=1,inplace=True)
            print("Saving the out put file in: \n",os.path.join(cwd,Variable_Transformation.save_directory))
#            all_bins_df.to_excel(writer, sheet_name='WOE_Binning_Optimized',index=False)
            
            workbook  = writer.book
            # FORMATS--------------------------------------------------------------------------
            
            
            format_highlight1 = workbook.add_format(Variable_Transformation.cell_highlight1) 
            format_border2 = workbook.add_format(Variable_Transformation.border2) 
            #format_align_centre = workbook.add_format(Variable_Transformation.align_centre)
            format_table_header = workbook.add_format(Variable_Transformation.table_header)       
            format_cell_odd = workbook.add_format(Variable_Transformation.cell_odd)
            format_cell_even = workbook.add_format(Variable_Transformation.cell_even)
            format_percsignmultipled = workbook.add_format(Variable_Transformation.percsignmultipled)
            
            # SHEET: WOE_Binning--------------------------------------------------------------------------
            all_bins_df.rename(columns={'Population_Perc':'Population %',
                                'Bads_Perc':'Bads %',
                                'Goods_Perc':'Goods %',
                                'Bad_Rate':'Bad Rate',},inplace=True)
            all_bins_df.to_excel(writer, sheet_name='WOE_Binning_Optimized',index=False,startrow=4,startcol=1) 
        
            worksheet = writer.sheets['WOE_Binning_Optimized']
            worksheet.hide_gridlines(2)
            
            # applying formatting
                
            # table header       
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 5,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = 5,column_number = len(all_bins_df.columns)+1,fix_row=True),
                                         {'type': 'no_blanks','format': format_table_header})
            
            # logo
            worksheet.insert_image(ef.generate_excel_cell_name(row_number=2,column_number=2), os.path.join(os.getcwd(),'src','ks_logo.png'))
        
            # table cells  
            all_bins_df.reset_index(drop=True,inplace=True)        
    
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 6,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = len(all_bins_df)+6,column_number = len(all_bins_df.columns)+1,fix_row=True),
                                     {'type': 'formula',
                                      'criteria': "=" + ef.generate_excel_cell_name(row_number = 7,column_number = 2,fix_column=True)+'<>'+ef.generate_excel_cell_name(row_number = 6,column_number = 2,fix_column=True),
                                      'format': format_border2})
            
            rows = list(range(6,len(all_bins_df)+6))          
            
            for row in rows:
                worksheet.conditional_format(ef.generate_excel_cell_name(row_number = row,column_number = 3,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = row,column_number = len(all_bins_df.columns)+1,fix_row=True),
                                             {'type': 'no_blanks','format':  (format_cell_even if row%2==0 else format_cell_odd)})
            
            max_column_width = max([len(x) + 2 for x in all_bins_df['Variable']])
            worksheet.set_column(ef.generate_excel_cell_name(row_number = None,column_number = 2)+':'+ef.generate_excel_cell_name(row_number = None,column_number = 2),
                                     max_column_width)
            
            worksheet.set_column(ef.generate_excel_cell_name(row_number = None,column_number = 2)+':'+ef.generate_excel_cell_name(row_number = None,column_number = 2),
                                     max_column_width)
            
            
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 6,column_number = 2)+':'+ef.generate_excel_cell_name(row_number = len(all_bins_df)+6,column_number = 2),
                                         {'type': 'no_blanks','format': format_highlight1})
            
            worksheet.set_column(ef.generate_excel_cell_name(row_number = 6,column_number = 3,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = len(all_bins_df)+6,column_number = 3,fix_row=True),
                                         15,format_percsignmultipled)
            
            worksheet.set_column(ef.generate_excel_cell_name(row_number = 6,column_number = 9,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = len(all_bins_df)+6,column_number = 11,fix_row=True),
                                        11,format_percsignmultipled)
                
            worksheet.set_column(ef.generate_excel_cell_name(row_number = 6,column_number = len(all_bins_df.columns)+1,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = len(all_bins_df)+6,column_number = len(all_bins_df.columns)+1,fix_row=True),
                                         12,format_percsignmultipled)
                        
                
            
            mon_bins_df = pd.DataFrame()
            if monotonic==True:
                mon_bins_df = pd.concat(mon_bins, sort=False)
                mon_bins_df["Variable"] = mon_bins_df["Variable"].str.replace("_X","")
#                mon_bins_df[round_cols] = mon_bins_df[round_cols].round(2)
                mon_bins_df.drop("bins",axis=1,inplace=True)

                #mon_bins_df.to_excel(writer, sheet_name='WOE_Binning_Optimized_Monotonic',index=False)
                # SHEET: WOE_Binning--------------------------------------------------------------------------
                mon_bins_df.rename(columns={'Population_Perc':'Population %',
                                    'Bads_Perc':'Bads %',
                                    'Goods_Perc':'Goods %',
                                    'Bad_Rate':'Bad Rate',},inplace=True)
    
                mon_bins_df.to_excel(writer, sheet_name='WOE_Binning_Optimized_Monotonic',index=False,startrow=4,startcol=1) 
            
                worksheet = writer.sheets['WOE_Binning_Optimized_Monotonic']
                worksheet.hide_gridlines(2)
                
                # applying formatting
                    
                # table header       
                worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 5,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = 5,column_number = len(mon_bins_df.columns)+1,fix_row=True),
                                             {'type': 'no_blanks','format': format_table_header})
                
                # logo
                worksheet.insert_image(ef.generate_excel_cell_name(row_number=2,column_number=2), os.path.join(os.getcwd(),'src','ks_logo.png'))
            
                # table cells  
                mon_bins_df.reset_index(drop=True,inplace=True)        
        
                worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 6,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = len(mon_bins_df)+6,column_number = len(mon_bins_df.columns)+1,fix_row=True),
                                         {'type': 'formula',
                                          'criteria': "=" + ef.generate_excel_cell_name(row_number = 7,column_number = 2,fix_column=True)+'<>'+ef.generate_excel_cell_name(row_number = 6,column_number = 2,fix_column=True),
                                          'format': format_border2})
                
                rows = list(range(6,len(mon_bins_df)+6))          
                
                for row in rows:
                    worksheet.conditional_format(ef.generate_excel_cell_name(row_number = row,column_number = 3,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = row,column_number = len(mon_bins_df.columns)+1,fix_row=True),
                                             {'type': 'no_blanks','format':  (format_cell_even if row%2==0 else format_cell_odd)})

                
                max_column_width = max([len(x) + 2 for x in mon_bins_df['Variable']])
                worksheet.set_column(ef.generate_excel_cell_name(row_number = None,column_number = 2)+':'+ef.generate_excel_cell_name(row_number = None,column_number = 2),
                                         max_column_width)
                
                
                worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 6,column_number = 2)+':'+ef.generate_excel_cell_name(row_number = len(mon_bins_df)+6,column_number = 2),
                                         {'type': 'no_blanks','format': format_highlight1})
            
                worksheet.set_column(ef.generate_excel_cell_name(row_number = 6,column_number = 3,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = len(mon_bins_df)+6,column_number = 3,fix_row=True),
                                             15,format_percsignmultipled)
                
                worksheet.set_column(ef.generate_excel_cell_name(row_number = 6,column_number = 9,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = len(mon_bins_df)+6,column_number = 11,fix_row=True),
                                            11,format_percsignmultipled)
                    
                worksheet.set_column(ef.generate_excel_cell_name(row_number = 6,column_number = len(all_bins_df.columns)+1,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = len(mon_bins_df)+6,column_number = len(all_bins_df.columns)+1,fix_row=True),
                                             12,format_percsignmultipled)
                        
                
                writer.save()
                writer.close()

            
        writer.save()
        writer.close()
        return all_bins_df, mon_bins_df

    def optimize(self, x, criterion=None, fix_depth=2, max_depth=None, cv=3, scoring=None, min_samples_leaf=None):
        """
        WoE bucketing optimization (continuous variables only)
        
        Input
        -----
        criterion: binary tree split criteria
        
        fix_depth: use tree of a fixed depth (2^fix_depth buckets)
        
        max_depth: maximum tree depth for a optimum cross-validation search
        
        cv: number of cv buckets
        
        scoring: scorer for cross_val_score
        
        min_samples_leaf: minimum number of observations in each of optimized buckets
        
        Output
        ------
        Return WoE class with optimized continuous variable split
        """
        if self.t_type == 'b':
            tree_type = tree.DecisionTreeClassifier
        else:
            tree_type = tree.DecisionTreeRegressor
        m_depth = int(np.log2(self.__qnt_num)) + 1 if max_depth is None else max_depth
        cont = self.df['labels'].apply(lambda z: not z.startswith('d_'))
        x_train = np.array(self.df[cont]['X'])
        y_train = np.array(self.df[cont]['Y'])
        x_train = x_train.reshape(x_train.shape[0], 1)
        if not min_samples_leaf:
            min_samples_leaf = self._min_block_size
        start = 1
        cv_scores = []
        if fix_depth is None:
            for i in range(start, m_depth):
                if criterion is None:
                    d_tree = tree_type(max_depth=i, min_samples_leaf=min_samples_leaf)
                else:
                    d_tree = tree_type(criterion=criterion, max_depth=i, min_samples_leaf=min_samples_leaf)
                scores = cross_val_score(d_tree, x_train, y_train, cv=cv, scoring=scoring)
                cv_scores.append(scores.mean())
            best = np.argmax(cv_scores) + start
        else:
            best = fix_depth
        final_tree = tree_type(max_depth=best, min_samples_leaf=min_samples_leaf)
        final_tree.fit(x_train, y_train)
        opt_bins = final_tree.tree_.threshold[final_tree.tree_.feature >= 0]
        opt_bins = np.sort(opt_bins)
        self.new_woe = WoE_Binning_Automated(self.__qnt_num, self._min_block_size, self.spec_values, self.v_type, opt_bins, self.t_type)
        return self.new_woe.fit(self.df['X'], self.df['Y'])


class WoE_Transformation:
    """
    Basic functionality for WoE bucketing of continuous and discrete variables. 
    
    Use this functionality to transform the variables to their WOE values using either Percentile bins or Automated bins or Optimized bins. 
    
    To use Percentile bins, pass transformation_type="Optimize" in the class initialization and monotonic=False in the 'woe_transformation' method. 
    
    To use Automated bins, pass transformation_type="Optimize" in the class initialization and monotonic=True in the 'woe_transformation' method.
    
    To use Optimized bins, pass transformation_type="Optimize".

        Sample Codes:
    
        import Feature_Transformation as ft
        woe = ft.WoE_Transformation(transformation_type='Automated',v_type="c")
        
        # selected_num_cols: The following functionality is for only numerical variables as v_type is passes as "c" in the above class. Please put the names of the continuous variables in place of 'selected_num_cols'
        woe_transform_test_num = woe.woe_transformation(train_o[selected_num_cols+[target]], test_o[selected_num_cols+[target]], monotonic=True, target=target)
    """

    def __init__(self, transformation_type = "Automated", qnt_num=16, min_block_size=16, spec_values=[float("-inf"), float("inf")], v_type='c', bins=None, t_type='b', out_file="WOE_Transformation",save=False):
        """
        Description of Parameters
        -------------------------
        qnt_num: Number of buckets (quartiles) for continuous variable split
        
        min_block_size: min number of obs in bucket (continuous variables), incl. optimization restrictions
        
        spec_values: List or Dictionary {'label': value} of special values (frequent items etc.)
        
        v_type: 'c' for continuous variable, 'd' - for discrete
        
        bins: Predefined bucket borders for continuous variable split
        
        t_type : Binary 'b' or continous 'c' target variable
        """
        self.save = save
        self.transformation_type = transformation_type
        self.__qnt_num = qnt_num  # Num of buckets/quartiles
        self._predefined_bins = None if bins is None else np.array(bins)  # user bins for continuous variables
        self.v_type = v_type  # if 'c' variable should be continuous, if 'd' - discrete
        self._min_block_size = min_block_size  # Min num of observation in bucket
        self._gb_ratio = None  # Ratio of good and bad in the sample
        self.bins = None  # WoE Buckets (bins) and related statistics
        self.df = None  # Training sample DataFrame with initial data and assigned woe
        self.qnt_num = None  # Number of quartiles used for continuous part of variable binning
        self.t_type = t_type  # Type of target variable
        if type(spec_values) == dict:  # Parsing special values to dict for cont variables
            self.spec_values = {}
            for k, v in spec_values.items():
                if v.startswith('d_'):
                    self.spec_values[k] = v
                else:
                    self.spec_values[k] = 'd_' + v
        else:
            if spec_values is None:
                self.spec_values = {}
            else:
                self.spec_values = {i: 'd_' + str(i) for i in spec_values}
        self.out_file = out_file
        
    def woe_transformation(self, train, test=None, target="y", ID=None, monotonic=False, hypothesis=0, criterion=None, fix_depth=2, max_depth=None, cv=3, scoring=None, min_samples_leaf=None, replace_missing=0):
        """
        WoE transformation based on the parameters chosen, i.e., if Automated, the monotonic bins are used to transform the data. If Optimize, the optimized bins from DecisionTree split are used
        
        Input
        -----
        df: Dataframe to be trained/fitted along with target variable
        
        y: target variable
        
        monotonic: if True, forcfully monotonise the woe bins
        
        hypothesis: 0, if the independent variable have direct relationship with Target variable or 1 if the relationship is inverse

        criterion: binary tree split criteria
        
        fix_depth: use tree of a fixed depth (2^fix_depth buckets)
        
        max_depth: maximum tree depth for a optimum cross-validation search
        
        cv: number of cv buckets
        
        scoring: scorer for cross_val_score
        
        min_samples_leaf: minimum number of observations in each of optimized buckets
        
        Output
        ------
        WoE class
        """
        if test is None:
            test=train.copy()

        if ID is None:
            train.reset_index(drop=True, inplace=True)
    #        train.replace([float("inf"), float("-inf")], np.nan,inplace=True)
            test.reset_index(drop=True, inplace=True)
    #        test.replace([float("inf"), float("-inf")], np.nan,inplace=True)

        if ID is not None:
            train.set_index(ID, inplace=True)
            test.set_index(ID, inplace=True)
            
        cols = list(train.columns)
        cols.remove(target)
#        df = df[cols]
        all_cols = [""]*len(list(train[cols].columns))
        for i,col in enumerate(list(train[cols].columns)):
            if str(train[col].dtypes)=="object":
                train[col]=train[col].astype(str)
            
            if str(test[col].dtypes)=="object":
                test[col]=test[col].astype(str)
                
            if str(type(self.v_type))=="<class 'str'>":
                self.v_type = self.v_type
                
            if str(type(self.v_type))=="<class 'dict'>":
                self.v_type = self.v_type[col]
            
            df_trans = self.transform(x_train=train[col], y_train=train[target], x_test=None, y_test=None, monotonic=monotonic, hypothesis=hypothesis, criterion=criterion, fix_depth=fix_depth, max_depth=max_depth, cv=cv, scoring=scoring, min_samples_leaf=min_samples_leaf, replace_missing=replace_missing)
            
            if test is not None:
                df_trans = self.transform(x_train=train[col], y_train=train[target], x_test=test[col], y_test=test[target], monotonic=monotonic, hypothesis=hypothesis, criterion=criterion, fix_depth=fix_depth, max_depth=max_depth, cv=cv, scoring=scoring, min_samples_leaf=min_samples_leaf, replace_missing=replace_missing)

            all_cols[i] = df_trans
            all_cols[i].reset_index(drop=True,inplace=True)
#            all_bins[i]["Variable"]=col
        all_cols_df = pd.concat(all_cols, axis=1, sort=False)
#        all_cols_df = all_cols_df.astype(float)
#        all_cols_df = all_cols_df.round(2)
        if self.save==True:
            filename = self.out_file
            filename = ef.create_version(base_filename=filename, path = os.path.join(cwd,Variable_Transformation.save_directory))
            filename += ".xlsx"
            
            with pd.ExcelWriter(os.path.join(cwd,Variable_Transformation.save_directory,filename), engine = 'xlsxwriter') as writer:
                print("Saving the out put file in: \n",os.path.join(cwd,Variable_Transformation.save_directory))
                all_cols_df.to_excel(writer, sheet_name='WOE_Transformation',index=False)
                writer.save()
                writer.close()
        
        return all_cols_df

    def transform(self, x_train, y_train, x_test, y_test, monotonic=False, hypothesis=0, criterion=None, fix_depth=2, max_depth=None, cv=3, scoring=None, min_samples_leaf=None, replace_missing=0):
        """
        Transforms input variable according to previously fitted rule
        Input:
        ------
        :param x: input variable
        :param manual_woe: one can change fitted woe with manual values by providing dict {label: new_woe_value}
        :replace_missing: replace woe for labels not observable in traning dataset by this value
        :return: DataFrame with transformed with original and transformed variables
        """
        if x_test is None:
            x_test = x_train
        if y_test is None:
            y_test = y_train
            
        if self.transformation_type == "Automated":
            woe = WoE_Binning_Automated(qnt_num=self.__qnt_num, min_block_size=self._min_block_size, spec_values=self.spec_values, v_type=self.v_type, bins=None, t_type=self.t_type, out_file="WOE_Binning_Automated_Trans")
            woe.fit(x_train, y_train)
            if monotonic==True:
                if self.v_type=="d":
                    pass
                hypothesis = 0
#                    print(x_train.name,"  : Discrete Variable. No Monotonic. Returning same bins")
                if self.v_type=="c":
                    bins_fit = woe.out_bins[~woe.out_bins.Labels.str.contains("d_")]
                    if len(bins_fit)>2:
                        coeff = np.polyfit(bins_fit.Labels.astype(float), list(bins_fit.WOE.values), deg=1)
                        
                        if coeff[-2]>=0:
                            hypothesis = 0
                        elif coeff[-2]<0:
                            hypothesis = 1
                    else:
                        hypothesis = 0

#                print("Monotonic----------")
                woe = woe.force_monotonic(x=x_train.name, hypothesis=hypothesis)            

            self.bins = woe.bins
            self.spec_values = woe.spec_values
            self.v_type = woe.v_type
            
        if self.transformation_type == "Optimize":
            woe_opt = WoE_Binning_Optimization(qnt_num=self.__qnt_num, min_block_size=self._min_block_size, spec_values=self.spec_values, v_type=self.v_type, bins=None, t_type=self.t_type, out_file="WOE_Binning_Optimized_Trans")
            woe = WoE_Binning_Automated(qnt_num=self.__qnt_num, min_block_size=self._min_block_size, spec_values=self.spec_values, v_type=self.v_type, bins=None, t_type=self.t_type, out_file="WOE_Binning_Automated_Trans")
            
            woe.fit(x_train, y_train)
#            print("print---------------------", woe.df)
            woe_opt.df = woe.df
            woe_opt.optimize(x=x_train, criterion=criterion, fix_depth=fix_depth, max_depth=max_depth, cv=cv, scoring=scoring, min_samples_leaf=min_samples_leaf)

            self.bins = woe_opt.new_woe.bins
            self.spec_values = woe_opt.spec_values
            self.v_type = woe_opt.v_type

            if monotonic==True:
#                print("Monotonic----------")
                if self.v_type=="d":
                    pass
                hypothesis = 0
#                    print(x_train.name,"  : Discrete Variable. No Monotonic. Returning same bins")
                if self.v_type=="c":
                    bins_fit = woe_opt.new_woe.out_bins[~woe_opt.new_woe.out_bins.Labels.str.contains("d_")]
                    if len(bins_fit)>2:
                        coeff = np.polyfit(bins_fit.Labels.astype(float), list(bins_fit.WOE.values), deg=1)
                        
                        if coeff[-2]>=0:
                            hypothesis = 0
                        elif coeff[-2]<0:
                            hypothesis = 1
                    else:
                        hypothesis = 0

                mon_woe = woe_opt.force_monotonic(x=x_train.name, hypothesis=hypothesis)
                self.bins = mon_woe.new_woe.bins
                self.spec_values = mon_woe.spec_values
                self.v_type = mon_woe.v_type
        
        if not isinstance(x_test, pd.Series):
            raise TypeError("pandas.Series type expected")
        if self.bins is None:
            raise Exception('Fit the model first, please')
        df = pd.DataFrame({"X": x_test, 'order': np.arange(x_test.size)})
        # splitting to discrete and continous pars
        df_sp_values, df_cont = self._split_sample(df)
        # Replacing original with manual woe
        tr_bins = self.bins[['woe', 'labels']].copy()
    
        if replace_missing is not None:
            tr_bins = tr_bins.append({'labels': 'd__transform_missing_replacement__', 'woe': replace_missing},
                                     ignore_index=True)
    
        # function checks existence of special values, raises error if sp do not exist in training set
        def get_sp_label(x_):
            if x_ in self.spec_values.keys():
                return self.spec_values[x_]
            else:
                str_x = 'd_' + str(x_)
                if str_x in list(self.bins['labels']):
                    return str_x
                else:
                    if replace_missing is not None:
                        return 'd__transform_missing_replacement__'
                    else:
                        raise ValueError('Value {} does not exist in the training set'.format(str_x))
    
        # assigning labels to discrete part
        df_sp_values['labels'] = df_sp_values['X'].apply(get_sp_label)
        # assigning labels to continuous part
        c_bins = self.bins[self.bins['labels'].apply(lambda z: not z.startswith('d_'))]
        if self.v_type != 'd':
            cuts = pd.cut(df_cont['X'], bins=np.append(c_bins["bins"], (float("inf"),)), labels=c_bins["labels"])
            df_cont['labels'] = cuts.astype(str)
        # Joining continuous and discrete parts
        df = df_sp_values.append(df_cont)
        # assigning woe
        df = pd.merge(df, tr_bins[['woe', 'labels']].drop_duplicates(), left_on=['labels'], right_on=['labels'])
        # returning to original observation order
        df.sort_values('order', inplace=True)
        df["Y"] = y_test
        df.columns = [x_test.name+"_"+col if col!="X" else x_test.name for col in df.columns]
#        df.set_index(x_test.index, inplace=True)
        df[x_test.name+"_woe"] = df[x_test.name+"_woe"].astype(float)
#        df[x_test.name+"_woe"] = df[x_test.name+"_woe"].round(2)
        return df[[x_test.name, x_test.name+"_woe"]]

    def _split_sample(self, df):
        """
        Helper Function
        """
        if self.v_type == 'd':
            return df, None
#        sp_values_flag = df['X'].isin(self.spec_values.keys()).values | df['X'].isnull().values| np.isfinite(df['X']).values
        sp_values_flag = df['X'].isin(self.spec_values.keys()).values | df['X'].isnull().values
        df_sp_values = df[sp_values_flag].copy()
        df_cont = df[np.logical_not(sp_values_flag)].copy()
        return df_sp_values, df_cont


class WoE_Transformation_Manual:
    """
    Functionality to transform the independent variables into woe by using User Manual Bins
    
        Sample Codes:
    
        import Feature_Transformation as ft
        woe_transform = ft.WoE_Transformation_Manual()
        
        # The variables in the test_data should match with the variables in the woedata
        woe_transform.woe_data(target,test_data,woedata)
    """

    def __init__(self):
        """
        Initiate the class and call the method "woe_data" to transform continuous variables and "woe_data_cat" to transform the categorical variables
        """
        pass
    
    def woe_data(self,Target, df2,woedata, ID=None):
        """
        This function creates woe data for numeric variables using the binning excel sheets provided by the User
        
            Attributes
            -------
            Target: Y variable name
            df2: Original dataset
            woedata: Binning sheet
            
            Returns
            -------       
            pandas DataFrame, containing original data along with woe varibles that are binned in the binning sheet
            
            Code
            -----
            
            import Feature_Transformation as ft
            woedata = pd.read_csv("WOE_Manual_Bins.csv")
            # The variables in the test_data should match with the variables in the woedata
            c=ft.WoE_Transformation_Manual().woe_data(target,test_data,woedata)
        
        """
        if ID is None:
            df2.reset_index(drop=True, inplace=True)
        if ID is not None:
            df2.set_index(ID, inplace=True)
        X=[x for x in woedata.Variable.unique() if x!=Target]
#        print(X)
        woe_final = pd.DataFrame(index=range(len(df2)))
        for i in X:
            print(i)
            d=woedata.loc[woedata.Variable==i,:]
            d_cont = d[~d["Min"].astype(str).str.contains("d_")]
            d_cat = d[d["Min"].astype(str).str.contains("d_")]
            d_cat["Min"] = d_cat["Min"].apply(lambda x: re.sub("^d_","",x))
            df2_cat = pd.DataFrame(df2[i])
            df2_cat = df2_cat[df2_cat[i].isin(d_cat["Min"].to_list())]
            df2_cat_woe = self.woe_data_cat(Target,df2_cat,d_cat)
            
            agg=d_cont.groupby(["bins"])['Bads %','Goods %'].sum()
#            print(agg)
            agg['WOE']=np.log(agg["Bads %"]/agg["Goods %"])
#            print(agg)
            agg['bins']=d_cont.groupby(["bins"])['bins'].mean()
#            print(agg)
#            print(agg)
    
            subst=pd.DataFrame(df2[i])
            subst=subst[~(subst[i].isin(d_cat["Min"].to_list()))]
    
            subst[i+"_woe"]=0
    
            for j in range(0,len(d_cont.bins.unique())):
                if j==len(d_cont.bins.unique())-1:
                    m1=min(d_cont.loc[d_cont.bins==d_cont.bins.unique()[j],'Min'])
                    subst.loc[subst[i]>m1,i+"_woe"]=float(agg.loc[agg['bins']==d_cont.bins.unique()[j],"WOE"].values)
                else:
                    m1=min(d_cont.loc[d_cont.bins==d_cont.bins.unique()[j],'Min'])
                    m2=min(d_cont.loc[d_cont.bins==d_cont.bins.unique()[j+1],'Min'])
                    subst.loc[(subst[i]>m1) & (subst[i]<=m2),i+"_woe"]=float(agg.loc[agg['bins']==d_cont.bins.unique()[j],"WOE"].values)
#            df2[i+"_woe"]=subst[i+"_woe"]
#            print(len(pd.concat([df2_cat_woe,subst[[i,i+"_woe"]]])))
            
            woe_final = woe_final.merge(pd.concat([df2_cat_woe,subst[[i,i+"_woe"]]]).reset_index(drop=True),left_index=True,right_index=True)
#        df2 = df2.round(2)
        return woe_final
    
    def woe_data_cat(self,Target, df2,woedata, ID=None):
        """
        This function creates woe data for categorical variables using the binning excel sheets provided by the User
        
            Attributes
            -------
            Target: Y variable name
            df2: Original dataset
            woedata: Binning sheet
            
            Returns
            -------
            pandas DataFrame, containing original data along with woe varibles that are binned in the binning sheet
            
            Code
            -------       
            import Feature_Transformation as ft
            woedata = pd.read_csv("WOE_Manual_Bins.csv")
            
            # The variables in the test_data should match with the variables in the woedata
            c=ft.WoE_Transformation_Manual().woe_data_cat(target,test_data,woedata)
            
        """
        if ID is None:
            df2.reset_index(drop=True, inplace=True)
        if ID is not None:
            df2.set_index(ID, inplace=True)
            
        X=[x for x in woedata.Variable.unique() if x!=Target]
        for i in X:
            d=woedata.loc[woedata.Variable==i,:]
            d["Min"] = d["Min"].apply(lambda x: re.sub("^d_","",x))
            agg=d.groupby(["bins"])['Bads %','Goods %'].sum()
            agg['WOE']=np.log(agg['Bads %']/agg['Goods %'])
#            print(agg)
            agg['bins_f']=d.groupby(["bins"])['bins'].mean()
            
            subst=pd.DataFrame(df2[i])
#            print(type(d["Min"][0]))
            #subst[i+"woe"]=0
            subst.fillna("non",inplace=True)
            df2.fillna("non",inplace=True)
            d.replace("nan","non",inplace=True)
            subst.replace("nan","non",inplace=True)
            df2.replace("nan","non",inplace=True)
            subst[i]=subst[i].astype(object)
            df2[i]=df2[i].astype(object)

#            print(d.Min.unique())
#            print(subst[i].unique())
            s=subst.merge(d[['Min','bins']],left_on=i,right_on='Min')
#            print(agg)
#            print(s)
            
            s1=s.merge(agg[['WOE','bins_f']],left_on="bins",right_on='bins_f')
            s1.drop_duplicates(subset=None, keep='first', inplace=True)
#            print(s1)
            df2=df2.merge(s1[['WOE',i]],left_on=i,right_on=i)
            df2=df2.rename(columns={"WOE": i+"_woe"})
            df2.replace("non",np.nan,inplace=True)
            df2.replace([float("-inf"),float("inf")],0,inplace=True)
#        df2 = df2.round(2)
        return df2

            
