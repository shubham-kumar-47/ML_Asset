"""
Project
-------
Feature Selection
 
Introduction
----------
This module performs functions which can be used to select important features for any ML model.
It basically has four components-

1. VARCLUS

2. PCA

3. Information Value

4. VIF

####VARCLUS: 
It is an unsupervised algorithm which uses PCA and clustering to group similar features in the same cluster. 'varclus'
function can be used for this algorithm.

####PCA: 
It is an unsupervised algorithm which finds latent variables(PCs). These latent variables are weighted average of actual
variables. One can find important features by collecting all the high weightage variables from important PCs. 'PCA' function
can be used for this algorithm.


####Information Value(IV): 
It basically shows the bivariate strength of relationship between the independent variable and
the response variable. "IV" function can be used for this algorithm.


####VIF: 
It addresses the problem of multicollinearity. It measures the strength of relationship between the variable under 
consideration and the rest of the X variables. 'VIF' can be used for this algorithm

Dependencies
------------
    python>-3.6
    
    pandas==0.24.2
    
    numpy==1.18.4
    
    sklearn==0.20.3
    
    statsmodels
    
    varclushi

**Additional Files**
--------------------
    
    1. excel_formatting.py 

Please ensure that the above files are also present in the working directory and packages mentioned in the dependencies section are installed.

How to run?
-----------
    Sample Codes:    
    
    # Importing Feature Selection Module
    import feature_selection as fs
    
    # Creating Feature Selection Object
    a=fs.feature_selection() 
    
    For varclus:
    
    rvc=a.varclus(train_o,target_id=target,maxeigval=1,maxcluster=5)
            
    For PCA:
    
    var_list, pc_weights = a.PCA(train_o, target, threshold=40)
    
    For IV:
    
    rvc=a.IV(train_o,target_id=target,numeric_bivariate_output_file_name="bivariate_num",categorical_bivariate_output_file_name="bivariate_cat")
    
    For VIF:
    
    # This function is recurcise in nature, would take longer for larger datasets
    v=a.vif_fun(train_o],target,threshold=10)
    
    For varclusIV:
     
    rvc=a.varclusIV(train_o,target, maxeigval=1, maxcluster=40)
    
Author
------
Created on Wednesday Jun  17 12:08:27 2020

@author: Kumar, Shubham

"""

import os
import numpy as np
import pandas as pd
from varclushi import VarClusHi
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
#from eda2 import eda
#from sklearn.feature_selection import VarianceThreshold
#from sklearn.svm import SVC
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.feature_selection import RFECV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from .excel_formatting import excel_formatting as ef
import warnings
warnings.filterwarnings("ignore")


class feature_selection:

    save_directory = 'results/Feature Selection'
    
    if os.path.isdir("results")==False:
        os.mkdir("results")
    
    if os.path.isdir(save_directory)==False:
        os.mkdir(save_directory)
    
#    Formats - xlsxwriter
    round0 = ef.formatting_xlsxwriter(format_name='round0')
    round2 = ef.formatting_xlsxwriter(format_name='round2')
    round3 = ef.formatting_xlsxwriter(format_name='round3')
    percsign = ef.formatting_xlsxwriter(format_name='percsign')
    cell_highlight1 = ef.formatting_xlsxwriter(format_name='cell_highlight1')
    cell_highlight2 = ef.formatting_xlsxwriter(format_name='cell_highlight2')
    cell_anomaly_bad = ef.formatting_xlsxwriter(format_name='cell_anomaly_bad')
    table_header= ef.formatting_xlsxwriter(format_name='table_header')
    table_title = ef.formatting_xlsxwriter(format_name='table_title')
    cell_even = ef.formatting_xlsxwriter(format_name='cell_even')
    cell_odd = ef.formatting_xlsxwriter(format_name='cell_odd')
    align_centre = ef.formatting_xlsxwriter(format_name='align_centre')
    border1 = ef.formatting_xlsxwriter(format_name='border1')
    
    
    def __init__(self):
        """This is the initiallisation of class. It needs to be done before calling any function from the class.
    
        """
        pass
        
    def PCA(self,data,target_id,threshold):
        """This Function performs PCA and selects most important variables using the following procedure:
        
           1. Performing PCA on the given data
           
           2. Finding the variation explained by each PC
           
           3. Selecting the PCs which constitute 'threshold'(for example 0.8 etc) amount of information
           
           4. A PC is a weighted mean of original variables. Using the weight, top 5 important variables for each PC can be found
           
           5. Point 4 is applied on each PC and against each PC one can get top 5 important variable. Unique set of these
           variable is taken as the variable set
    
                Attributes
                ----------
                data: A Pandas Dataframe
                target_id: Name of the target variable, if not present pass empty string ("")
                threshold: Amount of information to be retained(range:0 to 1)
    
                Returns
                -------
                varible data set: Dataframe of important varibles
                PC weights: Weight Matrix of PC weights 
                
                Code
                -------
                import feature_selection as fs
                
                a=fs.feature_selection()
                var_list, pc_weights = a.PCA(train_o, target, threshold=40)
    
        """
        x = [x for x in data.columns if data[x].dtypes != 'object' and x != target_id]  
        data2=data[x]
        data2.replace([np.inf, -np.inf], np.nan,inplace=True) 
        data2.fillna(data2.mean(),inplace=True)
        scaled_data = pd.DataFrame(scale(data2), columns=data2.columns, index = data.index)
        pca1 = PCA()
        pca=pca1.fit(scaled_data.values)
        pcweights=pd.DataFrame(pca.components_,columns=data2.columns).round(3)
        #variance_explained = pca.explained_variance_ratio_
        cumulative_variance_explained = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
        cum_var_exp_df=pd.DataFrame(cumulative_variance_explained, index = pd.Series(["PC_"+str(i) for i in range(pca.n_components_)]),columns=["Cumulative Variance Explained"])
        cum_var_df_thres=cum_var_exp_df.loc[cum_var_exp_df['Cumulative Variance Explained']<threshold]
        rng=len(cum_var_df_thres)
        pc=pd.DataFrame()
        for i in range(rng):
            PC_variance = pd.DataFrame([pd.Series(scaled_data.columns),pd.Series(pca.components_[i]), pd.Series(pca.components_[i]).abs()]).T
            PC_variance.columns = ["Variable","Value","Sort Column"]
            pc=pc.append(PC_variance.sort_values(["Sort Column"] ,ascending=False).head())
        
        
        pca_vars = pd.DataFrame({'Important Variables':pd.Series(pc['Variable'].unique())})
        
        ef.reset_header_default()
        filename = 'PCA'
        filename = ef.create_version(base_filename=filename, path = os.path.join(os.getcwd(),feature_selection.save_directory))
        filename += '.xlsx'
        
        writer = pd.ExcelWriter(os.path.join(os.getcwd(),feature_selection.save_directory,filename), engine='xlsxwriter')
        workbook  = writer.book
        
        # FORMATS--------------------------------------------------------------------------
        
        
        format_highlight1 = workbook.add_format(feature_selection.cell_highlight1)        
        
        format_table_header = workbook.add_format(feature_selection.table_header)
        
        format_cell_even = workbook.add_format(feature_selection.cell_even)
        format_cell_odd = workbook.add_format(feature_selection.cell_odd)
        
        # SHEET: Important Variables--------------------------------------------------------------------------
        
        pca_vars.to_excel(writer, sheet_name='Important Variables',index='True',startrow=4,startcol=1) 
    
        worksheet = writer.sheets['Important Variables']
        worksheet.hide_gridlines(2)
        
        # applying formatting
            
        # table header
        worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 5,column_number = 3,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = 5,column_number = len(pca_vars)+3,fix_row=True),
                                     {'type': 'no_blanks','format': format_table_header})
        
        # logo
        worksheet.insert_image(ef.generate_excel_cell_name(row_number=2,column_number=2), os.path.join(os.getcwd(),'src','ks_logo.png'))
        if len(pca_vars)>0:            
            
            
            # table cells                       
            rows = list(range(5,len(pca_vars)+6))          
            
            for row in rows:
                worksheet.conditional_format(ef.generate_excel_cell_name(row_number = row,column_number = 3,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = row,column_number = len(pca_vars)+3,fix_row=True),
                                             {'type': 'no_blanks','format': (format_cell_even if row%2==0 else format_cell_odd)})
            
            max_column_width = max([len(x) + 2 for x in pca_vars['Important Variables']])
            worksheet.set_column(ef.generate_excel_cell_name(row_number = None,column_number = 3)+':'+ef.generate_excel_cell_name(row_number = None,column_number = 3),
                                 max_column_width)
            
        
        # SHEET: PC Weights--------------------------------------------------------------------------
        
        pcweights.to_excel(writer, sheet_name='PC Weights',index_label='PC',startrow=4,startcol=1) 
    
        worksheet = writer.sheets['PC Weights']
        worksheet.hide_gridlines(2)
        
        # applying formatting
            
         # table header
        worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 5,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = 5,column_number = len(pcweights.columns)+2,fix_row=True), {'type': 'no_blanks','format': format_table_header})
        
        # logo
        worksheet.insert_image(ef.generate_excel_cell_name(row_number=2,column_number=2), os.path.join(os.getcwd(),'src','ks_logo.png'))
        
        if len(pcweights)>0:            
            
            
            # table cells                       
            rows = list(range(5,len(pcweights)+6))          
            
            for row in rows:
                worksheet.conditional_format(ef.generate_excel_cell_name(row_number = row,column_number = 3,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = row,column_number = len(pcweights.columns)+2,fix_row=True),
                                             {'type': 'no_blanks','format': (format_cell_even if row%2==0 else format_cell_odd)})
            
        
            
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 6,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = len(pcweights)+5,column_number = 2,fix_row=True),
                                         {'type': 'no_blanks','format': format_highlight1})
            
        writer.save()
        writer.close()
            
    
        return([pca_vars,pcweights])

    def varclus(self,data,target_id,maxeigval,maxcluster,save_results=True):
        """This Function performs the standard PROC VARCLUS code in python 
        
                Attributes
                ----------
                data: A Pandas Dataframe
                target_id: Name of the target variable, if not present pass empty string ("")
                maxeigval: Maximum eigen value
                maxcluster: Maximum number of clusters to be retained
                save_results: string, default=False, If True, will save output excel
    
                Returns
                -------
                dataset: A dataframe of variables along with their clusters and 1-R^2 ratio value
                
                Code
                -------
                import feature_selection as fs
                
                a=fs.feature_selection()
                rvc=a.varclus(train_o,target_id=target,maxeigval=1,maxcluster=5)
                
                Usage
                -------
                It can be used to select different type of variables(as each cluster is different from one another).
                
                R^2 ratio denotes ratio of R^2 of relationship between variable under consideration and other cluster variables
                and R^2 of relationship between variable under consideration and own cluster variables.
                
                This means a variable with low R^2 ratio is strongly related to its cluster and less related other clusters.
                
                One can select one or two variables from each cluster with low R^2 ratio. 
                
                
        """
            
        x = [x for x in data.columns if data[x].dtypes != 'object' and x != target_id and len(data[x].unique())!=1]  
        data2=data[x]
        data2.replace([np.inf, -np.inf], np.nan,inplace=True) 
        data2.fillna(data2.mean(),inplace=True)
        demo1_vc = VarClusHi(data2,maxeigval2=maxeigval,maxclus=maxcluster)
        demo1_vc.varclus()
        df=pd.DataFrame(demo1_vc.rsquare)
        df['RS_Own']=np.round(df['RS_Own'],2)
        df['RS_NC']=np.round(df['RS_NC'],2)
        df['RS_Ratio']=np.round(df['RS_Ratio'],2)
        
        if save_results==False:
            return(df)
        
        ef.reset_header_default()
        filename = 'Varclus'
        filename = ef.create_version(base_filename=filename, path = os.path.join(os.getcwd(),feature_selection.save_directory))
        filename += '.xlsx'
        
        writer = pd.ExcelWriter(os.path.join(os.getcwd(),feature_selection.save_directory,filename), engine='xlsxwriter')
        workbook  = writer.book
        
        # FORMATS--------------------------------------------------------------------------
        
        
        format_border1 = workbook.add_format(feature_selection.border1)  
        format_align_centre = workbook.add_format(feature_selection.align_centre)
        format_table_header = workbook.add_format(feature_selection.table_header)
        format_cell_odd = workbook.add_format(feature_selection.cell_odd)
        
        # SHEET: Varclus--------------------------------------------------------------------------
        df.to_excel(writer, sheet_name='Varclus',index=False,startrow=4,startcol=1) 
    
        worksheet = writer.sheets['Varclus']
        worksheet.hide_gridlines(2)
        
        # applying formatting
            
        # table header
        worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 5,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = 5,column_number = len(df.columns)+1,fix_row=True),
                                     {'type': 'no_blanks','format': format_table_header})
        
        # logo
        worksheet.insert_image(ef.generate_excel_cell_name(row_number=2,column_number=2), os.path.join(os.getcwd(),'src','ks_logo.png'))
        
        if len(df)>0:            
            
            
            # table cells
            df.reset_index(drop=True,inplace=True)        
            start_row = 6
            end_row = 0
            for i,r in df.iterrows():
                if i>1:
                    if df['Cluster'][i]>df['Cluster'][i-1]:
                        end_row  = start_row + i-1  
                        worksheet.conditional_format(ef.generate_excel_cell_name(row_number = start_row,column_number = len(df.columns)+1,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = end_row,column_number = len(df.columns)+1,fix_row=True),
                                                 {'type': '3_color_scale',
                                                  'min_color': "#538ED5",
                                                  'mid_color': "#8DB4E3",
                                                  'max_color': "#C5D9F1"})
                        start_row = i             
            
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 7,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = len(df)+6,column_number = len(df.columns)+1,fix_row=True),
                                     {'type': 'formula',
                                      'criteria': "=" + ef.generate_excel_cell_name(row_number = 7,column_number = 2,fix_column=True)+'>'+ef.generate_excel_cell_name(row_number = 6,column_number = 2,fix_column=True),
                                      'format': format_border1})
            
            rows = list(range(6,len(df)+6))          
            
            for row in rows:
                worksheet.conditional_format(ef.generate_excel_cell_name(row_number = row,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = row,column_number = len(df.columns)+1,fix_row=True),
                                             {'type': 'no_blanks','format': format_cell_odd})
            
        max_column_width = max([len(x) + 2 for x in df['Variable']])
        worksheet.set_column(ef.generate_excel_cell_name(row_number = None,column_number = 3)+':'+ef.generate_excel_cell_name(row_number = None,column_number = 3),
                                 max_column_width)
        
        worksheet.set_column(ef.generate_excel_cell_name(row_number = None,column_number = 2)+':'+ef.generate_excel_cell_name(row_number = None,column_number = 2),
                                 None,format_align_centre)
            
            
        writer.save()
        writer.close()
        return(df)

    def __WOE(self,data, varList, type0='Con', target_id='y'):
        """This Function performs bivariate on numeric as well as categorical variables
        
                Attributes
                -------
                data: pandas DataFrame, mostly refer to ABT(Analysis Basics Table)
                varList: variable list
                type0: Continuous or Discontinuous(Category), 'con' is the required input for Continuous
                target_id: y flag when gen the train data, if not present pass empty string ("")
                resfile: download path of the result file of WOE and IV
    
                Returns
                -------       
                pandas DataFrame, result of woe and iv value according y flag
    
        """
        data=data.astype({target_id: 'int32'})
        result = pd.DataFrame()
        for var in varList:
            print(var)
            data.replace( np.nan,-999999,inplace=True) 
            try:
                if type0.upper() == "CON".upper():
                    #data.replace( np.nan,-999999,inplace=True) 
                    if len(list(data[var].unique()))>10:
                        df, retbins = pd.qcut(data[var], q=10, retbins=True, duplicates="drop")
                        tmp = pd.crosstab(df, data[target_id])
                        tmp2 = pd.crosstab(df, data[target_id]).apply(lambda x: x / sum(x), axis=0)
                    else:
                        
                        df = data[var]
                        tmp = pd.crosstab(data[var], data[target_id])
                        tmp2 = pd.crosstab(data[var], data[target_id]).apply(lambda x: x / sum(x),axis=0)
                else:
                    #data.replace( np.nan,-999999,inplace=True) 
                    df = data[var]
                    tmp = pd.crosstab(data[var], data[target_id])
                    tmp2 = pd.crosstab(data[var], data[target_id]).apply(lambda x: x / sum(x),axis=0)
    
    
                res = tmp.merge(tmp2, how="left", left_index=True, right_index=True)
                res['count'] = res['0_x']+res['1_x']
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
                    ['name','count','Population_perc', 'low', 'high', '0_x', '1_x', '1_y', 'Bad_perc', 'Good_perc', 'WOE',
                     'dif',
                     'IV', 'IV_sum']]
                result = result.append(res)
    
            except Exception as e:
                print(e, var)
        result.rename(columns = {'0_x':'Goods','1_x':'Bads','1_y':'Badrate'}, inplace = True)
        #result['count'] = result['Goods']+result['Bads']
        #result.to_excel(resfile)
        result.round()
        #result.IV.replace([np.inf, -np.inf], np.nan,inplace=True)
        #result.WOE.replace([np.inf, -np.inf], np.nan,inplace=True)
        print(result)
        result['Badrate']=np.round(result['Badrate'],2)
        result.sort_values(['IV_sum'],inplace=True,ascending=False)
        return result


    def IV(self,data,target_id='y',numeric_bivariate_output_file_name="bivariate_num",categorical_bivariate_output_file_name="bivariate_cat"):
        """This Function performs bivariate analysis on the given data 
        
                Attributes
                ----------
                data: A Pandas Dataframe
                target_id: Name of the target variable, if not present pass empty string ("")
                numeric_bivariate_output_file_name: Numeric Bivariate output file name
                categorical_bivariate_output_file_name: Categorical Bivariate output file name
                
                Returns
                -------
                A list of two dataframes: Numeric Bivariate and Categorical Bivariate
                
                Code
                -------
                import feature_selection as fs
                
                a=fs.feature_selection()
                rvc=a.IV(train_o,target_id=target,numeric_bivariate_output_file_name="bivariate_num",categorical_bivariate_output_file_name="bivariate_cat")
        """
        
        result = []
        ef.reset_header_default()
        filename = 'IV'
        filename = ef.create_version(base_filename=filename, path = os.path.join(os.getcwd(),feature_selection.save_directory))
        filename += '.xlsx'
        
        writer = pd.ExcelWriter(os.path.join(os.getcwd(),feature_selection.save_directory,filename), engine='xlsxwriter')
        workbook  = writer.book
        
        # FORMATS--------------------------------------------------------------------------
        
        format_table_header = workbook.add_format(feature_selection.table_header)
        format_cell_odd = workbook.add_format(feature_selection.cell_odd)
        format_cell_even = workbook.add_format(feature_selection.cell_even)
        
        
        X = [x for x in data.columns if data[x].dtypes != 'object' and x != target_id] 
        
        if len(X)>0:
            
            woe_num=self.__WOE(data, X, type0='Con', target_id=target_id)     
            result.append(woe_num)
        
    
        X = [x for x in data.columns if data[x].dtypes == 'object' and x != target_id]  
        
        if len(X)>0:
            
            woe_cat=self.__WOE(data, X, type0='Cat', target_id=target_id) 
            woe_cat.reset_index(inplace=True,drop=True)
            result.append(woe_cat)
    
               
        #woe_num.to_excel(numeric_bivariate_output_file_name+".xlsx")
        #woe_cat.to_excel(categorical_bivariate_output_file_name+".xlsx")
        return (result)
    
    
    


    def varclusIV(self,data,target_id,maxeigval,maxcluster, resultfile='result'):
        """
        This Function performs bivariate and VARCLUS on the numeric dataset
        
            Attributes
            ----------       
            data: pandas DataFrame, mostly refer to ABT(Analysis Basics Table)
            varList: variable list
            type0: Continuous or Discontinuous(Category), 'con' is the required input for Continuous
            target_id: y flag when gen the train data, if not present pass empty string ("")
            resultfile: Output file name
            
            Returns
            -------
            pandas DataFrame, containing variables along with their clusters, 1-R^2 ratio and IV values
            
            Code
            -------       
            import feature_selection as fs
            
            a=fs.feature_selection()
            rvc=a.varclusIV(train_o,target, maxeigval=1, maxcluster=40)
            
            Usage
            -------
            It can be used to select different type of variables(as each cluster is different from one another).
            
            R^2 ratio denotes ratio of R^2 of relationship between variable under consideration and other cluster variables
            and R^2 of relationship between variable under consideration and own cluster variables.
            
            This means a variable with low R^2 ratio is strongly related to its cluster and less related other clusters.
            
            One can select one or two variables from each cluster with low R^2 ratio. 
            
            This function also provides IV of each variable along with R^2 ratio so both these quatities can be used while 
            selecting the variable. 
            
            One or two variables can be choosen from each cluster with low R^2 ratio and high IV.
            
        
        """
        X = [x for x in data.columns if data[x].dtypes != 'object' and x != target_id]  
    
        result=self.__WOE(data, X, type0='Con', target_id=target_id)        
        
        dd=self.varclus(data,target_id,maxeigval,maxcluster,save_results=False)
    
    
        ####merging IV and Varclus
        d1=dd.merge(result[['name','IV_sum']], how = "left", left_on = ['Variable'],right_on=['name'])
        d1.drop_duplicates(subset=None, keep='first', inplace=True)
        d1.drop(['name'],axis=1,inplace=True)
        
        d1.rename(columns={"IV_sum":"IV Sum",
                           "RS_Own":"1-RS_Own",
                           "RS_NC":"1-RS_NC",
                           "RS_Ratio":"1-RS_Ratio",},inplace=True)
        
        ef.reset_header_default()
        filename = 'Varclus IV'
        filename = ef.create_version(base_filename=filename, path = os.path.join(os.getcwd(),feature_selection.save_directory))
        filename += '.xlsx'
        
        writer = pd.ExcelWriter(os.path.join(os.getcwd(),feature_selection.save_directory,filename), engine='xlsxwriter')
        workbook  = writer.book
        
        # FORMATS--------------------------------------------------------------------------
        
        
        format_border1 = workbook.add_format(feature_selection.border1)  
        format_align_centre = workbook.add_format(feature_selection.align_centre)
        format_table_header = workbook.add_format(feature_selection.table_header)       
        format_cell_odd = workbook.add_format(feature_selection.cell_odd)
        
        # SHEET: Varclus--------------------------------------------------------------------------
        d1.to_excel(writer, sheet_name='Varclus',index=False,startrow=4,startcol=1) 
    
        worksheet = writer.sheets['Varclus']
        worksheet.hide_gridlines(2)
        
        # applying formatting
            
        # table header       
        worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 5,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = 5,column_number = len(d1.columns)+1,fix_row=True),
                                     {'type': 'no_blanks','format': format_table_header})
        
        # logo
        worksheet.insert_image(ef.generate_excel_cell_name(row_number=2,column_number=2), os.path.join(os.getcwd(),'src','ks_logo.png'))
    
        # table cells  
        d1.reset_index(drop=True,inplace=True)        
        start_row = 6
        end_row = 0
        for i,r in d1.iterrows():
            if i>1:
                if d1['Cluster'][i]>d1['Cluster'][i-1]:
                    end_row  = start_row + i-1  
                    worksheet.conditional_format(ef.generate_excel_cell_name(row_number = start_row,column_number = len(d1.columns)+1,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = end_row,column_number = len(d1.columns)+1,fix_row=True),
                                             {'type': '3_color_scale'})
                    worksheet.conditional_format(ef.generate_excel_cell_name(row_number = start_row,column_number = len(d1.columns),fix_row=True)+':'+ef.generate_excel_cell_name(row_number = end_row,column_number = len(d1.columns),fix_row=True),
                                             {'type': '3_color_scale',
                                              'min_color': "#538ED5",
                                              'mid_color': "#8DB4E3",
                                              'max_color': "#C5D9F1"})
                    start_row = i

        worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 7,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = len(d1)+6,column_number = len(d1.columns)+1,fix_row=True),
                                 {'type': 'formula',
                                  'criteria': "=" + ef.generate_excel_cell_name(row_number = 7,column_number = 2,fix_column=True)+'>'+ef.generate_excel_cell_name(row_number = 6,column_number = 2,fix_column=True),
                                  'format': format_border1})
        
        rows = list(range(6,len(d1)+6))          
        
        for row in rows:
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number = row,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = row,column_number = len(d1.columns)+1,fix_row=True),
                                         {'type': 'no_blanks','format': format_cell_odd})
        
        max_column_width = max([len(x) + 2 for x in d1['Variable']])
        worksheet.set_column(ef.generate_excel_cell_name(row_number = None,column_number = 3)+':'+ef.generate_excel_cell_name(row_number = None,column_number = 3),
                                 max_column_width)
        
        
        worksheet.set_column(ef.generate_excel_cell_name(row_number = None,column_number = 2)+':'+ef.generate_excel_cell_name(row_number = None,column_number = 2),
                                 None,format_align_centre)
            
        writer.save()
        writer.close()
        
        return(d1)   


    def vif_fun(self,data,target_id,threshold=2,resultfile='VIF_Result'):
        """
        This function finds vif the variable set 
        
            Attributes
            -------       
            data: Dataset
            target: target variable name, if not present pass empty string ("")
            resultfile: Output file name
            
    
            Returns
            -------       
            A list of two dataframes: 
            A dataframe of variables and VIF factors
            A dataframe of shortlisted variables and VIF factors after recursively removing the variables with VIF greater than
            given threshold
            
            Code
            -------       
            import feature_selection as fs
            
            a=fs.feature_selection()
            v=a.vif_fun(train_o[list(train_o.columns)[:5]],target,threshold=10)
            
            Usage
            -------   
            If a variable has VIF value of greater than 2, it means varible is moderately related to other variables
            
        """
        
        x = [x for x in data.columns if data[x].dtypes != 'object' and x != target_id]  
        data2=data[x]
        data2.replace([np.inf, -np.inf], np.nan,inplace=True) 
        data2.fillna(data2.mean(),inplace=True)
        
    
        x_data=data2
        variables=x_data.columns
        vif_overall = pd.DataFrame()
        vif_overall["Feature"] = x_data.columns
    
        vif_overall["VIF Factor"]= np.round([variance_inflation_factor(x_data[variables].values, x_data.columns.get_loc(var)) for var in x_data.columns],2)
        vif_overall.sort_values('VIF Factor', axis=0, ascending=False, inplace=True)
        #if max(vif['VIF Factor'])>2:
        x_data.drop(x_data.columns[np.argmax(vif_overall['VIF Factor'])],axis=1,inplace=True)
    
        max_vif=max(vif_overall['VIF Factor'])
        
        df1created=False
        while max_vif>threshold and x_data.shape[1]>1:
            variables=x_data.columns
            vif = pd.DataFrame()
            vif["Feature"] = x_data.columns
    
            vif["VIF Factor"]= np.round([variance_inflation_factor(x_data[variables].values, x_data.columns.get_loc(var)) for var in x_data.columns],2)
            vif.sort_values('VIF Factor', axis=0, ascending=False, inplace=True)
        
            #if max(vif['VIF Factor'])>2:
            x_data.drop(x_data.columns[np.argmax(vif['VIF Factor'])],axis=1,inplace=True)
            max_vif=max(vif['VIF Factor'])
            df1created=True
            
        ef.reset_header_default()
        filename = 'VIF'
        filename = ef.create_version(base_filename=filename, path = os.path.join(os.getcwd(),feature_selection.save_directory))
        filename += '.xlsx'
        
        writer = pd.ExcelWriter(os.path.join(os.getcwd(),feature_selection.save_directory,filename), engine='xlsxwriter')
        workbook  = writer.book
        
        # FORMATS--------------------------------------------------------------------------
        
        format_table_header = workbook.add_format(feature_selection.table_header)
        format_cell_odd = workbook.add_format(feature_selection.cell_odd)
        format_cell_even = workbook.add_format(feature_selection.cell_even)
        
        if(df1created==True):  
#            writer = pd.ExcelWriter(resultfile+'.xlsx', engine='xlsxwriter')
#            vif_overall.to_excel(writer,sheet_name='VIF(All Variables)')
#            vif.to_excel(writer,sheet_name='VIF(Shortlisted Variables)')
#            writer.save()
            
            
        
            # SHEET: All Variables--------------------------------------------------------------------------
            vif_overall.to_excel(writer, sheet_name='All Variables',index=False,startrow=4,startcol=1) 
    
            worksheet = writer.sheets['All Variables']
            worksheet.hide_gridlines(2)
            
            # applying formatting
                
            # table header
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 5,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = 5,column_number = len(vif_overall.columns)+1,fix_row=True),
                                         {'type': 'no_blanks','format': format_table_header})
            
            # logo
            worksheet.insert_image(ef.generate_excel_cell_name(row_number=2,column_number=2), os.path.join(os.getcwd(),'src','ks_logo.png'))
            
            if len(vif_overall)>0:            
                
                
                # table cells
                
                rows = list(range(6,len(vif_overall)+6))          
                
                for row in rows:
                    worksheet.conditional_format(ef.generate_excel_cell_name(row_number = row,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = row,column_number = len(vif_overall.columns)+1,fix_row=True),
                                                 {'type': 'no_blanks','format': (format_cell_even if row%2==0 else format_cell_odd)})
                
                max_column_width = max([len(x) + 2 for x in vif_overall['Feature']])
                worksheet.set_column(ef.generate_excel_cell_name(row_number = None,column_number = 2)+':'+ef.generate_excel_cell_name(row_number = None,column_number = 2),
                                         max_column_width)
                
            
            # SHEET: All Variables--------------------------------------------------------------------------
            vif.to_excel(writer, sheet_name='Shortlisted Variables',index=False,startrow=4,startcol=1) 
    
            worksheet = writer.sheets['Shortlisted Variables']
            worksheet.hide_gridlines(2)
            
            # applying formatting
                
            # table header
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 5,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = 5,column_number = len(vif.columns)+1,fix_row=True),
                                         {'type': 'no_blanks','format': format_table_header})
            
            # logo
            worksheet.insert_image(ef.generate_excel_cell_name(row_number=2,column_number=2), os.path.join(os.getcwd(),'src','ks_logo.png'))
            
            if len(vif)>0:            
                
                
                # table cells
                
                rows = list(range(6,len(vif)+6))          
                
                for row in rows:
                    worksheet.conditional_format(ef.generate_excel_cell_name(row_number = row,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = row,column_number = len(vif.columns)+1,fix_row=True),
                                                 {'type': 'no_blanks','format': (format_cell_even if row%2==0 else format_cell_odd)})
                
                max_column_width = max([len(x) + 2 for x in vif['Feature']])
                worksheet.set_column(ef.generate_excel_cell_name(row_number = None,column_number = 2)+':'+ef.generate_excel_cell_name(row_number = None,column_number = 2),
                                         max_column_width)
                
                
            writer.save()
            writer.close()
    
    
            return [vif_overall, vif, x_data.columns]   
            
        else:
            # SHEET: All Variables--------------------------------------------------------------------------
            vif.to_excel(writer, sheet_name='Shortlisted Variables',index=False,startrow=4,startcol=1) 
    
            worksheet = writer.sheets['Shortlisted Variables']
            worksheet.hide_gridlines(2)
            
            # applying formatting
                
            # table header
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 5,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = 5,column_number = len(vif.columns)+1,fix_row=True),
                                         {'type': 'no_blanks','format': format_table_header})
            
            # logo
            worksheet.insert_image(ef.generate_excel_cell_name(row_number=2,column_number=2), os.path.join(os.getcwd(),'src','ks_logo.png'))
            
            if len(vif)>0:            
                
                
                # table cells
                
                rows = list(range(6,len(vif)+6))          
                
                for row in rows:
                    worksheet.conditional_format(ef.generate_excel_cell_name(row_number = row,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = row,column_number = len(vif.columns)+1,fix_row=True),
                                                 {'type': 'no_blanks','format': format_cell_odd})
                
            max_column_width = max([len(x) + 2 for x in vif['Feature']])
            worksheet.set_column(ef.generate_excel_cell_name(row_number = None,column_number = 2)+':'+ef.generate_excel_cell_name(row_number = None,column_number = 2),
                                     max_column_width)
                
                
            writer.save()
            writer.close()
    
            return vif_overall
            

