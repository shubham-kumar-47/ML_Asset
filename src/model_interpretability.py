# -*- coding: utf-8 -*-
"""
Project
-------
Model Interpretation

Description
-----------
### Reason codes
This module has various mechanisms to find the top variables of the data by which the model predicts the output for a particular row.
The reasoning of the model can be done using any one of the following algorithms:

#### **VWOE** (**V**ariable importance induced **W**eight **O**f **E**vidence)
 
Feature rank is calculated for each variable at different level with the product of WOE and the feature importance value of those variable respectively.

Variable importance refers to how much a given model "uses" that variable to make accurate predictions. The more a model relies on a variable to make predictions, the more important it is for the model

Feature rank = WOE * Variable Importance

Using this Feature rank we find the high contributing variables for each predictions, given the list of variables and their levels used in the prediction we map it to their Feature rank and sort the top predictors.


#### **LIME** (**L**ocal **I**nterpretable **M**odel-Agnostic **E**xplanation)

LIME builds sparse linear models around each prediction to explain how the black box model works in that local vicinity.
The explanation reflect the behaviour of the classifier "around" the instance being predicted.

In order to figure out what parts of the interpretable input are contributing to the prediction, we perturb the input around its neighborhood and see how the model's predictions behave. 
We then weight these perturbed data points by their proximity to the original example, and learn an interpretable model on those and the associated predictions.


#### **SHAP** (**SH**apley **A**dditive ex**P**lanations)

A game-theoretic approach to explain the output of any machine learning model.
It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions.

SHAP provides multiple explainers for different kind of models.

- TreeExplainer: Support XGBoost, LightGBM, CatBoost and scikit-learn models by Tree SHAP.
- DeepExplainer (DEEP SHAP): Support TensorFlow and Keras models by using DeepLIFT and Shapley values.
- GradientExplainer: Support TensorFlow and Keras models.
- KernelExplainer (Kernel SHAP): Applying to any models by using LIME and Shapley values.


#### Advantages of **SHAP** over VWOE and LIME

- VWOE file creation takes a lot of time, as it needs a lot of input data to be created before final reason code calculation. LIME also takes a lot of time, If we have 1000 test claims, it will create 1000 models to get important variables. SHAP takes seconds.
- LIME is not trustworthy as its randomization can’t be forced and hence gives different result every time.
- VWOE doesn’t differentiate between a classification and regression model (Variable Importance, has a flavor though). In LIME it needs to be specified but SHAP can understand it based on the structure of the tree.
- SHAP is available for Deep Learning models also.


The interpretability module has the option to choose the type of algorithm to be used.

### Separation score and Top Variable Distribution
This module also provides funtions to calculate the separation scores and top variable distribution dataframe.
Separation score is calculated for the top variables between top and bottom decile.
The different variable distribution to the multiple top(n) columns are also calculated and stored.

.. note:: This module can be used for comparing the top variable distribution interpreted by all three models (LIME, SHAP and VWOE). By applying business knowledge, we have the ability to choose the right interpretation for the model.


Please refer to the class methods to understand their functionality.

Dependencies
------------
    lime==0.2.0.0
    numpy==1.18.4
    python>-3.6
    pandas==0.24.2
    shap==0.35.0
    xgboost==1.0.0 
    

How to run?
-----------
    Sample Codes:    

    >>> from model_interpretability import Reason_code as r
    >>> reasoning_dataframe = r.reasons_code(train_data,unique_id,vwoe,binning,top_n,test_data,model_fit,features,model_type,absent_remove,target_name,reason_type)

    
Authors
-------
Created on Fri Jun  15 12:35:17 2020

@author: Kumar, Shubham

"""


import pandas as pd
import numpy as np
import lime
import shap
import re
import lime.lime_tabular
import copy
import os
from .excel_formatting import excel_formatting as ef
import warnings
warnings.filterwarnings("ignore")


class Reason_code:
    """
    This class contains the different mechanisms to find the top variables of the data by which the model predicts the output for a particular row. 

    Please ensure that the packages mentioned in the requirements.txt are installed.
    """
    
    
    __save_directory = 'results/Model_Interpretation_Results'
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
    
    @classmethod
    def reasons_code(cls,train_data,unique_id,vwoe,binning_flag,topn,test_data,model_fit,features,model_type,absent_remove,target_name,algo="all"):     
        '''
        This function returns top(n) variables for the prediction of the row.

        Parameters
        ----------
        train_data: dataframe
            X and Y variables in training dataset.

        unique_id: string
            Name of the unique identifier column in train and test data.

        vwoe: dataframe
            Dataframe containing vwoe values for each level of each column. This can be generated using the function `vwoe_generate()` present in the same class.

        binning_flag: boolean
            This defines if the numerical variables are binned or not.

            `True` - If numerical variables are binned

            `False` - If numerical variables are not binned

        topn: integer
            This defines the number of top columns contributing to the prediction.
        
        test_data: dataframe
            X and Y variables in training dataset.
        
        model_fit: model object
            Model object for which reason is generated. Only Tree based model is supported now.

        features: list
            List of features used for fitting the model
        
        model_type: string
            It defines the type of model. It has two options - classification or regression

            `"classification"` - If it is a classification model

            `"regression"` - If it is a regression problem
        
        absent_remove: boolean
            This defines if the data containing values like 0 or null must be removed from the data or not.

            `True` - If values like 0 or null must be removed in the data for predicting reason

            `False` - If all values including 0s must be considered
        
        target_name: string
            Name of the column containing target (y) variable.
        
        algo: string, default = "all"
            This defines the type of algorithm used for reasoning the model. It includes four options - vwoe, lime, shap, all

            `"all"` - Gives reason using all the algorithms (VWOE,Lime,Shap)
            
            `"vwoe"`/`"lime"`/`"shap"` - Gives reason using only the specified algorithm

        Returns
        -------

        reasoning_dataframe: dataframe
            Contains top variables for all instances of test data with the original Y variable and predicted probability for events.
            This dataframe is also stored in the Model_Interpretation_Results folder. (File name : Model_Interpretation.csv)
        
        Example
        -------

        >>> from model_interpretability import Reason_code as r
        >>> reasoning_dataframe = r.reasons_code(train_data,unique_id,vwoe,binning,top_n,test_data,model_fit,features,model_type,absent_remove,target_name,reason_type)

		

        '''

        y = target_name
        mode_lime = model_type
        ID = unique_id
        binning = binning_flag
        model1 = model_fit
        
        test_data["index"]=test_data["index"].astype('str')
        train_data["index"]=train_data["index"].astype('str')
        
        #if (np.isfinite(train_data).values.all()==False):
        #    raise Exception('There may be missing values or inf values present in your data.\nPlease ensure that such values are not present in your input.')


        def reason_code(score_data,ID,vwoe,binning,topn,absent_remove):
                vwoe['Type']=vwoe['Type'].apply(lambda x:x.lower())
                vwoe.columns=['Variable','Value','Featur_rank','Lower','Upper','Type']
                vwoe['Type']=np.where(vwoe['Type']=='charcter','character',vwoe['Type'])
                #vwoe.drop(['indi',],inplace=True,axis=1)
                commen=list(np.intersect1d(list(score_data.columns),vwoe.Variable.unique()))
                commen.insert(0,ID)
                score_data=score_data[commen]
                #if(((len(uncommen)==1) & (uncommen[0]==ID))|(len(uncommen)==0)):
                def numeric_replace(x):
                    col=x.variable
                    #col=col+'_bin'
                    ##print(col)
                    value=float(x.value)
                    #lis=list()
                    ##print(value)
                    df=vwoe[vwoe['Variable']==col]
                    df.Upper=df['Upper'].astype(float)
                    df.Lower=df['Lower'].astype(float)
                    max_val=df.Upper.max()
                    min_val=df.Lower.min()
        #                 #print(col,value)
                    ##print(type(x))
                    #rint(df.dtypes)
                    if((value<max_val) & (value>min_val)):
                        t=float(df[(value<=df['Upper']) & (value>df['Lower'])]['Featur_rank'])
                    elif(value==min_val):
                        #print(df[(df['Lower']==min_val) & (df['Upper']!=min_val)]['Featur_rank'])
                        t=float(df[(df['Lower']==min_val) & (df['Upper']!=min_val)]['Featur_rank'])
                    else:
                        t=float(df[df['Upper']==max_val]['Featur_rank'])
                    #lis.append(t)
                    return(t)
        
        
                if(binning==False):   
                    numeric=list(vwoe[vwoe['Type']=='numeric']['Variable'].unique())
                    numeric=[i[:-4] if i[:-4]=='_bin' else i for i in numeric ]
                    numeric=list(np.intersect1d(numeric,list(score_data.columns)))
                    fin=pd.DataFrame()
                    if(len(numeric)>1):
                        numeric_data=score_data[numeric+[ID]]
                        num_data=pd.melt(numeric_data, id_vars =[ID], value_vars = numeric) 
                        t=num_data.apply(numeric_replace,axis=1)
                        num_data['Featur_rank']=t
                        num_data=num_data[~num_data['Featur_rank'].isna()]
                        num_data.columns=[ID, 'Variable', 'Value', 'Featur_rank']
                        fin=num_data.copy()
                    ###### for character columns
                    charactr=list(vwoe[vwoe['Type']=='character']['Variable'].unique())
                    char_rank=vwoe[vwoe['Type']=='character'][['Variable','Value','Featur_rank']]
                    charactr1=list(np.intersect1d(charactr,list(score_data.columns)))
                    if(len(charactr1)>1):
                        charcter_data=score_data[charactr1+[ID]]
                        c=pd.melt(charcter_data, id_vars =[ID], value_vars = charactr1) 
                        c.columns=[ID,'Variable','Value']
                        ty=type(c['Value'][0])
                        char_rank.Value=char_rank.Value.astype(ty)
                        char_data=pd.merge(c,char_rank,on=['Variable','Value'],how='left')
                        char_data=char_data[~char_data['Featur_rank'].isna()]
                        fin=fin.append(char_data)
                    if(len(charactr1)==0):
                        fin=num_data
                    fin=fin.sort_values(['Featur_rank'],ascending=False)
                    if(absent_remove==True):
                        fin = fin[fin['Value']>0]
                    g = fin.groupby(ID)
                    fin['top'] = g['Featur_rank'].cumcount()+1
                    fin=fin[fin['top']<=topn]
                    original_df = fin.pivot(index=ID, columns='top')['Variable'].reset_index()
                    l=['top_'+str(i+1) for i in range((original_df.shape[1])-1)]
                    l.insert(0,ID)
                    original_df.columns=l
                    
                    return(original_df)
                else:  
                    score_data=score_data.astype(str)
                    cols=list(score_data.columns)
                    cols.remove(ID)
                    melted=pd.melt(score_data, id_vars =[ID], value_vars = cols)
                    melted.columns=[ID,'Variable','Value']
                    melted['Value']=melted['Value'].astype(str)
                    vwoe['Value']=vwoe['Value'].astype(str)
                    wo=pd.merge(melted,vwoe,on=['Variable','Value'],how='left')
                    fin=wo[[ID,'Variable','Value','Featur_rank']]
                    fin=fin.sort_values(['Featur_rank'],ascending=False)
                    g = fin.groupby(ID)
                    fin['top'] = g['Featur_rank'].cumcount()+1
                    fin=fin[fin['top']<=topn]
                    original_df = fin.pivot(index=ID, columns='top')['Variable'].reset_index()
                    l=['top_'+str(i+1) for i in range((original_df.shape[1])-1)]
                    l.insert(0,ID)
                    original_df.columns=l
                    return(original_df)
        feature1 =features.copy() 
        feature1.insert(0,ID)
        model=copy.deepcopy(model1)
        if((algo=='all')|(algo=='vwoe')):
            v_woe_out=reason_code(test_data[feature1].copy(),ID,vwoe,binning,topn,absent_remove)
            #print(v_woe_out.head(5))
            v_woe_out['model']='VWOE'
    
        def lime_y(reason_df,model_fit,model_features,mode1,test_ds,ID,vwoe,absent_remove):

            reason_df_column_names = list(reason_df.columns)

            X = np.array(reason_df[model_features])
            model_fit.fit(X,reason_df[y])
            ##print(list(vwoe[vwoe['Type']=='character']['Variable'].unique()))
            cat_cols=list(vwoe[vwoe['Type']=='character']['Variable'].unique())
            cf=list()
            for i in range(len(test_ds.columns)):
                if((list(test_ds.columns)[i]) in cat_cols ):
                    cf.append(i)
            explainer = lime.lime_tabular.LimeTabularExplainer(X, feature_names=model_features,verbose=True,categorical_names=cat_cols,categorical_features=cf, mode=mode1,random_state=12345)
            test_ds=test_ds.set_index(ID)
            test_ds=test_ds[model_features]
            num_features1=len(list(set(model_features)))
            count=0
            temp1=pd.DataFrame()
            for i in (range(len(test_ds))):
                count=count+1
    #             inside_df = pd.DataFrame(train_data[model_features].values[i]).T
    #             inside_df.columns = train_data[model_features].columns
                np.random.seed(0)
                if(mode1=='classification'):
                    exp = explainer.explain_instance(test_ds[model_features].values[i],model_fit.predict_proba ,num_features=num_features1)
                else:
                    exp = explainer.explain_instance(test_ds[model_features].values[i],model_fit.predict ,num_features=num_features1)
                temp = pd.DataFrame(exp.as_list(),columns=['Var','Coef'])
                temp['Values'] = temp['Var'].apply(lambda x:x.split()[-1])
                temp['row_number'] = test_ds.index.values[i]
                temp = temp.set_index('row_number')
                temp = temp.sort_values(by='Coef',ascending=False)
                if (count == 1) & (i ==1):
                    temp1 = temp.copy()
                else:
                    temp1 = temp1.append(temp)
                    count=count+1
            imp=list()
            for i in range(len(temp1)):
                if ('=' in list(temp1['Values'])[i]):
                   imp.append(re.findall(r'=(.*)',list(temp1['Values'])[i])[0])
                else:
                    imp.append(list(temp1['Values'])[i])

            temp1['Values']=imp
            temp1['Values'] = temp1['Values'].astype(str).astype(float)
            if(absent_remove==True):
                temp2 = temp1[(~((temp1.Coef==0 )|(temp1['Coef'].isin([np.nan,'None','NULL','none','null'])))&(temp1['Values']>0))]
            else:
                temp2 = temp1[~((temp1.Coef==0 )|(temp1['Coef'].isin([np.nan,'None','NULL','none','null'])))]
            score_row_v2 = temp2.copy()
            score_row_v2=score_row_v2.reset_index().reset_index()
            score_row_v2 = score_row_v2[['row_number','Coef','Var']].drop_duplicates()
            score_row_v3 = score_row_v2.sort_values(['row_number','Coef'],ascending = False)
            score_row_v3=score_row_v3.groupby(['row_number'])['Var'].apply(lambda x:list(x)).reset_index()
            kk = pd.DataFrame(score_row_v3.Var.values.tolist(), index= score_row_v3.index)
            mm =list(range(len(kk.columns)))
            kk.columns = ["reason_code_model"+str(i+1) for i in mm]
            final_reason_code_model = pd.concat([score_row_v3[['row_number']],kk],axis=1)
            for i in range(1,final_reason_code_model.shape[1]):
                final_reason_code_model['reason_code_model'+str(i)]=final_reason_code_model['reason_code_model'+str(i)].str.lstrip().str.rstrip()
            rep1=final_reason_code_model.iloc[:,0:topn+1]
    #         #print(rep1.head())

            # rep1.to_csv("REP_1.csv",index=False)

            def get_column_name(x,reason_df_column_names):
                for column in reason_df_column_names:
                    if column in x:
                        return column
                return x

            for i in rep1.columns:
                #print(i)
                if(i!='row_number'):

                    rep1[i] = rep1[i].apply(lambda x : get_column_name(x,reason_df_column_names))
                    #rep1[i]=rep1[i].apply(lambda x: (re.findall(r'(.*)[\>\<\=]',x)[0]).strip() if len(re.findall(r'[><]?(.*)[><=]',x))==0 else (re.findall(r'[><]?(.*)[><=]',x)[0]).strip())
                    # rep1[i]=rep1[i].apply(lambda x: (re.sub(r'>|<|=',' ',x)))
                    # rep1[i]=rep1[i].apply(lambda x: (re.sub(r'[0-9]+\.[0-9]+','',x)))
                    # rep1[i]=rep1[i].apply(lambda x: (re.sub(r'\s[0-9]+','',x)).strip())
            
            # rep1.to_csv("REP_1_changed.csv",index=False)

            l=['top_'+str(i+1) for i in range((rep1.shape[1])-1)]
            l.insert(0,ID)
    #         #print(rep1.head())
            rep1.columns=l
            return(rep1)
        
        if((algo=='all')|(algo=='lime')):
            lime_out=lime_y(train_data.copy(),model,features,mode_lime,test_data.copy(),ID,vwoe,absent_remove)
            lime_out['model']='LIME'
            
        def shapy(score_data,test_data,model,topn,ID,features,absent_remove):
            explainer = shap.TreeExplainer(model,random_state=12345)
            score_data1=score_data.copy()        
            cat_cols=list(vwoe[vwoe['Type']=='character']['Variable'].unique())
            score_data1.drop(ID,inplace=True,axis=1)
            test_data1=test_data.copy()
            test_data1.drop(ID,inplace=True,axis=1)
    #         #print(score_data1.columns)
    #         explainer = shap.KernelExplainer(model.predict, score_data[features])        
    #         #print(aa)
            if(mode_lime=='classification'):
                shap_values = explainer.shap_values(test_data[features])
                # separate condition for Random Forest - pick any one is fine
                if len(shap_values)!=len(test_data):
                    shap_values = shap_values[0]
                
                temp=pd.DataFrame(shap_values,columns=test_data[features].columns)    

            else:
                shap_values = explainer.shap_values(test_data[features])
    #             #print(len(shap_values)
                ##print(shap)
                temp=pd.DataFrame(shap_values,columns=test_data[features].columns)
            
            # make each value mod
            temp = temp.apply(lambda x : abs(x))

            temp[ID]=list(test_data[ID])
            cols=list(temp.columns)
            cols.remove(ID)
            melted=pd.melt(temp, id_vars =[ID], value_vars = cols)
            melted = melted.sort_values(by='value',ascending=False)
    #         #print('yayyy')
            if(absent_remove==True):
                data_melt=pd.melt(test_data, id_vars =[ID], value_vars = cols)
                data_melt.drop_duplicates(inplace=True)
                data_melt.columns=[ID,'variable','original_value']
                melted1=pd.merge(melted,data_melt,on=[ID,'variable'])    
                melted1['remove_flag']=np.where((melted1['variable'].isin(cat_cols))&(melted1['original_value'].isin([0,np.nan,'None','NULL','none','null'])),1,0)
                melted=melted1[melted1['remove_flag']==0]
            
    #         #print(temp[ID])
            g = melted.groupby(ID)
            melted['top'] = g['value'].cumcount()+1
            melted=melted[melted['top']<=topn].reset_index(drop=True)
            original_df = melted.pivot(index=ID, columns='top')['variable'].reset_index()
            l=['top_'+str(i+1) for i in range((original_df.shape[1])-1)]
            l.insert(0,ID)
            original_df.columns=l
            return(original_df)
    #     #print(test_data.head())
        if((algo=='all')|(algo=='shap')):
            shaps_out=shapy(train_data.copy(),test_data.copy(),model,topn,ID,features,absent_remove)
            shaps_out['model']='SHAP'
        
        if((mode_lime=='classification')):
            # CHANGED!! - removed np.array()
            s=model.predict_proba(np.array(test_data[features]))
            # s=model.predict_proba(test_data[features])
            
            s=[i[1] for i in s]
            t=pd.DataFrame(list(s),columns=['predicted_prob'])
            #print(test_data[ID],t.head())
            t[ID]=test_data[ID]
            t[y]=test_data[y]
        else:
            s=model.predict(np.array(test_data[features]))
            t=pd.DataFrame(list(s),columns=['predicted_prob'])
            t[ID]=test_data[ID]
            t[y]=test_data[y]


        if((algo=='all')):
            merge1=v_woe_out.append(lime_out)
            merge2=merge1.append(shaps_out)
            merge2=pd.merge(merge2,t,on=ID,how='left')
            
        elif(algo=='vwoe'):
            merge2=pd.merge(v_woe_out,t,on=ID,how='left')
        elif(algo=='lime'):
            merge2=pd.merge(lime_out,t,on=ID,how='left')
        else:
            merge2=pd.merge(shaps_out,t,on=ID,how='left')

        # Merge 2 - reorder
        merge2 = merge2.sort_values(by=[ID,'model'], ascending=[True,True])


        # Add code here to save the formatted excel file
        # ----------------------------------------------

        ef.reset_header_default()
        filename = 'Model_Interpretation'
        filename = ef.create_version(base_filename=filename, path = os.path.join(os.getcwd(),Reason_code.__save_directory))
        filename += '.xlsx'
        
        writer = pd.ExcelWriter(os.path.join(os.getcwd(),Reason_code.__save_directory,filename), engine='xlsxwriter')
        workbook  = writer.book
        
        # FORMATS--------------------------------------------------------------------------
        
        
        format_cell_highlight1 = workbook.add_format(Reason_code.__cell_highlight1)        
        
        format_table_header = workbook.add_format(Reason_code.__table_header)
        
        format_cell_even = workbook.add_format(Reason_code.__cell_even)
        format_cell_odd = workbook.add_format(Reason_code.__cell_odd)
        
        # SHEET: Model_Interpretation--------------------------------------------------------------------------
        
        merge2.to_excel(writer, sheet_name='Model_Interpretation',index=False,startrow=4,startcol=1) 
    
        worksheet = writer.sheets['Model_Interpretation']
        worksheet.hide_gridlines(2)
        
        # applying formatting
            
        # table header
        worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 5,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = 5,column_number = len(merge2)+2,fix_row=True),
                                     {'type': 'no_blanks','format': format_table_header})
        
        # logo
        worksheet.insert_image(ef.generate_excel_cell_name(row_number=2,column_number=2), os.path.join(os.getcwd(),'src','ks_logo.png'))

        # table cells formating - alternate row colours
        if len(merge2)>0:            
            
            # table cells                       
            rows = list(range(5,len(merge2)+6))          
            
            for row in rows:
                worksheet.conditional_format(ef.generate_excel_cell_name(row_number = row,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = row,column_number = len(merge2)+2,fix_row=True),
                                             {'type': 'no_blanks','format': (format_cell_even if row%2==0 else format_cell_odd)})
            
            
            # max_column_width = max([len(x) + 2 for x in merge2['Important Variables']])
            # worksheet.set_column(ef.generate_excel_cell_name(row_number = None,column_number = 3)+':'+ef.generate_excel_cell_name(row_number = None,column_number = 3),max_column_width)
        
        
        writer.save()
        writer.close()        
        # --------------------------------------------------------
        

        # merge2.to_csv("Model_Interpretation_Results/Model_Interpretation.csv",index=False)

        reasoning_dataframe = merge2
        return(reasoning_dataframe)
        
    @classmethod
    def similarity_score(cls,reasoning_dataframe,unique_id,prob_list,topn,no_bins):  
        '''
        This function generates the seperation score for top variables between top and bottom decile. It also generates the top variable distribution.

        Parameters
        ----------
        reasoning_dataframe: dataframe
            It is the output obtained from reasons_code function. The data should contain top variables for all instances of test data with the original Y variable and predicted probability for events. 
        
        unique_id: string
            Name of the unique identifier column in train and test data.
            
        prob_list: list
            List of probabilities obtained from the model for test data.

        topn: integer
            This defines the number of top columns contributing to the prediction.
            
        no_bins: Integer
            This defines the number of bins to be created while calculating seperation score.

        Returns
        -------

        separation_score: dataframe
            Contains the seperation score for top variables between top and bottom decile.
            This dataframe is also stored in the Model_Interpretation_Results folder. (File name : Separation_Score.csv)

        top_var_distribution: dataframe
            Contains the top variable distribution values to the prediction.
            This dataframe is also stored in the Model_Interpretation_Results folder. (File name : Top_Variable_Distribution.csv)

        Example
        -------

        >>> from model_interpretability import Reason_code as r
        >>> separation_score, top_var_distribution = r.similarity_score(reasoning_dataframe,unique_id,prob_list,topn,no_of_bins)


        '''

        all_out = reasoning_dataframe
        ID = unique_id
        
#        if (np.isfinite(reasoning_dataframe).values.all()==False):
#            raise Exception('There may be missing values or inf values present in your data.\nPlease ensure that such values are not present in your input.')


        for_score=all_out.pivot(index=ID, columns='model').reset_index()
        s=[i[0] if i[0]==ID else str(i[0])+'_'+str(i[1]) for i in for_score.columns]
        for_score.columns=s    
        for_score['pred_prob']=prob_list       
        for_score.sort_values(by='pred_prob',inplace=True,ascending=False)
        q=pd.qcut(for_score['pred_prob'],no_bins,retbins=True,duplicates='drop')    
        bin_vals=q[1].tolist()
        bin_vals[0]=bin_vals[0]-0.05
        seen=set()
        bin_vals= [x for x in bin_vals if not (x in seen or seen.add(x))]
        for_score['bucket']=pd.DataFrame(pd.cut(for_score['pred_prob'],bins=bin_vals,labels=False))   
    
        fin=pd.DataFrame()
        lis=list()

        reason_type_list = reasoning_dataframe['model'].unique().tolist()

        for j in reason_type_list:
            tem=pd.DataFrame()
            c=0
            for i in range(topn):  
                top_d=set(for_score[for_score['bucket']==(max(for_score['bucket']))]['top_'+str(i+1)+'_'+j])
                if(c==0):
                    tem=for_score['top_'+str(i+1)+'_'+j].value_counts().reset_index()
                    tem.columns=['variable','top_'+str(i+1)]
                    count = tem['top_'+str(i+1)]
                    tem['top_'+str(i+1)]=tem['top_'+str(i+1)]/len(for_score)
                    tem['model']=j
                    tem['Count'+'_top_'+str(i+1)] = count
                    c=c+1
                else:
                    t=for_score['top_'+str(i+1)+'_'+j].value_counts().reset_index()
                    t.columns=['variable','top_'+str(i+1)]
                    count = t.copy()
                    count.columns=['variable','Count'+'_top_'+str(i+1)]
                    t['top_'+str(i+1)]=t['top_'+str(i+1)]/len(for_score)
                    t['model']=j
    
                    #print(t)
                    tem=pd.merge(tem.copy(),t,on=['variable','model'],how='outer')
                    c=c+1
                    tem=pd.merge(tem,count,on=['variable'],how='left')
                low_d=set(for_score[for_score['bucket']==(min(for_score['bucket']))]['top_'+str(i+1)+'_'+j])
                #print(top_d,low_d)
                diff_num=len(top_d.union(low_d) - top_d.intersection(low_d))
                tot_prod=len(top_d.union(low_d))
                lis.append([j,'top_'+str(i+1),(diff_num/tot_prod)*100,len(top_d),len(low_d),tot_prod])
    
            if(c==1):
                fin=tem
            else:
                fin=fin.append(tem)
        mod=fin.model
        del fin['model']      
        fin.insert(1,'model',mod)
        sep_out=pd.DataFrame(lis,columns=['model','top','seperation_score','Top_Feature_count','Bottom_Feature_count','Total_Feature_Count'])
        fin=fin.fillna(0)

        #---------------------------------------------------  

        # # saving seperation score an dtop decile csv
        # sep_out.to_csv("Model_Interpretation_Results/Separation_Score.csv",index=False)
        # fin.to_csv("Model_Interpretation_Results/Top_Variable_Distribution.csv",index=False)


        # Add code here to save the formatted excel file - Separation_Score
        # -----------------------------------------------------------------

        ef.reset_header_default()
        filename = 'Separation_Score'
        filename = ef.create_version(base_filename=filename, path = os.path.join(os.getcwd(),Reason_code.__save_directory))
        filename += '.xlsx'
        
        writer = pd.ExcelWriter(os.path.join(os.getcwd(),Reason_code.__save_directory,filename), engine='xlsxwriter')
        workbook  = writer.book
        
        # FORMATS--------------------------------------------------------------------------
        
        
        format_highlight1 = workbook.add_format(Reason_code.__cell_highlight1)        
        
        format_table_header = workbook.add_format(Reason_code.__table_header)
        
        format_cell_even = workbook.add_format(Reason_code.__cell_even)
        format_cell_odd = workbook.add_format(Reason_code.__cell_odd)
        format_border2 = workbook.add_format(Reason_code.__border2)
        
        # SHEET: Separation_Score--------------------------------------------------------------------------
        
        sep_out.to_excel(writer, sheet_name='Separation_Score',index=False,startrow=4,startcol=1) 
    
        worksheet = writer.sheets['Separation_Score']
        worksheet.hide_gridlines(2)
        
        # applying formatting
            
        # table header
        worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 5,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = 5,column_number = len(sep_out)+2,fix_row=True),
                                     {'type': 'no_blanks','format': format_table_header})
        
        # logo
        worksheet.insert_image(ef.generate_excel_cell_name(row_number=2,column_number=2), os.path.join(os.getcwd(),'src','ks_logo.png'))
        
        # divider between variable names
        worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 6,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = len(sep_out)+6,column_number = len(sep_out.columns)+1,fix_row=True),
                                      {'type': 'formula',
                                       'criteria': "=" + ef.generate_excel_cell_name(row_number = 7,column_number = 2,fix_column=True)+'<>'+ef.generate_excel_cell_name(row_number = 6,column_number = 2,fix_column=True),
                                       'format': format_border2})
            

        # table cells formating - alternate row colours
        if len(sep_out)>0:            
            
            # table cells                       
            rows = list(range(5,len(sep_out)+6))          
            
            for row in rows:
                worksheet.conditional_format(ef.generate_excel_cell_name(row_number = row,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = row,column_number = len(sep_out)+2,fix_row=True),
                                             {'type': 'no_blanks','format': (format_cell_even if row%2==0 else format_cell_odd)})
            
            
            # max_column_width = max([len(x) + 2 for x in merge2['Important Variables']])
            # worksheet.set_column(ef.generate_excel_cell_name(row_number = None,column_number = 3)+':'+ef.generate_excel_cell_name(row_number = None,column_number = 3),max_column_width)
        
        writer.save()
        writer.close()


        # Add code here to save the formatted excel file - Top_Variable_Distribution
        # -----------------------------------------------------------------

        ef.reset_header_default()
        filename = 'Top_Variable_Distribution'
        filename = ef.create_version(base_filename=filename, path = os.path.join(os.getcwd(),Reason_code.__save_directory))
        filename += '.xlsx'
        
        writer = pd.ExcelWriter(os.path.join(os.getcwd(),Reason_code.__save_directory,filename), engine='xlsxwriter')
        workbook  = writer.book
        
        # FORMATS--------------------------------------------------------------------------
        
        
        format_highlight1 = workbook.add_format(Reason_code.__cell_highlight1)        
        
        format_table_header = workbook.add_format(Reason_code.__table_header)
        
        format_cell_even = workbook.add_format(Reason_code.__cell_even)
        format_cell_odd = workbook.add_format(Reason_code.__cell_odd)
        
        # SHEET: Top_Variable_Distribution--------------------------------------------------------------------------
        
        fin.to_excel(writer, sheet_name='Top_Variable_Distribution',index=False,startrow=4,startcol=1) 
    
        worksheet = writer.sheets['Top_Variable_Distribution']
        worksheet.hide_gridlines(2)
        
        # applying formatting
            
        # table header
        worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 5,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = 5,column_number = len(fin)+2,fix_row=True),
                                     {'type': 'no_blanks','format': format_table_header})
        
        # logo
        worksheet.insert_image(ef.generate_excel_cell_name(row_number=2,column_number=2), os.path.join(os.getcwd(),'src','ks_logo.png'))

        # table cells formating - alternate row colours
        if len(fin)>0:            
            
            # table cells                       
            rows = list(range(5,len(fin)+6))          
            
            for row in rows:
                worksheet.conditional_format(ef.generate_excel_cell_name(row_number = row,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = row,column_number = len(fin)+2,fix_row=True),
                                             {'type': 'no_blanks','format': (format_cell_even if row%2==0 else format_cell_odd)})
            
            
            # max_column_width = max([len(x) + 2 for x in merge2['Important Variables']])
            # worksheet.set_column(ef.generate_excel_cell_name(row_number = None,column_number = 3)+':'+ef.generate_excel_cell_name(row_number = None,column_number = 3),max_column_width)
        
        writer.save()
        writer.close()

        # --------------------------------------------------

        separation_score = sep_out
        top_var_distribution = fin

        return(separation_score,top_var_distribution)

    @classmethod
    def vwoe_generate(cls,train_data,char_cols,target_name,feature_imp,binning):
        '''

        Generates a VWOE file based on weight of evidence of a level and model's feature importance.

        
        Parameters
        ----------
        train_data: dataframe
            X and Y variables in training dataset.

        char_cols: list
            List of character columns used as features in the model

        target_name: string
            Name of the column containing target (y) variable.
            
        feature_imp: dataframe
            Dataframe containing feature importance obtained from model. The dataframe should contain two columns :'importance' and 'feature'.
        
        binning: boolean
            This defines if the numerical variables have been binned or not.

            `True` - If binning has been done for numerical variables.
            
            `False` - If binning has not been done for numerical variables.

            
        Returns
        ----------
        vwoe_dataframe: dataframe
            Contains the feature rank for each level of each feature.

        Example
        -------

        >>> from model_interpretability import Reason_code as r
        >>> vwoe = r.vwoe_generate(train_data,char_cols,target_name,feature_imp,binning)

        '''

        X_Var = train_data
        y = target_name
        
        if (np.isfinite(train_data).values.all()==False):
            raise Exception('There may be missing values or inf values present in your data.\nPlease ensure that such values are not present in your input.')

        Char_Cols = char_cols
        Num_Cols = list(np.setdiff1d(X_Var.columns,char_cols))
        if(y in Char_Cols):
            Char_Cols.remove(y)
        elif(y in Num_Cols):
            Num_Cols.remove(y)
        
        if(len(Num_Cols)>1):
            X_Num  = X_Var.filter(items=Num_Cols).fillna(-999999999).copy()
        if(len(Char_Cols)>1):
            X_Char = X_Var.filter(items=Char_Cols).fillna('MISSING').copy()
            X_Char=X_Char.astype(str)
        
        
        def bining(inti,i):
            q=pd.qcut(inti[i],10,retbins=True,duplicates='drop',labels=False)
            bin_vals=q[1].tolist()
            if(-999999999.0 in inti[i].unique()):
                bin_vals.insert(0,-999999999)
                bin_vals[0]=float(bin_vals[0]-1)
            else:
                bin_vals.insert(0,-999999999)
                bin_vals.insert(0,-999999999)
                bin_vals[0]=float(bin_vals[0]-1)
            seen = set()
            bin_vals= [x for x in bin_vals if not (x in seen or seen.add(x))]
            inti[i]=pd.DataFrame(pd.cut(inti[i],bins=bin_vals))
            #del inti[i]   
        if((binning==False) &(len(Num_Cols)>1)):
            for i in X_Num.columns:
                bining(X_Num,i) 
        
        
        if((len(Num_Cols)>1)&(len(Char_Cols)>1)):
            df=pd.concat([X_Num,X_Char],axis=1)
        elif(len(Num_Cols)>1):
            df=X_Num.copy()
        else:
            df=X_Char.copy()
            
        df1=pd.concat([df,X_Var[y]],axis=1)
        def calc_woe(df, feature, target, pr=False):
            lst = []
            df[feature] = df[feature].fillna("NULL")
            for i in range(df[feature].nunique()):
                val = list(df[feature].unique())[i]
                lst.append([feature,                                                        # Variable
                            val,                                                            # Value
                            df[df[feature] == val].count()[feature],                        # All
                            df[(df[feature] == val) & (df[target] == 0)].count()[feature],  # Good (think: Fraud == 0)
                            df[(df[feature] == val) & (df[target] == 1)].count()[feature]]) # Bad (think: Fraud == 1)
            data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Good', 'Bad'])
            data['Share'] = data['All'] / data['All'].sum()
            data['Bad Rate'] = data['Bad'] / data['All']
            data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
            data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
            data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])
            data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}})
            data = data.sort_values(by=['Variable', 'Value'], ascending=[True, True])
            data.index = range(len(data.index))
            if((feature in (X_Num.columns)) &(binning==False)):
                data['Lower']=data['Value'].apply(lambda x:x.left)
                data['Upper']=data['Value'].apply(lambda x:x.right)
                data['Type']='Numeric'
            else:
                data['Lower']='-'
                data['Upper']='-'
                data['Type']='charcter'
            return data
        
        d=pd.DataFrame(columns=['Variable','Value','All','Good','Bad','Share','Bad Rate','Distribution Good','Distribution Bad','WoE','Lower','Upper','Type'])
        
        for i in df1.columns:
            if i!=y:
                data = calc_woe(df1, i,y)
                d=d.append(data,sort=False)
            #p=p.append(pd.DataFrame({'variable':[i],'iv':[iv]}),sort=False)
        
        ################## getting feature importance 
        woe_file=d[['Variable','Value','WoE','Lower','Upper','Type']]
        feature_imp=feature_imp[['feature','importance']]
        
        V_WOE=woe_file.merge(feature_imp,left_on='Variable',right_on='feature',how='left')
        V_WOE['Featur_rank']=V_WOE['WoE']*V_WOE['importance']
        
        final_VWOE=V_WOE[['Variable','Value','Featur_rank','Lower','Upper','Type']]
        
        vwoe_dataframe = final_VWOE
        return(vwoe_dataframe)