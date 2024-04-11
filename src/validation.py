"""
    
    **Project**
    -----------
    Model Validation
    
    **Description**
    ---------------
    This class contains functions for : 
    
    - KS 
    - PSI
    - CSI
    - Concordance Discordance
    
    
    ###**KS**
    The *calculate_ks* function calculates the KS for train set (by dividing data into equal deciles) and 
    test set (by dividing data into equal deciles or by dividing based on train score cutoff).
    
    The *KS statistic* is used as a measure of the ability of the model to separate good and bad accounts. 
    To calculate KS manually, each dataset is divided into 10 groups (deciles). Since the score values occur in discrete intervals, each decile contains approximately 10% of
    the dataset. Once the deciles are obtained, the cumulative percentage of defaults and nondefaults is calculated across the 10 deciles. The KS is the maximum difference between the
    cumulative percentage of defaults and non-defaults. 
    
    The point where the difference is between the cumulative percentage of the defaults and non-defaults is the largest is the value of the KS statistic.     
    It is important to see from the figure above that the shape of the KS curves for both the development and validation datasets have approximately the same shape, 
    indicating that the maximal separation occurs around the same point in the second decile. 
    
    ###**Population Stability Index**
    The *calculate_psi* function checks the PSI stability. 
    
    *Population Stability Index or PSI* quantifies shifts in population dynamics over time. As models are based on historical datasets, 
    it is necessary to ensure that present-day population features are sufficiently similar to
    the historical population from which the model is based. A higher PSI corresponds to greater
    shifts in population. Generally, PSI values greater than 0.25 indicate a significant shift in the population, 
    while values less than 0.10 indicate a minimal shift in the population. Values between 0.10 and 0.25 indicate a minor shift
        
    ###**Characteristic Stability Index**
    The *calculate_csi* function performs the Characteristic Stability Index calculation for all the binned variables of the data.
    
    *Characteristic Stability Index or CSI* is the measure of the change in the distribution of a variable between the development and recent data. 
    
    ###**Concordance Discordance**
    The *concordance_discordance* function calculates *concordant%, discordant% and tied%*.
    
    - Concordant % : Percentage of pairs where the observation with the desired outcome (event) has a higher predicted probability than the observation without the outcome (non-event).
    - Discordant % : Percentage of pairs where the observation with the desired outcome (event) has a lower predicted probability than the observation without the outcome (non-event).    
    - Tied % : Percentage of pairs where the observation with the desired outcome (event) has same predicted probability than the observation without the outcome (non-event).
        
    
    
    Please refer to each class and the corresponding methods to understand their functionality in detail.
    
    **Dependencies**
    ----------------
        Install the following python libraries: 
        
        numpy==1.19.0
        pandas==1.0.3
        packaging==20.4

    **Additional Files**
    --------------------
        
        1. excel_formatting.py 
    
    Please ensure that the above files are also present in the same folder and packages mentioned in the dependencies section are installed.
        
    **Authors**
    -----------
    Created on Fri Jun  12 18:28:27 2020 
    
    @author: Kumar, Shubham
    
"""
import numpy as np
import os
import pandas as pd
from .excel_formatting import excel_formatting as ef
import packaging.version
import warnings
warnings.filterwarnings("ignore")
        
class validation:
    """
    This class contains functions to validate the model efficiency and perform stability checks.


    """
    
    __save_directory = 'results/Model Validation'
    if os.path.isdir("results")==False:
        os.mkdir("results")
    
    if os.path.isdir(__save_directory)==False:
        os.mkdir(__save_directory)
    
    # FORMATS--------------------------------------------------------------------------
            
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
    def __get_population_csi(cls,y_train_probability,column_name):

        y_train_probability['index'] = y_train_probability.index

        grouped = y_train_probability.groupby(column_name,as_index=False)
        #print("\n\n",grouped.count())
        gdf = pd.DataFrame()
        gdf[column_name] = grouped.groups.keys()
        gdf['Count'] = grouped.count()['index']
        gdf['Pop %'] = gdf['Count'] / y_train_probability.shape[0]
        gdf['Pop %'].replace(0,0.0001,inplace=True)

        return gdf
    
    @classmethod
    def __reset_header_default(cls):
        version = packaging.version.parse(pd.__version__)
        if version < packaging.version.parse('0.18'):
            pd.pandas.core.format = None
        elif version < packaging.version.parse('0.20'):
            pd.pandas.formats.format = None
        else:
            pd.pandas.io.formats.excel = None
            
    @classmethod
    def __make_bin_tuples(cls,x1, x2):
        
        """This function returns tuples with non equal elements"""
        
        if x1!=x2:
            return (x1,x2)
        
    @classmethod
    def __make_ks_table(cls,
                     y_train_probability,
                     y_test_probability,
                     column_to_groupby,
                     table_type="Train",
                     cut_type="Score"):
        
        """This function returns decile or percentile ks chart
        
        y_train_probability: dataframe (with 3 columns "Target","Probability")
                             "Target","Probability" containing the y value and proba score respectively.
                             
        y_test_probability: dataframe (with 2 columns "Target","Probability") 
                            "Target","Probability" containing the y value and proba score respectively.
        
        column_to_groupby: string
            Accepted values = {'Decile','Percentile'}
        
        table_type: string, default='Train'
            Accepted values = {'Test','Train'}
            
        cut_type: string, default='Score'
            Accepted values = {'Test','Train'}
            
        
        """
    
        if table_type == 'Train':
            y_probability = y_train_probability.copy()
        if table_type == 'Test':
            y_probability = y_test_probability.copy()
        
        KS = pd.DataFrame()
        grouped_decile_wise = y_probability.groupby(column_to_groupby,as_index=False)
        
        if type(list(grouped_decile_wise.groups.keys())[0]) == type(0):
            KS['MaxScore'] = grouped_decile_wise.max()['Probability']
            KS['MaxScore'].iloc[0] = 1.0
            KS['MinScore'] = KS['MaxScore'].shift(-1)
            KS['MinScore'].fillna(0.0,inplace=True)
        else:
            idx = pd.IntervalIndex(list(grouped_decile_wise.groups.keys()))
            KS = pd.DataFrame({'MaxScore': idx.right,'MinScore': idx.left})
    
        KS['ScoreRange'] = KS['MinScore'].round(decimals=2).astype(str) + '-' + KS['MaxScore'].round(decimals=2).astype(str)
        
        KS['Total'] = grouped_decile_wise.count()['Target']
        KS['Event'] = grouped_decile_wise.sum()['Target']
        KS['NonEvent'] = KS['Total'] - KS['Event']
        
        KS['BadRate'] = KS['Event'] / KS['Total']
        
        KS['Population%'] = KS['Total']*100 / KS['Total'].sum()
        KS['Cum_Population%'] = KS['Population%'].cumsum()
        
        KS['Cum_Event'] = KS['Event'].cumsum()
        KS['Cum_NonEvent'] = KS['NonEvent'].cumsum()
        
        KS['Cum_Event%'] = KS['Cum_Event']*100 / KS['Event'].sum()
        KS['Cum_NonEvent%'] = KS['Cum_NonEvent']*100 / KS['NonEvent'].sum()
        
        KS['KS'] = KS['Cum_Event%'] - KS['Cum_NonEvent%']
        overall_bad_rate = KS['Event'].sum() / KS['Total'].sum()
        KS['Lift'] = KS['BadRate'] / overall_bad_rate
        KS['BadRate'] = KS['BadRate']*100
        if table_type == 'Test' and  column_to_groupby=='Decile':
            if cut_type =='Score':
                KS['PSI'] = cls.calculate_psi(y_train_probability[['Target','Probability']],
                                                  y_test_probability[['Target','Probability']],
                                                  bin_type=column_to_groupby)[1]
#            else:
#                try:
#                    KS['PSI'] = cls.calculate_psi(y_train_probability[['Target','Probability']],
#                                                  y_test_probability[['Target','Probability']],
#                                                  bin_type=column_to_groupby)[1]
#                except:
#                    print(cut_type)
        
        KS.index = KS.index + 1
        
        return KS
    
    
    @classmethod
    def calculate_psi(cls,
                      y_train_probability,
                      y_test_probability,
                      bin_type='Decile'):
        """
        This function checks the PSI stability.

        Parameters
        ----------
        
        y_train_probability: pandas dataframe
            Dataframe with 2 columns "Target","Probability". "Target","Probability" containing the y value and proba score respectively.

        y_test_probability: pandas dataframe
            Dataframe with 2 columns "Target","Probability". "Target","Probability" containing the y value and proba score respectively.
        
        bin_type: string, default='Decile'
            Accepted values = {'Decile','Percentile'}

        Returns
        -------

        psi_score: float
            PSI score (summation of psi values of deciles).

        decile_wise_psi_values: list
            PSI value for each decile level.

        """
        
        if (np.isfinite(y_train_probability).values.all()==False) or (np.isfinite(y_test_probability).values.all()==False):
            raise Exception('There may be missing values or inf values present in your data.\nPlease ensure that such values are not present in your input.')
            

        if bin_type=='Decile':
            no_of_bins = 10
        elif bin_type=='Percentile':
            no_of_bins = 100
        else:
            raise Exception('bin_type parameter value is invalid. \nPlease refer to the documentation for accepted values.')
        
        # calculation of decile - equal cuts - train
        # ==========================================
        #y_train_probability.reset_index(inplace=True)
        y_train_probability.sort_values(by=['Probability'],ascending=False,inplace=True)
        y_train_probability.reset_index(drop=True,inplace=True)
        qcut = pd.qcut(y_train_probability.index,no_of_bins,retbins=True,duplicates='drop',labels=False)
        y_train_probability[bin_type]=qcut[0]
        
        

        grouped = y_train_probability.groupby(bin_type,as_index=False)

        KSTrain = pd.DataFrame()
        KSTrain[bin_type] = grouped.groups.keys()
        KSTrain['Count'] = grouped.count().Target
        KSTrain['Pop %'] = KSTrain['Count'] / y_train_probability.shape[0]
        KSTrain['Pop %'].replace(0,0.0001,inplace=True)
        
        KSTrain['MaxScore'] = grouped.max()['Probability']
        KSTrain['MaxScore'].iloc[0] = 1.0
        KSTrain['MinScore'] = KSTrain['MaxScore'].shift(-1)
        KSTrain['MinScore'].fillna(0.0,inplace=True)
        

        pop_df_train = KSTrain.copy()
        #print(KSTrain[['MinScore','MaxScore']])


        # calculation of decile - score cuts - test
        # ==========================================
        y_test_probability.reset_index(inplace=True)
        y_test_probability.sort_values(by=['Probability'],ascending=False,inplace=True)
        y_test_probability.reset_index(drop=True,inplace=True)
        
        # cannot have repeating intervals
        bins = KSTrain[['MinScore','MaxScore']].apply(lambda x: cls.__make_bin_tuples(x['MinScore'],x['MaxScore']), axis=1).tolist()
        # to remove the None elements
        bins = list(filter(None.__ne__, bins))
        bins[0] = (bins[0][0],1.00)
        bins[-1] = (0.00,bins[-1][1])
        bins = pd.IntervalIndex.from_tuples(bins)
        y_test_probability[bin_type]=pd.cut(y_test_probability.Probability,bins = bins,retbins=True,
                                            duplicates='drop',labels=False,include_lowest=True)[0]
        
        KSTest = pd.DataFrame()
        grouped = y_test_probability.groupby(bin_type,as_index=False)
        
        KSTest[bin_type] = grouped.groups.keys()
        KSTest['Count'] = grouped.count().Target
        KSTest['Pop %'] = KSTest['Count'] / y_test_probability.shape[0]
        KSTest['Pop %'].replace(0,0.0001,inplace=True)

        pop_df_test = KSTest.copy()

        decile_wise_psi_values = list(map(lambda e_perc,a_perc : (e_perc - a_perc) * np.log(e_perc / a_perc) ,pop_df_train['Pop %'],pop_df_test['Pop %']))
        
        decile_wise_psi_values = [value*100 for value in decile_wise_psi_values]

        psi_score = sum(decile_wise_psi_values)
        #print(KSTest)

        #return psi_score, decile_wise_psi_values        
        return psi_score,decile_wise_psi_values

    @classmethod
    def calculate_ks(cls,
                     y_train_probability,
                     y_test_probability,
                     iteration_number = None,
                     test_calculation_method = 'ScoreCutOff',
                     save_charts = True, include_plots=True,
                     save_folder = '',
                     path = ''):
        
        """
        This function calculates the KS for train set (by dividing data into equal deciles) and 
        test set (by dividing data into equal deciles or by dividing based on train score cutoff).
        
        Please note thet by default this function creates a csv for every iteration and hence may be time consuming.
        Set the save_charts flag to False to disable the save feature. 
        
        Make sure that folder specified already exists if save_folder parameter is passed.
        
        Parameters
        ----------
        
        y_train_probability: dataframe (with 2 columns "Target","Probability")
                             "Target","Probability" containing the y value and proba score respectively.
                             
        y_test_probability: dataframe (with 2 columns "Target","Probability") 
                            "Target","Probability" containing the y value and proba score respectively.
                            
        iteration_number: integer, default=1
                          grid search iteration number
                          
        test_calculation_method: string
                                 Possible values: {"ScoreCutOff", "EqualDeciles"}
                                 default="ScoreCutOff"
                                 Whether to divide test set into equal size deciles or divide into deciles using score cuts.     
        save_charts: boolean
                     default = True
                     True, to save KS charts.
        
        include_plots: boolean
                       default = True
                       True, to save KS Lift plot for every iteration. 
                       If True, save_charts cannot be False.
                       
        save_folder: string
                     default = ''
                     Folder name where the output files must be saved
                     By default will save files in the same location as the code.
                     
        Returns
        -------
        
        KS_train_decile: dataframe 
            Contains decile wise KS chart for train dataset.
        
        KS_test_decile: dataframe
                 Contains decile wise KS chart for test dataset.
        
        """
        
        if (np.isfinite(y_train_probability).values.all()==False) or (np.isfinite(y_test_probability).values.all()==False):
            raise Exception('There may be missing values or inf values present in your data.\nPlease ensure that such values are not present in your input.')
            
        # reset index and save an original version of y_train_probability and y_test_probability
        y_train_probability.reset_index(drop=True, inplace=True)
        y_test_probability.reset_index(drop=True, inplace=True)
        
        y_train_probability_original = y_train_probability.copy()
        y_test_probability_original = y_test_probability.copy()
        
        # if the code is run only for KS charts check version, else make gridsearch iteration file
        if iteration_number == None:
            filename = "KS_Tables"
            filename = ef.create_version(base_filename=filename, path = os.path.join(os.getcwd(),validation.__save_directory))
        else:
            filename = "Iteration_"+str(iteration_number)
            
            
        
        # KS for Train set
        # Decile Chart
        
        y_train_probability.sort_values(by = ['Probability'], ascending=False, inplace=True)
        y_train_probability.reset_index(drop=True, inplace=True)
        qcut = pd.qcut(y_train_probability.index, 10, retbins=True, duplicates='drop', labels=False)
        y_train_probability['Decile']=qcut[0]
        KS_train_decile = cls.__make_ks_table(y_train_probability,y_test_probability,'Decile')
        #print(KS_train_decile[['MaxScore','MinScore']])
               
        
        # KS for Test set        
        # Decile Chart
        # Divide data into equal population deciles
        y_train_probability = y_train_probability_original.copy()
        y_test_probability = y_test_probability_original.copy()
        
        y_test_probability.sort_values(by = ['Probability'], ascending=False, inplace=True)
        y_test_probability.reset_index(drop=True, inplace=True)
        
        qcut = pd.qcut(y_test_probability.index, 10, retbins=True, duplicates='drop', labels=False)
        y_test_probability['Decile'] = qcut[0]
        KS_test_decile_equalcuts = cls.__make_ks_table(y_train_probability,y_test_probability,'Decile',table_type='Test',cut_type='Equal')
        #KS_test_decile_equalcuts.drop(columns=['PSI'],inplace=True)
        
        
        # Divide data into deciles based on train score cutoffs
        y_train_probability = y_train_probability_original.copy()
        y_test_probability = y_test_probability_original.copy()
        
        y_test_probability.sort_values(by=['Probability'], ascending=False, inplace=True)
        y_test_probability.reset_index(drop=True, inplace=True)
        # cannot have repeating intervals
        bins = KS_train_decile[['MinScore','MaxScore']].apply(lambda x: cls.__make_bin_tuples(x['MinScore'],x['MaxScore']), axis=1).tolist()
        # to remove the None elements
        bins = list(filter(None.__ne__, bins))
        bins[0] = (bins[0][0],1.00)
        bins[-1] = (0.00,bins[-1][1])
        bins = pd.IntervalIndex.from_tuples(bins)
        y_test_probability['Decile']=pd.cut(y_test_probability['Probability'],bins = bins,retbins=True,
                                             duplicates='drop',labels=False,include_lowest=True)[0]
        KS_test_decile_scorecutoff = cls.__make_ks_table(y_train_probability,y_test_probability,'Decile',table_type='Test')            
        
        
        if save_charts == True:
            
            # KS for Train set
            # Percentile Chart
            y_train_probability = y_train_probability_original.copy()
            y_test_probability = y_test_probability_original.copy()
            y_train_probability.sort_values(by = ['Probability'], ascending=False, inplace=True)
            y_train_probability.reset_index(drop=True, inplace=True)
            qcut = pd.qcut(y_train_probability.index, 100, retbins=True, duplicates='drop', labels=False)
            y_train_probability['Percentile']=qcut[0]
            KS_train_percentile = cls.__make_ks_table(y_train_probability,y_test_probability,'Percentile')
            
            # KS for Test set        
            # Percentile Chart
            # Divide data into equal population percentiles
            y_train_probability = y_train_probability_original.copy()
            y_test_probability = y_test_probability_original.copy()
            y_train_probability.sort_values(by = ['Probability'], ascending=False, inplace=True)
            y_train_probability.reset_index(drop=True, inplace=True)
            qcut = pd.qcut(y_test_probability.index, 100, retbins=True, duplicates='drop', labels=False)
            y_test_probability['Percentile'] = qcut[0]
            KS_test_percentile_equalcuts = cls.__make_ks_table(y_train_probability,y_test_probability,'Percentile',table_type='Test',cut_type='Equal')
            
            # Divide data into percentiles based on train score cutoffs
            y_train_probability = y_train_probability_original.copy()
            y_test_probability = y_test_probability_original.copy()
            y_test_probability.sort_values(by=['Probability'], ascending=False, inplace=True)
            y_test_probability.reset_index(drop=True, inplace=True)
            # cannot have repeating intervals
            bins = KS_train_percentile[['MinScore','MaxScore']].apply(lambda x: cls.__make_bin_tuples(x['MinScore'],x['MaxScore']), axis=1).tolist()
            # to remove the None elements
            bins = list(filter(None.__ne__, bins))
            bins[0] = (bins[0][0],1.00)
            bins[-1] = (0.00,bins[-1][1])
            bins = pd.IntervalIndex.from_tuples(bins)
            y_test_probability['Percentile']=pd.cut(y_test_probability['Probability'],bins = bins,retbins=True,
                                                 duplicates='drop',labels=False,include_lowest=True)[0]
            KS_test_percentile_scorecutoff = cls.__make_ks_table(y_train_probability,y_test_probability,'Percentile',table_type='Test')     
            

            
            ef.reset_header_default()
            if iteration_number == None:
                writer = pd.ExcelWriter(os.path.join(os.getcwd(),
                                                 validation.__save_directory,
                                                 save_folder,
                                                 filename +".xlsx"), engine='xlsxwriter')               
            else:
                writer = pd.ExcelWriter(os.path.join(os.getcwd(),
                                                 "results/Model Tuning",
                                                 save_folder,
                                                 filename +".xlsx"), engine='xlsxwriter')  
            
            
            workbook  = writer.book
            
            # FORMATS--------------------------------------------------------------------------
            
            
            format_round0 = workbook.add_format(validation.__round0)
#            format_round1 = workbook.add_format(validation.__round1)
            format_round2 = workbook.add_format(validation.__round2)
            format_round3 = workbook.add_format(validation.__round3)
            format_percsign = workbook.add_format(validation.__percsign)
#            format_percsignmultipled = workbook.add_format(validation.__percsignmultipled)
            format_cell_highlight1 = workbook.add_format(validation.__cell_highlight1)
            format_cell_highlight2 = workbook.add_format(validation.__cell_highlight2)
#            format_cell_highlight3 = workbook.add_format(validation.__cell_highlight3)
#            format_cell_highlight4 = workbook.add_format(validation.__cell_highlight4)
#            format_cell_highlight5 = workbook.add_format(validation.__cell_highlight5)
#            format_cell_highlight6 = workbook.add_format(validation.__cell_highlight6)
            format_table_header= workbook.add_format(validation.__table_header)
            format_table_title = workbook.add_format(validation.__table_title)
            format_cell_even = workbook.add_format(validation.__cell_even)
            format_cell_odd = workbook.add_format(validation.__cell_odd)
            format_align_centre = workbook.add_format(validation.__align_centre)
            format_cell_anomaly_bad = workbook.add_format(validation.__cell_anomaly_bad)
#            format_cell_anomaly_good = workbook.add_format(validation.__cell_anomaly_good)
#            format_border1 = workbook.add_format(validation.__border1)
#            format_border2 = workbook.add_format(validation.__border2)

            
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            
            # SHEET: DecileTables--------------------------------------------------------------------------
            
            
            KS_train_decile.to_excel(writer, sheet_name='DecileTables',index_label='Decile',startrow=4,startcol=1) 
            KS_test_decile_equalcuts.to_excel(writer, sheet_name='DecileTables',index_label='Decile',startrow=17,startcol=1) 
            KS_test_decile_scorecutoff.to_excel(writer, sheet_name='DecileTables',index_label='Decile',startrow=30,startcol=1)
            worksheet = writer.sheets['DecileTables']
            worksheet.hide_gridlines(2)
            
            worksheet.merge_range(3,1,3,5, 'Train KS: Equal Decile Cuts', cell_format=format_table_title)
            worksheet.merge_range(16,1,16,5,'Test KS: Equal Deciles Cuts', cell_format=format_table_title)
            worksheet.merge_range(29,1,29,5,'Test KS: Score Cuts', cell_format=format_table_title)
            
            # applying formatting
            
            max_column_number = len(KS_test_decile_scorecutoff.columns)+1
            # table header
            worksheet.conditional_format('B$5:'+alphabet[max_column_number]+'$5', {'type': 'no_blanks','format': format_table_header})
            worksheet.conditional_format('B$18:'+alphabet[max_column_number]+'$18', {'type': 'no_blanks','format': format_table_header})
            worksheet.conditional_format('B$31:'+alphabet[max_column_number]+'$31', {'type': 'no_blanks','format': format_table_header})
            
            # logo
            worksheet.insert_image('B1', os.path.join(os.getcwd(),'src','ks_logo.png'))
            
            
            
            # conditional formatting to highlight badrate break 
            # NOTE: Doing this before even odd row formatting because formatting cannnot be overwritten
            worksheet.conditional_format('I7:I15',
                                         {'type':     'formula',
                                          'criteria': '=$I7>$I6',
                                          'format':   format_cell_anomaly_bad})
    
            worksheet.conditional_format('I20:I28',
                                         {'type':     'formula',
                                          'criteria': '=$I20>$I19',
                                          'format':   format_cell_anomaly_bad})
            worksheet.conditional_format('I33:I41',
                                         {'type':     'formula',
                                          'criteria': '=$I33>$I32',
                                          'format':   format_cell_anomaly_bad})
    
            # table cells                       
            rows = list(range(6,16))+list(range(19,29))+list(range(32,42))          
            
            for row in rows:
                worksheet.conditional_format('C$'+str(row)+':O$'+str(row), {'type': 'no_blanks','format': (format_cell_even if row%2==0 else format_cell_odd)})
                worksheet.conditional_format('Q$'+str(row)+':'+alphabet[max_column_number]+'$'+str(row), {'type': 'no_blanks','format': (format_cell_even if row%2==0 else format_cell_odd)})
                
            
            worksheet.set_column('C:D', 10, format_round3)
            worksheet.set_column('E:E', 13,format_align_centre)
            worksheet.set_column('F:G', 9)
            worksheet.set_column('H:H', 11)
            worksheet.set_column('I:I', 12, format_percsign)
            worksheet.set_column('J:J', 13, format_percsign)
            worksheet.set_column('K:K', 17, format_percsign)
            worksheet.set_column('L:M', 14, format_round0)
            worksheet.set_column('N:O', 16, format_percsign)
            worksheet.set_column('P:Q', 8, format_round2)
            worksheet.set_column('R:R', 8, format_percsign)
            
            worksheet.conditional_format('P6:P15', {'type': 'no_blanks','format': format_cell_highlight2})
            worksheet.conditional_format('P19:P28', {'type': 'no_blanks','format': format_cell_highlight2})
            worksheet.conditional_format('P32:P41', {'type': 'no_blanks','format': format_cell_highlight2})
            
            worksheet.conditional_format('B6:B15', {'type': 'no_blanks','format': format_cell_highlight1})
            worksheet.conditional_format('B19:B28', {'type': 'no_blanks','format': format_cell_highlight1})
            worksheet.conditional_format('B32:B41', {'type': 'no_blanks','format': format_cell_highlight1})
    
           
            # SHEET: PercentileTables--------------------------------------------------------------------------
            KS_train_percentile.to_excel(writer, sheet_name='PercentileTables',index_label='Percentile',startrow=4,startcol=1) 
            KS_test_percentile_equalcuts.to_excel(writer, sheet_name='PercentileTables',index_label='Percentile',startrow=107,startcol=1)  
            KS_test_percentile_scorecutoff.to_excel(writer, sheet_name='PercentileTables',index_label='Percentile',startrow=210,startcol=1)  
            worksheet = writer.sheets['PercentileTables']
            
            worksheet.hide_gridlines(2)
            
            worksheet.merge_range(3,1,3,5, 'Train KS: Equal Decile Cuts', cell_format=format_table_title)
            worksheet.merge_range(106,1,106,5,'Test KS: Equal Deciles Cuts', cell_format=format_table_title)
            worksheet.merge_range(209,1,209,5,'Test KS: Score Cuts', cell_format=format_table_title)
            
            # applying formatting
            
             # table header
            worksheet.conditional_format('B$5:'+alphabet[max_column_number]+'$5', {'type': 'no_blanks','format': format_table_header})
            worksheet.conditional_format('B$108:'+alphabet[max_column_number]+'$108', {'type': 'no_blanks','format': format_table_header})
            worksheet.conditional_format('B$211:'+alphabet[max_column_number]+'$211', {'type': 'no_blanks','format': format_table_header})
            
            # logo
            worksheet.insert_image('B1', os.path.join(os.getcwd(),'src','ks_logo.png'))
            
            
            # conditional formatting to highlight badrate break 
            # NOTE: Doing this before even odd row formatting because formatting cannnot be overwritten
            # conditional formatting to highlight badrate break 
            # NOTE: Doing this before even odd row formatting because formatting cannnot be overwritten
            worksheet.conditional_format('I7:I15',
                                         {'type':     'formula',
                                          'criteria': '=$I7>$I6',
                                          'format':   format_cell_anomaly_bad})
    
            worksheet.conditional_format('I110:I208',
                                         {'type':     'formula',
                                          'criteria': '=$I110>$I109',
                                          'format':   format_cell_anomaly_bad})
            worksheet.conditional_format('I213:I311',
                                         {'type':     'formula',
                                          'criteria': '=$I213>$I212',
                                          'format':   format_cell_anomaly_bad})
 
            
            # table cells                       
            rows = list(range(6,106))+list(range(109,209))+list(range(211,312))          
            
            for row in rows:
                worksheet.conditional_format('C$'+str(row)+':O$'+str(row), {'type': 'no_blanks','format': (format_cell_even if row%2==0 else format_cell_odd)})
                worksheet.conditional_format('Q$'+str(row)+':'+alphabet[max_column_number]+'$'+str(row), {'type': 'no_blanks','format': (format_cell_even if row%2==0 else format_cell_odd)})
                
            
            worksheet.set_column('B:B', 13)
            worksheet.set_column('C:D', 13, format_round3)
            worksheet.set_column('E:E', 13,format_align_centre)
            worksheet.set_column('F:G', 9)
            worksheet.set_column('H:H', 11)
            worksheet.set_column('I:I', 12, format_percsign)
            worksheet.set_column('J:J', 13, format_percsign)
            worksheet.set_column('K:K', 17, format_percsign)
            worksheet.set_column('L:M', 14, format_round0)
            worksheet.set_column('N:O', 16, format_percsign)
            worksheet.set_column('P:Q', 8, format_round2)
            
            worksheet.conditional_format('P6:P105', {'type': 'no_blanks','format': format_cell_highlight2})
            worksheet.conditional_format('P109:P208', {'type': 'no_blanks','format': format_cell_highlight2})
            worksheet.conditional_format('P212:P311', {'type': 'no_blanks','format': format_cell_highlight2})
            
            worksheet.conditional_format('B6:B105', {'type': 'no_blanks','format': format_cell_highlight1})
            worksheet.conditional_format('B109:B208', {'type': 'no_blanks','format': format_cell_highlight1})
            worksheet.conditional_format('B212:B311', {'type': 'no_blanks','format': format_cell_highlight1})
    
            
            

            # SHEET: Plots--------------------------------------------------------------------------
            if include_plots == True:
                worksheet = workbook.add_worksheet("Plots")
                worksheet.hide_gridlines(2)
                worksheet.insert_image('B1', os.path.join(os.getcwd(),'src','ks_logo.png'))
                chart1 = workbook.add_chart({'type': 'line'})
                chart1.add_series({'name':'=DecileTables!$N$5',
                                   'categories':'=DecileTables!$B$6:$B$15',
                                   'values':'=DecileTables!$N$6:$N$15',
                                   'line':{'color': '#002E8A'}})
        
                chart1.add_series({'name':'=DecileTables!$O$5',
                                   'categories':'=DecileTables!$B$6:$B$15',
                                   'values':'=DecileTables!$O$6:$O$15',
                                   'line':{'color': '#A9DA74'}})
                chart1.set_title ({'name': 'Train'})
                chart1.set_x_axis({'name': 'Decile'})
                chart1.set_y_axis({'name': ' '})
                
                chart1.set_style(1)
                worksheet.insert_chart('B4', chart1, {'x_offset': 0, 'y_offset': 0})
                
                
                chart2 = workbook.add_chart({'type': 'line'})
                chart2.add_series({'name':'=DecileTables!$N$31',
                                   'categories':'=DecileTables!$B$32:$B$41',
                                   'values':'=DecileTables!$N$32:$N$41',
                                   'line':{'color': '#002E8A'}})
        
                chart2.add_series({'name':'=DecileTables!$O$31',
                                   'categories':'=DecileTables!$B$32:$B$41',
                                   'values':'=DecileTables!$O$32:$O$41',
                                   'line':{'color': '#A9DA74'}})
                chart2.set_title ({'name': 'Test'})
                chart2.set_x_axis({'name': 'Decile'})
                chart2.set_y_axis({'name': ' '})
                
                chart2.set_style(1)
                worksheet.insert_chart('J4', chart2, {'x_offset': 0, 'y_offset': 0})
                
    
            writer.save()
            writer.close()
            
            if iteration_number == None:
                print("KS tables stored at: \n"+os.path.join(os.getcwd(),validation.__save_directory,save_folder))
                print("\nFilename : \n"+filename)
                
        if test_calculation_method == "EqualDeciles":
           KS_test_decile = KS_test_decile_equalcuts.copy()
        elif test_calculation_method == "ScoreCutOff":
            KS_test_decile = KS_test_decile_scorecutoff.copy()
        else:
            raise Exception('test_calculation_method parameter value is invalid. \nPlease refer to the documentation for accepted values.')
            
        
        return (KS_train_decile,KS_test_decile)
        
    
    @classmethod
    def concordance_disconcordance(cls,x,y,model):
        """
        This function calculates concordant%, discordant% and tied%.
        
        Parameters
        ----------
        x: array-like, sparse matrix of shape (n_samples, n_features)
    		Data, where n_samples is the number of samples and n_features is the number of features.
        
        y: array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target variable for supervised learning problems.
        
        model_object: estimator object
            Model object for the ml algorithm used.
        
        Returns
        -------

        percent_concordant: float
            Percentage of pairs where the observation with the desired outcome (event) 
            has a higher predicted probability than the observation without the outcome (non-event).

        percent_discordant: float
            Percentage of pairs where the observation with the desired outcome (event)
            has a lower predicted probability than the observation without the outcome (non-event).

        percent_tied: float
            Percentage of pairs where the observation with the desired outcome (event) 
            has same predicted probability than the observation without the outcome (non-event).
        
        """
        
        if (np.isfinite(x).values.all()==False) or (np.isfinite(y).values.all()==False):
            raise Exception('There may be missing values or inf values present in your data.\nPlease perform the necessary imputation using the impute function from the EDA module first.')
            
        
        data=pd.DataFrame({'actual':list(y),'predicted_prob':model.predict_proba(x)[:, 1]})
    
        con = 0
        dis = 0
        tied = 0
        for i in data[data['actual']==1]['predicted_prob'].tolist():
            con = con + sum(np.where(data[data['actual']==0]['predicted_prob'] < i, 1, 0))
            dis = dis + sum(np.where(data[data['actual']==0]['predicted_prob'] > i, 1, 0))
            tied = tied + sum(np.where(data[data['actual']==0]['predicted_prob'] == i, 1, 0))
     
        no_of_events = len(data[data['actual']==1])
        no_of_non_events = len(data[data['actual']==0])
        
        percent_concordant = 100*(con)/(no_of_events*no_of_non_events)
        percent_discordant = 100*(dis)/(no_of_events*no_of_non_events)
        percent_tied = 100*(tied)/(no_of_events*no_of_non_events)
        
        return percent_concordant, percent_discordant, percent_tied
      

    @classmethod
    def calculate_csi(cls, x_train, x_test, column_names):
        """
        This function performs the Characteristic Stability Index calculation for all the binned variables of the data.

        Parameters
        ----------

        x_train: pandas dataframe
            Binned train dataset (including bins columns).

        x_test: pandas dataframe
            Binned test dataset (including bins columns).

        column_names: list
            List of column names in the train and test dataset on which CSI is calculated.

        Returns
        -------

        csi_score: list
            List of float values that correspond to the CSI values for each column name specified.
        
        bin_wise_csi_list: list
            List of dataframes of the format ["<column_name>", "csi value"].


        """
        if (np.isfinite(x_train).values.all()==False) or (np.isfinite(x_test).values.all()==False):
            raise Exception('There may be missing values or inf values present in your data.\nPlease perform the necessary imputation using the impute function from the EDA module first.')
            
            
        #csi_score = []
        #bin_wise_csi_list = []
        result=pd.DataFrame() 
        for column in column_names:

            # calculation of pop - train
            pop_df_train = cls.__get_population_csi(x_train,column)
            pop_df_train['Pop %'].fillna(0.0001)
            pop_df_train.rename(columns = {column:'Bin','Pop %':'Train Population %'}, inplace = True) 

            # calculation of pop - test
            pop_df_test = cls.__get_population_csi(x_test,column)
            pop_df_test['Pop %'].fillna(0.0001)
            pop_df_test.rename(columns = {column:'Bin','Pop %':'Test Population %'}, inplace = True) 

            final_df = pop_df_train.merge(pop_df_test,how='left',on='Bin')
            final_df['Test Population %'].fillna(0.0000001,inplace=True) 
            final_df.rename(columns = {'Bin':'Level'}, inplace = True)  
            		
            final_df['CSI Value'] = (final_df['Train Population %'] - final_df['Test Population %']) * np.log(final_df['Train Population %'] / final_df['Test Population %'])
            final_df['CSI Value'] = final_df['CSI Value'] *100
            final_df['CSI Sum']=sum(final_df['CSI Value'])
            final_df['Variable']=column
            result=result.append(final_df)	
            result=result[['Variable','Level','Train Population %','Test Population %','CSI Value','CSI Sum']]
            
            
        filename = 'CSI'
        filename = ef.create_version(base_filename=filename, path = os.path.join(os.getcwd(),validation.__save_directory))
        filename += ".xlsx"
        
        
        ef.reset_header_default()
        
        writer = pd.ExcelWriter(os.path.join(os.getcwd(),
                                             validation.__save_directory,
                                             filename), engine='xlsxwriter')               
                
        
        workbook  = writer.book
        
        # FORMATS--------------------------------------------------------------------------
            
            
#        format_round0 = workbook.add_format(validation.__round0)
#        format_round1 = workbook.add_format(validation.__round1)
#        format_round2 = workbook.add_format(validation.__round2)
#        format_round3 = workbook.add_format(validation.__round3)
        format_percsign = workbook.add_format(validation.__percsign)
#        format_percsignmultipled = workbook.add_format(validation.__percsignmultipled)
#        format_cell_highlight1 = workbook.add_format(validation.__cell_highlight1)
#        format_cell_highlight2 = workbook.add_format(validation.__cell_highlight2)
#        format_cell_highlight3 = workbook.add_format(validation.__cell_highlight3)
        format_cell_highlight4 = workbook.add_format(validation.__cell_highlight4)
        format_cell_highlight5 = workbook.add_format(validation.__cell_highlight5)
        format_cell_highlight6 = workbook.add_format(validation.__cell_highlight6)
        format_table_header= workbook.add_format(validation.__table_header)
#        format_table_title = workbook.add_format(validation.__table_title)
        format_cell_even = workbook.add_format(validation.__cell_even)
        format_cell_odd = workbook.add_format(validation.__cell_odd)
#        format_align_centre = workbook.add_format(validation.__align_centre)
#        format_cell_anomaly_bad = workbook.add_format(validation.__cell_anomaly_bad)
#        format_cell_anomaly_good = workbook.add_format(validation.__cell_anomaly_good)
#        format_border1 = workbook.add_format(validation.__border1)
        format_border2 = workbook.add_format(validation.__border2)

        
        
        
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        # SHEET: DecileTables--------------------------------------------------------------------------
        
        
        result.to_excel(writer, sheet_name='CSI',index=False,startrow=4,startcol=1) 
        worksheet = writer.sheets['CSI']
        worksheet.hide_gridlines(2)
        
        # applying formatting
        
        max_column_number = len(result.columns)+1
        # table header
        worksheet.conditional_format('B$5:'+alphabet[max_column_number]+'$5', {'type': 'no_blanks','format': format_table_header})
        
        # logo
        worksheet.insert_image('B2', os.path.join(os.getcwd(),'src','ks_logo.png'))
        
        # table cells 
        
        result.reset_index(drop=True,inplace=True)

        worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 6,column_number = len(result.columns)+1)+':'+ef.generate_excel_cell_name(row_number = len(result)+5,column_number = len(result.columns)+1),
                                             {'type': 'cell',
                                              'criteria': '<=',
                                              'value': 10,
                                              'format':format_cell_highlight5})
                                                        
        worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 6,column_number = len(result.columns)+1)+':'+ef.generate_excel_cell_name(row_number = len(result)+5,column_number = len(result.columns)+1),
                                 {'type': 'cell',
                                  'criteria': 'between',
                                  'minimum':  10,
                                  'maximum':  25,
                                  'format':   format_cell_highlight6})
                                               
        worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 6,column_number = len(result.columns)+1)+':'+ef.generate_excel_cell_name(row_number = len(result)+5,column_number = len(result.columns)+1),
                                 {'type': 'cell',
                                  'criteria': '>=',
                                  'value': 25,
                                  'format':format_cell_highlight4})        
#        start_row = 6
#        end_row = 6
#        for i,r in result.iterrows():
#            if i>1:
#                if (result['Variable'][i]!=result['Variable'][i-1]) or (i==len(result)-1):
#                    if i==len(result)-1:
#                        end_row  = i + 6
#                    else:
#                        end_row  = i-1 + 6
#                    worksheet.conditional_format(ef.generate_excel_cell_name(row_number = start_row,column_number = len(result.columns)+1)+':'+ef.generate_excel_cell_name(row_number = end_row,column_number = len(result.columns)+1),
#                                             {'type': 'cell',
#                                              'criteria': '<=',
#                                              'value': 0.01,
#                                              'format':format_cell_highlight5})
#                                                        
#                    worksheet.conditional_format(ef.generate_excel_cell_name(row_number = start_row,column_number = len(result.columns)+1)+':'+ef.generate_excel_cell_name(row_number = end_row,column_number = len(result.columns)+1),
#                                             {'type': 'cell',
#                                              'criteria': 'between',
#                                              'minimum':  0.01,
#                                              'maximum':  0.025,
#                                              'format':   format_cell_highlight4})
#                                                           
#                    worksheet.conditional_format(ef.generate_excel_cell_name(row_number = start_row,column_number = len(result.columns)+1)+':'+ef.generate_excel_cell_name(row_number = end_row,column_number = len(result.columns)+1),
#                                             {'type': 'cell',
#                                              'criteria': '>=',
#                                              'value': 0.025,
#                                              'format':format_cell_highlight6})
#                                                         
#                    start_row = end_row + 1 

        worksheet.conditional_format(ef.generate_excel_cell_name(row_number = 6,column_number = 2,fix_row=True)+':'+ef.generate_excel_cell_name(row_number = len(result)+6,column_number = len(result.columns)+1,fix_row=True),
                                     {'type': 'formula',
                                      'criteria': "=" + ef.generate_excel_cell_name(row_number = 7,column_number = 2,fix_column=True)+'<>'+ef.generate_excel_cell_name(row_number = 6,column_number = 2,fix_column=True),
                                      'format': format_border2})           
        rows = range(6,len(result)+6)
        
        for row in rows:
            worksheet.conditional_format('B$'+str(row)+':B$'+str(row), {'type': 'no_blanks','format': format_cell_odd})
            worksheet.conditional_format('C$'+str(row)+':'+alphabet[max_column_number]+'$'+str(row), {'type': 'no_blanks','format': (format_cell_even if row%2==0 else format_cell_odd)})
            
        
        max_column_length = max([len(val)+2 for val in result.Variable])
        worksheet.set_column('B:B',max_column_length)
        worksheet.set_column('D:E', 20, format_percsign)
        worksheet.set_column('F:G', 13, format_percsign)
        
        writer.save()
        writer.close()


        #result.to_csv(os.path.join(os.getcwd(),validation.__save_directory,filename + 'csv'),index=False)
        print("CSI result stored at: \n"+os.path.join(os.getcwd(),validation.__save_directory))
        print("\nFile name: \n"+filename)
	
        	
        return result







