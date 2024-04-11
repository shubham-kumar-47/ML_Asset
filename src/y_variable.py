# -*- coding: utf-8 -*-
"""
    Project
    -------
    Y Variable 

    Introduction
    ------------
    Y Variable analysis is done to identify the defaulters who will fail to repay the loan in the in the given time frame. 
    
    This module performs Roll Rate & Vintage calculation for Y Variable.
    
    
    Roll Rate: Roll rate analysis examines the movement of customers or accounts between different states over time. In the context of credit, 
               these states often represent different stages of delinquency or default. By analyzing roll rates, lenders can gain insights into the performance 
               of their portfolios and identify trends or areas of concern.
                a. Roll Forward: percentage of account moving to higher delinquency bucket.
                b. Roll Backward: percentage of account moving to lower delinquency bucket.
                c. Same Bucket: percentage of account staying in same bucket.
                d. Total Live: the number of active loan.
    
    Vintage: Vintage analysis involves grouping loans or accounts based on the time they were originated or booked (the vintage) and then analyzing the performance 
             of each vintage over time. This allows lenders to assess how the quality of their lending has evolved over time and to make adjustments to underwriting 
             standards or risk management practices if necessary.
    
    
    
    Author
    ----------
    @author: Kumar, Shubham
    
    Dependencies
    ------------
        python>-3.6
        pandas==0.24.2
        numpy==1.18.4
    
"""

import os 
import warnings
import numpy as np  
import pandas as pd
import datetime as dt
from src.excel_formatting import excel_formatting as ef 

class Y_Variable():
    
    # create directories if not available
    __save_directory = 'results/Y_Variable'

    if os.path.isdir("results")==False: 
        os.mkdir("results")

    if os.path.isdir(__save_directory)==False: 
        os.mkdir(__save_directory)
            
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
        """
        This is the initiallisation of class. It needs to be done before calling any function from the class.

        """
        pass

    
    def __roll_rate(self, data, m1, m2, target_variable, dpd_var, mob_column):
        """ It helps to identify the proxy of bad as earliest as possible.
                a. Roll Forward: percentage of account moving to higher delinquency bucket.
                b. Roll Backward: percentage of account moving to lower delinquency bucket.
                c. Same Bucket: percentage of account staying in same bucket.
                d. Total Live: the number of active loan. 
                
            
        """
        if None in data: 
            assert ValueError('Please provide the required arguments.')
 
        bins = [-np.inf, 0, 60, 90, 120, 150, 180, 99999]
        labels = ['0', '1','2','3','4' ,'5', '6']
            #target_variable = 'LoanAccountNo'
            #dpd_var = 'AccountDPD'

        data['DPD_Bucket'] = pd.cut(data[dpd_var], bins=bins, labels=labels)

        # subset the given dataset according to months
        mob_1, mob_2 = data[data[mob_column] == m1], data[data[mob_column] == m2] 

        # join the subset based on the target variable
        mob_x, mob_y = mob_column + '_x', mob_column + '_y'
        tmp_df = pd.merge(mob_1, mob_2, on = target_variable, how='left')[[mob_x, mob_y, 'DPD_Bucket_x', 'DPD_Bucket_y']]

        tmp_df['DPD_Bucket_x'] = tmp_df['DPD_Bucket_x'].astype(str).str.replace('nan', 'NULL')
        tmp_df['DPD_Bucket_y'] = tmp_df['DPD_Bucket_y'].astype(str).str.replace('nan', 'NULL') 

        # apply crosstab to DPD_Buckets 
        table = pd.crosstab(tmp_df['DPD_Bucket_x'], tmp_df['DPD_Bucket_y'], dropna=False, margins=True, margins_name='Grand Total')
        table.index.name = 'SumOfCount'
        # Total Live 
        table['Total Live'] = np.subtract(table['Grand Total'], table['NULL'])

        table_res = table[[ele for ele in table.columns.tolist() if ele.isnumeric()]] 

        # Roll Back
        table['Roll Back %'] = np.round(np.divide(np.sum(np.tril(table_res, k=-1), axis=1), table['Total Live'].values) * 100, 2)

        # Roll Forward
        table['Roll Forward %'] = np.round(np.divide(np.sum(np.triu(table_res, k=1), axis=1) , table['Total Live'].values)*100, 2) 

        # Same Bucket
        table['Same Bucket %'] = np.round(np.divide(np.diag(table), table['Total Live'].values) * 100,2)   

        # Net Forward
        table['Net Forward %'] = np.round(np.divide(np.sum(np.triu(table_res, k=0), axis=1), table['Total Live'].values) * 100,2)
        table['Net Forward %'][0] = table['Roll Forward %'][0]

        # Cleaning Up Unnecessary Values
        table.iloc[-1:,-5:] = ' '
        return table 


    def roll_rate(
        self,
        data,
        target_variable,
        dpd_var,
        mob_column,
        mob_bucket = [[2,4], [2,6], [4,6], [4,8], [4,10], [6,8], [6,10], [6,12]]
    ):
        """ It helps to identify the proxy of bad as earliest as possible.
                a. Roll Forward: percentage of account moving to higher delinquency bucket.
                b. Roll Backward: percentage of account moving to lower delinquency bucket.
                c. Same Bucket: percentage of account staying in same bucket.
                d. Total Live: the number of active loan. 
                
            
        """
        
        # Add code here to save the formatted excel file
        ef.reset_header_default()
        filename = 'Roll Rate'
        filename = ef.create_version(base_filename=filename, path = os.path.join(os.getcwd(), self.__save_directory))
        filename += '.xlsx'

        writer = pd.ExcelWriter(os.path.join(os.getcwd(), self.__save_directory,filename), engine='xlsxwriter')
        workbook  = writer.book
        
        format_table_title = workbook.add_format(self.table_title)
        format_table_header = workbook.add_format(self.table_header) 
        
        # get mob from user
        #if None in (data, mob_bucket): raise ValueError("Please provide dataset and mob bucket.") 

        start_row = 6
        for m1, m2 in mob_bucket:
            res = self.__roll_rate(data, m1, m2, target_variable, dpd_var, mob_column)
            res.to_excel(writer, sheet_name='Roll Rate', startrow = start_row , startcol=1)
            worksheet = writer.sheets['Roll Rate']
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number = start_row, column_number = 2, fix_row=True)+':'+ef.generate_excel_cell_name(row_number = start_row, column_number = len(res.columns)+2, fix_row=True), {'type': 'no_blanks','format': format_table_header})
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number = start_row+1, column_number = 2, fix_row=True)+':'+ef.generate_excel_cell_name(row_number = start_row+1, column_number = len(res.columns)+2, fix_row=True), {'type': 'no_blanks','format': format_table_title})
            worksheet.conditional_format(ef.generate_excel_cell_name(row_number = start_row+1 + res.shape[0], column_number = 2, fix_row=True)+':'+ef.generate_excel_cell_name(row_number = start_row+1 + res.shape[0] , column_number = len(res.columns)+2, fix_row=True), {'type': 'no_blanks','format': format_table_title})
            worksheet.write_string(start_row - 1, 1, 'M' + str(m1) +'-'+ 'M' + str(m2))
            start_row += (res.shape[0] + 3) 
        # logo
        worksheet = writer.sheets['Roll Rate']
        worksheet.hide_gridlines(2)
        worksheet.insert_image(ef.generate_excel_cell_name(row_number=2,column_number=2), os.path.join(os.getcwd(),'src','bridgei2i_logo.png'))
        writer.save()
        writer.close()
        
        
    def vintage(self, data, target_variable, subset_on, mob_var, threshold, target_column):
        ############ Vintage Code ############
#         data = data[data[target_variable] >= threshold] 
#         data.sort_values([mob_var],ascending=True, inplace=True)
#         data.drop_duplicates(subset=subset_on, keep = 'first', inplace = True)  
#         result = data.groupby(mob_var)[subset_on].count().reset_index()
#         result[subset_on] = result[subset_on].cumsum()
#         result.columns = ['MOB', 'OverAll']
#         result['OverAll(%)'] = result['OverAll']/np.max(result['OverAll'])

        data = data[data[target_variable] >= threshold] 
        data.sort_values([mob_var],ascending=True, inplace=True)
        data.drop_duplicates(subset=subset_on, keep = 'first', inplace = True)  
        result = data.groupby([mob_var, target_column])[[subset_on]].count().reset_index().pivot(index=mob_var, columns = target_column, values = subset_on)  
        result['Overall'] = result.sum(1)
        result = result.fillna(0)
        # result = np.cumsum(axis=0).astype(int)
        ############ End of Vintage Code ############
        
        # create directories if not available
        __save_directory = 'results/Y_Variable'
        if os.path.isdir("results")==False: 
            os.mkdir("results")
        if os.path.isdir(__save_directory)==False: 
            os.mkdir(__save_directory)

        # Add code here to save the formatted excel file
        ef.reset_header_default()
        filename = 'Vintage'
        filename = ef.create_version(base_filename=filename, path = os.path.join(os.getcwd(),__save_directory))
        filename += '.xlsx'

        writer = pd.ExcelWriter(os.path.join(os.getcwd(),__save_directory,filename), engine='xlsxwriter')
        workbook  = writer.book
        format_table_title = workbook.add_format(self.table_title)
        result.to_excel(writer, sheet_name='Vintage', startrow=5 , startcol=1, index=True)
        # logo
        worksheet = writer.sheets['Vintage']
        worksheet.hide_gridlines(2)
        worksheet.conditional_format(
            ef.generate_excel_cell_name(
                row_number = 6, 
                column_number = 2,
                fix_row=True,
                fix_column=True)+':'+ef.generate_excel_cell_name(
                row_number = 6,
                column_number = len(result.columns)+2,
                fix_row=True,
                fix_column=True), {'type': 'no_blanks','format': format_table_title}
        )
        
        worksheet.insert_image(ef.generate_excel_cell_name(row_number=2,column_number=2), os.path.join(os.getcwd(),'src','bridgei2i_logo.png'))
        writer.save()
        writer.close()
