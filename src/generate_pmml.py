# coding: utf-8

import os
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelBinarizer

def generate_pmml(X_train,Y_train,model,cat_columns,cont_columns,output_file_name="Model"):
    """
    This function generates the PMML file for the given model.
    
    Dependency: 
    This file (generate_pmml.py) must be present in the folder in which the function is called.
    
    Parameters
    ----------
    
    X_train: dataframe
        X variables in training dataset
        
    Y_train: dataframe
        Y variable in training dataset
        
    model: object
        Model object to be used to convert into pmml format
        
    cat_columns: list
        Defines the list of categorical variable names
        
    cont_columns: list
        Defines the list of numerical/ continuous variable names
        
    output_file_name: string, default = "Model"
        Name of the PMML file to be saved. (It saves in the folder "PMML_files")
        
    Example
    -------
    
    >>> from generate_pmml import generate_pmml
    >>> generate_pmml(X_train,Y_train,model_object,["A","B","C"],["D","E"],"GBM_model")
    
    """
    
    __save_directory = 'results/PMML Files'
    if os.path.isdir("results")==False:
        os.mkdir("results")
    
    if os.path.isdir(__save_directory)==False:
        os.mkdir(__save_directory)
    
        
    try:

        mapper = -1
        if len(cat_columns)!=0 and len(cont_columns)!=0:
            mapper = DataFrameMapper([([cat_column], [CategoricalDomain(), LabelBinarizer()]) for cat_column in cat_columns] + [(cont_columns, ContinuousDomain())])
        elif len(cat_columns)!=0:
            mapper = DataFrameMapper([([cat_column], [CategoricalDomain(), LabelBinarizer()]) for cat_column in cat_columns])


        classifier = model

        if mapper!=-1:
            pipeline = PMMLPipeline([("mapper", mapper),("classifier", classifier)])
        else:
            pipeline = PMMLPipeline([("classifier", classifier)])


        pipeline.fit(X_train, Y_train)


        sklearn2pmml(pipeline, __save_directory+"/{}.pmml".format(output_file_name), with_repr = True)

    except Exception as e:
        print("Error : ",e)

