import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import src.target_variable as tv 

"""### Target Variable / Y Variable """   
data = pd.read_csv('data\ovrd_data.csv', index_col=0)
y_var = tv.Y_Variable()

y_var.roll_rate(
    data,
    target_variable='LoanAccountNo', 
    dpd_var='AccountDPD',
    mob_column = 'MOB',
    mob_bucket = [[2,4], [2,6], [4,6], [4,8], [4,10], [6,8], [6,10], [6,12]]
)

y_var.vintage(
    data, 
    target_variable='AccountDPD', 
    subset_on = "LoanAccountNo", 
    mob_var = 'MOB', 
    threshold = 30, 
    target_column = 'LOB'
)

y_var

"""### Import Data"""

#-------------------- Drop the unneccesary varibles in the csv itself
# Importing Dataset
data = pd.read_csv("data\dev_model_data.csv",index_col=0)
data.drop(["agreement_no", "DPD_24mob0P", "y"], axis=1, inplace=True)
# data.reset_index(drop=True,inplace=True)
data.head()

# Set the target variable
target = "dpd_24mob0P_6b"
ID = "appl_id"

"""### EDA"""

# Importing the EDA module
import src.eda as eda

# Creating EDA object
eda_obj = eda.model_EDA()

"""#### Data Split"""

# Splitting data into train/test
# methods: [random, stratify]
train, test = eda_obj.data_split(dataset = data.copy(), method = 'stratify', test_size = 0.25, strata = [target], random_state=42)

print("Target Distribution\ntrain: %f \ntest: %f"%(train[target].sum()/train[target].count(), test[target].sum()/test[target].count()))

"""#### Univariate Analysis"""

# Univariate Analysis
# missing_cuoff, n_levels for categorical variables
# To understand Variation, perc_cutoff
categorical, continous, var_list = eda.model_EDA.univariate_analysis(data = train.copy(),ID=ID, missing_cutoff = 10, n_levels = 20, perc_cutoff = [5,95])

var_list

# Intial Selection of variables
# Reading the Updated Univariate Files
continous = pd.read_excel("results/EDA/Univariate_v1.xlsx",sheet_name="Continuous", index_col=1, skiprows=4).reset_index().dropna(axis=1)
categorical = pd.read_excel("results/EDA/Univariate_v1.xlsx",sheet_name="Categorical", index_col=1, skiprows=4).reset_index().dropna(axis=1)

selected_num_cols = eda_obj.final_var_selection(continous.reset_index(),variable_col_name='Variable',Var_Select_Name='var_select',Variable_Selection_Label='Y', fileName="Continuous")
selected_cat_cols = eda_obj.final_var_selection(categorical.reset_index(),variable_col_name='Variable',Var_Select_Name='var_select',Variable_Selection_Label='Y', fileName="Categorical")

final_var_list = selected_num_cols + selected_cat_cols
final_var_list

print("Number of Variables selected from Univariate Analysis: ",len(final_var_list))

"""#### Missing Values Imputation"""

# Add function for the final selection of variables
train = train[final_var_list+[ID]]
test = test[final_var_list+[ID]]

# Imputing missing values
# Categorical imputed with mode
# Continuous can be imputed with [mean, median, mice, knn]
# An integer value of number of neighbours for knn imputation: strata
train_imputed, test_imputed = eda.model_EDA.impute(df_train = train, target=target, ID=ID ,df_test = test, method = 'mean', strata = 50)

"""#### Outlier Treatment"""

# Outlier Treatment
# method: ['cap_floor', 'kmeans', 'zscore']
# strata: based on the method list of percentile values for cap_floor, percantage for kmeans and integer for zscore
train_o, test_o, train_ex, test_ex = eda.model_EDA.outlier_treatment(train_data = train_imputed,test_data = test_imputed, method = 'cap_floor', strata = [0.05,0.95])

"""### Feature Selection"""

# Importing Feature Selection Module
import src.feature_selection as fs

# Creating Feature Selection Object
a=fs.feature_selection()

"""#### PCA"""

# PCA - Variable Reduction
# threshold: cutoff for choosing the top variables
var_list, pc_weights = a.PCA(train_o, target, threshold=40)

"""#### VIF Calculation"""

# VIF Calculation
# Threshold to shortlist the variables
v=a.vif_fun(train_o[var_list["Important Variables"].to_list()[:5]],target,threshold=10)

"""#### Variable Clustering - IV"""

# Calculates IV for all the variables
# For clustering, eigen value cutoff and maxcluster to be given
rvc=a.varclusIV(train_o,target, maxeigval=1, maxcluster=40)

"""#### Final Variable Selection"""

# Final Selection of Numerical Variables from Varclus IV
#varclus_df = pd.read_excel("data/Varclus IV_Demo_Internal.xlsx", index_col=1, skiprows=4).reset_index().dropna(axis=1)

#selected_num_cols = eda_obj.final_var_selection(varclus_df,variable_col_name='Variable',Var_Select_Name='Select',Variable_Selection_Label='Y', fileName="Continuous")

# taking variables selected by pca
selected_num_cols = var_list['Important Variables'].tolist().copy()
selected_num_cols

final_var_list = selected_num_cols + selected_cat_cols
final_var_list

"""### Variable Transformation"""

# Importing Feature Transformation Module
import src.Feature_Transformation as ft

# Preparing Data for Transformation
train_f = train_o[selected_num_cols+selected_cat_cols+[target]]
test_f = test_o[selected_num_cols+selected_cat_cols+[target]]

# Train Data Transformation
# transform_var = pd.read_csv("Continuous_Trans.csv",index_col=0).T.to_dict(orient="records")[0]
# transform_var = transform_var
trans = ft.Variable_Transformation(train_f[selected_num_cols])
train_trans = trans.variable_transformation()

# Test Data Transformation
trans = ft.Variable_Transformation(train_f[selected_num_cols], test_f[selected_num_cols])
test_trans = trans.variable_transformation()
train_trans

encode = ft.Encoding()
encode.fit(train_f[selected_cat_cols])
encode.transform(train_f[selected_cat_cols])

bin = ft.Binning_Perc()
bin.fit(train_f[selected_num_cols])
bin.transform(test_f[selected_num_cols])

"""### WOE Automated Binning"""

# WOE Automated Binning for continuous variables
woe = ft.WoE_Binning_Automated(v_type='c')
binned_df, mon_binned_df = woe.woe_bins(train_f[selected_num_cols+[target]], target=target, monotonic=True)

"""### WOE Optimized Binning"""

# WOE Optimized Binning for continuous variables
woe = ft.WoE_Binning_Optimization(v_type='c', spec_values=[float("-inf"), float("inf")])
bins_opt, bins_opt_mon = woe.woe_bins_optimized(train_f[selected_num_cols+[target]], target=target, monotonic=True)

"""### WOE Manual Binning"""

# Manual WOE Bins
woe_data = pd.read_excel("data/WOE_Binning_Automated_Demo.xlsx",sheet_name="WOE_Binning_Automated", index_col=1, skiprows=4).reset_index().dropna(axis=1)
woe_manual_transformed = ft.WoE_Transformation_Manual().woe_data(target,test_f[selected_num_cols],woedata=woe_data)

"""### WOE Tranformation (Automated)"""

# WOE Transformation
woe = ft.WoE_Transformation(transformation_type='Automated',v_type="c",save=True)
woe_transform_train_num = woe.woe_transformation(train_f[selected_num_cols+[target]], monotonic=True, target=target)

woe = ft.WoE_Transformation(transformation_type='Automated',v_type="c",save=True)
woe_transform_test_num = woe.woe_transformation(train_f[selected_num_cols+[target]], test_f[selected_num_cols+[target]], monotonic=True, target=target)

# cat_cols
woe = ft.WoE_Transformation(transformation_type='Automated',v_type="d",save=True)
woe_transform_train_cat = woe.woe_transformation(train_f[selected_cat_cols+[target]], target=target)

woe = ft.WoE_Transformation(transformation_type='Automated',v_type="d",save=True)
woe_transform_test_cat = woe.woe_transformation(train_f[selected_cat_cols+[target]], test_f[selected_cat_cols+[target]], target=target)

"""### Model Tuning"""

# Importing Module
import src.model_tuning as mt

# data_encoded_train, data_encoded_test, woe_transform_mon_train, woe_transform_mon_test
model_train_y = train_f[target].reset_index(drop=True)
model_test_y = test_f[target].reset_index(drop=True)

# Preparing train test data for modelling
# 
woe_transform_train_num = woe_transform_train_num[list(set(woe_transform_train_num.columns)-set(selected_num_cols))]
woe_transform_test_num = woe_transform_test_num[list(set(woe_transform_test_num.columns)-set(selected_num_cols))]
woe_transform_train_cat = woe_transform_train_cat[list(set(woe_transform_train_cat.columns)-set(selected_cat_cols))]
woe_transform_test_cat = woe_transform_test_cat[list(set(woe_transform_test_cat.columns)-set(selected_cat_cols))]

model_train = woe_transform_train_num.merge(woe_transform_train_cat, right_index=True, left_index=True)
model_test = woe_transform_test_num.merge(woe_transform_test_cat, right_index=True, left_index=True)

model_train.replace([np.inf,np.inf*-1],0,inplace=True)
model_test.replace([np.inf,np.inf*-1],0,inplace=True)

"""### GridSearch - Model Tuning"""

# Gridsearch - GBM
# 
out_df, model_list = mt.model_tuning.grid_search_GBM(model_train,
    model_test,
    model_train_y,
    model_test_y,
    subsample=[1.0],
    learning_rate=[0.1, 0.01],
    n_estimators=range(100,400,100),
    criterion=['friedman_mse'],
    max_depth=[3],
    min_samples_split=[2],
    min_samples_leaf=[1],
    max_features=[None],
    random_state=0,
    test_calculation_method='ScoreCutOff',
    save_charts=True,
    include_plots=True)

"""### Recursive Feature Elimination"""

# RFE
result, selected_columns, model_object = mt.model_tuning.recursive_feature_elimination_ks(model_train,
    model_train_y,
    model_test,
    model_test_y,
    model_list[1])

# Selected columns from RFE
selected_columns

#cross_validation_scores = mt.model_tuning.cross_validation(model_train[selected_columns], model_train_y, model_object, score_cutoff=0.14)

"""### Validation"""

# Importing Module
import src.validation as mv

# Creating object
mv_object = mv.validation()

# Preparing Data for Validation
y_test_probablity = pd.DataFrame(list(model_object.predict_proba(model_test[selected_columns])[:,1]),columns=["Probability"])
y_test_probablity["Target"] = list(model_test_y)

y_train_probablity = pd.DataFrame(list(model_object.predict_proba(model_train[selected_columns])[:,1]),columns=["Probability"])
y_train_probablity["Target"] = list(model_train_y)

"""### CSI and PSI Validation"""

# Model Validation - CSI check
csi_check = mv_object.calculate_csi(model_train,model_test,selected_columns)

# mv_object.calculate_ks(model_train_y,model_test_y)
psi_score, decile_wise_psi_values = mv_object.calculate_psi(y_train_probablity,y_test_probablity)

"""### Interpretibility"""

# Importing Module
import src.model_interpretability as mi


# Create Object
mi_obj = mi.Reason_code

# Preparing Data for Interpretibility Module
train_data = model_train.merge(model_train_y, left_index=True, right_index=True)
train_data = train_data[list(selected_columns)+[target]].reset_index()[:50]

test_data = model_test.merge(model_test_y, left_index=True, right_index=True)
test_data = test_data[list(selected_columns)+[target]].reset_index()[:50]

# Calculating Feature Importance
features = list(selected_columns)
feat_imp = pd.DataFrame({'importance':model_object.feature_importances_})
feat_imp['feature'] = train_data[features].columns
feature_imp = feat_imp.sort_values(by='importance', ascending=False)

"""### Model Interpretibility"""

vwoe = mi_obj.vwoe_generate(train_data[list(selected_columns)+[target]],[],target,feature_imp,True)

# This function provides a csv file with model interpretability information
reasoning_dataframe = mi_obj.reasons_code(train_data=train_data.copy(),unique_id="index",vwoe=vwoe.copy(),binning_flag=True,topn=5,
                                          test_data=test_data.copy(),model_fit=model_object,features=list(selected_columns),model_type='classification',absent_remove=False,target_name=target,algo="vwoe")

"""### Top Variable Distribution"""

# Seperaction score and top variable distribution values are stored
prob_list = model_object.predict_proba(test_data[selected_columns])[:,1]
separation_score, top_var_distribution = mi_obj.similarity_score(reasoning_dataframe.copy(),"index",prob_list,5,10)

"""### PMML"""

import src.generate_pmml as pmml

pmml.generate_pmml(model_train,model_train_y,model_object,[],list(selected_columns))

from pypmml import Model

model = Model.fromFile('results/PMML Files/Model.pmml')
result = model.predict(model_train[:1])
