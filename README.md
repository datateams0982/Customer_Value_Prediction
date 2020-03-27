# Customer_Value_Prediction

# QUERY:
- Query data from ODS, including asset data, trading data and demographic data.
- Asset data is queried and saved year by year.
- Trading data is queried with demographic data, along with labeling.
- Asset data should be processed year by year by asset_process.ipynb

# Process:
- Preprocess_func.py is the function needed for preprocessing
- Preprocess step: Trading_Process (filling missing value, create variables, trading data binning) -> Sampling (Down Sampling active data, checking sample representiveness) -> Sample_Process (Lookback and combine with asset data) -> Training_Preparation (Transform data to training form, including for autoML and local CNN training)

# Visualization:
- Plotting distribution of active and churn data.

# Model:
- Model_func.py: Including functions for model evaluation
- Model.ipynb: Experiment of first stage model
- Evaluation_local.ipynb: Evaluation for first stage model
- Evaluation.ipynb: Evaluation for google autoML

# Evaluation:
- Resampling active data for further evaluation.
