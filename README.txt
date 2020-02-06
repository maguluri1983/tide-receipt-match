

This GIT repository contains the data science case study sent over by the team from tide.

Objective:
- Train a model that is able to reorder the partial matches provided in dataset
- Transactions that are most probable to be the right match are expected at the top
- Transaction that are least probable to be the right match are expected at the bottom

List of files:
1 - README
2 - data.csv => Original input file provided for case study
3 - Receipt Matching Interview Test-2.pdf => Case Study overview document
4 - match_receipt_eda_analysis.ipynb => Jupyter Notebook with Exploratory data analysis and Modeling Details
5 - predict_matching_receipts.py => Python program with all the functionality, to make predictions with new file
6 - model_file.pk => Model file saved
7 - validation_file.csv => Validation file (a copy of data.csv with matched_transaction_id removed)
8 - predictions.csv => Predictions file with reordered feature transaction ids for a reciept

Assumptions Made:
1 - Input file has and will have delimiter of ":" and thousands seperator as ","
2 - No presence of null values in the input file
3 - Random sampling of receipt ids will result in similar distributions of transaction ids
4 - CompanyIDs stay the same as in training and no new companyIDs get added in Test
5 - matched_transaction_id will not be provided in validation/test file

Validating the Model:
1 - Please refer to the Jupyter notebook file to walk through the entire analysis
2 - To validate the python program, follow the below steps
      => To change the file on which the program is trained, Please update "data.csv" to any new training file
      => To change the validation file, please update "validation_file.csv" to any new validation file
      => Similarly update "predictions.csv" to any new predictions file with reorderd entries

Executing the python program:
python predict_matching_receipts.py

Sample Output:
Model Info: XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.1, max_delta_step=0, max_depth=10,
              min_child_weight=1, missing=None, n_estimators=100, n_jobs=-1,
              nthread=None, objective='binary:logistic', random_state=1234,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)
Model Metrics:
F1-Score: 0.756
Balanced Accuracy: 0.837
ROC_AUC_Score: 0.983
