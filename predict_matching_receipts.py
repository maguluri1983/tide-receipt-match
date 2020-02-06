import warnings, pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt, pickle
from sklearn.model_selection import train_test_split
from os import path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.metrics import roc_auc_score
from pylab import rcParams
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score, \
                            balanced_accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore")

class receipt_match:
    outcome_feature = 'Match'
    data_split_seed = 123
    model_seed      = 1234
    
    def __init__(self,train_file):
        self.train_file     = train_file
        self.input_data     = pd.read_csv(train_file,delimiter=':', thousands=',')
        self.total_receipts = len(self.input_data['receipt_id'].unique())
        self.model_file     = None
        
    def model_stats(self,test_ratio):
        np.random.seed(self.data_split_seed)
        test_receipt_count  = round(self.total_receipts*test_ratio)
        train_receipt_count = self.total_receipts - test_receipt_count
        train_receipts      = np.random.choice(self.input_data['receipt_id'].unique(),
                                               train_receipt_count,replace=False).tolist()
        test_receipts  = [x for x in self.input_data['receipt_id'].unique().tolist() if x not in train_receipts]
        train_df       = self.input_data[self.input_data['receipt_id'].isin(train_receipts)]
        test_df        = self.input_data[self.input_data['receipt_id'].isin(test_receipts)]

        train_df = self.FeatureEngineering(train_df); test_df = self.FeatureEngineering(test_df)
        drop_cols = ['receipt_id']
        train_df.drop(drop_cols,axis=1,inplace=True); test_df.drop(drop_cols,axis=1,inplace=True)
        
        scaler = MinMaxScaler(); scaler.fit(train_df)
        train_df_scaled = pd.DataFrame(scaler.transform(train_df),columns=train_df.columns)
        test_df_scaled  = pd.DataFrame(scaler.transform(test_df),columns=test_df.columns)
        
        X_train  = train_df_scaled.loc[:,train_df_scaled.columns != self.outcome_feature]
        y_train  = train_df_scaled.loc[:,train_df_scaled.columns == self.outcome_feature]
        X_test   = test_df_scaled.loc[:,test_df_scaled.columns != self.outcome_feature]
        y_test   = test_df_scaled.loc[:,test_df_scaled.columns == self.outcome_feature]
        
        xgb_clf = XGBClassifier(max_depth=10,random_state=self.model_seed,n_estimators=100,n_jobs=-1)
        xgb_clf.fit(X_train, y_train);
        
        predictions    = xgb_clf.predict(X_test)
        predict_proba  = xgb_clf.predict_proba(X_test)[:,1]
        
        metric_f1 = round(f1_score(y_test,predictions,average=None).tolist()[1],3)
        bal_acc   = round(balanced_accuracy_score(y_test,predictions),3)
        roc_score = round(roc_auc_score(y_test,predict_proba),3)
        
        print("Model Info:",xgb_clf)
        
        print("Model Metrics:\nF1-Score:",metric_f1,
              "\nBalanced Accuracy:", bal_acc, 
              "\nROC_AUC_Score:", roc_score)
        
    def FeatureEngineering(self,temp_df):
        
        #Match Feature
        if (('matched_transaction_id' in temp_df.columns) and 
            ('feature_transaction_id' in temp_df.columns)):
            temp_df['Match'] = temp_df.apply(lambda row: 1 \
                                            if row['matched_transaction_id'] == row['feature_transaction_id'] \
                                            else 0, axis = 1)
        #CompanyID OneHotEncoding
        temp_df = pd.concat([temp_df,pd.get_dummies(temp_df['company_id'],
                                                    prefix='compid')],axis=1).drop('company_id',axis=1)

        #ReceiptMatchesCount
        receipt_counts = temp_df.groupby(['receipt_id']).size().to_dict()
        temp_df['ReceiptMatchesCount'] = temp_df.apply(lambda row: receipt_counts[row['receipt_id']],axis=1)

        feature_list = ['DateMappingMatch','AmountMappingMatch','DescriptionMatch',
                        'DifferentPredictedTime', 'TimeMappingMatch', 'PredictedNameMatch', 
                        'ShortNameMatch', 'DifferentPredictedDate', 'PredictedAmountMatch', 'PredictedTimeCloseMatch']

        #StatisticsFeatures
        mean_df   = temp_df.groupby(['receipt_id']).mean()
        median_df = temp_df.groupby(['receipt_id']).median()
        std_df    = temp_df.groupby(['receipt_id']).std()
        rank_df   = temp_df.groupby(['receipt_id']).rank(pct=True)
        for each_feature in feature_list:
            temp_dict = mean_df[each_feature].to_dict()
            temp_df[each_feature+'_mean']   = temp_df.apply(lambda row: temp_dict[row['receipt_id']],axis=1)
            temp_dict = median_df[each_feature].to_dict()
            temp_df[each_feature+'_median'] = temp_df.apply(lambda row: temp_dict[row['receipt_id']],axis=1)
            temp_dict = std_df[each_feature].to_dict()
            temp_df[each_feature+'_std']    = temp_df.apply(lambda row: temp_dict[row['receipt_id']],axis=1).fillna(0)
            temp_df[each_feature+'_rank']   = rank_df[each_feature]

        #Drop feature_transaction_id and matched_transaction_id columns
        for each_col in ['feature_transaction_id','matched_transaction_id']:
            if each_col in temp_df.columns:
                temp_df.drop(each_col,axis=1,inplace=True)
                
        return temp_df
        
    def fit(self,model_file):
        self.model_file = model_file
        feature_eng_df = self.FeatureEngineering(self.input_data)
        feature_eng_df.drop(['receipt_id'],axis=1,inplace=True)
        X_train  = feature_eng_df.loc[:,feature_eng_df.columns != self.outcome_feature]
        y_train  = feature_eng_df.loc[:,feature_eng_df.columns == self.outcome_feature]
        self.scaler = MinMaxScaler()
        self.scaler.fit(X_train)
        X_train_scaled = pd.DataFrame(self.scaler.transform(X_train),columns=X_train.columns)
        self.xgb_clf = XGBClassifier(max_depth=10,random_state=self.model_seed,n_estimators=100,n_jobs=-1)
        self.xgb_clf.fit(X_train_scaled, y_train)
        pickle.dump(self.xgb_clf, open(model_file, 'wb'))
        
    def predict(self,validation_file,predictions_file):
        if self.model_file is None:
            print("Please fit the model before making predictions")
        else:
            if path.exists(validation_file):
                if self.xgb_clf is None:
                    self.xgb_clf = pickle.load(open(self.model_file,'rb'))
                val_df       = pd.read_csv(validation_file,delimiter=':', thousands=',')
                fe_df        = self.FeatureEngineering(val_df).drop(['receipt_id'],axis=1)
                fe_df_scaled = pd.DataFrame(self.scaler.transform(fe_df),columns=fe_df.columns)
                X_test       = fe_df_scaled.loc[:,fe_df_scaled.columns != self.outcome_feature]
                y_predict_proba = self.xgb_clf.predict_proba(X_test)[:,1]
                val_df['Match_Probability'] = y_predict_proba
                val_df = val_df.groupby(['receipt_id']).apply(lambda x: x.sort_values(['Match_Probability'],
                                                              ascending=False)).reset_index(drop=True)
                val_df.drop(['Match_Probability'],axis=1,inplace=True)
                val_df.to_csv(predictions_file,index=False)
            else:
                print("Validation file doesnt exist. Please check again")
                
if __name__ == "__main__":
     match_obj = receipt_match("data.csv")
     match_obj.model_stats(0.3)
     match_obj.fit('model_file.pk')
     match_obj.predict('validation_file.csv','predictions.csv')