# This script shows you how to make a submission using a few
# useful Python libraries.
# It gets a public leaderboard score of 0.76077.
# Maybe you can tweak it and do better...?

import pandas as pd
import xgboost as xgb
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import accuracy_score

# We'll impute missing values using the median for numeric columns and the most
        # common value for string columns.
        # This is based on some nice code by 'sveitser' at http://stackoverflow.com/a/25562948


class DataFrameImputer(TransformerMixin):
    def fit(self, X):
        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
                              index=X.columns)
        return self

    def transform(self, X):
        return X.fillna(self.fill)


class BlackBox:

    def __init__(self):
        self.train_X = None
        self.train_Y = None
        self.validation_X = None
        self.validation_Y = None
        self.test_X = None
        self.gbm = None
        self.test_X_original = None
        self.train_original = None

    def get_data(self):
        if self.train_X is not None:
            return self.train_X, self.train_Y, self.validation_X, self.validation_Y, self.test_X

        # Load the data
        self.test_X_original = pd.read_csv('/Users/yz/Projects/kaggle/datasets/titanic/test.csv', header=0)
        self.train_orginal = pd.read_csv('/Users/yz/Projects/kaggle/datasets/titanic/train.csv', header=0)

        feature_columns_to_use = ['Pclass', 'Sex', 'Age', 'Fare', 'Parch']
        non_numeric_columns = ['Sex']

        # Join the features from train and test together before imputing missing values,
        # in case their distribution is slightly different
        big_X = self.train_orginal[feature_columns_to_use].append(self.test_X_original[feature_columns_to_use])
        big_X_imputed = DataFrameImputer().fit_transform(big_X)

        # XGBoost doesn't (yet) handle categorical features automatically, so we need to change
        # them to columns of integer values.
        # See http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing for more
        # details and options
        le = LabelEncoder()
        for feature in non_numeric_columns:
            big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])

        # Prepare the inputs for the model
        split_pos = int(self.train_orginal.shape[0] * (2/3))
        self.train_X = big_X_imputed[0:split_pos].as_matrix()
        self.validation_X = big_X_imputed[split_pos:self.train_orginal.shape[0]].as_matrix()

        self.train_Y = self.train_orginal['Survived'][0:split_pos].as_matrix()
        self.validation_Y = self.train_orginal['Survived'][split_pos:self.train_orginal.shape[0]].as_matrix()

        self.test_X = big_X_imputed[self.train_orginal.shape[0]::].as_matrix()

        return self.train_X, self.train_Y, self.validation_X, self.validation_Y, self.test_X

    def run(self, max_depth, n_estimators, learning_rate, reg_lambda):
        train_X, train_Y, validation_X, validation_Y, test_X = self.get_data()

        self.gbm = xgb.XGBClassifier(max_depth=int(max_depth), n_estimators=int(n_estimators),
                                learning_rate=learning_rate,
                                reg_lambda=reg_lambda)\
            .fit(train_X, train_Y)

        tmp_Y = self.gbm.predict(validation_X)

        return accuracy_score(validation_Y, tmp_Y)

    def generate(self):
        submission = pd.DataFrame({'PassengerId': self.test_X_original['PassengerId'],
                                   'Survived': self.gbm.predict(self.test_X)})
        submission.to_csv("submission.csv", index=False)

# You can experiment with many other options here, using the same .fit() and .predict()
# methods; see http://scikit-learn.org
# This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost

'''
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)
predictions = gbm.predict(test_X)

# Kaggle needs the submission to have a certain format;
# see https://www.kaggle.com/c/titanic-gettingStarted/download/gendermodel.csv
# for an example of what it's supposed to look like.
submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],
                            'Survived': predictions })
submission.to_csv("submission.csv", index=False)
'''
if __name__ == '__main__':
    bb = BlackBox()
    acc = bb.run(7, 322, 0.0111, 7.5333)
    print(acc)

    bb.generate()

    print("done")