import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve



class CleaningPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.LotFrontage_median = X['LotFrontage'].median()
        self.most_common_electrical = X['Electrical'].mode()[0]
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        
        # Remove duplicates
        X_transformed.drop_duplicates(inplace=True)
        
        # MSSubClass as category
        # This is stored as a number, but behaves as a category
        X_transformed['MSSubClass'] = X_transformed['MSSubClass'].astype('object')
        
        # Fill LotFrontage NaN with median
        # The NaN values here mostly mean incomplete data, so it will be filled with the median (because it is continuous data)
        X_transformed.loc[X_transformed['LotFrontage'].isna(), 'LotFrontage'] = self.LotFrontage_median
        
        # MasVnrType NaN handling
        # NaN here means that it does not exist, so MasVnrArea should be 0
        X_transformed.loc[X_transformed['MasVnrType'].isna(), 'MasVnrArea'] = 0
        
        # Electrical NaN handling
        # NaN here means that it is unknown, so we will fill it with the most common value
        X_transformed.loc[X_transformed['Electrical'].isna(), 'Electrical'] = self.most_common_electrical
        
        # GarageYrBlt binning
        # When a house does not have a garage, the year for it is NaN. Giving all of these values some year value would be wrong, so I decided to put these years into bins, where the NaNs will all go inside the same bin
        bins = [0, 1900, 1920, 1940, 1960, 1980, 2000, 2020]
        labels = ['None', '1900-1919', '1920-1939', '1940-1959', '1960-1979', '1980-1999', '2000-2019']
        X_transformed['GarageYrBltInt'] = pd.cut(X_transformed['GarageYrBlt'].fillna(0), bins=bins, labels=labels, right=False).astype('object')
        X_transformed.drop(columns=['GarageYrBlt'], inplace=True)
        
        # Fill remaining NaN with 'None' for cat and 0 for num
        # Fill category columns with NaN with 'None' and numerical columns with 0
        for col in X_transformed.columns:
            if X_transformed[col].dtype == 'str' or X_transformed[col].dtype == 'object':
                X_transformed[col] = X_transformed[col].fillna('None')
            else:
                X_transformed[col] = X_transformed[col].fillna(0)
        
        return X_transformed


    
class FeatureEngineeringPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, one_hot_cats=None, target_cats=None, frequency_cats=None):
        self.one_hot_cats = one_hot_cats
        self.target_cats = target_cats
        self.frequency_cats = frequency_cats
    
    def fit(self, X, y):
        if self.target_cats is not None:
            x_tmp = pd.concat([X, y], axis=1)
            self.target_mean_map = {}
            for col in self.target_cats:
                self.target_mean_map[col] = x_tmp.groupby(col)[y.name].mean()
        if self.frequency_cats is not None:
            self.frequency_map = {}
            for col in self.frequency_cats:
                self.frequency_map[col] = X[col].value_counts(normalize=True)
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        
        # One-hot encoding
        if self.one_hot_cats is not None:
            one_hot_encoded = X_transformed = pd.get_dummies(X_transformed, columns=self.one_hot_cats, drop_first=True)
        # Target encoding
        if self.target_cats is not None:
            for col in self.target_cats:
                X_transformed[col] = X_transformed[col].map(self.target_mean_map[col])
            X_transformed = X_transformed.fillna(0)
        # Frequency encoding
        if self.frequency_cats is not None:
            for col in self.frequency_cats:
                X_transformed[col] = X_transformed[col].map(self.frequency_map[col])
            X_transformed = X_transformed.fillna(0)
        # Label encoding
        encoder = LabelEncoder()
        for col in X_transformed.select_dtypes(include=['object']).columns:
            X_transformed[col] = encoder.fit_transform(X_transformed[col])
        
        return X_transformed



class FeatureSelectionPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, selected_features=None):
        self.selected_features = selected_features
    
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        if self.selected_features is not None:
            for col in self.selected_features:
                if col not in X_transformed.columns:
                    X_transformed[col] = False
            return X_transformed[['Id'] + self.selected_features]
        return X_transformed


    
class ExtensiveFeatureSelectionPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.8, n_features_to_select=10):
        self.threshold = threshold
        self.n_features_to_select = n_features_to_select
    
    def fit(self, X, y):
        X_transformed = X.copy()
        to_drop = correlation_filter(X_transformed, y, threshold=self.threshold)
        a_X = X.drop(columns=to_drop)
        self.selected_features = rfe(a_X, y, n_features_to_select=self.n_features_to_select)
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        return X_transformed[['Id'] + self.selected_features]



def check_columns(df):
    cat_cols = df.select_dtypes(include=['object']).columns
    num_cols = df.select_dtypes(exclude=['object']).columns
    
    print(f"Categorical columns: {len(cat_cols)}")
    print(cat_cols)
    print(f"Numerical columns: {len(num_cols)}")
    print(num_cols)


    
def grid_search_results(grid_search):
    result = pd.DataFrame(grid_search.cv_results_)
    result = result.loc[:, ~result.columns.str.startswith('split')]
    result = result.loc[:, ~result.columns.str.startswith('std')]
    result = result.loc[:, ~result.columns.str.endswith('time')]
    result = result.sort_values(by='rank_test_neg_mean_squared_error')
    result = result.drop(columns=['rank_test_neg_root_mean_squared_error', 'rank_test_neg_mean_absolute_error', 'rank_test_neg_median_absolute_error', 'rank_test_r2', 'rank_test_explained_variance'])
    result['MSE_test'] = -result['mean_test_neg_mean_squared_error']
    result['RMSE_test'] = -result['mean_test_neg_root_mean_squared_error']
    result['MAE_test'] = -result['mean_test_neg_mean_absolute_error']
    result['MedAE_test'] = -result['mean_test_neg_median_absolute_error']
    result['R2_test'] = result['mean_test_r2']
    result['EV_test'] = result['mean_test_explained_variance']
    result = result.drop(columns=['mean_test_neg_mean_squared_error', 'mean_test_neg_root_mean_squared_error', 'mean_test_neg_mean_absolute_error', 'mean_test_neg_median_absolute_error', 'mean_test_r2', 'mean_test_explained_variance'])
    result['MSE_train'] = -result['mean_train_neg_mean_squared_error']
    result['RMSE_train'] = -result['mean_train_neg_root_mean_squared_error']
    result['MAE_train'] = -result['mean_train_neg_mean_absolute_error']
    result['MedAE_train'] = -result['mean_train_neg_median_absolute_error']
    result['R2_train'] = result['mean_train_r2']
    result['EV_train'] = result['mean_train_explained_variance']
    result = result.drop(columns=['mean_train_neg_mean_squared_error', 'mean_train_neg_root_mean_squared_error', 'mean_train_neg_mean_absolute_error', 'mean_train_neg_median_absolute_error', 'mean_train_r2', 'mean_train_explained_variance'])
    result['Rank'] = result['rank_test_neg_mean_squared_error']
    result = result.drop(columns=['rank_test_neg_mean_squared_error'])
    result.head(10)
    return result



def plot_grid_search_results(grid_search):
    results = grid_search_results(grid_search)
    
    # Plot RMSE for train and test
    plt.figure(figsize=(12, 6))
    plt.plot(results['Rank'], results['RMSE_train'], label='Train RMSE', marker='o')
    plt.plot(results['Rank'], results['RMSE_test'], label='Test RMSE', marker='o')
    plt.xlabel('Rank')
    plt.ylabel('RMSE')
    plt.title('Grid Search Results: RMSE')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot R2 for train and test
    plt.figure(figsize=(12, 6))
    plt.plot(results['Rank'], results['R2_train'], label='Train R2', marker='o')
    plt.plot(results['Rank'], results['R2_test'], label='Test R2', marker='o')
    plt.xlabel('Rank')
    plt.ylabel('R2')
    plt.title('Grid Search Results: R2')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def residual_plot(model, X, y):
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    plt.figure(figsize=(12, 6))
    plt.scatter(y_pred, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    plt.show()
    
def predicted_vs_actual_plot(model, X, y):
    y_pred = model.predict(X)
    
    plt.figure(figsize=(12, 6))
    plt.scatter(y, y_pred)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.grid(True)
    plt.show()
    
def learning_curve_plot(model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5):    
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, train_sizes=train_sizes, cv=cv)
    
    # Calculate the mean and standard deviation of the training and testing scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot the learning curve
    plt.figure(figsize=(12, 6))
    plt.plot(train_sizes, train_mean, label='Training Score', marker='o')
    plt.plot(train_sizes, test_mean, label='Cross-Validation Score', marker='o')
    
    # Plot the fill between the standard deviation
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


    
def correlation_filter(X, y, threshold=0.8, log=False):
    corr = X.drop(columns=['Id']).corr().abs()

    to_drop_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i):
            if corr.iloc[i, j] > threshold:
                to_drop_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

    if log:
        print("Highly correlated pairs:")
        for col1, col2, corr_val in to_drop_pairs:
            print(f"{col1} and {col2}: {corr_val}")
        
    to_drop = set()
    for col1, col2, _ in to_drop_pairs:
        if abs(X[col1].corr(y)) > abs(X[col2].corr(y)):
            to_drop.add(col2)
        else:
            to_drop.add(col1)
            
    to_drop = list(to_drop)
    if log:
        print(f"Columns to drop: {to_drop}")
    
    return to_drop

def rfe(X, y, n_features_to_select=10, log=False):
    X_t = X.drop(columns=['Id'])

    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=n_features_to_select, step=1)
    rfe.fit(X_t, y)

    selected_features = X_t.columns[rfe.support_].tolist()
    
    if log:
        print(f"Selected features: {selected_features}")
    
    return selected_features
