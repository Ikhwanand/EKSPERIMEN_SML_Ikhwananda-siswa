import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



def preprocessing_dataset(dataset_path=None, dataframe=None):
    if dataset_path:
        df = pd.read_csv(dataset_path)
    else:
        df = dataframe
    
    df = df.copy()
    
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['O']).columns.tolist()
    
    # Hapus 'CustomerID' dan 'Churn' dari fitur numerik jika ada
    if 'CustomerID' in numeric_features:
        numeric_features.remove('CustomerID')
    
    if 'Churn' in categorical_features:
        categorical_features.remove('Churn')
        
    # Buat pipeline untuk preprocessing numerik
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), # Mengisi missing values dengan mean
        ('scaler', StandardScaler()) # Scaling fitur numerik
    ])
    
    # Buat pipeline untuk preprocessing kategorikal
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Mengisi missing values dengan modus
        ('onehot', OneHotEncoder(handle_unknown='ignore')) # One-hot encoding figure kategorikal
    ])
    
    # Gabungkan preprocessing untuk kolom numerik dan kategorikal
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    preprocessing_data = preprocessor.fit_transform(df)
    
    # Ubah menjadi DataFrame hasil preprocessing
    preprocessing_df = pd.DataFrame(preprocessing_data)
    
    return pd.concat([preprocessing_df, df['Churn']], axis=1)


