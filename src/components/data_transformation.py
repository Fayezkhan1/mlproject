import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.DataTransformation_config=DataTransformationConfig()

    def get_data_tranformer_object(self):
        try:
            numericals_columns=['writing_score','reading_score']
            categorical_columns=[
                'gender',
                'race_ethinicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            cat_pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_enoder",OneHotEncoder()),
                    ("scaler",StandardScaler())
                ]
            )
            
            logging.info("numerical columns standard scaling completed")
            loggin.info("categorical columns encoding completed")
            
        except:
            pass
