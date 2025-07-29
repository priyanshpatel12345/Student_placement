import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file:str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformer_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            numerical_cols = ["age", "cgpa", "communication_skill", "attendance_percentage"]
            categorical_cols = ["gender", "department", "year", "internship_done"]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaling", StandardScaler(with_mean=False))
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoding", OneHotEncoder(handle_unknown='ignore')),
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_cols),
                    ("cat_pipeline", cat_pipeline, categorical_cols)
                ]
            )

            return preprocessor
    
        except Exception as e:
            raise CustomException(e, sys)    
        
    def initiate_data_transformation(self, train_data, test_data):
        try:
            train_df = pd.read_csv(train_data)
            test_df = pd.read_csv(test_data)

            logging.info("Read train and test data Completed")

            logging.info("obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformation_object()

            target_feature = "placement"

            input_feature_train_df = train_df.drop(columns=[target_feature], axis=1)
            target_feature_train_df = train_df[target_feature]


            input_feature_test_df = test_df.drop(columns=[target_feature], axis=1)
            target_feature_test_df = test_df[target_feature]

            target_feature_train_df = target_feature_train_df.map({"No":0, "Yes":1})
            target_feature_test_df = target_feature_test_df.map({"No":0, "Yes":1})

            print(target_feature_train_df)
            logging.info(
                "Applying preprocessing object on training dataFrame and testing Dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.concatenate(
            [input_feature_train_arr, np.array(target_feature_train_df).reshape(-1, 1)],
            axis=1
            )

            test_arr = np.concatenate(
            [input_feature_test_arr, np.array(target_feature_test_df).reshape(-1, 1)],
            axis=1
            )

            logging.info("saved Preprocessing Object")

            save_object(
                file_path=self.data_transformer_config.preprocessor_obj_file,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformer_config.preprocessor_obj_file
            )

        except Exception as e:
            raise CustomException(e, sys)