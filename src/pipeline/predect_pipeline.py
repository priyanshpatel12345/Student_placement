import sys
import pandas as pd
import os

from src.exception import CustomException
from src.utils import load_object

class predictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
        age: int,
        cgpa: float,
        communication_skill: int,
        attendance_percentage:float,
        gender: str,
        department: str,
        year: str,
        internship_done: str):

        self.age = age
        self.cgpa = cgpa
        self.communication_skill = communication_skill
        self.attendance_percentage = attendance_percentage
        self.gender = gender
        self.department = department
        self.year = year
        self.internship_done = internship_done
     
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
            "age": [self.age],
            "cgpa": [self.cgpa],
            "communication_skill": [self.communication_skill],
            "attendance_percentage": [self.attendance_percentage],
            "gender": [self.gender],
            "department": [self.department],
            "year": [self.year],
            "internship_done": [self.internship_done]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)