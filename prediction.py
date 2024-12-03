import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization
import os

def predict_hemoglobin_level(values):
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model("xgb_model.json")
    print("Model loaded successfully!")


    combined_left_eye = sum(values['left_eye'])/ 3
    combined_left_palm = sum(values['right_palm']) / 3
    combined_right_nail = sum(values['right_fingernail']) / 3

    features = np.array([
        combined_left_eye,
        combined_left_palm,
        combined_right_nail,
        values['right_fingernail'][0],
        values['right_fingernail'][1],
        values['right_fingernail'][2]
    ]).reshape(1, -1)
    predicted_hb_level = xgb_model.predict(features)
    return predicted_hb_level[0]
