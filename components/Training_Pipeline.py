import pandas as pd
from components.Data_Transformation import data_preprocessing, text_vectorization
from components.Model_Training import model_trainer
from sklearn.feature_extraction.text import TfidfVectorizer

Dataset_file_path = 'E:\\DataScience_projects\\Movie_recommendation-2\\Dataset\\IMDB Dataset.csv'
X,y = data_preprocessing(Dataset_file_path)

X_tv = text_vectorization(X)


accuracy = model_trainer(X_tv,y)
print("Accuracy of the model is : ", accuracy)
