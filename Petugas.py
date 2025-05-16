import joblib as jb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, classification_report as sk_classification_report, accuracy_score

FILEDIR = "Model/Dataset/data_augmentasi_petugas.csv"

def split_data_to_test(test_size=0.2, random_state=42):
    try:
        data = pd.read_csv(FILEDIR)
        X = data.drop(columns=['Status Pelaporan'])  
        y = data['Status Pelaporan']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error in split_data_to_test: {str(e)}")
        return None, None, None, None

def read_model(filename):
    try:
        Model = jb.load(filename)
        print(f"Model loaded from {filename}")
        return Model
    except Exception as e:
        print(f"Error in read_model: {str(e)}")
        return None

def result(FILENAME, Lokasi, Identitas_Petugas):
    try:
        Model_loaded = read_model(FILENAME)
        if Model_loaded is None:
            return [], [], [], []

        errors = []
        if not Lokasi:
            errors.append("Lokasi is empty.")
        if not Identitas_Petugas:
            errors.append("Identitas_Petugas is empty.")

        if errors:
            print("Error:", " ".join(errors))
            return [], [], [], []

        NewData = pd.DataFrame({
            'Lokasi': Lokasi if isinstance(Lokasi, list) else [Lokasi],
            'Identitas Petugas': Identitas_Petugas if isinstance(Identitas_Petugas, list) else [Identitas_Petugas]
        })

        y_predictions = Model_loaded.predict(NewData)

        if hasattr(Model_loaded, 'predict_proba'):
            confidence_scores = Model_loaded.predict_proba(NewData).max(axis=1)

        else:
            confidence_scores = [None] * len(y_predictions)

        NewData['Status Pelaporan'] = y_predictions
        NewData['Akurasi Prediksi'] = confidence_scores

        print(NewData[['Lokasi', 'Identitas Petugas', 'Akurasi Prediksi', 'Status Pelaporan']])

        return NewData['Lokasi'].tolist(), NewData['Identitas Petugas'].tolist(), NewData['Akurasi Prediksi'].tolist(), NewData['Status Pelaporan'].tolist()

    except Exception as e:
        print(f"Error in result function: {str(e)}")
        return [], [], [], []
