import joblib as jb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, classification_report as sk_classification_report, accuracy_score

FILEDIR = "Model/Dataset/DataFIxParkirLiarv1.csv"

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

def result(FILENAME, Jenis_Kendaraan, Deskripsi_Masalah, waktu, y_true=None):
    try:
        Model_loaded = read_model(FILENAME)

        errors = []
        if not Jenis_Kendaraan:
            errors.append("Jenis_Kendaraan is empty.")
        if not Deskripsi_Masalah:
            errors.append("Deskripsi_Masalah is empty.")
        if not waktu:
            errors.append("Waktu is empty.")

        if errors:
            print("Error:", " ".join(errors))
            return [], [], [], [], []

        NewData = pd.DataFrame({
            'Deskripsi Masalah': Deskripsi_Masalah if isinstance(Deskripsi_Masalah, list) else [Deskripsi_Masalah],
            'Jenis Kendaraan': Jenis_Kendaraan if isinstance(Jenis_Kendaraan, list) else [Jenis_Kendaraan],
            'waktu': waktu if isinstance(waktu, list) else [waktu] 
        })

        y_predictions = Model_loaded.predict(NewData)

        if hasattr(Model_loaded, 'predict_proba'):
            confidence_scores = Model_loaded.predict_proba(NewData).max(axis=1)
        else:
            confidence_scores = ['N/A'] * len(y_predictions)

        NewData['Status Pelaporan'] = y_predictions
        NewData['Akurasi Prediksi'] = confidence_scores

        print(NewData[['Deskripsi Masalah', 'Jenis Kendaraan', 'Status Pelaporan', 'Akurasi Prediksi', 'waktu']])

        return (
            NewData['Deskripsi Masalah'].tolist(),
            NewData['Jenis Kendaraan'].tolist(),
            NewData['Status Pelaporan'].tolist(),
            NewData['waktu'].tolist(),
            NewData['Akurasi Prediksi'].tolist()
        )

    except Exception as e:
        print(f"Error in result function: {str(e)}")
        return [], [], [], [], []
