import joblib as jb
import pandas as pd
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, classification_report as sk_classification_report, accuracy_score

def read_model(filename):
    Model = jb.load(filename)
    print(f"Model loaded from {filename}")
    return Model

def confusion_matrix(model, X_test, y_test, return_accuracy=False):
    y_pred = model.predict(X_test)
    cm = sk_confusion_matrix(y_test, y_pred)

    if return_accuracy:
        accuracy = cm.diagonal().sum() / cm.sum()
        return cm, accuracy

    return cm

def classification_report(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = sk_classification_report(y_test, y_pred, output_dict=True)
    return report

def result(FILENAME, Lokasi, Identitas_Petugas, y_true=None):
    try:
        # Load the model
        Model_loaded = read_model(FILENAME)

        errors = []
        if not Lokasi:
            errors.append("Lokasi is empty.")
        if not Identitas_Petugas:
            errors.append("Identitas_Petugas is empty.")

        if errors:
            print("Error:", " ".join(errors))
            return [], []  # Return empty lists if inputs are invalid

        # Create a DataFrame for new data with the correct column names
        NewData = pd.DataFrame({
            'Lokasi': Lokasi if isinstance(Lokasi, list) else [Lokasi],
            'Identitas Petugas': Identitas_Petugas if isinstance(Identitas_Petugas, list) else [Identitas_Petugas]
        })

        # Make predictions
        y_predictions = Model_loaded.predict(NewData)

        # Calculate accuracy if y_true is provided
        if y_true is not None:
            accuracy = accuracy_score(y_true, y_predictions)
            NewData['Akurasi Prediksi'] = [accuracy] * len(y_predictions)
        else:
            NewData['Akurasi Prediksi'] = ['N/A'] * len(y_predictions)

        # Append predicted statuses to DataFrame
        NewData['Status Pelaporan'] = y_predictions
        print(NewData[['Lokasi', 'Identitas Petugas', 'Akurasi Prediksi', 'Status Pelaporan']])

        # Return the necessary values
        return NewData['Lokasi'].tolist(), NewData['Identitas Petugas'].tolist(), NewData['Status Pelaporan'].tolist()

    except Exception as e:
        print(f"Error in result function: {str(e)}")
        return [], [], []
