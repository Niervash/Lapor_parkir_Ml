import joblib as jb
import pandas as pd
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, classification_report as sk_classification_report

def read_model(filename):
    model = jb.load(filename)
    print(f"Model loaded from {filename}")
    return model

def confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = sk_confusion_matrix(y_test, y_pred)
    return cm

def classification_report(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = sk_classification_report(y_test, y_pred, output_dict=True)
    return report

def result(FILENAME, Jenis_Kendaraan, Deskripsi_Masalah, waktu):
    try:
        # Load the model
        model_loaded = read_model(FILENAME)

        errors = []
        if not Jenis_Kendaraan:
            errors.append("Jenis_Kendaraan is empty.")
        if not Deskripsi_Masalah:
            errors.append("Deskripsi_Masalah is empty.")
        if not waktu:
            errors.append("Waktu is empty.")

        if errors:
            print("Error:", " ".join(errors))
            return [], [], []  # Return empty lists if inputs are invalid

        # Create a DataFrame for new data with the correct column names
        NewData = pd.DataFrame({
            'Deskripsi Masalah': Deskripsi_Masalah if isinstance(Deskripsi_Masalah, list) else [Deskripsi_Masalah],
            'Jenis Kendaraan': Jenis_Kendaraan if isinstance(Jenis_Kendaraan, list) else [Jenis_Kendaraan],
            'waktu': waktu if isinstance(waktu, list) else [waktu] 
        })

        # Make predictions
        y_predictions = model_loaded.predict(NewData)

        # Append predicted statuses to DataFrame
        NewData['Status Pelaporan'] = y_predictions
        # Print relevant columns (adjust based on your needs)
        print(NewData[['Deskripsi Masalah', 'Jenis Kendaraan', 'Status Pelaporan','waktu']])

        # Return the necessary values
        return NewData['Deskripsi Masalah'].tolist(), NewData['Jenis Kendaraan'].tolist(),NewData['waktu'].tolist(), y_predictions.tolist()
    
    except Exception as e:
        print(f"Error in result function: {str(e)}")
        return [], [], []