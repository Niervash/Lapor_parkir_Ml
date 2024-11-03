import joblib as jb
import pandas as pd
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, classification_report as sk_classification_report

def read_model(filename):
    Model = jb.load(filename)
    print(f"Model loaded from {filename}")
    return Model

def confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = sk_confusion_matrix(y_test, y_pred)
    return cm


def classification_report(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = sk_classification_report(y_test, y_pred, output_dict=True)
    return report

def result(FILENAME, Lokasi, Identitas_Petugas):
    try:
        # Load the model
        Model_loaded = read_model(FILENAME)

        # Create a DataFrame for new data with the correct column names
        NewData = pd.DataFrame({
            'Lokasi': Lokasi,
            'Identitas Petugas': Identitas_Petugas  # Ensure column name matches the model's expectations
        })

        # Make predictions
        y_predictions = Model_loaded.predict(NewData)

        # Append predicted statuses to DataFrame
        NewData['Status Pelaporan'] = y_predictions
        print(NewData[['Lokasi', 'Identitas Petugas', 'Status Pelaporan']])

        # Return the necessary values
        return Lokasi, NewData['Identitas Petugas'].tolist(), y_predictions.tolist()
    except Exception as e:
        print(f"Error in result function: {str(e)}")
        return [], [], []