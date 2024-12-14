import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import io
import pygwalker as pyg
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
import pickle
import xgboost as xgb
import os
import zipfile #Added this line


# App title
st.title("Cement Compression Strength: Out-of-Sample Prediction")

# Function to load and clean data (cached)
@st.cache_data
def load_data(file, sheet_name):
    # Load the selected sheet
    data = pd.read_excel(file, sheet_name=sheet_name)

    # Clean column names: remove extra spaces and capitalize
    data.columns = data.columns.str.strip().str.upper()

    # Ensure 'FECHA' is in datetime format and truncate to date only
    if 'FECHA' in data.columns:
        data['FECHA'] = pd.to_datetime(data['FECHA'], errors='coerce').dt.date

    return data

# Function to filter data by date (cached)
@st.cache_data
def filter_data_by_date(data, cutoff_date):
    return data[data['FECHA'] >= cutoff_date]

# Add tabs to the app
tab1, tab2 = st.tabs(["Data Input", "Predictions"])

# Tab 1: Data Input
with tab1:
    # File uploader for Excel files
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

    if uploaded_file is not None:
        try:
            # Load the Excel file and retrieve sheet names
            excel_data = pd.ExcelFile(uploaded_file)
            sheet_names = excel_data.sheet_names

            # Let the user select a sheet
            st.subheader("Select the sheet to load")
            selected_sheet = st.selectbox("Choose a sheet:", sheet_names)

            # Load the data (cached)
            data = load_data(uploaded_file, selected_sheet)

            # Ensure necessary columns exist
            required_columns = ['FECHA', 'MOLINO', 'TIPO', 'SO3', 'P.F', 'BLAINE', 'FINURA', 'F. CLINKER', 'C3S CLINKER', 'C2S CLINKER', 'C3A CLINKER', 'C4AF CLINKER']
            if all(col in data.columns for col in required_columns):
                 # Drop rows with invalid dates
                data = data.dropna(subset=['FECHA'])

                # Display the data before filtering
                st.subheader("Preview of Original Data (with Cleaned Column Names)")
                st.write(data)

                # Filter by date
                st.subheader("Filter Data by Date")
                cutoff_date = st.date_input(
                    "Select a cutoff date:",
                    value=data['FECHA'].min() if not data.empty else None,
                )

                 # Filter the data by date (cached)
                filtered_data = filter_data_by_date(data, cutoff_date)

                 # Cleaning: Remove rows with 0, negative, or invalid values
                st.subheader("Clean Data")
                st.markdown("Removing rows containing **0**, **negative values**, or **invalid values** in any column.")
                cleaned_data = filtered_data.copy()
                cleaned_data = cleaned_data.replace(0, pd.NA)  # Replace 0 with NaN
                cleaned_data = cleaned_data.applymap(
                    lambda x: pd.NA if isinstance(x, (int, float)) and x < 0 else x
                )  # Replace negatives with NaN
                cleaned_data = cleaned_data.dropna()  # Drop rows with any NaN values
                numfeatures = [col for col in cleaned_data.columns if col not in ['MOLINO', 'TIPO','FECHA']]
                for numfeature in numfeatures:
                    cleaned_data[numfeature] = pd.to_numeric(cleaned_data[numfeature], errors='coerce')
                # Display the cleaned data
                st.subheader("Cleaned Data")
                st.write(cleaned_data)
            else:
                missing_cols = [col for col in required_columns if col not in data.columns]
                st.error(f"The following required columns are missing from the dataset: {', '.join(missing_cols)}")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
    else:
        st.info("Please upload and clean the data in Tab 1 first.")


# Tab 2: Predictions
with tab2:
    st.subheader("Upload Trained Models")
    # File uploader for zip files
    uploaded_models_zip = st.file_uploader("Upload your models (.zip)", type=["zip"])

    if uploaded_models_zip is not None:
        try:
            # Initialize session state for models if not already present
            if 'loaded_models' not in st.session_state:
                st.session_state.loaded_models = {}
            
            # Load Models
            st.markdown("#### Loading Models...")
            with zipfile.ZipFile(uploaded_models_zip, 'r') as zip_file:
                 for file_name in zip_file.namelist():
                     if file_name.endswith('.pkl'): #ensure only pkl files are read
                         with zip_file.open(file_name) as model_file:
                             model = pickle.load(model_file)
                             parts = file_name.replace('.pkl', '').split('_')
                             if len(parts) == 4:
                                 _, molino, tipo, model_name = parts
                                 if (molino, tipo) not in st.session_state.loaded_models:
                                     st.session_state.loaded_models[(molino, tipo)] = {}
                                 st.session_state.loaded_models[(molino, tipo)][model_name] = model
                             else:
                                st.error(f"Skipping model with invalid filename format: {file_name}")
                                 
            st.success("Models loaded successfully!")
        except Exception as e:
            st.error(f"An error occurred while processing the model file: {e}")

        if 'cleaned_data' in locals() and 'loaded_models' in st.session_state and st.session_state.loaded_models:
           st.markdown("### Performing Out-of-Sample Predictions")
           
           # Make a copy of the data to avoid modifying the original
           prediction_data = cleaned_data.copy()
            
           # Create empty columns to store the predictions
           prediction_data['R1D'] = pd.NA
           prediction_data['R3D'] = pd.NA
           prediction_data['R7D'] = pd.NA
           prediction_data['R28D'] = pd.NA

           # Predict for each row in the dataframe
           for index, row in prediction_data.iterrows():
               molino = row['MOLINO']
               tipo = row['TIPO']

                # Check if model exists for this MOLINO and TIPO combination
               if (molino, tipo) in st.session_state.loaded_models:
                   models = st.session_state.loaded_models[(molino, tipo)]
                   features = [col for col in prediction_data.columns if col not in ['R1D', 'R3D', 'R7D', 'R28D', 'MOLINO', 'TIPO', 'FECHA']]
                
                   try:
                       # Stage 1: Predict R1D
                       X_R1D = row[features].values.reshape(1, -1)
                       R1D_pred = models['R1D'].predict(X_R1D)[0]
                       prediction_data.loc[index,'R1D'] = R1D_pred

                       # Stage 2: Predict R3D
                       X_R3D = row[features].to_list()
                       X_R3D.append(R1D_pred)
                       X_R3D = np.array(X_R3D).reshape(1, -1)
                       R3D_pred = models['R3D'].predict(X_R3D)[0]
                       prediction_data.loc[index,'R3D'] = R3D_pred

                        # Stage 3: Predict R7D
                       X_R7D = row[features].to_list()
                       X_R7D.append(R1D_pred)
                       X_R7D.append(R3D_pred)
                       X_R7D = np.array(X_R7D).reshape(1, -1)
                       R7D_pred = models['R7D'].predict(X_R7D)[0]
                       prediction_data.loc[index,'R7D'] = R7D_pred

                       # Stage 4: Predict R28D
                       X_R28D = row[features].to_list()
                       X_R28D.append(R1D_pred)
                       X_R28D.append(R3D_pred)
                       X_R28D.append(R7D_pred)
                       X_R28D = np.array(X_R28D).reshape(1, -1)
                       R28D_pred = models['R28D'].predict(X_R28D)[0]
                       prediction_data.loc[index,'R28D'] = R28D_pred
                   except Exception as e:
                       st.error(f"Error during prediction for MOLINO: {molino}, TIPO: {tipo} - {e}")
                
               else:
                  st.warning(f"No model found for MOLINO: {molino}, TIPO: {tipo}. Skipping prediction.")
           st.subheader("Prediction Results")
           st.write(prediction_data)

           # Function to convert the DataFrame to an Excel file
           def to_excel(df):
                output = BytesIO()
                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                df.to_excel(writer, sheet_name='Predictions', index=False)
                writer.close()
                processed_data = output.getvalue()
                return processed_data

           #Download button
           st.download_button(
                label="Download Predictions (.xlsx)",
                data=to_excel(prediction_data),
                file_name='predictions.xlsx',
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
           )

        elif 'cleaned_data' not in locals():
            st.info("Please upload and clean the data in Tab 1 first.")
        elif 'loaded_models' not in st.session_state or not st.session_state.loaded_models:
            st.info("Please upload the models in Tab 2 first.")

# Footer
st.markdown("---")
st.markdown("Developed by CepSA with ❤️ using Streamlit.")
