import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import io
import pygwalker as pyg
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import sklearn.metrics as mt
import pickle
import xgboost as xgb

# App title
st.title("Cement Compression Strength: Data Viewer, Filtering, and Cleaning")

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
tab1, tab2, tab3, tab4 = st.tabs(["Data Cleaning", "Visualizations", "Descriptive Analytics", "Model Training"])

# Tab 1: Data Cleaning
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
            required_columns = ['FECHA', 'MOLINO', 'TIPO', 'R1D', 'R3D', 'R7D', 'R28D']
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

# Tab 2: Visualizations
with tab2:
    if 'cleaned_data' in locals():
        st.subheader("Resistance Data Visualizations")
        st.markdown("Filter by **MOLINO** and **TIPO** and choose a chart type to visualize resistance data.")

        # Filter options
        molinos = ['All'] + list(cleaned_data['MOLINO'].dropna().unique())
        selected_molino = st.selectbox("Select MOLINO:", molinos)
        tipos = ['All'] + list(cleaned_data['TIPO'].dropna().unique())
        selected_tipo = st.selectbox("Select TIPO:", tipos)

        # Filter the data
        filtered_viz_data = cleaned_data.copy()
        if selected_molino != 'All':
            filtered_viz_data = filtered_viz_data[filtered_viz_data['MOLINO'] == selected_molino]
        if selected_tipo != 'All':
            filtered_viz_data = filtered_viz_data[filtered_viz_data['TIPO'] == selected_tipo]

        # Select chart type
        chart_type = st.radio("Select Chart Type:", ['Histogram', 'Boxplot', 'Trend Chart'])

        # Resistance columns
        resistance_columns = ['R1D', 'R3D', 'R7D', 'R28D']

        # Generate visualizations
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()

        for i, col in enumerate(resistance_columns):
            if chart_type == 'Histogram':
                axes[i].hist(filtered_viz_data[col], bins=20, alpha=0.7, color='blue')
                axes[i].set_title(f'{col} - Histogram')
                axes[i].set_xlabel('Resistance')
                axes[i].set_ylabel('Frequency')
            elif chart_type == 'Boxplot':
                axes[i].boxplot(filtered_viz_data[col].dropna(), vert=True)  # Set boxplot to vertical
                axes[i].set_title(f'{col} - Boxplot')
                axes[i].set_ylabel('Resistance')
            elif chart_type == 'Trend Chart':
                axes[i].plot(filtered_viz_data['FECHA'], filtered_viz_data[col], marker='o', linestyle='-')
                axes[i].set_title(f'{col} - Trend Chart')
                axes[i].set_xlabel('Date')
                axes[i].set_ylabel('Resistance')
                axes[i].tick_params(axis='x', rotation=90)  # Rotate x-axis labels vertically

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Please upload and clean the data in Tab 1 first.")

# Tab 3: Descriptive Analytics
with tab3:
    if 'cleaned_data' in locals():
        st.subheader("Descriptive Analytics")
        st.markdown("Detailed descriptive statistics for resistance columns and frequency counts for **MOLINO** and **TIPO**.")

        # Descriptive statistics for resistance columns
        resistance_columns = ['R1D', 'R3D', 'R7D', 'R28D']
        st.subheader("Resistance Summary Statistics")
        summary_stats = {}
        for col in resistance_columns:
            column_data = cleaned_data[col].dropna()
            summary_stats[col] = {
                "Mean": column_data.mean(),
                "Median": column_data.median(),
                "Range": column_data.max() - column_data.min(),
                "Standard Deviation": column_data.std(),
                "Quantiles (25%, 50%, 75%)": column_data.quantile([0.25, 0.5, 0.75]).to_dict()
            }
        st.write(pd.DataFrame(summary_stats).transpose())

        # Frequency counts for MOLINO and TIPO
        st.subheader("Frequency Counts for MOLINO and TIPO")
        molino_counts = cleaned_data['MOLINO'].value_counts()
        tipo_counts = cleaned_data['TIPO'].value_counts()

        st.markdown("**MOLINO Counts**")
        st.write(molino_counts)

        st.markdown("**TIPO Counts**")
        st.write(tipo_counts)

        # Segmented descriptive statistics
        st.subheader("Segmented Descriptive Statistics")
        st.markdown("Descriptive statistics for resistance columns segmented by **MOLINO**, **TIPO**, and combinations of both.")

        # Segment by MOLINO
        st.subheader("Segmented by MOLINO")
        for molino, group_data in cleaned_data.groupby('MOLINO'):
            st.markdown(f"**MOLINO: {molino}**")
            molino_stats = {}
            for col in resistance_columns:
                column_data = group_data[col].dropna()
                molino_stats[col] = {
                    "Mean": column_data.mean(),
                    "Median": column_data.median(),
                    "Range": column_data.max() - column_data.min(),
                    "Standard Deviation": column_data.std(),
                    "Quantiles (25%, 50%, 75%)": column_data.quantile([0.25, 0.5, 0.75]).to_dict()
                }
            st.write(pd.DataFrame(molino_stats).transpose())

        # Segment by TIPO
        st.subheader("Segmented by TIPO")
        for tipo, group_data in cleaned_data.groupby('TIPO'):
            st.markdown(f"**TIPO: {tipo}**")
            tipo_stats = {}
            for col in resistance_columns:
                column_data = group_data[col].dropna()
                tipo_stats[col] = {
                    "Mean": column_data.mean(),
                    "Median": column_data.median(),
                    "Range": column_data.max() - column_data.min(),
                    "Standard Deviation": column_data.std(),
                    "Quantiles (25%, 50%, 75%)": column_data.quantile([0.25, 0.5, 0.75]).to_dict()
                }
            st.write(pd.DataFrame(tipo_stats).transpose())

        # Segment by MOLINO and TIPO combinations
        st.subheader("Segmented by MOLINO and TIPO")
        for (molino, tipo), group_data in cleaned_data.groupby(['MOLINO', 'TIPO']):
            st.markdown(f"**MOLINO: {molino}, TIPO: {tipo}**")
            combined_stats = {}
            for col in resistance_columns:
                column_data = group_data[col].dropna()
                combined_stats[col] = {
                    "Mean": column_data.mean(),
                    "Median": column_data.median(),
                    "Range": column_data.max() - column_data.min(),
                    "Standard Deviation": column_data.std(),
                    "Quantiles (25%, 50%, 75%)": column_data.quantile([0.25, 0.5, 0.75]).to_dict()
                }
            st.write(pd.DataFrame(combined_stats).transpose())
    else:
        st.info("Please upload and clean the data in Tab 1 first.")

# Tab 4:  Model Training


# Add a fourth tab for predictive modeling
#tab4 = st.tab("Predictive Modeling")

with tab4:
    if 'cleaned_data' in locals():
        st.subheader("Predictive Modeling by MOLINO and TIPO")
        st.markdown(
            "This section trains sequential models to predict the resistances (R1D, R3D, R7D, R28D) "
            "for each combination of **MOLINO** and **TIPO** using all available data."
        )

        # Ensure directories for saving models
        model_dir = "trained_models"
        os.makedirs(model_dir, exist_ok=True)

        # Segment data by MOLINO and TIPO
        for (molino, tipo), group_data in cleaned_data.groupby(['MOLINO', 'TIPO']):
            st.markdown(f"### MOLINO: {molino}, TIPO: {tipo}")

            # Drop rows with missing values (data cleaning handled in Tab 1)
            segment_data = group_data.dropna()

            # Initialize dictionary for storing trained models
            models = {}

            # Prepare variables for sequential modeling
            features = [col for col in segment_data.columns if col not in ['R1D', 'R3D', 'R7D', 'R28D', 'MOLINO', 'TIPO']]

            # Stage 1: Train model for R1D
            st.markdown("#### Stage 1: Predict R1D")
            X_R1D = segment_data[features]
            y_R1D = segment_data['R1D']
            model_R1D = XGBRegressor(objective='reg:squarederror')
            model_R1D.fit(X_R1D, y_R1D)
            models['R1D'] = model_R1D

            # Display feature importance for R1D
            st.markdown("**Feature Importance for R1D**")
            importance_R1D = pd.DataFrame({
                'Feature': X_R1D.columns,
                'Importance': model_R1D.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            st.write(importance_R1D)

            # Plot feature importance for R1D
            fig, ax = plt.subplots()
            ax.barh(importance_R1D['Feature'], importance_R1D['Importance'])
            ax.set_title("Feature Importance for R1D")
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
            st.pyplot(fig)

            # Stage 2: Train model for R3D
            st.markdown("#### Stage 2: Predict R3D")
            X_R3D = segment_data[features + ['R1D']]
            y_R3D = segment_data['R3D']
            model_R3D = XGBRegressor(objective='reg:squarederror')
            model_R3D.fit(X_R3D, y_R3D)
            models['R3D'] = model_R3D

            # Display feature importance for R3D
            st.markdown("**Feature Importance for R3D**")
            importance_R3D = pd.DataFrame({
                'Feature': X_R3D.columns,
                'Importance': model_R3D.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            st.write(importance_R3D)

            # Plot feature importance for R3D
            fig, ax = plt.subplots()
            ax.barh(importance_R3D['Feature'], importance_R3D['Importance'])
            ax.set_title("Feature Importance for R3D")
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
            st.pyplot(fig)

            # Stage 3: Train model for R7D
            st.markdown("#### Stage 3: Predict R7D")
            X_R7D = segment_data[features + ['R1D', 'R3D']]
            y_R7D = segment_data['R7D']
            model_R7D = XGBRegressor(objective='reg:squarederror')
            model_R7D.fit(X_R7D, y_R7D)
            models['R7D'] = model_R7D

            # Display feature importance for R7D
            st.markdown("**Feature Importance for R7D**")
            importance_R7D = pd.DataFrame({
                'Feature': X_R7D.columns,
                'Importance': model_R7D.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            st.write(importance_R7D)

            # Plot feature importance for R7D
            fig, ax = plt.subplots()
            ax.barh(importance_R7D['Feature'], importance_R7D['Importance'])
            ax.set_title("Feature Importance for R7D")
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
            st.pyplot(fig)

            # Stage 4: Train model for R28D
            st.markdown("#### Stage 4: Predict R28D")
            X_R28D = segment_data[features + ['R1D', 'R3D', 'R7D']]
            y_R28D = segment_data['R28D']
            model_R28D = XGBRegressor(objective='reg:squarederror')
            model_R28D.fit(X_R28D, y_R28D)
            models['R28D'] = model_R28D

            # Display feature importance for R28D
            st.markdown("**Feature Importance for R28D**")
            importance_R28D = pd.DataFrame({
                'Feature': X_R28D.columns,
                'Importance': model_R28D.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            st.write(importance_R28D)

            # Plot feature importance for R28D
            fig, ax = plt.subplots()
            ax.barh(importance_R28D['Feature'], importance_R28D['Importance'])
            ax.set_title("Feature Importance for R28D")
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
            st.pyplot(fig)

            # Save models to .pkl files
            #for response, model in models.items():
            #    model_path = os.path.join(model_dir, f"model_{molino}_{tipo}_{response}.pkl")
            #    joblib.dump(model, model_path)
            #    st.success(f"Model for {response} saved to {model_path}")

    else:
        st.info("Please upload and clean the data in Tab 1 first.")



# Footer
st.markdown("---")
st.markdown("Developed by CepSA with ❤️ using Streamlit.")
