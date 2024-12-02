import streamlit as st
import pandas as pd

# App title
st.title("Cement Compression Strength: Data Viewer, Filtering, and Cleaning")

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

        # Load the selected sheet
        data = pd.read_excel(uploaded_file, sheet_name=selected_sheet)

        # Ensure necessary columns exist
        required_columns = ['Fecha', 'Molino', 'Tipo']
        if all(col in data.columns for col in required_columns):
            # Convert 'Fecha' column to datetime format
            data['Fecha'] = pd.to_datetime(data['Fecha'], errors='coerce')
            data = data.dropna(subset=['Fecha'])

            # Display the data before filtering
            st.subheader("Preview of Original Data")
            st.write(data)

            # Filter by date
            st.subheader("Filter Data by Date")
            cutoff_date = st.date_input(
                "Select a cutoff date:",
                value=data['Fecha'].min().date() if not data.empty else None,
            )
            filtered_data = data[data['Fecha'] >= pd.Timestamp(cutoff_date)]

            # Filter by Molino
            st.subheader("Filter Data by Molino")
            unique_molinos = filtered_data['Molino'].dropna().unique()
            selected_molino = st.selectbox("Select Molino:", ["All"] + list(unique_molinos))
            if selected_molino != "All":
                filtered_data = filtered_data[filtered_data['Molino'] == selected_molino]

            # Filter by Tipo
            st.subheader("Filter Data by Tipo")
            unique_tipos = filtered_data['Tipo'].dropna().unique()
            selected_tipo = st.selectbox("Select Tipo:", ["All"] + list(unique_tipos))
            if selected_tipo != "All":
                filtered_data = filtered_data[filtered_data['Tipo'] == selected_tipo]

            # Cleaning: Remove rows with 0 or invalid values
            st.subheader("Clean Data")
            st.markdown("Removing rows containing **0** or invalid values in any column.")
            cleaned_data = filtered_data.replace(0, pd.NA).dropna()

            # Column Selection for Exclusion
            st.subheader("Select Columns to Exclude")
            available_columns = list(cleaned_data.columns)
            columns_to_remove = st.multiselect(
                "Select columns to exclude from training data:",
                available_columns,
            )

            # Exclude the selected columns
            training_data = cleaned_data.drop(columns=columns_to_remove)

            # Display the cleaned and final training data
            st.subheader("Final Training Data")
            st.write(training_data)
        else:
            missing_cols = [col for col in required_columns if col not in data.columns]
            st.error(f"The following required columns are missing from the dataset: {', '.join(missing_cols)}")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload an Excel file to proceed.")

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit.")
