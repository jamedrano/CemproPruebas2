import streamlit as st
import pandas as pd

# App title
st.title("Cement Compression Strength: Data Viewer, Filtering, and Cleaning")

# Add tabs to the app
tab1, tab2, tab3 = st.tabs(["Data Cleaning", "Tab 2", "Tab 3"])

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

            # Load the selected sheet
            data = pd.read_excel(uploaded_file, sheet_name=selected_sheet)

            # Clean column names: remove extra spaces and capitalize
            data.columns = data.columns.str.strip().str.upper()

            # Ensure necessary columns exist
            required_columns = ['FECHA']
            if all(col in data.columns for col in required_columns):
                # Convert 'FECHA' column to datetime format
                data['FECHA'] = pd.to_datetime(data['FECHA'], errors='coerce')
                data = data.dropna(subset=['FECHA'])

                # Display the data before filtering
                st.subheader("Preview of Original Data (with Cleaned Column Names)")
                st.write(data)

                # Filter by date
                st.subheader("Filter Data by Date")
                cutoff_date = st.date_input(
                    "Select a cutoff date:",
                    value=data['FECHA'].min().date() if not data.empty else None,
                )
                filtered_data = data[data['FECHA'] >= pd.Timestamp(cutoff_date)]

                # Cleaning: Remove rows with 0, negative, or invalid values
                st.subheader("Clean Data")
                st.markdown("Removing rows containing **0**, **negative values**, or **invalid values** in any column.")
                cleaned_data = filtered_data.copy()
                cleaned_data = cleaned_data.replace(0, pd.NA)  # Replace 0 with NaN
                cleaned_data = cleaned_data.applymap(
                    lambda x: pd.NA if isinstance(x, (int, float)) and x < 0 else x
                )  # Replace negatives with NaN
                cleaned_data = cleaned_data.dropna()  # Drop rows with any NaN values

                # Display the number of valid rows remaining
                st.subheader("Summary of Cleaning Results")
                num_rows = cleaned_data.shape[0]
                st.markdown(f"**Number of valid rows left in the cleaned data:** {num_rows}")

                # Display the cleaned data
                st.subheader("Cleaned Data")
                st.write(cleaned_data)
            else:
                missing_cols = [col for col in required_columns if col not in data.columns]
                st.error(f"The following required columns are missing from the dataset: {', '.join(missing_cols)}")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
    else:
        st.info("Please upload an Excel file to proceed.")

# Placeholder content for other tabs
with tab2:
    st.subheader("Tab 2 Content")
    st.write("This is Tab 2. Additional functionality can be added here.")

with tab3:
    st.subheader("Tab 3 Content")
    st.write("This is Tab 3. Additional functionality can be added here.")

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit.")
