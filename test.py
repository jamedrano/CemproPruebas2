import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# App title
st.title("Cement Compression Strength: Data Viewer, Filtering, and Cleaning")

# Add tabs to the app
tab1, tab2, tab3 = st.tabs(["Data Cleaning", "Visualizations", "Tab 3"])

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
            required_columns = ['FECHA', 'MOLINO', 'TIPO', 'R1D', 'R3D', 'R7D', 'R28D']
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
                axes[i].boxplot(filtered_viz_data[col].dropna(), vert=False)
                axes[i].set_title(f'{col} - Boxplot')
                axes[i].set_xlabel('Resistance')
            elif chart_type == 'Trend Chart':
                axes[i].plot(filtered_viz_data['FECHA'], filtered_viz_data[col], marker='o', linestyle='-')
                axes[i].set_title(f'{col} - Trend Chart')
                axes[i].set_xlabel('Date')
                axes[i].set_ylabel('Resistance')

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Please upload and clean the data in Tab 1 first.")

# Placeholder content for Tab 3
with tab3:
    st.subheader("Tab 3 Content")
    st.write("This is Tab 3. Additional functionality can be added here.")

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit.")
