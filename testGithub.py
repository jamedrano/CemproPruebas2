import streamlit as st
from github import Github
import base64
import os

# Streamlit app title
st.title("File Uploader to GitHub")

github_token = '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'
repo_name = 'jamedrano/CemproPruebas2'


# File upload widget
uploaded_file = st.file_uploader("Upload a file", type=None)

if uploaded_file and github_token and repo_name:
    st.success("Ready to upload!")
    # Action button to upload the file
    if st.button("Upload to GitHub"):
        try:
            # Authenticate to GitHub
            g = Github(github_token)
            repo = g.get_repo(repo_name)
            
            # Prepare file content
            file_name = uploaded_file.name
            file_path = file_name

            content = uploaded_file.read()
            b64_data = base64.b64encode(content).decode('utf-8')

            # Create the file in the repository
            repo.create_file(
                path=file_path,
                message=f"Add {file_name}",
                content=b64_data,
                branch="main"  # Default branch
            )

            st.success(f"File '{file_name}' uploaded successfully to {repo_name} in {folder_path or 'root'}!")
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("Please fill out all fields and upload a file.")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Ensure the GitHub token has `repo` permissions for private repositories.")
