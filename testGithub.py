import streamlit as st
from github import Github
import base64
import os

# Streamlit app title
st.title("File Uploader to GitHub")

repo_name = 'jamedrano/CemproPruebas2'


# File upload widget
uploaded_file = st.file_uploader("Upload a file", type=None)

token_file = st.file_uploader("Upload token file (e.g., token.txt)", type=["txt"])
if token_file:
    github_token = token_file.read().decode("utf-8").strip()
    st.write(github_token)
else:
    st.write("subir el archivo del token")

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
            try:
                existing_file = repo.get_contents(file_path, ref="main")
                # File exists: update it
                repo.update_file(
                    path=file_path,
                    message=f"Update {file_name}",
                    content=uploaded_file.getvalue(),
                    sha=existing_file.sha,  # Use the file's SHA to update
                    branch="main"
                )
                st.success(f"File '{file_name}' updated successfully in {repo_name}!")
            except Exception as e:
                if "404" in str(e):  # File does not exist
                    # Create the file
                    repo.create_file(
                        path=file_path,
                        message=f"Add {file_name}",
                        content=uploaded_file.getvalue(),
                        branch="main"
                    )
                    st.success(f"File '{file_name}' created successfully in {repo_name}!")
                else:
                    raise e  # Re-raise other errors      

            
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("Please fill out all fields and upload a file.")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Ensure the GitHub token has `repo` permissions for private repositories.")
