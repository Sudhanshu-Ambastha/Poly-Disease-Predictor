import streamlit as st

def set_bg_from_url(url, opacity=1):
    st.markdown(
        f"""
        <style>
            body {{
                background: url('{url}') no-repeat center center fixed;
                background-size: cover;
                opacity: 0.875;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

