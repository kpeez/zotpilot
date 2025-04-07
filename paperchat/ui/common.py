"""
Common UI elements and utilities for Streamlit.
"""

import streamlit as st


def set_css_styles():
    """
    Set global CSS styles for the app.
    """
    st.markdown(
        """
        <style>
            .stButton button {
                width: 100%;
            }

            .app-header {
                font-weight: 500;
                margin-bottom: 1rem;
            }

            .action-buttons {
                display: flex;
                flex-direction: column;
                gap: 10px;
            }

            .info-box {
                padding: 1rem;
                border-radius: 0.5rem;
                background-color: #f0f2f6;
                margin-bottom: 1rem;
            }

            /* For settings sections */
            .settings-section {
                margin-bottom: 1.5rem;
                padding-bottom: 1rem;
                border-bottom: 1px solid #f0f2f6;
            }

            /* For text with subtle emphasis */
            .text-subtle {
                color: #606C86;
                font-size: 0.9rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title, subtitle=None):
    """
    Render a consistent page header.

    Args:
        title: Main title text
        subtitle: Optional subtitle text
    """
    st.markdown(f"<h1 class='app-header'>{title}</h1>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p class='text-subtle'>{subtitle}</p>", unsafe_allow_html=True)
    st.divider()


def show_success(message):
    """
    Show a success message.

    Args:
        message: Success message text
    """
    st.success(message)


def show_error(message):
    """
    Show an error message.

    Args:
        message: Error message text
    """
    st.error(message)


def show_info(message):
    """
    Show an info message.

    Args:
        message: Info message text
    """
    st.info(message)


def show_warning(message):
    """
    Show a warning message.

    Args:
        message: Warning message text
    """
    st.warning(message)
