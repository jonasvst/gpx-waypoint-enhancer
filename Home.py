import streamlit as st

st.set_page_config(
    page_title="UltraToolkit",
    page_icon="🚴",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("🚴 UltraToolkit")
st.subheader("Survival tools for ultra-distance cycling.")

st.markdown("""
### **Welcome**
This platform hosts specialized utilities for ultra-distance racing (TCR, etc.).

👈 **Select a tool from the sidebar to get started.**

#### **Currently Available:**
* **📍 GPS Enricher:** Scans your track for water, food, and sleep.
""")

st.sidebar.success("Select a tool above")
st.sidebar.info("v1.0 - verest.ch")
