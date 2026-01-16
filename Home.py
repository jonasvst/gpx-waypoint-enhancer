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
This is a collection of utilities designed to help you plan, survive, and finish ultra-distance races like the **Transcontinental (TCR)**.

👈 **Select a tool from the sidebar to get started.**

#### **Available Tools:**
* **📍 GPX Enricher:** Scans your track for water, food, and sleep.
""")

st.sidebar.success("Select a tool above")
