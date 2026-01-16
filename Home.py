import streamlit as st

st.set_page_config(page_title="UltraToolkit", page_icon="🚴", layout="centered")

st.title("🚴 UltraToolkit")
st.caption("v1.0 | verest.ch")

st.markdown("### Survival tools for ultra-distance cycling.")
st.markdown("Select a tool below to get started:")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📍 GPS Enricher")
    st.info("Scan GPX tracks for roadside water, food, and sleep.")
    if st.button("Open Enricher 🚀"):
        st.switch_page("pages/1_📍_GPS_Enricher.py")

with col2:
    st.markdown("#### 🔜 More Tools")
    st.info("Weather planners and packing lists coming soon.")
    st.button("Coming Soon", disabled=True)

st.markdown("---")
st.markdown("##### 📬 Support")
st.caption("Built for the TCR community. Contact: **jonas@verest.ch**")
