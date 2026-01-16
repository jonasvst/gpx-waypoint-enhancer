import streamlit as st
import gpxpy
import requests
import time
import pandas as pd
import pydeck as pdk
import concurrent.futures
from math import radians, cos, sin, asin, sqrt
from io import BytesIO

# --- PAGE CONFIG ---
st.set_page_config(page_title="GPS Enricher", page_icon="📍", layout="centered")

if 'status' not in st.session_state:
    st.session_state.status = 'idle'
if 'results' not in st.session_state:
    st.session_state.results = None

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; font-weight: bold; }
    div[data-testid="stExpander"] { border: 1px solid #e6e6e6; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

st.title("📍 GPS Enricher")

# --- 1. USER GUIDE ---
with st.expander("📘 **User Guide: Logic & Legend**"):
    st.markdown("""
    ### **1. Scanning Logic**
    * **Range:** Scans a **50m radius** along your track.
    * **Coverage:** Samples every **100m**.
    * **De-Cluttering:** Set a "minimum gap" (default 2km) to prevent icon overcrowding.

    ### **2. Data Extraction**
    Extracts from OpenStreetMap: Brand names, **Opening Hours**, **Phone Numbers**, and attributes like water drinkability.

    ### **3. Support**
    Contact: **jonas@verest.ch**
    """)

# --- 2. UPLOAD ---
is_disabled = st.session_state.status == 'running'
uploaded_file = st.file_uploader("📂 **Step 1:** Drop your GPX file here", type=['gpx'], disabled=is_disabled)

# --- 3. SETTINGS ---
if uploaded_file:
    st.subheader("⚙️ Configuration")
    MIN_GAP_KM = st.slider("🧹 **Map Density (Min Gap)**", 0.0, 10.0, 2.0, 0.5, format="%.1f km", disabled=is_disabled)

    amenity_config = {
        "Water": {"label": "💧 Water & Taps", "query": 'node["amenity"~"drinking_water|fountain|watering_place"](around:{radius},{coords});node["natural"~"spring"](around:{radius},{coords});node["man_made"~"water_tap|water_well"](around:{radius},{coords});', "icon": "Water", "color": [0, 128, 255]},
        "Cemetery": {"label": "⚰️ Cemeteries (Water)", "query": 'node["amenity"~"grave_yard"](around:{radius},{coords})', "icon": "Water", "color": [0, 100, 255]},
        "Toilets": {"label": "🚽 Toilets", "query": 'node["amenity"~"toilets"](around:{radius},{coords})', "icon": "Restroom", "color": [150, 150, 150]},
        "Shops": {"label": "🛒 Supermarkets", "query": 'node["shop"~"supermarket|convenience|kiosk|bakery|general"](around:{radius},{coords})', "icon": "Convenience Store", "color": [0, 200, 0]},
        "Fuel": {"label": "⛽ Fuel Stations", "query": 'node["amenity"~"fuel"](around:{radius},{coords})', "icon": "Gas Station", "color": [255, 140, 0]},
        "Food": {"label": "🍔 Restaurants", "query": 'node["amenity"~"restaurant|fast_food|cafe"](around:{radius},{coords})', "icon": "Food", "color": [0, 200, 0]},
        "Sleep": {"label": "🛏️ Hotels", "query": 'node["tourism"~"hotel|hostel|guest_house"](around:{radius},{coords})', "icon": "Lodging", "color": [128, 0, 128]},
        "Camping": {"label": "⛺ Campsites", "query": 'node["tourism"~"camp_site"](around:{radius},{coords})', "icon": "Campground", "color": [34, 139, 34]},
        "Bike": {"label": "🔧 Bike Shops", "query": 'node["shop"~"bicycle"](around:{radius},{coords})', "icon": "Bike Shop", "color": [255, 0, 0]},
        "Pharm": {"label": "💊 Pharmacies", "query": 'node["amenity"~"pharmacy"](around:{radius},{coords})', "icon": "First Aid", "color": [255, 0, 0]},
        "ATM": {"label": "🏧 ATMs", "query": 'node["amenity"~"atm"](around:{radius},{coords})', "icon": "Generic", "color": [0, 100, 0]},
        "Train": {"label": "🚆 Train Stations", "query": 'node["railway"~"station|halt"](around:{radius},{coords})', "icon": "Generic", "color": [50, 50, 50]}
    }

    selected_keys = [k for i, (k, v) in enumerate(amenity_config.items()) if st.checkbox(v["label"], value=(k in ["Water", "Cemetery", "Toilets", "Shops", "Fuel"]), key=f"chk_{k}", disabled=is_disabled)]

    st.markdown("---")
    c_start, c_stop = st.columns([3, 1])
    if st.session_state.status in ['idle', 'complete'] and c_start.button("🚀 Start Scan", type="primary"):
        if not selected_keys: st.error("Please select amenities.")
        else:
            st.session_state.status = 'running'
            st.session_state.results = None
            st.rerun()
    if st.session_state.status == 'running' and c_stop.button("🛑 Cancel"):
        st.session_state.status = 'idle'
        st.rerun()

    # --- HELPERS ---
    def haversine(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        return 6371000 * (2 * asin(sqrt(a)))

    def fetch_batch(args):
        pts, keys = args
        coords = ",".join([f"{round(p.latitude,5)},{round(p.longitude,5)}" for p in pts])
        query = f"[out:json][timeout:25];({''.join([amenity_config[k]['query'].format(radius=50, coords=coords)+';' for k in keys])});out body;"
        for url in ["https://overpass.kumi.systems/api/interpreter", "https://api.openstreetmap.fr/oapi/interpreter"]:
            try:
                r = requests.post(url, data={'data': query}, timeout=30)
                if r.status_code == 200: return r.json().get('elements', [])
            except: continue
        return []

    # --- ENGINE ---
    if st.session_state.status == 'running':
        status = st.status("Scanning track...", expanded=True)
        try:
            gpx = gpxpy.parse(uploaded_file)
            raw_pts = [p for t in gpx.tracks for s in t.segments for p in s.points]
            scan_pts, last = [], None
            for p in raw_pts:
                if not last or haversine(last.longitude, last.latitude, p.longitude, p.latitude) >= 100:
                    scan_pts.append(p); last = p
            
            batches = [scan_pts[i:i+25] for i in range(0, len(scan_pts), 25)]
            found_raw, seen = [], set()
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as exc:
                futures = {exc.submit(fetch_batch, (b, selected_keys)): i for i, b in enumerate(batches)}
                for i, f in enumerate(concurrent.futures.as_completed(futures)):
                    for item in f.result():
                        if item['id'] not in seen: seen.add(item['id']); found_raw.append(item)

            final_pois, locs = [], {k: [] for k in selected_keys}
            for item in found_raw:
                tags = item.get('tags', {})
                # Robust Category ID
                cat = None
                if tags.get("amenity") == "grave_yard": cat = "Cemetery"
                elif tags.get("amenity") in ["drinking_water", "fountain", "watering_place"] or tags.get("natural") == "spring": cat = "Water"
                elif tags.get("amenity") == "toilets": cat = "Toilets"
                elif "shop" in tags: cat = "Shops"
                elif tags.get("amenity") == "fuel": cat = "Fuel"
                elif tags.get("amenity") in ["restaurant", "cafe", "fast_food"]: cat = "Food"
                elif "tourism" in tags: cat = "Sleep" if tags.get("tourism") != "camp_site" else "Camping"
                
                if not cat or cat not in selected_keys: continue
                lat, lon = item['lat'], item['lon']
                if any(sqrt((alat-lat)**2 + (alon-lon)**2) < (MIN_GAP_KM/111.0) for (alat, alon) in locs[cat]): continue

                # BUILD DESCRIPTION
                name = tags.get('name') or tags.get('brand') or tags.get('operator') or cat
                bits = []
                if tags.get('opening_hours'): bits.append(f"🕒 {tags.get('opening_hours')}")
                if tags.get('phone'): bits.append(f"📞 {tags.get('phone')}")
                if tags.get('drinking_water') == 'yes': bits.append("🚰 Potable")
                desc_str = f"{cat} | " + " | ".join(bits) if bits else cat

                final_pois.append({
                    "km": 0, "cat": cat, "name": name, "lat": lat, "lon": lon, "desc": desc_str,
                    "hours": tags.get('opening_hours', ""), "phone": tags.get('phone', ""),
                    "symbol": amenity_config[cat]["icon"], "color": amenity_config[cat]["color"]
                })
                locs[cat].append((lat, lon))

            # Store in session
            g_out = gpxpy.gpx.GPX()
            for t in gpx.tracks: g_out.tracks.append(t)
            for p in final_pois:
                w = gpxpy.gpx.GPXWaypoint(latitude=p['lat'], longitude=p['lon'], name=p['name'])
                w.description, w.symbol, w.type = p['desc'], p['symbol'], p['cat']
                g_out.waypoints.append(w)

            st.session_state.results = {
                "pois": final_pois, "gpx_bytes": g_out.to_xml().encode('utf-8'),
                "path": [[p.longitude, p.latitude] for p in raw_pts[::30]],
                "slat": raw_pts[0].latitude, "slon": raw_pts[0].longitude
            }
            st.session_state.status = 'complete'
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}"); st.session_state.status = 'idle'

    # --- RESULTS ---
    if st.session_state.status == 'complete' and st.session_state.results:
        res = st.session_state.results
        st.subheader("📊 Results")
        map_data = [{"coordinates": [p['lon'], p['lat']], "color": p['color'], "info": f"**{p['name']}**\n{p['desc']}"} for p in res['pois']]
        st.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(latitude=res['slat'], longitude=res['slon'], zoom=11),
            layers=[
                pdk.Layer("PathLayer", [{"path": res['path']}], get_path="path", get_color=[255, 0, 0], width_min_pixels=2),
                pdk.Layer("ScatterplotLayer", map_data, get_position="coordinates", get_fill_color="color", get_radius=200, pickable=True)
            ], tooltip={"text": "{info}"}
        ))
        
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Download GPX", res['gpx_bytes'], "enriched.gpx", "application/gpx+xml", type="primary")
        if st.button("🔄 New Scan"):
            st.session_state.status = 'idle'; st.rerun()
