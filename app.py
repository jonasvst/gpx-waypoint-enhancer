import streamlit as st
import gpxpy
import requests
import time
import random
import pandas as pd
import pydeck as pdk
import concurrent.futures
from math import radians, cos, sin, asin, sqrt
from io import BytesIO

# --- PAGE CONFIG ---
st.set_page_config(page_title="TCR Survival Tool V6", page_icon="🚴", layout="centered")

if 'running' not in st.session_state:
    st.session_state.running = False

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; }
    .stProgress > div > div > div > div { background-color: #ff4b4b; }
    </style>
""", unsafe_allow_html=True)

st.title("🚴 TCR Survival Map V6 (Visualizer)")
st.markdown("Scans your track for amenities and visualizes them directly on the route.")

# --- 1. FILE UPLOAD ---
uploaded_file = st.file_uploader(
    "📂 Step 1: Drop your GPX file here", 
    type=['gpx'], 
    disabled=st.session_state.running
)

# --- 2. SMART SETTINGS ---
with st.expander("⚙️ Step 2: Search Logic & Filters", expanded=True):
    
    search_mode = st.select_slider(
        "🎯 Search Tolerance",
        options=["Strict", "Balanced", "Wide"],
        value="Balanced",
        help="Strict = On-road only (30-50m). Balanced = Easy access (100m). Wide = Survival mode (500m).",
        disabled=st.session_state.running
    )

    if search_mode == "Strict":
        radii = {"Water": 35, "Springs": 35, "Toilets": 40, "Shops": 80, "Food": 50, "Fuel": 80, "Sleep": 100, "Pharmacy": 80, "Bike Shop": 80}
    elif search_mode == "Balanced":
        radii = {"Water": 60, "Springs": 60, "Toilets": 80, "Shops": 150, "Food": 100, "Fuel": 150, "Sleep": 250, "Pharmacy": 150, "Bike Shop": 150}
    else: 
        radii = {k: 500 for k in ["Water", "Springs", "Toilets", "Shops", "Food", "Fuel", "Sleep", "Pharmacy", "Bike Shop"]}

    col1, col2 = st.columns(2)
    with col1:
        MIN_GAP_KM = st.slider("Min. Gap (De-clutter)", 0, 50, 10, 5, disabled=st.session_state.running)
    with col2:
        SAMPLE_STEP = st.selectbox("Precision", [250, 500, 1000], index=1, format_func=lambda x: f"Check every {x}m", disabled=st.session_state.running)

    st.markdown("---")
    
    amenity_config = {
        "Water": {"query": """node["amenity"~"drinking_water|fountain"]""", "icon": "Water", "radius_key": "Water", "color": [0, 128, 255]},
        "Springs": {"query": """node["natural"~"spring"]""", "icon": "Water", "radius_key": "Springs", "color": [0, 128, 255]},
        "Toilets": {"query": """node["amenity"~"toilets"]""", "icon": "Restroom", "radius_key": "Toilets", "color": [150, 150, 150]},
        "Shops": {"query": """node["shop"~"supermarket|convenience|kiosk|general|bakery"]""", "icon": "Convenience Store", "radius_key": "Shops", "color": [0, 200, 0]},
        "Food": {"query": """node["amenity"~"restaurant|fast_food|cafe"]""", "icon": "Food", "radius_key": "Food", "color": [0, 200, 0]},
        "Fuel": {"query": """node["amenity"~"fuel"]""", "icon": "Gas Station", "radius_key": "Fuel", "color": [255, 140, 0]},
        "Sleep": {"query": """node["tourism"~"hotel|hostel|guest_house|camp_site"]""", "icon": "Lodging", "radius_key": "Sleep", "color": [128, 0, 128]},
        "Pharmacy": {"query": """node["amenity"~"pharmacy"]""", "icon": "First Aid", "radius_key": "Pharmacy", "color": [255, 0, 0]},
        "Bike Shop": {"query": """node["shop"~"bicycle"]""", "icon": "Bike Shop", "radius_key": "Bike Shop", "color": [255, 0, 0]}
    }
    
    defaults = ["Water", "Shops", "Fuel"]
    c1, c2, c3, c4 = st.columns(4)
    cols = [c1, c2, c3, c4]
    selected_types = []
    
    for i, (key, val) in enumerate(amenity_config.items()):
        with cols[i % 4]:
            if st.checkbox(key, value=(key in defaults), disabled=st.session_state.running):
                selected_types.append(key)

# --- BACKEND LOGIC ---
BATCH_SIZE = 20
WORKERS = 4
OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"
MIRRORS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://api.openstreetmap.fr/oapi/interpreter",
    "https://overpass-api.de/api/interpreter"
]
HEADERS = {"User-Agent": "TCR-Tool/6.0", "Referer": "https://streamlit.io/"}

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371000 * c

def fetch_batch(batch_data):
    points, active_types_config = batch_data
    if not points: return []
    
    coord_list = [f"{round(p.latitude,5)},{round(p.longitude,5)}" for p in points]
    coord_string = ",".join(coord_list)
    
    parts = []
    for label, config in active_types_config.items():
        base = config["query"]
        rad = config["radius"]
        parts.append(f'{base}(around:{rad},{coord_string});')
    
    query_body = "\n".join(parts)
    final_query = f"[out:json][timeout:25];({query_body});out body;"
    
    for url in MIRRORS:
        try:
            resp = requests.post(url, data={'data': final_query}, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                return resp.json().get('elements', [])
            elif resp.status_code == 429:
                time.sleep(2)
        except:
            continue
    return []

if uploaded_file and selected_types:
    if st.button("🚀 Start Scan", disabled=st.session_state.running):
        st.session_state.running = True
        st.rerun()

    if st.session_state.running:
        status_box = st.status("Processing...", expanded=True)
        try:
            gpx = gpxpy.parse(uploaded_file)
            all_points = []
            for track in gpx.tracks:
                for segment in track.segments:
                    all_points.extend(segment.points)
            
            # Smart Downsample for API
            query_points = []
            last_point = None
            for point in all_points:
                if last_point is None:
                    query_points.append(point)
                    last_point = point
                    continue
                dist = haversine(last_point.longitude, last_point.latitude, point.longitude, point.latitude)
                if dist >= SAMPLE_STEP:
                    query_points.append(point)
                    last_point = point
            
            # Prepare Batch Config
            active_config = {}
            for t in selected_types:
                active_config[t] = {
                    "query": amenity_config[t]["query"],
                    "radius": radii[amenity_config[t]["radius_key"]]
                }

            batches = [query_points[i:i + BATCH_SIZE] for i in range(0, len(query_points), BATCH_SIZE)]
            status_box.write(f"Scanning {len(query_points)} locations...")
            
            progress_bar = status_box.progress(0)
            found_raw = []
            unique_ids = set()
            
            batch_args = [(b, active_config) for b in batches]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
                future_to_batch = {executor.submit(fetch_batch, arg): i for i, arg in enumerate(batch_args)}
                completed = 0
                for future in concurrent.futures.as_completed(future_to_batch):
                    completed += 1
                    progress_bar.progress(completed / len(batches))
                    data = future.result()
                    for item in data:
                        if item['id'] not in unique_ids:
                            unique_ids.add(item['id'])
                            found_raw.append(item)
            
            # THINNING & CATEGORIZING
            status_box.write("Categorizing results...")
            final_list = []
            
            parsed_items = []
            for item in found_raw:
                tags = item.get('tags', {})
                matched_cat = "Other"
                # Strict matching order
                for cat_label in selected_types:
                    cfg = amenity_config[cat_label]
                    q = cfg["query"]
                    # Simplified matching
                    if "drinking_water" in q and tags.get("amenity") == "drinking_water": matched_cat = cat_label; break
                    if "spring" in q and tags.get("natural") == "spring": matched_cat = cat_label; break
                    if "toilets" in q and tags.get("amenity") == "toilets": matched_cat = cat_label; break
                    if "shop" in q and "shop" in tags: matched_cat = cat_label; break
                    if "fuel" in q and tags.get("amenity") == "fuel": matched_cat = cat_label; break
                    if "camp_site" in q and "tourism" in tags: matched_cat = cat_label; break
                    if "hotel" in q and "tourism" in tags: matched_cat = cat_label; break
                    if "pharmacy" in q and tags.get("amenity") == "pharmacy": matched_cat = cat_label; break
                    if "bicycle" in q and "shop" in tags: matched_cat = cat_label; break
                
                parsed_items.append({
                    "data": item,
                    "cat": matched_cat,
                    "lat": item["lat"],
                    "lon": item["lon"]
                })

            spatial_memory = []
            min_deg = MIN_GAP_KM / 111.0
            
            for obj in parsed_items:
                cat = obj['cat']
                lat = obj['lat']
                lon = obj['lon']
                
                too_close = False
                for (mlat, mlon, mcat) in spatial_memory:
                    if mcat == cat:
                        if sqrt((mlat-lat)**2 + (mlon-lon)**2) < min_deg:
                            too_close = True
                            break
                if not too_close:
                    final_list.append(obj)
                    spatial_memory.append((lat, lon, cat))
            
            status_box.update(label=f"Done! Found {len(final_list)} items.", state="complete", expanded=False)

            # --- VISUALIZATION (MAP) ---
            st.subheader("📍 Route Preview")
            
            # 1. Prepare Track Path (Downsample heavily for UI performance)
            ui_track_points = [[p.longitude, p.latitude] for p in all_points[::10]] 
            
            # 2. Prepare Scatter Points (Amenities)
            map_data = []
            for item in final_list:
                cat = item['cat']
                tags = item['data'].get('tags', {})
                name = tags.get('name', cat)
                desc = f"{cat}: {tags.get('amenity', tags.get('shop', ''))}"
                color = amenity_config[cat]['color'] if cat in amenity_config else [200, 200, 0]
                
                map_data.append({
                    "name": name,
                    "coordinates": [item['lon'], item['lat']],
                    "color": color,
                    "desc": desc
                })
            
            # 3. Render PyDeck
            view_state = pdk.ViewState(
                latitude=ui_track_points[0][1],
                longitude=ui_track_points[0][0],
                zoom=6,
                pitch=0
            )
            
            layer_track = pdk.Layer(
                "PathLayer",
                [{"path": ui_track_points}],
                get_path="path",
                get_color=[255, 75, 75], # Red Track
                width_min_pixels=3,
                pickable=False
            )
            
            layer_points = pdk.Layer(
                "ScatterplotLayer",
                data=map_data,
                get_position="coordinates",
                get_fill_color="color",
                get_radius=200, # Size in meters
                radius_min_pixels=5,
                radius_max_pixels=20,
                pickable=True
            )
            
            r = pdk.Deck(
                layers=[layer_track, layer_points],
                initial_view_state=view_state,
                tooltip={"text": "{name}\n{desc}"}
            )
            
            st.pydeck_chart(r)
            st.caption("🔴 Red Line = Your Route | 🔵 Water | 🟢 Food | 🟠 Fuel | 🟣 Sleep")

            # --- DOWNLOAD ---
            for obj in final_list:
                cat = obj['cat']
                tags = obj['data'].get('tags', {})
                name = tags.get('name', cat)
                desc = f"{cat}: {tags.get('amenity', tags.get('shop', 'poi'))}"
                
                wpt = gpxpy.gpx.GPXWaypoint(latitude=obj['lat'], longitude=obj['lon'], name=name)
                wpt.description = desc
                wpt.type = cat
                wpt.symbol = amenity_config[cat]['icon'] if cat in amenity_config else "Waypoint"
                gpx.waypoints.append(wpt)

            output_io = BytesIO()
            output_io.write(gpx.to_xml().encode('utf-8'))
            output_io.seek(0)
            
            st.success(f"Generated {len(final_list)} POIs.")
            st.download_button("⬇️ Download GPX", output_io, "TCR_Enhanced.gpx", "application/gpx+xml", type="primary")
            
            if st.button("Start New Scan"):
                st.session_state.running = False
                st.rerun()

        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.running = False
