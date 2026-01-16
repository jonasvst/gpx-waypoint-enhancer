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
st.set_page_config(page_title="TCR Survival Tool V7", page_icon="🚴", layout="wide")

if 'running' not in st.session_state:
    st.session_state.running = False

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; }
    .stProgress > div > div > div > div { background-color: #ff4b4b; }
    /* Compact sliders */
    div[data-testid="stSlider"] { padding-top: 0rem; padding-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.title("🚴 TCR Survival Map V7 (Granular Control)")
st.markdown("Customize frequency and clustering for every single amenity category.")

# --- 1. FILE UPLOAD ---
uploaded_file = st.file_uploader(
    "📂 Step 1: Drop your GPX file here", 
    type=['gpx'], 
    disabled=st.session_state.running
)

# --- 2. GLOBAL SCAN SETTINGS ---
with st.expander("⚙️ Step 2: Scan Precision & Tolerance", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        # User requested 50m-500m incremental options
        SAMPLE_STEP = st.select_slider(
            "📡 Scan Frequency (Track Sampling)",
            options=[50, 100, 200, 500, 1000],
            value=200,
            format_func=lambda x: f"Check every {x}m",
            disabled=st.session_state.running,
            help="100m is very accurate but slower. 500m is faster."
        )

    with col2:
        # Search Tolerance Modes
        search_mode = st.select_slider(
            "🎯 Search Width (Detour Tolerance)",
            options=["Strict (30m)", "Tight (50m)", "Standard (100m)", "Relaxed (250m)", "Wide (500m)"],
            value="Standard (100m)",
            disabled=st.session_state.running
        )
        # Map labels to meters
        mode_map = {
            "Strict (30m)": 30, "Tight (50m)": 50, 
            "Standard (100m)": 100, "Relaxed (250m)": 250, 
            "Wide (500m)": 500
        }
        GLOBAL_RADIUS = mode_map[search_mode]

# --- 3. PER-CATEGORY CONFIGURATION ---
st.subheader("🛠️ Step 3: customize Your Amenities")
st.info("💡 **Gap:** Minimum distance between stops. **Cluster:** If checked, allows multiple items in the same village (2km radius) even if Gap is high.")

# Data structure for settings
amenity_config = {
    "Water": {"query": """node["amenity"~"drinking_water|fountain"]""", "icon": "Water", "color": [0, 128, 255]},
    "Springs": {"query": """node["natural"~"spring"]""", "icon": "Water", "color": [0, 128, 255]},
    "Shops": {"query": """node["shop"~"supermarket|convenience|kiosk|general|bakery"]""", "icon": "Convenience Store", "color": [0, 200, 0]},
    "Fuel": {"query": """node["amenity"~"fuel"]""", "icon": "Gas Station", "color": [255, 140, 0]},
    "Food": {"query": """node["amenity"~"restaurant|fast_food|cafe"]""", "icon": "Food", "color": [0, 200, 0]},
    "Toilets": {"query": """node["amenity"~"toilets"]""", "icon": "Restroom", "color": [150, 150, 150]},
    "Sleep": {"query": """node["tourism"~"hotel|hostel|guest_house"]""", "icon": "Lodging", "color": [128, 0, 128]},
    "Camping": {"query": """node["tourism"~"camp_site"]""", "icon": "Campground", "color": [34, 139, 34]},
    "Pharmacy": {"query": """node["amenity"~"pharmacy"]""", "icon": "First Aid", "color": [255, 0, 0]},
    "Bike Shop": {"query": """node["shop"~"bicycle"]""", "icon": "Bike Shop", "color": [255, 0, 0]}
}

# User Selections Storage
user_config = {}

# create a header row
h1, h2, h3, h4 = st.columns([2, 2, 1, 1])
h1.markdown("**Category**")
h2.markdown("**Min Gap (km)**")
h3.markdown("**Cluster?**")
h4.markdown("**Status**")

# Loop to create controls for each category
for cat, data in amenity_config.items():
    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    
    with c1:
        # Default checked items
        is_default = cat in ["Water", "Shops", "Fuel"]
        enabled = st.checkbox(f"**{cat}**", value=is_default, disabled=st.session_state.running, key=f"chk_{cat}")
    
    with c2:
        # Logic: Water defaults to 0km gap, Hotels to 20km
        def_gap = 0 if cat in ["Water", "Springs"] else 10
        gap = st.slider("Gap", 0, 100, def_gap, 5, key=f"gap_{cat}", label_visibility="collapsed", disabled=not enabled or st.session_state.running)
    
    with c3:
        # Cluster defaults to True for Sleep/Food, False for Water
        def_cluster = cat in ["Sleep", "Food", "Shops", "Camping"]
        allow_cluster = st.checkbox("🏘️", value=def_cluster, key=f"cls_{cat}", disabled=not enabled or st.session_state.running, help=f"Allow multiple {cat} in the same village?")
        
    with c4:
        if enabled:
            st.caption(f"✅ On")
        else:
            st.caption("❌")

    if enabled:
        user_config[cat] = {
            "gap_km": gap,
            "cluster": allow_cluster,
            "query": data["query"],
            "icon": data["icon"],
            "color": data["color"]
        }

st.markdown("---")

# --- BACKEND LOGIC ---
BATCH_SIZE = 15 # Smaller batch for higher precision
WORKERS = 4
OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"
MIRRORS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://api.openstreetmap.fr/oapi/interpreter",
    "https://overpass-api.de/api/interpreter"
]
HEADERS = {"User-Agent": "TCR-Tool/7.0", "Referer": "https://streamlit.io/"}

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371000 * c

def fetch_batch(batch_data):
    points, radius, active_queries = batch_data
    if not points: return []
    
    coord_list = [f"{round(p.latitude,5)},{round(p.longitude,5)}" for p in points]
    coord_string = ",".join(coord_list)
    
    # Build ONE massive query for all enabled types
    parts = []
    for q in active_queries:
        parts.append(f'{q}(around:{radius},{coord_string});')
    
    query_body = "\n".join(parts)
    final_query = f"[out:json][timeout:25];({query_body});out body;"
    
    for url in MIRRORS:
        try:
            resp = requests.post(url, data={'data': final_query}, headers=HEADERS, timeout=45)
            if resp.status_code == 200:
                return resp.json().get('elements', [])
            elif resp.status_code == 429:
                time.sleep(2)
        except:
            continue
    return []

if uploaded_file and user_config:
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
            
            # Resample
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
            
            # Prepare Batch Args
            # Collect all active queries
            active_queries = [cfg["query"] for cfg in user_config.values()]
            
            batches = [query_points[i:i + BATCH_SIZE] for i in range(0, len(query_points), BATCH_SIZE)]
            status_box.write(f"Scanning {len(query_points)} points with {WORKERS} workers...")
            
            progress_bar = status_box.progress(0)
            found_raw = []
            unique_ids = set()
            
            batch_args = [(b, GLOBAL_RADIUS, active_queries) for b in batches]
            
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
            
            # --- SMART FILTERING & CLUSTERING ---
            status_box.write("Applying Gap & Cluster Logic...")
            final_list = []
            
            # 1. Categorize Raw Items
            parsed_items = []
            for item in found_raw:
                tags = item.get('tags', {})
                matched_cat = "Other"
                
                # Reverse check which category this item belongs to
                for cat_name, cfg in user_config.items():
                    q = cfg["query"]
                    # Quick heuristic matching
                    if "drinking_water" in q and tags.get("amenity") == "drinking_water": matched_cat = cat_name; break
                    if "spring" in q and tags.get("natural") == "spring": matched_cat = cat_name; break
                    if "toilets" in q and tags.get("amenity") == "toilets": matched_cat = cat_name; break
                    if "shop" in q and "shop" in tags: matched_cat = cat_name; break
                    if "fuel" in q and tags.get("amenity") == "fuel": matched_cat = cat_name; break
                    if "camp_site" in q and "tourism" in tags: matched_cat = cat_name; break
                    if "hotel" in q and "tourism" in tags: matched_cat = cat_name; break
                    if "pharmacy" in q and tags.get("amenity") == "pharmacy": matched_cat = cat_name; break
                    if "bicycle" in q and "shop" in tags: matched_cat = cat_name; break
                
                if matched_cat != "Other":
                    parsed_items.append({
                        "data": item,
                        "cat": matched_cat,
                        "lat": item["lat"],
                        "lon": item["lon"]
                    })

            # 2. Apply Spatial Logic
            # We iterate through the list (which is roughly ordered by route)
            # For each item, we look at the 'last seen' item of that category.
            
            # To do this effectively without route distance, we use simple Euclidean
            # proximity to the *last accepted point*.
            
            last_accepted = {cat: None for cat in user_config.keys()}
            
            # Cluster Radius (Hardcoded to 2km for "Village")
            CLUSTER_DEG = 2000 / 111000.0 
            
            dropped_count = 0
            
            for obj in parsed_items:
                cat = obj['cat']
                cfg = user_config[cat]
                
                GAP_DEG = (cfg["gap_km"] * 1000) / 111000.0
                ALLOW_CLUSTER = cfg["cluster"]
                
                lat = obj['lat']
                lon = obj['lon']
                
                last = last_accepted[cat]
                
                keep = False
                
                if last is None:
                    keep = True
                else:
                    # Dist to last
                    dist = sqrt((last['lat']-lat)**2 + (last['lon']-lon)**2)
                    
                    if dist > GAP_DEG:
                        # It's far enough away -> New Stop
                        keep = True
                    elif ALLOW_CLUSTER and dist < CLUSTER_DEG:
                        # It's close to the last one (Same Village) -> Keep it!
                        keep = True
                    else:
                        # It's in the "dead zone" (between village and next gap)
                        keep = False
                
                if keep:
                    final_list.append(obj)
                    # We only update 'last_accepted' if we are STARTING a new cluster/stop.
                    # If we are adding to a cluster, the 'anchor' should arguably stay the same 
                    # OR move? If we move it, we might daisy chain. 
                    # Better Logic: If it was a 'Gap' jump, update anchor. 
                    # If it was a 'Cluster' add, update anchor? 
                    # Simple: Always update anchor.
                    last_accepted[cat] = obj
                else:
                    dropped_count += 1
            
            status_box.update(label=f"Done! kept {len(final_list)} items (Filtered {dropped_count}).", state="complete", expanded=False)

            # --- PREVIEW MAP ---
            st.subheader("📍 Preview")
            
            # Map Data
            map_data = []
            for item in final_list:
                cat = item['cat']
                tags = item['data'].get('tags', {})
                desc = f"{cat}: {tags.get('name', '')}"
                color = user_config[cat]['color']
                
                map_data.append({
                    "name": cat,
                    "coordinates": [item['lon'], item['lat']],
                    "color": color,
                    "desc": desc
                })
            
            view_state = pdk.ViewState(
                latitude=all_points[0].latitude,
                longitude=all_points[0].longitude,
                zoom=8
            )
            
            # Track Layer
            ui_track = [[p.longitude, p.latitude] for p in all_points[::20]]
            layer_track = pdk.Layer(
                "PathLayer", [{"path": ui_track}],
                get_path="path", get_color=[255, 0, 0], width_min_pixels=2
            )
            
            # Points Layer
            layer_points = pdk.Layer(
                "ScatterplotLayer", map_data,
                get_position="coordinates", get_fill_color="color",
                get_radius=300, pickable=True
            )
            
            st.pydeck_chart(pdk.Deck(
                layers=[layer_track, layer_points],
                initial_view_state=view_state,
                tooltip={"text": "{desc}"}
            ))

            # --- DOWNLOAD ---
            for obj in final_list:
                cat = obj['cat']
                tags = obj['data'].get('tags', {})
                name = tags.get('name', cat)
                desc = f"{cat}: {tags.get('amenity', tags.get('shop', 'poi'))}"
                
                wpt = gpxpy.gpx.GPXWaypoint(latitude=obj['lat'], longitude=obj['lon'], name=name)
                wpt.description = desc
                wpt.type = cat
                wpt.symbol = user_config[cat]['icon']
                gpx.waypoints.append(wpt)

            output_io = BytesIO()
            output_io.write(gpx.to_xml().encode('utf-8'))
            output_io.seek(0)
            
            st.download_button("⬇️ Download Final GPX", output_io, "TCR_Enhanced.gpx", "application/gpx+xml", type="primary")
            
            if st.button("Start New Scan"):
                st.session_state.running = False
                st.rerun()

        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.running = False
