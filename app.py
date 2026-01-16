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
st.set_page_config(page_title="TCR Tool V8 (Streamlined)", page_icon="🚴", layout="centered")

if 'running' not in st.session_state:
    st.session_state.running = False

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; }
    div[data-testid="stExpander"] { border: 0px; box-shadow: none; }
    </style>
""", unsafe_allow_html=True)

st.title("🚴 TCR Survival: Roadside Edition")
st.markdown("Finds amenities **strictly along your track**. No detours, no city clutter.")

# --- 1. UPLOAD ---
uploaded_file = st.file_uploader("📂 Drop GPX file", type=['gpx'], disabled=st.session_state.running)

# --- 2. SIMPLE SETTINGS ---
st.subheader("⚙️ Settings")
col1, col2 = st.columns(2)

with col1:
    # THE CRITICAL FILTER: Distance from road
    RADIUS = st.slider(
        "📏 Max Distance from Road (meters)", 
        20, 200, 40, 10, 
        help="Keep this low (30-50m) to avoid city noise. Only finds things you can see from the saddle.",
        disabled=st.session_state.running
    )

with col2:
    # THE CLEANER: Min Gap
    MIN_GAP_KM = st.slider(
        "🧹 Min Gap between stops (km)", 
        0, 20, 5, 1, 
        help="If we find a shop, ignore other shops for X km.",
        disabled=st.session_state.running
    )

# Amenities Configuration
amenity_config = {
    "Water": {"query": """node["amenity"~"drinking_water|fountain"]""", "icon": "Water", "color": [0, 128, 255]},
    "Springs": {"query": """node["natural"~"spring"]""", "icon": "Water", "color": [0, 128, 255]},
    "Shops": {"query": """node["shop"~"supermarket|convenience|kiosk|bakery"]""", "icon": "Convenience Store", "color": [0, 200, 0]},
    "Fuel": {"query": """node["amenity"~"fuel"]""", "icon": "Gas Station", "color": [255, 140, 0]},
    "Food": {"query": """node["amenity"~"fast_food|cafe|restaurant"]""", "icon": "Food", "color": [0, 200, 0]},
    "Toilets": {"query": """node["amenity"~"toilets"]""", "icon": "Restroom", "color": [150, 150, 150]},
    "Sleep": {"query": """node["tourism"~"hotel|hostel|camp_site"]""", "icon": "Lodging", "color": [128, 0, 128]},
    "Bike Shop": {"query": """node["shop"~"bicycle"]""", "icon": "Bike Shop", "color": [255, 0, 0]},
    "Pharmacy": {"query": """node["amenity"~"pharmacy"]""", "icon": "First Aid", "color": [255, 0, 0]}
}

# Simple Multi-Select
selected_types = st.multiselect(
    "Select what to scan for:", 
    options=list(amenity_config.keys()),
    default=["Water", "Shops", "Fuel"],
    disabled=st.session_state.running
)

# --- ENGINE ---
BATCH_SIZE = 25
SAMPLE_STEP = 150  # Hardcoded high frequency (check every 150m)
WORKERS = 4
HEADERS = {"User-Agent": "TCR-Tool/8.0", "Referer": "https://streamlit.io/"}
MIRRORS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://api.openstreetmap.fr/oapi/interpreter",
    "https://overpass-api.de/api/interpreter"
]

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371000 * c

def fetch_batch(args):
    points, queries, rad = args
    if not points: return []
    
    # 5 decimal places is ~1m precision
    coords = ",".join([f"{round(p.latitude,5)},{round(p.longitude,5)}" for p in points])
    
    # Build strict query
    parts = [f'{q}(around:{rad},{coords});' for q in queries]
    full_query = f"[out:json][timeout:25];({''.join(parts)});out body;"

    for url in MIRRORS:
        try:
            r = requests.post(url, data={'data': full_query}, headers=HEADERS, timeout=30)
            if r.status_code == 200: return r.json().get('elements', [])
            if r.status_code == 429: time.sleep(1)
        except: continue
    return []

if uploaded_file and st.button("🚀 Scan Route", disabled=st.session_state.running):
    if not selected_types:
        st.error("Select at least one amenity.")
    else:
        st.session_state.running = True
        st.rerun()

if st.session_state.running:
    status = st.status("Scanning...", expanded=True)
    try:
        gpx = gpxpy.parse(uploaded_file)
        
        # 1. Flatten Track
        all_pts = []
        for t in gpx.tracks:
            for s in t.segments:
                all_pts.extend(s.points)
        
        # 2. Resample (High Freq)
        scan_pts = []
        last = None
        for p in all_pts:
            if not last or haversine(last.longitude, last.latitude, p.longitude, p.latitude) >= SAMPLE_STEP:
                scan_pts.append(p)
                last = p
        
        # 3. Batch Scan
        status.write(f"Scanning {len(scan_pts)} points along track...")
        batches = [scan_pts[i:i+BATCH_SIZE] for i in range(0, len(scan_pts), BATCH_SIZE)]
        active_queries = [amenity_config[t]["query"] for t in selected_types]
        
        found_raw = []
        seen_ids = set()
        prog = status.progress(0)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as exc:
            futures = {exc.submit(fetch_batch, (b, active_queries, RADIUS)): i for i, b in enumerate(batches)}
            for i, f in enumerate(concurrent.futures.as_completed(futures)):
                prog.progress((i+1)/len(batches))
                for item in f.result():
                    if item['id'] not in seen_ids:
                        seen_ids.add(item['id'])
                        found_raw.append(item)
        
        # 4. Global De-Clutter
        status.write("Cleaning up duplicates...")
        final_pois = []
        
        # Helper to classify item
        def get_cat(item):
            tags = item.get('tags', {})
            for cat in selected_types:
                q = amenity_config[cat]["query"]
                if "drinking" in q and tags.get("amenity")=="drinking_water": return cat
                if "spring" in q and tags.get("natural")=="spring": return cat
                if "shop" in q and "shop" in tags: return cat
                if "fuel" in q and tags.get("amenity")=="fuel": return cat
                if "food" in q and tags.get("amenity") in ["restaurant","fast_food","cafe"]: return cat
                if "toilets" in q and tags.get("amenity")=="toilets": return cat
                if "hotel" in q and "tourism" in tags: return cat
                if "pharmacy" in q and tags.get("amenity")=="pharmacy": return cat
                if "bicycle" in q and "shop" in tags: return cat
            return None

        # Sort roughly by input order (not perfect but fast) - simplistic thinning
        # A better approach: check distance to ALL accepted items of same cat
        min_deg = MIN_GAP_KM / 111.0
        
        accepted_locs = {cat: [] for cat in selected_types} # list of (lat, lon)
        
        for item in found_raw:
            cat = get_cat(item)
            if not cat: continue
            
            lat, lon = item['lat'], item['lon']
            
            # Check gap
            too_close = False
            for (alat, alon) in accepted_locs[cat]:
                if sqrt((alat-lat)**2 + (alon-lon)**2) < min_deg:
                    too_close = True
                    break
            
            if not too_close:
                tags = item.get('tags', {})
                name = tags.get('name', cat)
                desc = f"{cat}: {tags.get('amenity', tags.get('shop', ''))}"
                
                final_pois.append({
                    "lat": lat, "lon": lon, 
                    "name": name, "desc": desc, 
                    "cat": cat
                })
                accepted_locs[cat].append((lat, lon))
        
        status.update(label=f"Done! Found {len(final_pois)} items.", state="complete")
        
        # 5. Preview Map
        st.subheader("📍 Preview")
        map_data = [{
            "coordinates": [p['lon'], p['lat']], 
            "color": amenity_config[p['cat']]['color'],
            "name": p['name']
        } for p in final_pois]
        
        # Simple Red Line Track
        path_data = [[p.longitude, p.latitude] for p in all_pts[::25]]
        
        st.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=all_pts[0].latitude, longitude=all_pts[0].longitude, zoom=8
            ),
            layers=[
                pdk.Layer("PathLayer", [{"path": path_data}], get_path="path", get_color=[255, 0, 0], width_min_pixels=2),
                pdk.Layer("ScatterplotLayer", map_data, get_position="coordinates", get_fill_color="color", get_radius=200, pickable=True)
            ],
            tooltip={"text": "{name}"}
        ))
        
        # 6. Build GPX
        for p in final_pois:
            wpt = gpxpy.gpx.GPXWaypoint(latitude=p['lat'], longitude=p['lon'], name=p['name'])
            wpt.description = p['desc']
            wpt.symbol = amenity_config[p['cat']]['icon']
            gpx.waypoints.append(wpt)
            
        out = BytesIO()
        out.write(gpx.to_xml().encode('utf-8'))
        out.seek(0)
        
        st.success(f"Added {len(final_pois)} POIs.")
        st.download_button("⬇️ Download GPX", out, "TCR_Roadside.gpx", "application/gpx+xml", type="primary")
        
        if st.button("Start Over"):
            st.session_state.running = False
            st.rerun()

    except Exception as e:
        st.error(f"Error: {e}")
        st.session_state.running = False
