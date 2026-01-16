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

# --- SESSION STATE INITIALIZATION ---
if 'status' not in st.session_state:
    st.session_state.status = 'idle'  # idle, running, complete
if 'results' not in st.session_state:
    st.session_state.results = None

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; font-weight: bold; }
    div[data-testid="stExpander"] { border: 1px solid #e6e6e6; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

st.title("📍 GPS Enricher")
st.markdown("Enrich your GPX track with detailed roadside amenities.")

# --- 1. USER GUIDE ---
with st.expander("📘 **User Guide: Logic & Legend**"):
    st.markdown("""
    ### **1. Scanning Logic**
    * **Range:** Scans a **50m radius** along your track for zero-detour stops.
    * **Coverage:** Samples every **100m**.
    * **De-Cluttering:** Set a "minimum gap" (default 2km) to prevent icon overcrowding.

    ### **2. Data Extraction**
    Extracts from OpenStreetMap: Brand names, **Opening Hours**, **Phone Numbers**, and attributes like water drinkability.

    ### **3. Support**
    Contact: **jonas@verest.ch**
    """)

# --- 2. UPLOAD ---
# Disable upload if we are currently running
is_disabled = st.session_state.status == 'running'
uploaded_file = st.file_uploader("📂 **Step 1:** Drop your GPX file here", type=['gpx'], disabled=is_disabled)

# --- 3. SETTINGS & ENGINE ---
if uploaded_file:
    # --- CONFIGURATION UI ---
    st.subheader("⚙️ Configuration")
    
    MIN_GAP_KM = st.slider(
        "🧹 **Map Density (Min Gap)**", 
        0.0, 10.0, 2.0, 0.5,
        format="%.1f km",
        help="Higher = Fewer icons. Lower = More data.",
        disabled=is_disabled
    )

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

    cols = st.columns(3)
    selected_keys = []
    defaults = ["Water", "Cemetery", "Toilets", "Shops", "Fuel"]

    for i, (key, cfg) in enumerate(amenity_config.items()):
        with cols[i % 3]:
            if st.checkbox(cfg["label"], value=(key in defaults), disabled=is_disabled):
                selected_keys.append(key)

    st.markdown("---")
    col_start, col_stop = st.columns([3, 1])
    
    with col_start:
        if st.session_state.status in ['idle', 'complete']:
            if st.button("🚀 Start Scan", type="primary"):
                if not selected_keys: st.error("Please select amenities.")
                else:
                    st.session_state.status = 'running'
                    st.session_state.results = None
                    st.rerun()

    with col_stop:
        if st.session_state.status == 'running':
            if st.button("🛑 Cancel"):
                st.session_state.status = 'idle'
                st.rerun()

    # --- HELPER FUNCTIONS ---
    def haversine(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        return 6371000 * (2 * asin(sqrt(a)))

    def calculate_track_distance(points):
        total_dist, last, points_with_dist = 0, None, []
        for p in points:
            if last: total_dist += haversine(last.longitude, last.latitude, p.longitude, p.latitude)
            points_with_dist.append({"lat": p.latitude, "lon": p.longitude, "cum_dist": total_dist})
            last = p
        return points_with_dist

    def get_nearest_km(poi_lat, poi_lon, track_data):
        min_dist, km_mark = float('inf'), 0
        for tp in track_data[::10]:
            d = (tp['lat'] - poi_lat)**2 + (tp['lon'] - poi_lon)**2
            if d < min_dist: min_dist, km_mark = d, tp['cum_dist']
        return km_mark / 1000.0

    def fetch_batch(args):
        points, active_keys = args
        if not points: return []
        coords = ",".join([f"{round(p.latitude,5)},{round(p.longitude,5)}" for p in points])
        parts = [amenity_config[k]["query"].format(radius=50, coords=coords) + ";" for k in active_keys]
        full_query = f"[out:json][timeout:25];({''.join(parts)});out body;"
        for url in ["https://overpass.kumi.systems/api/interpreter", "https://api.openstreetmap.fr/oapi/interpreter", "https://overpass-api.de/api/interpreter"]:
            try:
                r = requests.post(url, data={'data': full_query}, timeout=30)
                if r.status_code == 200: return r.json().get('elements', [])
            except: continue
        return []

    # --- CORE PROCESSING ---
    if st.session_state.status == 'running':
        status = st.status("Initializing Scan...", expanded=True)
        try:
            gpx = gpxpy.parse(uploaded_file)
            raw_pts = [p for t in gpx.tracks for s in t.segments for p in s.points]
            track_data = calculate_track_distance(raw_pts)
            
            scan_pts, last = [], None
            for p in raw_pts:
                if not last or haversine(last.longitude, last.latitude, p.longitude, p.latitude) >= 100:
                    scan_pts.append(p); last = p
            
            batches = [scan_pts[i:i+25] for i in range(0, len(scan_pts), 25)]
            found_raw, seen, prog = [], set(), status.progress(0)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as exc:
                futures = {exc.submit(fetch_batch, (b, selected_keys)): i for i, b in enumerate(batches)}
                for i, f in enumerate(concurrent.futures.as_completed(futures)):
                    prog.progress((i+1)/len(batches))
                    for item in f.result():
                        if item['id'] not in seen:
                            seen.add(item['id']); found_raw.append(item)
            
            # Enrich & Filter
            final_pois, locs = [], {k: [] for k in selected_keys}
            min_deg = MIN_GAP_KM / 111.0

            for item in found_raw:
                tags = item.get('tags', {})
                cat = next((k for k,v in amenity_config.items() if k in selected_keys and (tags.get("amenity") in v['query'] or tags.get("shop") in v['query'] or tags.get("tourism") in v['query'] or tags.get("railway") in v['query'])), None)
                if not cat: continue
                
                lat, lon = item['lat'], item['lon']
                if any(sqrt((alat-lat)**2 + (alon-lon)**2) < min_deg for (alat, alon) in locs[cat]): continue

                name = tags.get('name') or tags.get('brand') or tags.get('operator') or cat
                details = [f"🕒 {tags.get('opening_hours')}" if tags.get('opening_hours') else None, f"📞 {tags.get('phone')}" if tags.get('phone') else None]
                desc_str = f"{cat} | " + " | ".join(filter(None, details))
                km_mark = get_nearest_km(lat, lon, track_data)

                final_pois.append({
                    "km": km_mark, "cat": cat, "name": name, "lat": lat, "lon": lon, "desc": desc_str,
                    "hours": tags.get('opening_hours', ""), "phone": tags.get('phone', ""), "city": tags.get('addr:city', ""),
                    "symbol": amenity_config[cat]["icon"], "color": amenity_config[cat]["color"],
                    "gmap": f"https://www.google.com/maps?q={lat},{lon}"
                })
                locs[cat].append((lat, lon))

            final_pois.sort(key=lambda x: x['km'])
            
            # Save Results to State
            df = pd.DataFrame(final_pois)
            
            # Create Enriched GPX Waypoints
            for p in final_pois:
                wpt = gpxpy.gpx.GPXWaypoint(latitude=p['lat'], longitude=p['lon'], name=p['name'])
                wpt.description, wpt.symbol, wpt.type = p['desc'], p['symbol'], p['cat']
                gpx.waypoints.append(wpt)
            
            # Export to Memory
            out_gpx = BytesIO()
            out_gpx.write(gpx.to_xml().encode('utf-8'))
            
            csv_df = df[['km', 'cat', 'name', 'hours', 'phone', 'city', 'lat', 'lon', 'gmap']].copy()
            csv_df.columns = ['KM', 'Type', 'Name', 'Hours', 'Phone', 'City', 'Lat', 'Lon', 'Map Link']
            
            st.session_state.results = {
                "df": df, "pois": final_pois, "gpx_bytes": out_gpx.getvalue(),
                "csv_bytes": csv_df.to_csv(index=False).encode('utf-8'),
                "filename": f"{uploaded_file.name.split('.')[0]}_enriched.gpx",
                "path": [[p.longitude, p.latitude] for p in raw_pts[::30]],
                "start_lat": raw_pts[0].latitude, "start_lon": raw_pts[0].longitude
            }
            st.session_state.status = 'complete'
            status.update(label="Scan Complete!", state="complete")
            st.rerun()

        except Exception as e:
            st.error(f"Error: {e}"); st.session_state.status = 'idle'

    # --- 4. DISPLAY RESULTS ---
    if st.session_state.status == 'complete' and st.session_state.results:
        res = st.session_state.results
        st.subheader("📊 Results")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(res['df']['cat'].value_counts().reset_index().rename(columns={'count':'Count','cat':'Category'}), hide_index=True)
        with c2:
            map_data = [{"coordinates": [p['lon'], p['lat']], "color": p['color'], "info": f"{p['name']} (Km {p['km']:.1f})"} for p in res['pois']]
            st.pydeck_chart(pdk.Deck(
                initial_view_state=pdk.ViewState(latitude=res['start_lat'], longitude=res['start_lon'], zoom=8),
                layers=[
                    pdk.Layer("PathLayer", [{"path": res['path']}], get_path="path", get_color=[255, 0, 0], width_min_pixels=2),
                    pdk.Layer("ScatterplotLayer", map_data, get_position="coordinates", get_fill_color="color", get_radius=300, pickable=True)
                ], tooltip={"text": "{info}"}
            ))
        
        col_dl_gpx, col_dl_csv = st.columns(2)
        with col_dl_gpx:
            st.download_button("⬇️ Download Enriched GPX", res['gpx_bytes'], res['filename'], "application/gpx+xml", type="primary")
        with col_dl_csv:
            st.download_button("⬇️ Download CSV Table", res['csv_bytes'], "Cuesheet.csv", "text/csv")

        if st.button("🔄 Start New File"):
            st.session_state.status = 'idle'
            st.session_state.results = None
            st.rerun()
