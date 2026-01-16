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

if 'running' not in st.session_state:
    st.session_state.running = False

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; font-weight: bold; }
    div[data-testid="stExpander"] { border: 1px solid #e6e6e6; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

st.title("📍 GPS Enricher")
st.markdown("Turn any GPX track into a survival-ready route with water, food, and fuel stops.")

# --- 1. USER GUIDE ---
with st.expander("📘 **User Guide: How it works**"):
    st.markdown("""
    ### **The Logic**
    This tool is designed for **ultra-cycling efficiency**. It does not dump thousands of POIs on your map.
    
    1.  **Strict Roadside Scan:** We only look **50 meters** sideways from your track. If a fountain is 200m away, we ignore it (unless you detour).
    2.  **No Blind Spots:** We scan every **100 meters** along your route to ensure nothing is missed.
    3.  **Smart De-Clutter:** If we find a Gas Station, we ignore other Gas Stations for the next **2km** (configurable). This keeps your GPS screen readable in towns.
    
    ### **Categories**
    * **💧 Water:** Includes drinking fountains, springs, taps, and **Cemeteries** (often have taps).
    * **🛒 Shops:** Supermarkets, convenience stores, and bakeries.
    * **⛽ Fuel:** 24/7 stations (often with food/water).
    """)

# --- 2. UPLOAD ---
uploaded_file = st.file_uploader("📂 **Step 1:** Drop your GPX file here", type=['gpx'], disabled=st.session_state.running)

# --- 3. SETTINGS & FILTERS ---
if uploaded_file:
    st.subheader("⚙️ Configuration")
    
    # DENSITY CONTROL (The only technical setting left visible)
    MIN_GAP_KM = st.slider(
        "🧹 **Map Density (Min Gap)**", 
        0.0, 10.0, 2.0, 0.5,
        format="%.1f km",
        help="Prevents clutter. Example: If set to 2.0km, we won't mark two Supermarkets within 2km of each other.",
        disabled=st.session_state.running
    )

    st.caption("Select amenities to scan for:")

    # HIDDEN CONSTANTS
    RADIUS = 50        
    SAMPLE_STEP = 100  

    amenity_config = {
        "Water": {
            "label": "💧 Water & Taps",
            "query": """(
                node["amenity"~"drinking_water|fountain|watering_place"](around:{radius},{coords});
                node["natural"~"spring"](around:{radius},{coords});
                node["man_made"~"water_tap|water_well"](around:{radius},{coords});
            )""",
            "icon": "Water", "color": [0, 128, 255]
        },
        "Cemetery": {
            "label": "⚰️ Cemeteries",
            "query": """node["amenity"~"grave_yard"](around:{radius},{coords})""",
            "icon": "Water", "color": [0, 100, 255]
        },
        "Toilets": {"label": "🚽 Toilets", "query": """node["amenity"~"toilets"](around:{radius},{coords})""", "icon": "Restroom", "color": [150, 150, 150]},
        "Shops": {"label": "🛒 Supermarkets", "query": """node["shop"~"supermarket|convenience|kiosk|bakery|general"](around:{radius},{coords})""", "icon": "Convenience Store", "color": [0, 200, 0]},
        "Fuel": {"label": "⛽ Fuel Stations", "query": """node["amenity"~"fuel"](around:{radius},{coords})""", "icon": "Gas Station", "color": [255, 140, 0]},
        "Food": {"label": "🍔 Restaurants", "query": """node["amenity"~"restaurant|fast_food|cafe"](around:{radius},{coords})""", "icon": "Food", "color": [0, 200, 0]},
        "Sleep": {"label": "🛏️ Hotels", "query": """node["tourism"~"hotel|hostel|guest_house"](around:{radius},{coords})""", "icon": "Lodging", "color": [128, 0, 128]},
        "Camping": {"label": "⛺ Campsites", "query": """node["tourism"~"camp_site"](around:{radius},{coords})""", "icon": "Campground", "color": [34, 139, 34]},
        "Bike": {"label": "🔧 Bike Shops", "query": """node["shop"~"bicycle"](around:{radius},{coords})""", "icon": "Bike Shop", "color": [255, 0, 0]},
        "Pharm": {"label": "💊 Pharmacies", "query": """node["amenity"~"pharmacy"](around:{radius},{coords})""", "icon": "First Aid", "color": [255, 0, 0]},
        "ATM": {"label": "🏧 ATMs", "query": """node["amenity"~"atm"](around:{radius},{coords})""", "icon": "Generic", "color": [0, 100, 0]},
        "Train": {"label": "🚆 Train Stations", "query": """node["railway"~"station|halt"](around:{radius},{coords})""", "icon": "Generic", "color": [50, 50, 50]}
    }

    cols = st.columns(3)
    selected_keys = []
    defaults = ["Water", "Cemetery", "Toilets", "Shops", "Fuel"]

    for i, (key, cfg) in enumerate(amenity_config.items()):
        with cols[i % 3]:
            if st.checkbox(cfg["label"], value=(key in defaults), disabled=st.session_state.running):
                selected_keys.append(key)

    # --- ACTION BUTTONS ---
    st.markdown("---")
    col_start, col_stop = st.columns([3, 1])
    
    with col_start:
        if not st.session_state.running:
            if st.button("🚀 Start Scan", type="primary"):
                if not selected_keys: st.error("Please select at least one amenity.")
                else:
                    st.session_state.running = True
                    st.rerun()

    with col_stop:
        if st.session_state.running:
            if st.button("🛑 Cancel"):
                st.session_state.running = False
                st.rerun()

# --- ENGINE ---
BATCH_SIZE = 25
WORKERS = 4
HEADERS = {"User-Agent": "GPX-Enricher/17.0", "Referer": "https://streamlit.io/"}
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

def calculate_track_distance(points):
    total_dist = 0
    points_with_dist = []
    last = None
    for p in points:
        if last:
            dist = haversine(last.longitude, last.latitude, p.longitude, p.latitude)
            total_dist += dist
        points_with_dist.append({"lat": p.latitude, "lon": p.longitude, "cum_dist": total_dist})
        last = p
    return points_with_dist

def get_nearest_km(poi_lat, poi_lon, track_data):
    min_dist = float('inf')
    km_mark = 0
    # Optimization: Scan strided track points
    for tp in track_data[::10]:
        d = (tp['lat'] - poi_lat)**2 + (tp['lon'] - poi_lon)**2
        if d < min_dist:
            min_dist = d
            km_mark = tp['cum_dist']
    return km_mark / 1000.0

def fetch_batch(args):
    points, active_keys = args
    if not points: return []
    coords = ",".join([f"{round(p.latitude,5)},{round(p.longitude,5)}" for p in points])
    parts = []
    for key in active_keys:
        template = amenity_config[key]["query"]
        q = template.format(radius=RADIUS, coords=coords)
        parts.append(q + ";")
    full_query = f"[out:json][timeout:25];({''.join(parts)});out body;"
    for url in MIRRORS:
        try:
            r = requests.post(url, data={'data': full_query}, headers=HEADERS, timeout=30)
            if r.status_code == 200: return r.json().get('elements', [])
            if r.status_code == 429: time.sleep(1)
        except: continue
    return []

if st.session_state.running:
    status = st.status("Initializing Scan...", expanded=True)
    try:
        gpx = gpxpy.parse(uploaded_file)
        
        # 1. Flatten & Distance
        raw_pts = []
        for t in gpx.tracks:
            for s in t.segments:
                raw_pts.extend(s.points)
        track_data = calculate_track_distance(raw_pts)
        
        # 2. Resample
        scan_pts = []
        last = None
        for p in raw_pts:
            if not last or haversine(last.longitude, last.latitude, p.longitude, p.latitude) >= SAMPLE_STEP:
                scan_pts.append(p)
                last = p
        
        batches = [scan_pts[i:i+BATCH_SIZE] for i in range(0, len(scan_pts), BATCH_SIZE)]
        total_batches = len(batches)
        
        found_raw = []
        seen_ids = set()
        prog = status.progress(0)
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as exc:
            futures = {exc.submit(fetch_batch, (b, selected_keys)): i for i, b in enumerate(batches)}
            for i, f in enumerate(concurrent.futures.as_completed(futures)):
                completed = i + 1
                percent = completed / total_batches
                prog.progress(percent)
                
                elapsed = time.time() - start_time
                if completed > 0:
                    avg = elapsed / completed
                    rem = int(avg * (total_batches - completed))
                    m, s = divmod(rem, 60)
                    status.update(label=f"Scanning... {int(percent*100)}% (Est. {m}m {s}s)")
                
                for item in f.result():
                    if item['id'] not in seen_ids:
                        seen_ids.add(item['id'])
                        found_raw.append(item)
        
        # 3. Post-Processing
        status.write("Cleaning & Calculating KM markers...")
        final_pois = []
        accepted_locs = {k: [] for k in selected_keys}
        min_deg = MIN_GAP_KM / 111.0
        
        def get_cat(item):
            tags = item.get('tags', {})
            if "Cemetery" in selected_keys and tags.get("amenity") == "grave_yard": return "Cemetery"
            if "Water" in selected_keys:
                if tags.get("amenity") in ["drinking_water", "fountain", "watering_place"]: return "Water"
                if tags.get("natural") == "spring": return "Water"
                if tags.get("man_made") in ["water_tap", "water_well"]: return "Water"
            for k in selected_keys:
                if k in ["Water", "Cemetery"]: continue
                q = amenity_config[k]["query"]
                if "toilets" in q and tags.get("amenity") == "toilets": return k
                if "shop" in q and "shop" in tags: return k
                if "fuel" in q and tags.get("amenity") == "fuel": return k
                if "restaurant" in q and tags.get("amenity") in ["restaurant","fast_food","cafe"]: return k
                if "tourism" in q and "tourism" in tags: return k
                if "pharmacy" in q and tags.get("amenity") == "pharmacy": return k
                if "bicycle" in q and "shop" in tags: return k
                if "atm" in q and tags.get("amenity") == "atm": return k
                if "railway" in q and tags.get("railway") in ["station","halt"]: return k
            return None

        for item in found_raw:
            cat = get_cat(item)
            if not cat: continue
            
            lat, lon = item['lat'], item['lon']
            
            # De-clutter check
            too_close = False
            for (alat, alon) in accepted_locs[cat]:
                if sqrt((alat-lat)**2 + (alon-lon)**2) < min_deg:
                    too_close = True
                    break
            
            if not too_close:
                tags = item.get('tags', {})
                name = tags.get('name', cat)
                if cat == "Cemetery": name = "Cemetery (Check Tap)"
                
                km_mark = get_nearest_km(lat, lon, track_data)
                
                final_pois.append({
                    "lat": lat, "lon": lon,
                    "name": name, "cat": cat,
                    "km": km_mark,
                    "desc": f"{cat}: {tags.get('amenity', '')} (@ {km_mark:.1f}km)",
                    "symbol": amenity_config[cat]["icon"],
                    "gmap": f"http://googleusercontent.com/maps.google.com/?q={lat},{lon}"
                })
                accepted_locs[cat].append((lat, lon))
        
        final_pois.sort(key=lambda x: x['km'])
        status.update(label="Complete!", state="complete", expanded=False)
        
        # --- RESULTS ---
        st.subheader("📊 Results")
        
        # Summary Table
        df = pd.DataFrame(final_pois)
        if not df.empty:
            counts = df['cat'].value_counts().reset_index()
            counts.columns = ['Category', 'Count']
            c1, c2 = st.columns([1, 2])
            with c1:
                st.dataframe(counts, hide_index=True)
            with c2:
                # Map
                map_data = [{"coordinates": [p['lon'], p['lat']], "color": amenity_config[p['cat']]['color'], "name": p['name']} for p in final_pois]
                path_data = [[p.longitude, p.latitude] for p in raw_pts[::30]]
                st.pydeck_chart(pdk.Deck(
                    initial_view_state=pdk.ViewState(latitude=raw_pts[0].latitude, longitude=raw_pts[0].longitude, zoom=8),
                    layers=[
                        pdk.Layer("PathLayer", [{"path": path_data}], get_path="path", get_color=[255, 0, 0], width_min_pixels=2),
                        pdk.Layer("ScatterplotLayer", map_data, get_position="coordinates", get_fill_color="color", get_radius=200, pickable=True)
                    ], tooltip={"text": "{name}"}
                ))
            
            st.success(f"Added {len(final_pois)} waypoints to your route.")
            
            # Downloads
            col_gpx, col_csv = st.columns(2)
            
            # GPX Build
            for p in final_pois:
                wpt = gpxpy.gpx.GPXWaypoint(latitude=p['lat'], longitude=p['lon'], name=p['name'])
                wpt.description = p['desc']
                wpt.symbol = p['symbol']
                wpt.type = p['cat']
                gpx.waypoints.append(wpt)
            
            out_gpx = BytesIO()
            out_gpx.write(gpx.to_xml().encode('utf-8'))
            out_gpx.seek(0)
            
            with col_gpx:
                st.download_button("⬇️ Download GPX (Device)", out_gpx, "Enriched_Route.gpx", "application/gpx+xml", type="primary")
                
            # CSV Build
            csv_df = df[['km', 'cat', 'name', 'lat', 'lon', 'gmap']].copy()
            csv_df['km'] = csv_df['km'].round(1)
            csv_df.columns = ['KM', 'Category', 'Name', 'Latitude', 'Longitude', 'Google Maps Link']
            out_csv = csv_df.to_csv(index=False).encode('utf-8')
            
            with col_csv:
                st.download_button("⬇️ Download CSV (Cue Sheet)", out_csv, "Route_CueSheet.csv", "text/csv")
        
        else:
            st.warning("No amenities found nearby. Try increasing the Density Slider or checking more categories.")

        if st.button("🔄 Start New File"):
            st.session_state.running = False
            st.rerun()

    except Exception as e:
        st.error(f"Error: {e}")
        st.session_state.running = False
