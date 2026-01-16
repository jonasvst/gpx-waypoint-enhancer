import streamlit as st
import gpxpy
import requests
import time
import pydeck as pdk
import concurrent.futures
from math import radians, cos, sin, asin, sqrt
from io import BytesIO

# --- PAGE CONFIG ---
st.set_page_config(page_title="TCR Tool V12 (Simple)", page_icon="🚴", layout="centered")

if 'running' not in st.session_state:
    st.session_state.running = False

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; }
    div[data-testid="stExpander"] { display: none; } /* Hide expanders if any exist */
    </style>
""", unsafe_allow_html=True)

st.title("🚴 TCR Survival: Selector")
st.markdown("Select what you need. The tool finds roadside stops (50m range) automatically.")

# --- 1. UPLOAD ---
uploaded_file = st.file_uploader("📂 Drop GPX file", type=['gpx'], disabled=st.session_state.running)

# --- 2. SELECT CATEGORIES ---
st.subheader("🛠️ Select Amenities")

# Hardcoded Logic (Hidden from User)
RADIUS = 50        # Strict Roadside
SAMPLE_STEP = 100  # High Precision
MIN_GAP_KM = 0     # Show EVERYTHING (No De-clutter)

amenity_config = {
    "Water": {
        "label": "💧 Water (Taps/Springs/Cemeteries)",
        "query": """(
            node["amenity"~"drinking_water|fountain|watering_place"](around:{radius},{coords});
            node["natural"~"spring"](around:{radius},{coords});
            node["man_made"~"water_tap|water_well"](around:{radius},{coords});
            node["amenity"~"grave_yard"](around:{radius},{coords});
        )""",
        "icon": "Water", "color": [0, 128, 255]
    },
    "Toilets": {"label": "🚽 Toilets", "query": """node["amenity"~"toilets"](around:{radius},{coords})""", "icon": "Restroom", "color": [150, 150, 150]},
    "Shops": {"label": "🛒 Supermarkets & Kiosks", "query": """node["shop"~"supermarket|convenience|kiosk|bakery|general"](around:{radius},{coords})""", "icon": "Convenience Store", "color": [0, 200, 0]},
    "Fuel": {"label": "⛽ Fuel Stations", "query": """node["amenity"~"fuel"](around:{radius},{coords})""", "icon": "Gas Station", "color": [255, 140, 0]},
    "Food": {"label": "🍔 Restaurants/Fast Food", "query": """node["amenity"~"restaurant|fast_food|cafe"](around:{radius},{coords})""", "icon": "Food", "color": [0, 200, 0]},
    "Sleep": {"label": "🛏️ Hotels & Hostels", "query": """node["tourism"~"hotel|hostel|guest_house"](around:{radius},{coords})""", "icon": "Lodging", "color": [128, 0, 128]},
    "Camping": {"label": "⛺ Campsites", "query": """node["tourism"~"camp_site"](around:{radius},{coords})""", "icon": "Campground", "color": [34, 139, 34]},
    "Bike": {"label": "🔧 Bike Shops", "query": """node["shop"~"bicycle"](around:{radius},{coords})""", "icon": "Bike Shop", "color": [255, 0, 0]},
    "Pharm": {"label": "💊 Pharmacies", "query": """node["amenity"~"pharmacy"](around:{radius},{coords})""", "icon": "First Aid", "color": [255, 0, 0]},
    "ATM": {"label": "🏧 ATMs (Cash)", "query": """node["amenity"~"atm"](around:{radius},{coords})""", "icon": "Generic", "color": [0, 100, 0]},
    "Train": {"label": "🚆 Train Stations (Bailout)", "query": """node["railway"~"station|halt"](around:{radius},{coords})""", "icon": "Generic", "color": [50, 50, 50]}
}

# Display in 3 Columns
cols = st.columns(3)
selected_keys = []
defaults = ["Water", "Toilets", "Shops", "Fuel"]

for i, (key, cfg) in enumerate(amenity_config.items()):
    with cols[i % 3]:
        if st.checkbox(cfg["label"], value=(key in defaults), disabled=st.session_state.running):
            selected_keys.append(key)

# --- ENGINE ---
BATCH_SIZE = 25
WORKERS = 4
HEADERS = {"User-Agent": "TCR-Tool/12.0", "Referer": "https://streamlit.io/"}
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

# --- ACTIONS ---
col_start, col_stop = st.columns([3, 1])
with col_start:
    if uploaded_file and not st.session_state.running:
        if st.button("🚀 Start Scan", type="primary"):
            if not selected_keys: st.error("Select amenities first.")
            else:
                st.session_state.running = True
                st.rerun()

with col_stop:
    if st.session_state.running:
        if st.button("🛑 Cancel"):
            st.session_state.running = False
            st.rerun()

if st.session_state.running:
    status = st.status("Initializing...", expanded=True)
    try:
        gpx = gpxpy.parse(uploaded_file)
        all_pts = []
        for t in gpx.tracks:
            for s in t.segments:
                all_pts.extend(s.points)
        
        # Resample
        scan_pts = []
        last = None
        for p in all_pts:
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
                
                # Timer
                elapsed = time.time() - start_time
                if completed > 0:
                    est_seconds = int((elapsed / completed) * (total_batches - completed))
                    mins, secs = divmod(est_seconds, 60)
                    status.write(f"Scanning... {int(percent*100)}% (Est. {mins}m {secs}s remaining)")
                
                for item in f.result():
                    if item['id'] not in seen_ids:
                        seen_ids.add(item['id'])
                        found_raw.append(item)
        
        # Final Processing
        status.write("Processing results...")
        final_pois = []
        
        # Simple Cat Matcher
        def get_cat(item):
            tags = item.get('tags', {})
            # Prioritize specific checks
            if "Water" in selected_keys:
                if tags.get("amenity") == "grave_yard": return "Water"
                if tags.get("amenity") in ["drinking_water", "fountain", "watering_place"]: return "Water"
                if tags.get("natural") == "spring": return "Water"
                if tags.get("man_made") in ["water_tap", "water_well"]: return "Water"
            
            # Check others
            for k in selected_keys:
                if k == "Water": continue
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
            if cat:
                tags = item.get('tags', {})
                name = tags.get('name', cat)
                if cat == "Water" and tags.get("amenity") == "grave_yard":
                    name = "Cemetery (Check Tap)"
                
                final_pois.append({
                    "lat": item['lat'], "lon": item['lon'],
                    "name": name, "cat": cat,
                    "desc": f"{cat}: {tags.get('amenity', '')}",
                    "symbol": amenity_config[cat]["icon"]
                })
        
        status.update(label=f"Done! Found {len(final_pois)} items.", state="complete")
        
        # Preview Map
        st.subheader("📍 Preview")
        map_data = [{"coordinates": [p['lon'], p['lat']], "color": amenity_config[p['cat']]['color'], "name": p['name']} for p in final_pois]
        path_data = [[p.longitude, p.latitude] for p in all_pts[::30]] # Downsample for map speed
        
        st.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(latitude=all_pts[0].latitude, longitude=all_pts[0].longitude, zoom=8),
            layers=[
                pdk.Layer("PathLayer", [{"path": path_data}], get_path="path", get_color=[255, 0, 0], width_min_pixels=2),
                pdk.Layer("ScatterplotLayer", map_data, get_position="coordinates", get_fill_color="color", get_radius=200, pickable=True)
            ], tooltip={"text": "{name}"}
        ))
        
        # GPX
        for p in final_pois:
            wpt = gpxpy.gpx.GPXWaypoint(latitude=p['lat'], longitude=p['lon'], name=p['name'])
            wpt.description = p['desc']
            wpt.symbol = p['symbol']
            wpt.type = p['cat']
            gpx.waypoints.append(wpt)
            
        out = BytesIO()
        out.write(gpx.to_xml().encode('utf-8'))
        out.seek(0)
        
        st.success(f"Added {len(final_pois)} POIs.")
        st.download_button("⬇️ Download GPX", out, "TCR_Enhanced.gpx", "application/gpx+xml", type="primary")
        
        if st.button("🔄 Upload Another File"):
            st.session_state.running = False
            st.rerun()

    except Exception as e:
        st.error(f"Error: {e}")
        st.session_state.running = False
