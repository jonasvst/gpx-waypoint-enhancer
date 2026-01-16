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
st.set_page_config(page_title="GPS Enricher V18 (Deep Data)", page_icon="🕵️", layout="centered")

if 'running' not in st.session_state:
    st.session_state.running = False

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; font-weight: bold; }
    div[data-testid="stExpander"] { display: none; }
    </style>
""", unsafe_allow_html=True)

st.title("🕵️ GPS Enricher: Deep Data")
st.markdown("Extracts **Names, Opening Hours, Phone Numbers, and Fees** to create a detailed Cue Sheet.")

# --- 1. UPLOAD ---
uploaded_file = st.file_uploader("📂 Drop GPX file", type=['gpx'], disabled=st.session_state.running)

# --- 2. SETTINGS ---
st.subheader("⚙️ Settings")
MIN_GAP_KM = st.slider(
    "🧹 **Density (Min Gap km)**", 
    0.0, 10.0, 2.0, 0.5,
    help="0 = Show everything. 2.0 = Clean map.",
    disabled=st.session_state.running
)

# --- 3. CATEGORIES ---
st.caption("Select Amenities:")

# HIDDEN CONSTANTS
RADIUS = 50        
SAMPLE_STEP = 100  

amenity_config = {
    "Water": {
        "label": "💧 Water",
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

# --- ENGINE ---
BATCH_SIZE = 25
WORKERS = 4
HEADERS = {"User-Agent": "GPX-DeepData/18.0", "Referer": "https://streamlit.io/"}
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
    for tp in track_data[::10]: # Optimization
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

# --- ACTIONS ---
col_start, col_stop = st.columns([3, 1])
with col_start:
    if uploaded_file and not st.session_state.running:
        if st.button("🚀 Start Scan", type="primary"):
            if not selected_keys: st.error("Select amenities.")
            else:
                st.session_state.running = True
                st.rerun()

with col_stop:
    if st.session_state.running:
        if st.button("🛑 Cancel"):
            st.session_state.running = False
            st.rerun()

if st.session_state.running:
    status = st.status("Scanning...", expanded=True)
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
        total = len(batches)
        
        found_raw = []
        seen = set()
        prog = status.progress(0)
        start_t = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as exc:
            futures = {exc.submit(fetch_batch, (b, selected_keys)): i for i, b in enumerate(batches)}
            for i, f in enumerate(concurrent.futures.as_completed(futures)):
                comp = i + 1
                prog.progress(comp / total)
                
                # Timer
                elap = time.time() - start_t
                if comp > 0:
                    avg = elap / comp
                    rem = int(avg * (total - comp))
                    m, s = divmod(rem, 60)
                    status.update(label=f"Scanning... (Est. {m}m {s}s)")
                
                for item in f.result():
                    if item['id'] not in seen:
                        seen.add(item['id'])
                        found_raw.append(item)
        
        # 3. Smart Extraction
        status.write("Enriching data...")
        final_pois = []
        locs = {k: [] for k in selected_keys}
        min_deg = MIN_GAP_KM / 111.0
        
        def identify_cat(item):
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
            cat = identify_cat(item)
            if not cat: continue
            
            lat, lon = item['lat'], item['lon']
            
            # Gap check
            too_close = False
            for (alat, alon) in locs[cat]:
                if sqrt((alat-lat)**2 + (alon-lon)**2) < min_deg:
                    too_close = True
                    break
            
            if not too_close:
                tags = item.get('tags', {})
                
                # --- DEEP DATA EXTRACTION ---
                
                # 1. Name Strategy
                name = tags.get('name')
                if not name:
                    name = tags.get('brand') or tags.get('operator')
                    if not name:
                        # Fallback: Category + City?
                        city = tags.get('addr:city')
                        if city: name = f"{cat} ({city})"
                        else: 
                            if cat == "Cemetery": name = "Cemetery (Check Tap)"
                            else: name = cat

                # 2. Details String
                details = []
                # Hours
                hrs = tags.get('opening_hours')
                if hrs: details.append(f"🕒 {hrs}")
                
                # Phone
                ph = tags.get('phone') or tags.get('contact:phone')
                if ph: details.append(f"📞 {ph}")
                
                # Fee
                fee = tags.get('fee')
                if fee == 'yes': details.append("💵 Paid")
                elif fee == 'no': details.append("🆓 Free")
                
                # Drinking
                drink = tags.get('drinking_water')
                if drink == 'yes': details.append("🚰 Potable")
                elif drink == 'no': details.append("⚠️ Not Potable")
                
                # Access
                access = tags.get('access')
                if access == 'private': details.append("🚫 Private")
                elif access == 'permissive': details.append("⚠️ Permissive")
                
                # Indoor
                indoor = tags.get('indoor')
                if indoor == 'yes': details.append("🏠 Indoor")
                
                desc_str = f"{cat}"
                if details:
                    desc_str += " | " + " | ".join(details)
                
                km_mark = get_nearest_km(lat, lon, track_data)
                
                final_pois.append({
                    "km": km_mark,
                    "cat": cat,
                    "name": name,
                    "lat": lat, "lon": lon,
                    "desc": desc_str,
                    "hours": hrs or "",
                    "phone": ph or "",
                    "city": tags.get('addr:city', ""),
                    "symbol": amenity_config[cat]["icon"],
                    "gmap": f"http://googleusercontent.com/maps.google.com/?q={lat},{lon}"
                })
                locs[cat].append((lat, lon))
        
        final_pois.sort(key=lambda x: x['km'])
        status.update(label="Complete!", state="complete", expanded=False)
        
        # --- PREVIEW ---
        st.subheader("📊 Results")
        
        # Summary
        df = pd.DataFrame(final_pois)
        if not df.empty:
            c1, c2 = st.columns([1, 2])
            with c1:
                counts = df['cat'].value_counts().reset_index()
                counts.columns = ['Category', 'Count']
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
                    ], tooltip={"text": "{name}\n{desc}"}
                ))
            
            st.success(f"Enriched with {len(final_pois)} points.")
            
            # --- DOWNLOADS ---
            col_gpx, col_csv = st.columns(2)
            
            # GPX
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
                st.download_button("⬇️ Download GPX", out_gpx, "Enriched.gpx", "application/gpx+xml", type="primary")
            
            # CSV
            csv_df = df[['km', 'cat', 'name', 'hours', 'phone', 'city', 'lat', 'lon', 'gmap']].copy()
            csv_df['km'] = csv_df['km'].round(1)
            csv_df.columns = ['KM', 'Type', 'Name', 'Hours', 'Phone', 'City', 'Lat', 'Lon', 'Map Link']
            out_csv = csv_df.to_csv(index=False).encode('utf-8')
            with col_csv:
                st.download_button("⬇️ Download CSV", out_csv, "CueSheet.csv", "text/csv")

        else:
            st.warning("No amenities found.")

        if st.button("🔄 New File"):
            st.session_state.running = False
            st.rerun()

    except Exception as e:
        st.error(f"Error: {e}")
        st.session_state.running = False
