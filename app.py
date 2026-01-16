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
# 'status' handles the flow: idle -> running -> complete
if 'status' not in st.session_state:
    st.session_state.status = 'idle'
# 'results' stores the generated files and data to prevent reprocessing
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
    * **Range:** The tool scans a **50m radius** along your track to ensure zero-detour stops.
    * **Coverage:** It samples every **100m** to guarantee full coverage.
    * **De-Cluttering:** It applies a "minimum gap" filter (default 2km) to prevent icon stacking in towns.

    ### **2. Data Extraction**
    We extract deep details from OpenStreetMap:
    * **Identity:** Brand names ("Shell", "Coop") instead of generic labels.
    * **Logistics:** **Opening Hours** and **Phone Numbers**.
    * **Attributes:** Water drinkability, Toilet fees, etc.

    ### **3. Searchable Amenities**
    * **💧 Water:** Drinking fountains, springs, taps, and **Cemeteries** (marked with Water icon).
    * **🛒 Shops:** Supermarkets, convenience stores, bakeries.
    * **⛽ Fuel:** 24/7 stations (often with food/water).
    * **🍔 Food:** Restaurants, fast food, cafes.
    * **🚽 Toilets:** Public restrooms.
    * **🛏️ Sleep:** Hotels, hostels, guest houses.
    * **⛺ Camping:** Official campsites.
    * **💊 Pharmacy:** Medical supplies.
    * **🔧 Bike Shop:** Repairs and parts.
    * **🏧 ATM:** Cash machines.
    * **🚆 Train:** Stations (for emergency bail-out).

    ### **4. Support**
    For issues or feature requests: **jonas@verest.ch**
    """)

# --- 2. UPLOAD ---
is_busy = st.session_state.status == 'running'
uploaded_file = st.file_uploader("📂 **Step 1:** Drop your GPX file here", type=['gpx'], disabled=is_busy)

# --- 3. SETTINGS ---
if uploaded_file:
    st.subheader("⚙️ Configuration")
    
    MIN_GAP_KM = st.slider(
        "🧹 **Map Density (Min Gap)**", 
        0.0, 10.0, 2.0, 0.5,
        format="%.1f km",
        help="Controls map cleanliness. Higher = Fewer icons. Lower = More data.",
        disabled=is_busy
    )

    st.caption("Select amenities to scan for:")

    # CONSTANTS
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
            "label": "⚰️ Cemeteries (Water)",
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
            if st.checkbox(cfg["label"], value=(key in defaults), disabled=is_busy):
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

# --- ENGINE HELPERS ---
BATCH_SIZE = 25
WORKERS = 4
HEADERS = {"User-Agent": "GPX-Enricher/23.0", "Referer": "https://streamlit.io/"}
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

# --- 4. ENGINE (LOGIC) ---
if st.session_state.status == 'running':
    status_ui = st.status("Initializing Scan...", expanded=True)
    try:
        gpx = gpxpy.parse(uploaded_file)
        
        # 1. Resample
        raw_pts = []
        for t in gpx.tracks:
            for s in t.segments:
                raw_pts.extend(s.points)
        track_data = calculate_track_distance(raw_pts)
        
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
        prog = status_ui.progress(0)
        start_t = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as exc:
            futures = {exc.submit(fetch_batch, (b, selected_keys)): i for i, b in enumerate(batches)}
            for i, f in enumerate(concurrent.futures.as_completed(futures)):
                comp = i + 1
                prog.progress(comp / total)
                elapsed = time.time() - start_t
                if comp > 0:
                    avg = elapsed / comp
                    rem = int(avg * (total - comp))
                    m, s = divmod(rem, 60)
                    status_ui.update(label=f"Scanning... {int((comp/total)*100)}% (Est. {m}m {s}s)")
                for item in f.result():
                    if item['id'] not in seen:
                        seen.add(item['id'])
                        found_raw.append(item)
        
        # 2. Enrich
        status_ui.write("Enriching data...")
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
            
            too_close = False
            for (alat, alon) in locs[cat]:
                if sqrt((alat-lat)**2 + (alon-lon)**2) < min_deg:
                    too_close = True
                    break
            
            if not too_close:
                tags = item.get('tags', {})
                name = tags.get('name') or tags.get('brand') or tags.get('operator') or tags.get('addr:city')
                if not name:
                    name = "Cemetery (Check Tap)" if cat == "Cemetery" else cat
                
                details = []
                hrs = tags.get('opening_hours')
                if hrs: details.append(f"🕒 {hrs}")
                ph = tags.get('phone')
                if ph: details.append(f"📞 {ph}")
                if tags.get('fee') == 'yes': details.append("💵 Paid")
                elif tags.get('fee') == 'no': details.append("🆓 Free")
                drink = tags.get('drinking_water')
                if drink == 'yes': details.append("🚰 Potable")
                elif drink == 'no': details.append("⚠️ Not Potable")
                
                desc_str = f"{cat}"
                if details: desc_str += " | " + " | ".join(details)
                
                km_mark = get_nearest_km(lat, lon, track_data)
                
                final_pois.append({
                    "km": km_mark, "cat": cat, "name": name,
                    "lat": lat, "lon": lon,
                    "desc": desc_str,
                    "hours": hrs or "", "phone": ph or "", "city": tags.get('addr:city', ""),
                    "symbol": amenity_config[cat]["icon"],
                    "gmap": f"http://googleusercontent.com/maps.google.com/?q={lat},{lon}",
                    "color": amenity_config[cat]["color"]
                })
                locs[cat].append((lat, lon))
        
        final_pois.sort(key=lambda x: x['km'])
        
        # 3. Generate Files
        df_final = pd.DataFrame(final_pois)
        
        # GPX Creation
        for p in final_pois:
            wpt = gpxpy.gpx.GPXWaypoint(latitude=p['lat'], longitude=p['lon'], name=p['name'])
            wpt.description = p['desc']
            wpt.symbol = p['symbol']
            wpt.type = p['cat']
            gpx.waypoints.append(wpt)
        
        out_gpx = BytesIO()
        out_gpx.write(gpx.to_xml().encode('utf-8'))
        
        # CSV Creation
        csv_df = df_final[['km', 'cat', 'name', 'hours', 'phone', 'city', 'lat', 'lon', 'gmap']].copy()
        csv_df['km'] = csv_df['km'].round(1)
        csv_df.columns = ['KM', 'Type', 'Name', 'Hours', 'Phone', 'City', 'Lat', 'Lon', 'Map Link']
        out_csv = csv_df.to_csv(index=False).encode('utf-8')

        # SAVE TO SESSION STATE
        st.session_state.results = {
            "df": df_final,
            "gpx_bytes": out_gpx.getvalue(),
            "csv_bytes": out_csv,
            "filename": f"{uploaded_file.name.split('.')[0]}_enriched.gpx",
            "map_data": final_pois,
            "path_data": [[p.longitude, p.latitude] for p in raw_pts[::30]],
            "start_coords": [raw_pts[0].latitude, raw_pts[0].longitude]
        }
        st.session_state.status = 'complete'
        status_ui.update(label="Complete!", state="complete", expanded=False)
        st.rerun()

    except Exception as e:
        st.error(f"Error: {e}")
        st.session_state.status = 'idle'

# --- 5. RESULTS (STATIONARY DISPLAY) ---
if st.session_state.status == 'complete' and st.session_state.results:
    res = st.session_state.results
    st.subheader("📊 Results")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        # Avoid pandas version conflict on value_counts
        counts = res['df']['cat'].value_counts().reset_index()
        counts.columns = ['Category', 'Count']
        st.dataframe(counts, hide_index=True)
        
    with c2:
        pydeck_data = []
        for p in res['map_data']:
            info = f"**{p['name']}** ({p['cat']})\n📍 Km {p['km']:.1f}"
            if p['hours']: info += f"\n🕒 {p['hours']}"
            pydeck_data.append({
                "coordinates": [p['lon'], p['lat']],
                "color": p['color'],
                "info": info
            })
            
        st.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(latitude=res['start_coords'][0], longitude=res['start_coords'][1], zoom=8),
            layers=[
                pdk.Layer("PathLayer", [{"path": res['path_data']}], get_path="path", get_color=[255, 0, 0], width_min_pixels=2),
                pdk.Layer("ScatterplotLayer", pydeck_data, get_position="coordinates", get_fill_color="color", get_radius=200, pickable=True)
            ], tooltip={"text": "{info}"}
        ))
    
    st.success(f"Enriched with {len(res['map_data'])} points.")
    
    col_gpx, col_csv = st.columns(2)
    with col_gpx:
        # Pulls directly from memory - NO RERUNNING OF SCAN
        st.download_button("⬇️ Download GPX", res['gpx_bytes'], res['filename'], "application/gpx+xml", type="primary")
    with col_csv:
        st.download_button("⬇️ Download CSV", res['csv_bytes'], "CueSheet.csv", "text/csv")

    if st.button("🔄 New File"):
        st.session_state.status = 'idle'
        st.session_state.results = None
        st.rerun()
