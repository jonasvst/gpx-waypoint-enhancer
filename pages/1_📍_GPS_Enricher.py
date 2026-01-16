import streamlit as st
import gpxpy
import pandas as pd
import pydeck as pdk
import concurrent.futures
import requests
import time
from math import radians, cos, sin, asin, sqrt
from io import BytesIO

# --- CONFIG ---
st.set_page_config(page_title="GPS Enricher", page_icon="📍", layout="centered")

if "running" not in st.session_state:
    st.session_state.running = False

# --- CSS ---
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    div[data-testid="stExpander"] { border: 1px solid #e6e6e6; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# --- HEADER & NAV ---
c1, c2 = st.columns([1, 4])
with c1:
    if st.button("← Back"):
        st.switch_page("Home.py")
with c2:
    st.title("📍 GPS Enricher")

st.markdown("Enrich your GPX track with detailed roadside amenities.")

# --- USER GUIDE (Restored Legend) ---
with st.expander("📘 **User Guide: Logic & Legend**"):
    st.markdown("""
    **1. Smart Scanning:**
    * We scan **50m roadside** (zero detour) every **100m** (no blind spots).
    * **De-cluttering:** We hide duplicate stops (like 2 gas stations in 1 town) to keep your map readable.

    **2. Searchable Amenities:**
    * **💧 Water:** Fountains, Springs, Taps, and **Cemeteries**.
    * **🛒 Shops:** Supermarkets, Bakeries, Kiosks.
    * **⛽ Fuel:** 24/7 Stations (Food/Water).
    * **🍔 Food:** Restaurants, Fast Food.
    * **🚽 Toilets:** Public restrooms.
    * **🛏️ Sleep:** Hotels, Hostels.
    * **⛺ Camping:** Official campsites.
    * **💊 Pharmacy:** Medical supplies.
    * **🔧 Bike Shop:** Repairs.
    * **🏧 ATM:** Cash machines.
    * **🚆 Train:** Bail-out stations.
    """)

# --- CONFIGURATION ---
amenity_config = {
    "Water": {
        "label": "💧 Water",
        "desc": "Fountains, Springs, Taps, Cemeteries",
        "query": """(node["amenity"~"drinking_water|fountain|watering_place"](around:{radius},{coords});node["natural"~"spring"](around:{radius},{coords});node["man_made"~"water_tap|water_well"](around:{radius},{coords});node["amenity"~"grave_yard"](around:{radius},{coords});)""",
        "icon": "Water", "color": [0, 128, 255]
    },
    "Toilets": {"label": "🚽 Toilets", "desc": "Public Restrooms", "query": """node["amenity"~"toilets"](around:{radius},{coords})""", "icon": "Restroom", "color": [150, 150, 150]},
    "Shops": {"label": "🛒 Shops", "desc": "Supermarkets, Kiosks, Bakeries", "query": """node["shop"~"supermarket|convenience|kiosk|bakery|general"](around:{radius},{coords})""", "icon": "Convenience Store", "color": [0, 200, 0]},
    "Fuel": {"label": "⛽ Fuel", "desc": "Gas Stations (24/7)", "query": """node["amenity"~"fuel"](around:{radius},{coords})""", "icon": "Gas Station", "color": [255, 140, 0]},
    "Food": {"label": "🍔 Food", "desc": "Restaurants, Fast Food", "query": """node["amenity"~"restaurant|fast_food|cafe"](around:{radius},{coords})""", "icon": "Food", "color": [0, 200, 0]},
    "Sleep": {"label": "🛏️ Sleep", "desc": "Hotels, Hostels", "query": """node["tourism"~"hotel|hostel|guest_house"](around:{radius},{coords})""", "icon": "Lodging", "color": [128, 0, 128]},
    "Camping": {"label": "⛺ Camping", "desc": "Campsites", "query": """node["tourism"~"camp_site"](around:{radius},{coords})""", "icon": "Campground", "color": [34, 139, 34]},
    "Bike": {"label": "🔧 Bike", "desc": "Repair Shops", "query": """node["shop"~"bicycle"](around:{radius},{coords})""", "icon": "Bike Shop", "color": [255, 0, 0]},
    "Pharm": {"label": "💊 Pharmacy", "desc": "Medical Supplies", "query": """node["amenity"~"pharmacy"](around:{radius},{coords})""", "icon": "First Aid", "color": [255, 0, 0]},
    "ATM": {"label": "🏧 ATM", "desc": "Cash Machines", "query": """node["amenity"~"atm"](around:{radius},{coords})""", "icon": "Generic", "color": [0, 100, 0]},
    "Train": {"label": "🚆 Train", "desc": "Stations", "query": """node["railway"~"station|halt"](around:{radius},{coords})""", "icon": "Generic", "color": [50, 50, 50]}
}

uploaded_file = st.file_uploader("📂 **Upload GPX file**", type=["gpx"], disabled=st.session_state.running)

if uploaded_file:
    st.subheader("⚙️ Settings")
    MIN_GAP_KM = st.slider("🧹 Map Density (Min Gap km)", 0.0, 10.0, 2.0, 0.5, help="Prevents clutter. If set to 2.0km, we won't show two shops within 2km of each other.", disabled=st.session_state.running)
    
    st.caption("Select Amenities:")
    cols = st.columns(3)
    selected_keys = []
    defaults = ["Water", "Toilets", "Shops", "Fuel"]
    
    for i, (key, cfg) in enumerate(amenity_config.items()):
        with cols[i % 3]:
            # Tooltip AND Label description
            if st.checkbox(cfg["label"], value=(key in defaults), help=cfg["desc"], disabled=st.session_state.running):
                selected_keys.append(key)

    st.markdown("---")
    c_go, c_stop = st.columns([3, 1])
    with c_go:
        if not st.session_state.running:
            if st.button("🚀 Start Scan", type="primary"):
                if not selected_keys: st.error("Select amenities.")
                else:
                    st.session_state.running = True
                    st.rerun()
    with c_stop:
        if st.session_state.running:
            if st.button("🛑 Cancel"):
                st.session_state.running = False
                st.rerun()

# --- ENGINE CONSTANTS ---
RADIUS = 50
SAMPLE_STEP = 100
BATCH_SIZE = 25
WORKERS = 4
HEADERS = {"User-Agent": "GPX-Enricher/25.0", "Referer": "https://streamlit.io/"}
MIRRORS = ["https://overpass.kumi.systems/api/interpreter", "https://api.openstreetmap.fr/oapi/interpreter", "https://overpass-api.de/api/interpreter"]

# --- FUNCTIONS ---
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 6371000 * 2 * asin(sqrt(a))

def calculate_track_distance(points):
    total = 0
    res = []
    last = None
    for p in points:
        if last: total += haversine(last.longitude, last.latitude, p.longitude, p.latitude)
        res.append({"lat": p.latitude, "lon": p.longitude, "cum_dist": total})
        last = p
    return res

def get_nearest_km(lat, lon, track_data):
    min_d, km = float("inf"), 0
    for t in track_data[::10]:
        d = (t["lat"]-lat)**2 + (t["lon"]-lon)**2
        if d < min_d: min_d, km = d, t["cum_dist"]
    return km / 1000.0

def fetch_batch(args):
    pts, keys = args
    if not pts: return []
    coords = ",".join([f"{round(p.latitude,5)},{round(p.longitude,5)}" for p in pts])
    parts = []
    for k in keys:
        q = amenity_config[k]["query"].format(radius=RADIUS, coords=coords)
        parts.append(q + ";")
    full = f"[out:json][timeout:25];({''.join(parts)});out body;"
    for url in MIRRORS:
        try:
            r = requests.post(url, data={"data": full}, headers=HEADERS, timeout=30)
            if r.status_code == 200: return r.json().get("elements", [])
            time.sleep(1)
        except: continue
    return []

# --- EXECUTION ---
if st.session_state.running:
    status = st.status("Initializing...", expanded=True)
    try:
        gpx = gpxpy.parse(uploaded_file)
        raw = []
        for t in gpx.tracks:
            for s in t.segments: raw.extend(s.points)
        track_data = calculate_track_distance(raw)
        
        scan_pts = []
        last = None
        for p in raw:
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
                c = i + 1
                prog.progress(c/total)
                elap = time.time() - start_t
                if c > 0:
                    rem = int((elap/c)*(total-c))
                    m, s = divmod(rem, 60)
                    status.update(label=f"Scanning... (Est. {m}m {s}s)")
                for item in f.result():
                    if item["id"] not in seen:
                        seen.add(item["id"])
                        found_raw.append(item)
        
        status.write("Processing data...")
        final = []
        locs = {k: [] for k in selected_keys}
        min_deg = MIN_GAP_KM / 111.0
        
        def identify(tags):
            if "Cemetery" in selected_keys and tags.get("amenity") == "grave_yard": return "Cemetery"
            if "Water" in selected_keys:
                if tags.get("amenity") in ["drinking_water", "fountain", "watering_place"] or tags.get("natural") == "spring" or tags.get("man_made") in ["water_tap", "water_well"]: return "Water"
            for k in selected_keys:
                if k in ["Water", "Cemetery"]: continue
                q = amenity_config[k]["query"]
                if "toilets" in q and tags.get("amenity") == "toilets": return k
                if "shop" in q and tags.get("shop"): return k
                if "fuel" in q and tags.get("amenity") == "fuel": return k
                if "restaurant" in q and tags.get("amenity") in ["restaurant","fast_food","cafe"]: return k
                if "tourism" in q and tags.get("tourism") in ["hotel","hostel","guest_house"]: return k
                if "camp_site" in q and tags.get("tourism") == "camp_site": return k
                if "pharmacy" in q and tags.get("amenity") == "pharmacy": return k
                if "bicycle" in q and tags.get("shop") == "bicycle": return k
                if "atm" in q and tags.get("amenity") == "atm": return k
                if "railway" in q and tags.get("railway"): return k
            return None

        for item in found_raw:
            tags = item.get("tags", {})
            cat = identify(tags)
            if not cat: continue
            
            lat, lon = item["lat"], item["lon"]
            too_close = False
            for (alat, alon) in locs[cat]:
                if sqrt((alat-lat)**2 + (alon-lon)**2) < min_deg:
                    too_close = True; break
            
            if not too_close:
                name = tags.get("name") or tags.get("brand") or tags.get("operator")
                if not name:
                    city = tags.get("addr:city")
                    name = f"{cat} ({city})" if city else cat
                    if cat == "Cemetery" and name == cat: name = "Cemetery (Check Tap)"
                
                details = []
                if tags.get("opening_hours"): details.append(f"🕒 {tags['opening_hours']}")
                if tags.get("phone"): details.append(f"📞 {tags['phone']}")
                if tags.get("fee") == "yes": details.append("💵 Paid")
                if tags.get("drinking_water") == "yes": details.append("🚰 Potable")
                
                desc = f"{cat}" + (" | " + " | ".join(details) if details else "")
                km = get_nearest_km(lat, lon, track_data)
                
                final.append({
                    "km": km, "cat": cat, "name": name, "lat": lat, "lon": lon,
                    "desc": desc, "hours": tags.get("opening_hours",""), 
                    "phone": tags.get("phone",""), "city": tags.get("addr:city",""),
                    "symbol": amenity_config[cat]["icon"],
                    "gmap": f"http://googleusercontent.com/maps.google.com/?q={lat},{lon}"
                })
                locs[cat].append((lat, lon))
        
        final.sort(key=lambda x: x["km"])
        status.update(label="Done!", state="complete", expanded=False)
        
        st.subheader("📊 Results")
        if final:
            c1, c2 = st.columns([1, 2])
            with c1:
                df = pd.DataFrame(final)
                counts = df["cat"].value_counts().reset_index()
                counts.columns = ["Category", "Count"]
                st.dataframe(counts, hide_index=True)
            with c2:
                map_d = [{"coordinates": [p["lon"], p["lat"]], "color": amenity_config[p["cat"]]["color"], "info": f"**{p['name']}**\nKm {p['km']:.1f}"} for p in final]
                path = [[p.longitude, p.latitude] for p in raw[::30]]
                st.pydeck_chart(pdk.Deck(
                    initial_view_state=pdk.ViewState(latitude=raw[0].latitude, longitude=raw[0].longitude, zoom=8),
                    layers=[
                        pdk.Layer("PathLayer", [{"path": path}], get_path="path", get_color=[255, 0, 0], width_min_pixels=2),
                        pdk.Layer("ScatterplotLayer", map_d, get_position="coordinates", get_fill_color="color", get_radius=200, pickable=True)
                    ], tooltip={"text": "{info}"}
                ))
            
            st.success(f"Found {len(final)} items.")
            
            base = uploaded_file.name
            if base.lower().endswith(".gpx"): base = base[:-4]
            
            for p in final:
                wpt = gpxpy.gpx.GPXWaypoint(latitude=p["lat"], longitude=p["lon"], name=p["name"])
                wpt.description = p["desc"]
                wpt.symbol = p["symbol"]
                wpt.type = p["cat"]
                gpx.waypoints.append(wpt)
            
            out_gpx = BytesIO()
            out_gpx.write(gpx.to_xml().encode("utf-8"))
            out_gpx.seek(0)
            
            df_csv = df[["km","cat","name","hours","phone","city","lat","lon","gmap"]].copy()
            df_csv["km"] = df_csv["km"].round(1)
            out_csv = df_csv.to_csv(index=False).encode("utf-8")
            
            b1, b2 = st.columns(2)
            with b1: st.download_button("⬇️ GPX (Device)", out_gpx, f"{base}_enriched.gpx", "application/gpx+xml", type="primary")
            with b2: st.download_button("⬇️ CSV (Cue Sheet)", out_csv, f"{base}_cuesheet.csv", "text/csv")
        
        else: st.warning("No amenities found.")
        
        if st.button("🔄 Start Over"):
            st.session_state.running = False
            st.rerun()

    except Exception as e:
        st.error(f"Error: {e}")
        st.session_state.running = False
