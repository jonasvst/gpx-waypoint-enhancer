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

# --- 1. STATE MANAGEMENT ---
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

# --- 2. USER GUIDE ---
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

# --- 3. UPLOAD ---
is_disabled = st.session_state.status == 'running'
uploaded_file = st.file_uploader("📂 **Step 1:** Drop your GPX file here", type=['gpx'], disabled=is_disabled)

# --- 4. CONFIGURATION ---
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

    cols = st.columns(3)
    selected_keys = []
    for i, (key, cfg) in enumerate(amenity_config.items()):
        with cols[i % 3]:
            if st.checkbox(cfg["label"], value=(key in ["Water", "Cemetery", "Toilets", "Shops", "Fuel"]), key=f"sel_{key}", disabled=is_disabled):
                selected_keys.append(key)

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

    # --- 5. HELPERS ---
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
        pts, keys = args
        coords = ",".join([f"{round(p.latitude,5)},{round(p.longitude,5)}" for p in pts])
        parts = [amenity_config[k]["query"].format(radius=50, coords=coords) + ";" for k in keys]
        q = f"[out:json][timeout:25];({''.join(parts)});out body;"
        for url in ["https://overpass.kumi.systems/api/interpreter", "https://api.openstreetmap.fr/oapi/interpreter", "https://overpass-api.de/api/interpreter"]:
            try:
                r = requests.post(url, data={'data': q}, timeout=30)
                if r.status_code == 200: return r.json().get('elements', [])
            except: continue
        return []

    # --- 6. ENGINE ---
    if st.session_state.status == 'running':
        status_bar = st.status("Scanning track...", expanded=True)
        try:
            gpx = gpxpy.parse(uploaded_file)
            raw_pts = [p for t in gpx.tracks for s in t.segments for p in s.points]
            track_dist_data = calculate_track_distance(raw_pts)
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
                cat = None
                if tags.get("amenity") == "grave_yard": cat = "Cemetery"
                elif tags.get("amenity") in ["drinking_water", "fountain", "watering_place"] or tags.get("natural") == "spring": cat = "Water"
                elif tags.get("amenity") == "toilets": cat = "Toilets"
                elif tags.get("shop") in ["supermarket", "convenience", "kiosk", "bakery", "general"]: cat = "Shops"
                elif tags.get("amenity") == "fuel": cat = "Fuel"
                elif tags.get("amenity") in ["restaurant", "cafe", "fast_food"]: cat = "Food"
                elif tags.get("tourism") == "camp_site": cat = "Camping"
                elif tags.get("tourism") in ["hotel", "hostel", "guest_house"]: cat = "Sleep"
                elif tags.get("shop") == "bicycle": cat = "Bike"
                elif tags.get("amenity") == "pharmacy": cat = "Pharm"
                elif tags.get("amenity") == "atm": cat = "ATM"
                elif tags.get("railway") in ["station", "halt"]: cat = "Train"

                if not cat or cat not in selected_keys: continue
                lat, lon = item['lat'], item['lon']
                if any(sqrt((alat-lat)**2 + (alon-lon)**2) < (MIN_GAP_KM/111.0) for (alat, alon) in locs[cat]): continue

                name = tags.get('name') or tags.get('brand') or tags.get('operator') or (cat if cat != "Cemetery" else "Cemetery (Water)")
                bits = []
                if tags.get('opening_hours'): bits.append(f"🕒 {tags.get('opening_hours')}")
                if tags.get('phone'): bits.append(f"📞 {tags.get('phone')}")
                if tags.get('fee') == 'yes': bits.append("💵 Paid")
                if tags.get('drinking_water') == 'yes': bits.append("🚰 Potable")
                
                km_mark = get_nearest_km(lat, lon, track_dist_data)
                desc_str = f"{cat} | KM {km_mark:.1f}" + (" | " + " | ".join(bits) if bits else "")
                final_pois.append({"km": km_mark, "cat": cat, "name": name, "lat": lat, "lon": lon, "desc": desc_str, "hours": tags.get('opening_hours', ""), "phone": tags.get('phone', ""), "symbol": amenity_config[cat]["icon"], "color": amenity_config[cat]["color"]})
                locs[cat].append((lat, lon))

            final_pois.sort(key=lambda x: x['km'])
            g_out = gpxpy.gpx.GPX()
            for t in gpx.tracks: g_out.tracks.append(t)
            for p in final_pois:
                w = gpxpy.gpx.GPXWaypoint(latitude=p['lat'], longitude=p['lon'], name=p['name'])
                w.description, w.symbol, w.type = p['desc'], p['symbol'], p['cat']
                g_out.waypoints.append(w)
            
            # Use an empty dataframe with correct columns if no POIs found
            if not final_pois:
                df = pd.DataFrame(columns=["km", "cat", "name", "lat", "lon", "desc", "hours", "phone", "symbol", "color"])
            else:
                df = pd.DataFrame(final_pois)

            st.session_state.results = {
                "df": df, "pois": final_pois, "gpx_bytes": g_out.to_xml().encode('utf-8'),
                "path": [[p.longitude, p.latitude] for p in raw_pts[::30]],
                "slat": raw_pts[0].latitude, "slon": raw_pts[0].longitude, "fname": f"{uploaded_file.name.split('.')[0]}_enriched.gpx"
            }
            st.session_state.status = 'complete'
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}"); st.session_state.status = 'idle'

    # --- 7. DISPLAY ---
    if st.session_state.status == 'complete' and st.session_state.results:
        res = st.session_state.results
        st.subheader("📊 Results")
        
        if res['df'].empty:
            st.warning("No amenities found within 50m of your track. Try selecting more categories or check if the area is mapped on OSM.")
        else:
            c1, c2 = st.columns([1, 2])
            with c1:
                # Robust index reset and renaming
                counts = res['df']['cat'].value_counts().reset_index()
                counts.columns = ['Category', 'Count']
                st.dataframe(counts, hide_index=True)
            with c2:
                map_data = [{"coordinates": [p['lon'], p['lat']], "color": p['color'], "info": f"**{p['name']}**\n{p['desc']}"} for p in res['pois']]
                st.pydeck_chart(pdk.Deck(initial_view_state=pdk.ViewState(latitude=res['slat'], longitude=res['slon'], zoom=10), layers=[pdk.Layer("PathLayer", [{"path": res['path']}], get_path="path", get_color=[255, 0, 0], width_min_pixels=2), pdk.Layer("ScatterplotLayer", map_data, get_position="coordinates", get_fill_color="color", get_radius=250, pickable=True)], tooltip={"text": "{info}"}))
        
        dl1, dl2 = st.columns(2)
        dl1.download_button("⬇️ Download GPX", res['gpx_bytes'], res['fname'], "application/gpx+xml", type="primary")
        dl2.download_button("⬇️ Download CSV", res['df'].to_csv(index=False).encode('utf-8'), "Cuesheet.csv", "text/csv")
        if st.button("🔄 New Scan"):
            st.session_state.status = 'idle'; st.rerun()
