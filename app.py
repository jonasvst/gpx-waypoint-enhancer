import streamlit as st
import gpxpy
import requests
import time
import pandas as pd
import pydeck as pdk
import concurrent.futures
from math import radians, cos, sin, asin, sqrt
from io import BytesIO

# --- CONFIG ---
st.set_page_config(page_title="UltraToolkit", page_icon="🚴", layout="centered")

# --- CSS ---
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    div[data-testid="stExpander"] { border: 1px solid #e6e6e6; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

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
    pts, keys, config = args
    if not pts: return []
    coords = ",".join([f"{round(p.latitude,5)},{round(p.longitude,5)}" for p in pts])
    parts = []
    for k in keys:
        q = config[k]["query"].format(radius=50, coords=coords)
        parts.append(q + ";")
    full = f"[out:json][timeout:25];({''.join(parts)});out body;"
    mirrors = ["https://overpass.kumi.systems/api/interpreter", "https://api.openstreetmap.fr/oapi/interpreter", "https://overpass-api.de/api/interpreter"]
    for url in mirrors:
        try:
            r = requests.post(url, data={"data": full}, headers={"User-Agent": "GPX-Tool/30"}, timeout=30)
            if r.status_code == 200: return r.json().get("elements", [])
            time.sleep(1)
        except: continue
    return []

# --- VIEWS ---
def show_home():
    st.title("🚴 UltraToolkit")
    st.caption("v1.0 | verest.ch")
    st.info("👈 **Use the sidebar menu to open the GPS Enricher.**")
    st.markdown("### Survival tools for ultra-distance cycling.")
    st.markdown("This toolkit helps you plan logistics for races like TCR.")
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📍 GPS Enricher")
        st.write("Scan GPX tracks for water, food, and sleep.")
    with c2:
        st.markdown("#### 🔜 More Tools")
        st.write("Weather planners coming soon.")

def show_enricher():
    st.title("📍 GPS Enricher")
    
    with st.expander("📘 **User Guide: Logic & Legend**"):
        st.markdown("""
        **1. Smart Scanning:** 50m roadside scan (zero detour), every 100m.
        **2. Deep Data:** Extracts hours, phone numbers, and water quality.
        **3. Legend:**
        * 💧 **Water:** Fountains, Taps, Cemeteries.
        * 🛒 **Shops:** Supermarkets, Bakeries.
        * ⛽ **Fuel:** 24/7 Stations.
        * 🍔 **Food:** Restaurants.
        * 🚽 **Toilets:** Public restrooms.
        * 🛏️ **Sleep:** Hotels, Hostels.
        * 🚆 **Train:** Bail-out stations.
        """)

    uploaded_file = st.file_uploader("📂 Upload GPX", type=["gpx"])

    if uploaded_file:
        st.subheader("⚙️ Settings")
        min_gap = st.slider("🧹 Map Density (Min Gap km)", 0.0, 10.0, 2.0, 0.5)
        
        amenity_config = {
            "Water": {"label": "💧 Water", "query": """(node["amenity"~"drinking_water|fountain|watering_place"](around:{radius},{coords});node["natural"~"spring"](around:{radius},{coords});node["man_made"~"water_tap|water_well"](around:{radius},{coords});)""", "color": [0, 128, 255]},
            "Cemetery": {"label": "⚰️ Cemeteries", "query": """node["amenity"~"grave_yard"](around:{radius},{coords})""", "color": [0, 100, 255]},
            "Toilets": {"label": "🚽 Toilets", "query": """node["amenity"~"toilets"](around:{radius},{coords})""", "color": [150, 150, 150]},
            "Shops": {"label": "🛒 Shops", "query": """node["shop"~"supermarket|convenience|kiosk|bakery|general"](around:{radius},{coords})""", "color": [0, 200, 0]},
            "Fuel": {"label": "⛽ Fuel", "query": """node["amenity"~"fuel"](around:{radius},{coords})""", "color": [255, 140, 0]},
            "Food": {"label": "🍔 Food", "query": """node["amenity"~"restaurant|fast_food|cafe"](around:{radius},{coords})""", "color": [0, 200, 0]},
            "Sleep": {"label": "🛏️ Sleep", "query": """node["tourism"~"hotel|hostel|guest_house"](around:{radius},{coords})""", "color": [128, 0, 128]},
            "Train": {"label": "🚆 Train", "query": """node["railway"~"station|halt"](around:{radius},{coords})""", "color": [50, 50, 50]}
        }
        
        cols = st.columns(3)
        selected_keys = []
        defaults = ["Water", "Cemetery", "Toilets", "Shops", "Fuel"]
        for i, (k, v) in enumerate(amenity_config.items()):
            with cols[i % 3]:
                if st.checkbox(v["label"], value=(k in defaults)): selected_keys.append(k)

        if st.button("🚀 Start Scan", type="primary"):
            if not selected_keys:
                st.error("Select amenities.")
                return
                
            status = st.status("Processing...", expanded=True)
            try:
                gpx = gpxpy.parse(uploaded_file)
                raw = []
                for t in gpx.tracks:
                    for s in t.segments: raw.extend(s.points)
                track_data = calculate_track_distance(raw)
                
                scan_pts = []
                last = None
                for p in raw:
                    if not last or haversine(last.longitude, last.latitude, p.longitude, p.latitude) >= 100:
                        scan_pts.append(p)
                        last = p
                
                batches = [scan_pts[i:i+25] for i in range(0, len(scan_pts), 25)]
                found_raw = []
                seen = set()
                prog = status.progress(0)
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as exc:
                    futures = {exc.submit(fetch_batch, (b, selected_keys, amenity_config)): i for i, b in enumerate(batches)}
                    for i, f in enumerate(concurrent.futures.as_completed(futures)):
                        prog.progress((i+1)/len(batches))
                        for item in f.result():
                            if item["id"] not in seen:
                                seen.add(item["id"])
                                found_raw.append(item)
                
                status.write("Enriching data...")
                final = []
                locs = {k: [] for k in selected_keys}
                min_deg = min_gap / 111.0
                
                for item in found_raw:
                    tags = item.get("tags", {})
                    cat = None
                    if "Cemetery" in selected_keys and tags.get("amenity") == "grave_yard": cat = "Cemetery"
                    elif "Water" in selected_keys and (tags.get("amenity") in ["drinking_water", "fountain"] or tags.get("natural") == "spring"): cat = "Water"
                    else:
                        for k in selected_keys:
                            if k in ["Water", "Cemetery"]: continue
                            q = amenity_config[k]["query"]
                            if k == "Toilets" and tags.get("amenity") == "toilets": cat = k
                            elif k == "Shops" and tags.get("shop"): cat = k
                            elif k == "Fuel" and tags.get("amenity") == "fuel": cat = k
                            elif k == "Food" and tags.get("amenity") in ["restaurant","fast_food"]: cat = k
                            elif k == "Sleep" and tags.get("tourism") in ["hotel","hostel"]: cat = k
                            elif k == "Train" and tags.get("railway"): cat = k
                    
                    if not cat: continue
                    
                    lat, lon = item["lat"], item["lon"]
                    too_close = False
                    for (alat, alon) in locs[cat]:
                        if sqrt((alat-lat)**2 + (alon-lon)**2) < min_deg:
                            too_close = True; break
                    if too_close: continue
                    
                    name = tags.get("name") or tags.get("brand") or tags.get("operator")
                    if not name and tags.get("addr:city"): name = f"{cat} ({tags['addr:city']})"
                    if not name: name = f"{cat} (Check details)"
                    if cat == "Cemetery": name = "Cemetery (Check Tap)"
                    
                    details = []
                    if tags.get("opening_hours"): details.append(f"🕒 {tags['opening_hours']}")
                    if tags.get("phone"): details.append(f"📞 {tags['phone']}")
                    desc = f"{cat}" + (" | ".join(details) if details else "")
                    
                    km = get_nearest_km(lat, lon, track_data)
                    final.append({
                        "km": km, "cat": cat, "name": name, "lat": lat, "lon": lon,
                        "desc": desc, "hours": tags.get("opening_hours",""),
                        "phone": tags.get("phone",""), "city": tags.get("addr:city",""),
                        "symbol": "Water" if cat in ["Water", "Cemetery"] else "Waypoint"
                    })
                    locs[cat].append((lat, lon))
                
                final.sort(key=lambda x: x["km"])
                status.update(label=f"Done! Found {len(final)} items.", state="complete", expanded=False)
                
                st.subheader("📊 Results")
                c1, c2 = st.columns([1, 2])
                with c1:
                    df = pd.DataFrame(final)
                    if not df.empty:
                        counts = df["cat"].value_counts().reset_index()
                        counts.columns = ["Category", "Count"]
                        st.dataframe(counts, hide_index=True)
                with c2:
                    if final:
                        map_d = [{"coordinates": [p["lon"], p["lat"]], "color": amenity_config[p["cat"]]["color"], "info": f"**{p['name']}**"} for p in final]
                        path = [[p.longitude, p.latitude] for p in raw[::30]]
                        st.pydeck_chart(pdk.Deck(
                            initial_view_state=pdk.ViewState(latitude=raw[0].latitude, longitude=raw[0].longitude, zoom=8),
                            layers=[
                                pdk.Layer("PathLayer", [{"path": path}], get_path="path", get_color=[255, 0, 0], width_min_pixels=2),
                                pdk.Layer("ScatterplotLayer", map_d, get_position="coordinates", get_fill_color="color", get_radius=200, pickable=True)
                            ], tooltip={"text": "{info}"}
                        ))
                
                if final:
                    base = uploaded_file.name.replace(".gpx", "")
                    for p in final:
                        wpt = gpxpy.gpx.GPXWaypoint(latitude=p["lat"], longitude=p["lon"], name=p["name"])
                        wpt.description = p["desc"]
                        wpt.symbol = p["symbol"]
                        wpt.type = p["cat"]
                        gpx.waypoints.append(wpt)
                    out_gpx = BytesIO()
                    out_gpx.write(gpx.to_xml().encode("utf-8"))
                    out_gpx.seek(0)
                    
                    df_csv = df[["km", "cat", "name", "hours", "phone", "city", "lat", "lon"]].copy()
                    df_csv["km"] = df_csv["km"].round(1)
                    out_csv = df_csv.to_csv(index=False).encode("utf-8")
                    
                    b1, b2 = st.columns(2)
                    with b1: st.download_button("⬇️ GPX", out_gpx, f"{base}_enriched.gpx", "application/gpx+xml")
                    with b2: st.download_button("⬇️ CSV", out_csv, f"{base}_cuesheet.csv", "text/csv")

            except Exception as e:
                st.error(f"Error: {e}")

# --- MAIN NAVIGATION ---
if "page" not in st.session_state:
    st.session_state.page = "Home"

with st.sidebar:
    st.title("Menu")
    selection = st.radio("Go to", ["Home", "GPS Enricher"])
    if selection != st.session_state.page:
        st.session_state.page = selection
        st.rerun()

if st.session_state.page == "Home":
    show_home()
elif st.session_state.page == "GPS Enricher":
    show_enricher()
