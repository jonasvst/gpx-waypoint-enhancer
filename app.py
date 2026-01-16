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
st.set_page_config(page_title="GPS Enricher", page_icon="📍", layout="centered")

# --- CSS ---
st.markdown(\"\"\"
    <style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    div[data-testid="stExpander"] { border: 1px solid #e6e6e6; border-radius: 8px; }
    </style>
\"\"\", unsafe_allow_html=True)

st.title("📍 GPS Enricher")
st.markdown("Enrich your GPX track with detailed roadside amenities.")

# --- USER GUIDE ---
with st.expander("📘 **User Guide: Logic & Legend**"):
    st.markdown(\"\"\"
    ### **1. Smart Scanning Strategy**
    * **Zero Detour:** We scan a **50m radius** along your track.
    * **No Blind Spots:** We sample every **100m** to guarantee full coverage.
    * **Smart De-Clutter:** We hide duplicate stops (e.g. 2 gas stations in 1 town) to keep the map readable.

    ### **2. Deep Data Extraction**
    We extract logistics from OpenStreetMap:
    * **Identity:** Brand names ("Shell", "Coop") instead of generic labels.
    * **Logistics:** **Opening Hours** and **Phone Numbers**.
    * **Attributes:** Water drinkability, Toilet fees, etc.

    ### **3. Searchable Amenities**
    * **💧 Water:** Fountains, Springs, Taps, and **Cemeteries** (Water icon).
    * **🛒 Shops:** Supermarkets, Bakeries.
    * **⛽ Fuel:** 24/7 stations.
    * **🍔 Food:** Restaurants, Fast Food.
    * **🚽 Toilets:** Public restrooms.
    * **🛏️ Sleep:** Hotels, Hostels.
    * **⛺ Camping:** Official campsites.
    * **💊 Pharmacy:** Medical supplies.
    * **🔧 Bike Shop:** Repairs.
    * **🏧 ATM:** Cash machines.
    * **🚆 Train:** Bail-out stations.
    
    *Issues? Contact: jonas@verest.ch*
    \"\"\")

# --- UPLOAD ---
uploaded_file = st.file_uploader("📂 **Step 1: Upload GPX file**", type=["gpx"])

# --- SETTINGS & LOGIC ---
if uploaded_file:
    st.subheader("⚙️ Settings")
    min_gap = st.slider("🧹 Map Density (Min Gap km)", 0.0, 10.0, 2.0, 0.5, help="Prevents clutter. Higher = Cleaner map.")
    
    amenity_config = {
        "Water": {"label": "💧 Water", "desc": "Fountains, Taps, Springs", "query": \"\"\"(node["amenity"~"drinking_water|fountain|watering_place"](around:{radius},{coords});node["natural"~"spring"](around:{radius},{coords});node["man_made"~"water_tap|water_well"](around:{radius},{coords});)\"\"\", "color": [0, 128, 255]},
        "Cemetery": {"label": "⚰️ Cemeteries", "desc": "Water source", "query": \"\"\"node["amenity"~"grave_yard"](around:{radius},{coords})\"\"\", "color": [0, 100, 255]},
        "Toilets": {"label": "🚽 Toilets", "desc": "Restrooms", "query": \"\"\"node["amenity"~"toilets"](around:{radius},{coords})\"\"\", "color": [150, 150, 150]},
        "Shops": {"label": "🛒 Shops", "desc": "Supermarkets, Bakeries", "query": \"\"\"node["shop"~"supermarket|convenience|kiosk|bakery|general"](around:{radius},{coords})\"\"\", "color": [0, 200, 0]},
        "Fuel": {"label": "⛽ Fuel", "desc": "Gas Stations", "query": \"\"\"node["amenity"~"fuel"](around:{radius},{coords})\"\"\", "color": [255, 140, 0]},
        "Food": {"label": "🍔 Food", "desc": "Restaurants, Cafes", "query": \"\"\"node["amenity"~"restaurant|fast_food|cafe"](around:{radius},{coords})\"\"\", "color": [0, 200, 0]},
        "Sleep": {"label": "🛏️ Sleep", "desc": "Hotels, Hostels", "query": \"\"\"node["tourism"~"hotel|hostel|guest_house"](around:{radius},{coords})\"\"\", "color": [128, 0, 128]},
        "Camping": {"label": "⛺ Camping", "desc": "Campsites", "query": \"\"\"node["tourism"~"camp_site"](around:{radius},{coords})\"\"\", "color": [34, 139, 34]},
        "Bike": {"label": "🔧 Bike", "desc": "Repair Shops", "query": \"\"\"node["shop"~"bicycle"](around:{radius},{coords})\"\"\", "color": [255, 0, 0]},
        "Pharm": {"label": "💊 Pharmacy", "desc": "Medical", "query": \"\"\"node["amenity"~"pharmacy"](around:{radius},{coords})\"\"\", "color": [255, 0, 0]},
        "ATM": {"label": "🏧 ATM", "desc": "Cash", "query": \"\"\"node["amenity"~"atm"](around:{radius},{coords})\"\"\", "color": [0, 100, 0]},
        "Train": {"label": "🚆 Train", "desc": "Stations", "query": \"\"\"node["railway"~"station|halt"](around:{radius},{coords})\"\"\", "color": [50, 50, 50]}
    }
    
    st.caption("Select amenities:")
    cols = st.columns(3)
    selected_keys = []
    defaults = ["Water", "Cemetery", "Toilets", "Shops", "Fuel"]
    
    for i, (k, v) in enumerate(amenity_config.items()):
        with cols[i % 3]:
            if st.checkbox(v["label"], value=(k in defaults), help=v["desc"]): selected_keys.append(k)

    st.markdown("---")
    
    # --- ACTION ---
    c_go, c_stop = st.columns([3, 1])
    with c_go:
        start = st.button("🚀 Start Scan", type="primary")
    with c_stop:
        if st.button("🛑 Cancel"): st.rerun()

    if start:
        if not selected_keys:
            st.error("Select amenities first.")
        else:
            status = st.status("Initializing...", expanded=True)
            try:
                # 1. PARSE GPX
                gpx = gpxpy.parse(uploaded_file)
                raw = []
                for t in gpx.tracks:
                    for s in t.segments: raw.extend(s.points)
                
                # 2. HELPER CALCS
                def haversine(lon1, lat1, lon2, lat2):
                    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
                    a = sin((lat2-lat1)/2)**2 + cos(lat1) * cos(lat2) * sin((lon2-lon1)/2)**2
                    return 6371000 * 2 * asin(sqrt(a))

                # Track Distances
                track_data = []
                total_dist = 0
                last = None
                for p in raw:
                    if last: total_dist += haversine(last.longitude, last.latitude, p.longitude, p.latitude)
                    track_data.append({"lat": p.latitude, "lon": p.longitude, "cum_dist": total_dist})
                    last = p
                
                def get_nearest_km(lat, lon):
                    min_d, km = float("inf"), 0
                    for t in track_data[::10]: # optimize
                        d = (t["lat"]-lat)**2 + (t["lon"]-lon)**2
                        if d < min_d: min_d, km = d, t["cum_dist"]
                    return km / 1000.0

                # 3. RESAMPLE
                scan_pts = []
                last = None
                for p in raw:
                    if not last or haversine(last.longitude, last.latitude, p.longitude, p.latitude) >= 100:
                        scan_pts.append(p)
                        last = p
                
                # 4. FETCH
                def fetch(args):
                    pts, keys = args
                    if not pts: return []
                    coords = ",".join([f"{round(p.latitude,5)},{round(p.longitude,5)}" for p in pts])
                    parts = []
                    for k in keys:
                        q = amenity_config[k]["query"].format(radius=50, coords=coords)
                        parts.append(q + ";")
                    full = f"[out:json][timeout:25];({.join(parts)});out body;"
                    mirrors = ["https://overpass.kumi.systems/api/interpreter", "https://api.openstreetmap.fr/oapi/interpreter", "https://overpass-api.de/api/interpreter"]
                    for url in mirrors:
                        try:
                            r = requests.post(url, data={"data": full}, headers={"User-Agent": "GPX/29"}, timeout=30)
                            if r.status_code == 200: return r.json().get("elements", [])
                            time.sleep(1)
                        except: continue
                    return []

                batches = [scan_pts[i:i+25] for i in range(0, len(scan_pts), 25)]
                found_raw = []
                seen = set()
                prog = status.progress(0)
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as exc:
                    futures = {exc.submit(fetch, (b, selected_keys)): i for i, b in enumerate(batches)}
                    for i, f in enumerate(concurrent.futures.as_completed(futures)):
                        prog.progress((i+1)/len(batches))
                        for item in f.result():
                            if item["id"] not in seen:
                                seen.add(item["id"])
                                found_raw.append(item)
                
                # 5. PROCESS
                status.write("Enriching & Filtering...")
                final = []
                locs = {k: [] for k in selected_keys}
                min_deg = min_gap / 111.0
                
                def identify(tags):
                    if "Cemetery" in selected_keys and tags.get("amenity") == "grave_yard": return "Cemetery"
                    if "Water" in selected_keys and (tags.get("amenity") in ["drinking_water","fountain"] or tags.get("natural")=="spring"): return "Water"
                    for k in selected_keys:
                        if k in ["Water", "Cemetery"]: continue
                        q = amenity_config[k]["query"]
                        if k=="Toilets" and tags.get("amenity")=="toilets": return k
                        if k=="Shops" and tags.get("shop"): return k
                        if k=="Fuel" and tags.get("amenity")=="fuel": return k
                        if k=="Food" and tags.get("amenity") in ["restaurant","fast_food"]: return k
                        if k=="Sleep" and tags.get("tourism") in ["hotel","hostel"]: return k
                        if k=="Camping" and tags.get("tourism")=="camp_site": return k
                        if k=="Pharm" and tags.get("amenity")=="pharmacy": return k
                        if k=="Bike" and tags.get("shop")=="bicycle": return k
                        if k=="ATM" and tags.get("amenity")=="atm": return k
                        if k=="Train" and tags.get("railway"): return k
                    return None

                for item in found_raw:
                    tags = item.get("tags", {})
                    cat = identify(tags)
                    if not cat: continue
                    
                    lat, lon = item["lat"], item["lon"]
                    # Gap Check
                    too_close = False
                    for (alat, alon) in locs[cat]:
                        if sqrt((alat-lat)**2 + (alon-lon)**2) < min_deg:
                            too_close = True; break
                    if too_close: continue
                    
                    # Name
                    name = tags.get("name") or tags.get("brand") or tags.get("operator")
                    if not name and tags.get("addr:city"): name = f"{cat} ({tags[addr:city]})"
                    if not name: name = f"{cat} (Check Details)"
                    if cat == "Cemetery": name = "Cemetery (Check Tap)"
                    
                    # Deep Data
                    details = []
                    if tags.get("opening_hours"): details.append(f"🕒 {tags[opening_hours]}")
                    if tags.get("phone"): details.append(f"📞 {tags[phone]}")
                    if tags.get("fee")=="yes": details.append("💵 Paid")
                    if tags.get("drinking_water")=="yes": details.append("🚰 Potable")
                    
                    desc = f"{cat}" + (" | " + " | ".join(details) if details else "")
                    km = get_nearest_km(lat, lon)
                    
                    final.append({
                        "km": km, "cat": cat, "name": name, "lat": lat, "lon": lon,
                        "desc": desc, "hours": tags.get("opening_hours",""),
                        "phone": tags.get("phone",""), "city": tags.get("addr:city",""),
                        "symbol": "Water" if cat in ["Water", "Cemetery"] else "Waypoint"
                    })
                    locs[cat].append((lat, lon))
                
                final.sort(key=lambda x: x["km"])
                status.update(label=f"Done! Found {len(final)} items.", state="complete", expanded=False)
                
                # 6. RESULTS
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
                        map_d = [{"coordinates": [p["lon"], p["lat"]], "color": amenity_config[p["cat"]]["color"], "info": f"**{p[name]}**"} for p in final]
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
                    # GPX Export
                    for p in final:
                        wpt = gpxpy.gpx.GPXWaypoint(latitude=p["lat"], longitude=p["lon"], name=p["name"])
                        wpt.description = p["desc"]
                        wpt.symbol = p["symbol"]
                        wpt.type = p["cat"]
                        gpx.waypoints.append(wpt)
                    out_gpx = BytesIO()
                    out_gpx.write(gpx.to_xml().encode("utf-8"))
                    out_gpx.seek(0)
                    
                    # CSV Export
                    df_csv = df[["km", "cat", "name", "hours", "phone", "city", "lat", "lon"]].copy()
                    df_csv["km"] = df_csv["km"].round(1)
                    out_csv = df_csv.to_csv(index=False).encode("utf-8")
                    
                    b1, b2 = st.columns(2)
                    with b1: st.download_button("⬇️ GPX (Device)", out_gpx, f"{base}_enriched.gpx", "application/gpx+xml", type="primary")
                    with b2: st.download_button("⬇️ CSV (Cue Sheet)", out_csv, f"{base}_cuesheet.csv", "text/csv")
            
            except Exception as e:
                st.error(f"Error: {e}")
