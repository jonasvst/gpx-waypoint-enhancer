import streamlit as st
import gpxpy
import requests
import time
import random
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from io import BytesIO

# --- PAGE CONFIG ---
st.set_page_config(page_title="TCR Survival Tool V3", page_icon="🚴", layout="wide")

st.title("🚴 TCR Survival Map Generator")
st.markdown("""
**Turn a raw GPX track into a survival-ready route.** Select the amenities you need, set your search distance, and this tool will scan OpenStreetMap and embed the points directly into your file.
""")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Search Settings")

# 1. SEARCH RADIUS
SEARCH_RADIUS = st.sidebar.slider(
    "Max Distance from Track (meters)", 
    min_value=100, 
    max_value=5000, 
    value=500, 
    step=100,
    help="How far sideways from the road should we look?"
)

# 2. AMENITY SELECTION (Expanded List)
st.sidebar.subheader("What do you need?")

amenity_options = {
    "💧 Water Sources": """node["amenity"~"drinking_water|fountain"]""",
    "🌊 Natural Springs": """node["natural"~"spring"]""",
    "🚽 Toilets": """node["amenity"~"toilets"]""",
    "🛒 Food - Shops (Supermarket/Bakery)": """node["shop"~"supermarket|convenience|kiosk|general|bakery"]""",
    "🍔 Food - Meals (Restaurant/Fast Food)": """node["amenity"~"restaurant|fast_food|cafe"]""",
    "⛽ Fuel Stations (24/7 Survival)": """node["amenity"~"fuel"]""",
    "🛏️ Sleep - Hotels/Hostels": """node["tourism"~"hotel|hostel|guest_house"]""",
    "⛺ Sleep - Camping": """node["tourism"~"camp_site"]""",
    "🔧 Bike Shops": """node["shop"~"bicycle"]""",
    "💊 Pharmacies": """node["amenity"~"pharmacy"]"""
}

# Default selections (Essential Survival)
default_options = [
    "💧 Water Sources", 
    "🌊 Natural Springs", 
    "🛒 Food - Shops (Supermarket/Bakery)", 
    "⛽ Fuel Stations (24/7 Survival)"
]

selected_types = st.sidebar.multiselect(
    "Select Categories", 
    options=list(amenity_options.keys()),
    default=default_options
)

# 3. PRECISION SETTINGS
st.sidebar.subheader("Advanced")
SAMPLE_STEP = st.sidebar.select_slider(
    "Scan Precision (Gap between checks)",
    options=[100, 250, 500, 1000, 2000, 5000],
    value=250,
    format_func=lambda x: f"{x} meters",
    help="Smaller = More accurate but slower. Larger = Faster but might miss items in curves."
)

BATCH_SIZE = 15
OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (TCR-Tool; v3.0) Gecko/20100101",
    "Referer": "https://streamlit.io/"
}

# SYMBOL MAPPING (RideWithGPS Standard Icons)
SYMBOL_MAP = {
    "Water": "Water",
    "Spring": "Water",
    "Shop": "Convenience Store",
    "Restaurant": "Food",
    "Fuel": "Gas Station",
    "Toilet": "Restroom",
    "Hotel": "Lodging",
    "Camping": "Campground",
    "Bike Shop": "Bike Shop",
    "Pharmacy": "First Aid",
    "Other": "Waypoint"
}

# --- HELPERS ---
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371000 * c

def build_query_body(radius, coord_string, active_types):
    parts = []
    for user_label in active_types:
        base_query = amenity_options[user_label]
        query_part = f'{base_query}(around:{radius},{coord_string});'
        parts.append(query_part)
    return "\n".join(parts)

def get_amenities_batch(points_batch, radius, active_types):
    if not active_types: return []
    
    coord_list = [f"{round(p.latitude,5)},{round(p.longitude,5)}" for p in points_batch]
    coord_string = ",".join(coord_list)
    
    query_body = build_query_body(radius, coord_string, active_types)

    query = f"""
    [out:json][timeout:90];
    (
      {query_body}
    );
    out body;
    """
    try:
        time.sleep(random.uniform(1.0, 2.0))
        response = requests.post(OVERPASS_URL, data={'data': query}, headers=HEADERS, timeout=120)
        
        if response.status_code == 200:
            return response.json().get('elements', [])
        elif response.status_code == 429:
            time.sleep(5)
            return get_amenities_batch(points_batch, radius, active_types)
        else:
            return []
    except Exception:
        return []

# --- MAIN APP ---
uploaded_file = st.file_uploader("📂 Drop your GPX file here", type=['gpx'])

if uploaded_file is not None:
    gpx = gpxpy.parse(uploaded_file)
    all_points = []
    for track in gpx.tracks:
        for segment in track.segments:
            all_points.extend(segment.points)
            
    st.write(f"✅ **File Loaded:** {len(all_points)} track points detected.")
    
    if st.button("🚀 Start Scan"):
        if not selected_types:
            st.error("Please select at least one Amenity category!")
            st.stop()

        query_points = []
        last_point = None
        for point in all_points:
            if last_point is None or haversine(last_point.longitude, last_point.latitude, point.longitude, point.latitude) >= SAMPLE_STEP:
                query_points.append(point)
                last_point = point
        
        batches = [query_points[i:i + BATCH_SIZE] for i in range(0, len(query_points), BATCH_SIZE)]
        
        st.info(f"Scanning {len(query_points)} locations (in {len(batches)} batches). Please wait...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        found_amenities = []
        unique_ids = set()
        
        for i, batch in enumerate(batches):
            progress = (i + 1) / len(batches)
            progress_bar.progress(progress)
            status_text.markdown(f"**Scanning...** Batch {i+1}/{len(batches)} | Found: **{len(found_amenities)}** POIs")
            
            items = get_amenities_batch(batch, SEARCH_RADIUS, selected_types)
            
            for item in items:
                item_id = item.get('id')
                if item_id in unique_ids: continue
                unique_ids.add(item_id)
                
                tags = item.get('tags', {})
                
                # Determine Category based on tags
                category = "Other"
                amenity = tags.get('amenity')
                shop = tags.get('shop')
                tourism = tags.get('tourism')
                natural = tags.get('natural')
                
                if amenity in ['drinking_water', 'fountain']: category = "Water"
                elif natural == 'spring': category = "Spring"
                elif amenity == 'toilets': category = "Toilet"
                elif amenity == 'fuel': category = "Fuel"
                elif shop == 'bicycle': category = "Bike Shop"
                elif amenity == 'pharmacy': category = "Pharmacy"
                elif amenity in ['restaurant', 'fast_food', 'cafe']: category = "Restaurant"
                elif shop: category = "Shop"
                elif tourism in ['hotel', 'hostel', 'guest_house']: category = "Hotel"
                elif tourism == 'camp_site': category = "Camping"

                found_amenities.append({
                    "name": tags.get('name', category),
                    "lat": item.get('lat'),
                    "lon": item.get('lon'),
                    "category": category,
                    "desc": f"{category}: {tags.get('amenity', tags.get('shop', 'poi'))}"
                })

        st.success(f"🎉 Done! Found {len(found_amenities)} points of interest.")

        if found_amenities:
            st.subheader("📍 Preview Map")
            df = pd.DataFrame(found_amenities)
            st.map(df, latitude='lat', longitude='lon')

        for item in found_amenities:
            wpt = gpxpy.gpx.GPXWaypoint(latitude=item['lat'], longitude=item['lon'], name=item['name'])
            wpt.description = item['desc']
            wpt.type = item['category']
            wpt.symbol = SYMBOL_MAP.get(item['category'], "Waypoint")
            gpx.waypoints.append(wpt)
        
        output_io = BytesIO()
        output_io.write(gpx.to_xml().encode('utf-8'))
        output_io.seek(0)
        
        st.download_button(
            label="⬇️ Download Enhanced GPX",
            data=output_io,
            file_name=f"TCR_Enhanced_{int(time.time())}.gpx",
            mime="application/gpx+xml",
            type="primary"
        )
