import streamlit as st
import gpxpy
import requests
import time
import random
from math import radians, cos, sin, asin, sqrt
from io import BytesIO

# --- PAGE CONFIG ---
st.set_page_config(page_title="TCR Survival Map", page_icon="🚴")
st.title("🚴 TCR Survival Map Generator")
st.markdown("Upload your race track. This tool scans OpenStreetMap for **Water, Food, Fuel, and Campgrounds** and embeds them into your GPX.")

# --- SETTINGS ---
SEARCH_RADIUS = 500
SAMPLE_STEP = 250
BATCH_SIZE = 15
OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"

# HEADERS
HEADERS = {
    "User-Agent": "Mozilla/5.0 (TCR-Tool; v1.0) Gecko/20100101",
    "Referer": "https://streamlit.io/"
}

# SYMBOL MAP
SYMBOL_MAP = {
    "Water": "Water",
    "Shop": "Food",
    "Fuel/Services": "Gas Station",
    "Toilet": "Restroom",
    "Cemetery": "Water",
    "Camping": "Campground",
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

def get_amenities_batch(points_batch, radius):
    coord_list = [f"{round(p.latitude,5)},{round(p.longitude,5)}" for p in points_batch]
    coord_string = ",".join(coord_list)

    query = f"""
    [out:json][timeout:90];
    (
      node["amenity"~"drinking_water|fountain|toilets|fuel|grave_yard"](around:{radius},{coord_string});
      node["natural"~"spring"](around:{radius},{coord_string});
      node["shop"~"supermarket|convenience|kiosk|general"](around:{radius},{coord_string});
      node["tourism"~"camp_site"](around:{radius},{coord_string});
    );
    out body;
    """
    try:
        # Respectful pause for the free API
        time.sleep(random.uniform(1.0, 2.0))
        response = requests.post(OVERPASS_URL, data={'data': query}, headers=HEADERS, timeout=120)
        
        if response.status_code == 200:
            return response.json().get('elements', [])
        elif response.status_code == 429:
            time.sleep(5)
            return get_amenities_batch(points_batch, radius)
        else:
            return []
    except Exception:
        return []

# --- MAIN APP LOGIC ---
uploaded_file = st.file_uploader("Drop your GPX file here", type=['gpx'])

if uploaded_file is not None:
    if st.button("Start Scan (This takes ~10-15 mins)"):
        
        # 1. PARSE
        gpx = gpxpy.parse(uploaded_file)
        all_points = []
        for track in gpx.tracks:
            for segment in track.segments:
                all_points.extend(segment.points)
        
        # 2. RESAMPLE
        query_points = []
        last_point = None
        for point in all_points:
            if last_point is None or haversine(last_point.longitude, last_point.latitude, point.longitude, point.latitude) >= SAMPLE_STEP:
                query_points.append(point)
                last_point = point
        
        batches = [query_points[i:i + BATCH_SIZE] for i in range(0, len(query_points), BATCH_SIZE)]
        
        st.info(f"Route Length: {len(all_points)} points. Optimized to {len(batches)} scan batches.")
        
        # 3. SCAN LOOP WITH PROGRESS BAR
        progress_bar = st.progress(0)
        status_text = st.empty()
        found_amenities = []
        unique_ids = set()

        for i, batch in enumerate(batches):
            # Update UI
            progress = (i + 1) / len(batches)
            progress_bar.progress(progress)
            status_text.text(f"Scanning Batch {i+1}/{len(batches)}... Found {len(found_amenities)} POIs so far.")
            
            # Fetch Data
            items = get_amenities_batch(batch, SEARCH_RADIUS)
            
            for item in items:
                item_id = item.get('id')
                if item_id in unique_ids: continue
                unique_ids.add(item_id)
                
                tags = item.get('tags', {})
                category = "Other"
                amenity = tags.get('amenity')
                shop = tags.get('shop')
                
                # Logic
                if amenity in ['drinking_water', 'fountain']: category = "Water"
                elif tags.get('natural') == 'spring': category = "Water"
                elif amenity == 'toilets': category = "Toilet"
                elif amenity == 'fuel': category = "Fuel/Services"
                elif amenity == 'grave_yard': category = "Cemetery"
                elif shop: category = "Shop"
                elif tags.get('tourism') == 'camp_site': category = "Camping"

                found_amenities.append({
                    "name": tags.get('name', category),
                    "lat": item.get('lat'),
                    "lon": item.get('lon'),
                    "category": category,
                    "desc": f"{category}: {tags.get('amenity', tags.get('shop', 'poi'))}"
                })

        st.success(f"Done! Found {len(found_amenities)} survival points.")

        # 4. MERGE & DOWNLOAD
        for item in found_amenities:
            wpt = gpxpy.gpx.GPXWaypoint(latitude=item['lat'], longitude=item['lon'], name=item['name'])
            wpt.description = item['desc']
            wpt.type = item['category']
            wpt.symbol = SYMBOL_MAP.get(item['category'], "Waypoint")
            gpx.waypoints.append(wpt)
        
        # Create In-Memory File for Download
        output_io = BytesIO()
        output_io.write(gpx.to_xml().encode('utf-8'))
        output_io.seek(0)
        
        st.download_button(
            label="Download Enhanced GPX",
            data=output_io,
            file_name="TCR_Enhanced_Route.gpx",
            mime="application/gpx+xml"
        )
