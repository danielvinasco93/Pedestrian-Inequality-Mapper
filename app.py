import os
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
import h3
import streamlit as st
import pydeck as pdk
import altair as alt
import json
from shapely.geometry import MultiPoint
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# CSS Injection: Lock Viewport & Clean Desktop Web App Presentation
st.markdown("""
<style>
/* Streamlit header & navigation bar styling */
header[data-testid="stHeader"] {
    background: rgba(10, 15, 25, 0.85) !important;
    backdrop-filter: blur(8px) !important;
}
footer {
    display: none !important;
    visibility: hidden !important;
}

/* Viewport & Container Spacing Lock */
.main .block-container {
    padding-top: 1rem !important;
    padding-bottom: 0rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-height: 100vh !important;
}

[data-testid="stAppViewContainer"] {
    overflow: hidden !important;
}

/* Right Sidepanel Fixed Sidebar Styling (Commit 45a978a restored) */
div[data-testid="stColumn"]:nth-child(2) {
    position: fixed !important;
    right: 0 !important;
    top: 0 !important;
    bottom: 0 !important;
    height: 100vh !important;
    width: 380px !important;
    max-width: 380px !important;
    z-index: 9999 !important;
    background: #0f141c !important;
    border-left: none !important;
    overflow-y: auto !important;
    overscroll-behavior: contain !important;
    padding: 64px 16px 24px 16px !important;
    box-sizing: border-box !important;
}

/* Left Column / Canvas adjustment for fixed right panel */
div[data-testid="stColumn"]:nth-child(1) {
    width: calc(100% - 390px) !important;
}

/* EXPLICIT RESET FOR SIDEBAR COLUMNS TO PREVENT LEAKS */
[data-testid="stSidebar"] div[data-testid="stColumn"] {
    position: relative !important;
    right: auto !important;
    top: auto !important;
    bottom: auto !important;
    left: auto !important;
    height: auto !important;
    width: 100% !important;
    max-width: 100% !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    z-index: 1 !important;
    padding: 0 !important;
}
</style>
""", unsafe_allow_html=True)


# 1. LOAD DATA
@st.cache_resource
def load_data():
    ### Ruta del grafo
    graph_path = os.path.join("data", "processed", "weimar_master_routing_graph.graphml")
    G = nx.read_graphml(graph_path)
    for u, v, k, data in G.edges(keys=True, data=True):
        data['length'] = float(data.get('length', 1.0))
    
    ### Nodos y Aristas
    nodes_path = os.path.join("data", "processed", "weimar_nodes_final.geojson")
    gdf_nodes = gpd.read_file(nodes_path)
    gdf_nodes['lon'] = gdf_nodes.geometry.x
    gdf_nodes['lat'] = gdf_nodes.geometry.y
    gdf_nodes['node_id'] = list(G.nodes())
    gdf_nodes['str_id'] = gdf_nodes['node_id'].astype(str)
    
    # Enforce numeric data types on all POI and demographic columns
    target_numeric_cols = [
        'poi_Subsistence', 'poi_Health', 'poi_Education', 'poi_Recreation',
        'pop_total', 'pop_children', 'pop_adults', 'pop_elderly'
    ]
    for col in target_numeric_cols:
        if col in gdf_nodes.columns:
            gdf_nodes[col] = pd.to_numeric(gdf_nodes[col], errors='coerce').fillna(0).astype(int)
    
    edges_path = os.path.join("data", "processed", "weimar_edges.geojson")
    gdf_edges = gpd.read_file(edges_path)
    
    edge_map = [{'u': str(u), 'v': str(v)} for u, v in G.edges(keys=False)]
    df_map = pd.DataFrame(edge_map)
    if 'u' not in gdf_edges.columns:
        gdf_edges = pd.concat([gdf_edges, df_map], axis=1)

    # Precomputed dict mapping
    gini_mapping = gdf_nodes.set_index('osmid').to_dict('index') if 'osmid' in gdf_nodes.columns else {}
    
    return G, gdf_nodes, gdf_edges, gini_mapping, gdf_nodes

# Ajuste de llamada global
G, df_nodes, df_edges, gini_mapping, gdf_precomputed = load_data()

# Build spatial edge coordinate dictionary for true curved route rendering
edge_geom_dict = {}
for _, row in df_edges.iterrows():
    u_str, v_str = str(row['u']), str(row['v'])
    geom = row.get('geometry', None)
    if geom is not None and not geom.is_empty and hasattr(geom, 'coords'):
        coords = list(geom.coords)
        edge_geom_dict[(u_str, v_str)] = coords
        edge_geom_dict[(v_str, u_str)] = coords[::-1]

# --- DISTRICT BOUNDARIES LOADER ---
@st.cache_data  # CRITICAL: Do not remove this line
def load_district_boundaries():
    """Fetches Weimar district boundaries from a local GeoJSON file."""
    
    # Adjusted path to match your graph/nodes path structure
    local_path = os.path.join("data", "processed", "weimar_districts_final.geojson") 
    
    # Load and explicitly ensure the coordinate reference system is WGS84 for Pydeck
    gdf_districts = gpd.read_file(local_path)
    if gdf_districts.crs != "EPSG:4326":
        gdf_districts = gdf_districts.to_crs("EPSG:4326")
        
    return gdf_districts

@st.cache_data
def load_macro_summary():
    """Fetches precalculated city and district aggregates."""
    summary_path = os.path.join("data", "processed", "weimar_macro_summary.json")
    with open(summary_path, "r") as f:
        return json.load(f)

@st.cache_data
def load_h3_gdf():
    """Loads pre-computed H3 hexagon layer with raw demographic & vitality metrics."""
    h3_path = os.path.join("data", "processed", "weimar_h3_hexagons.geojson")
    if os.path.exists(h3_path):
        return gpd.read_file(h3_path)
    return None

@st.cache_data
def load_node_isochrones():
    """Loads pre-computed node isochrone dictionary for instant O(1) lookups."""
    iso_path = os.path.join("data", "processed", "weimar_node_isochrones.json")
    if os.path.exists(iso_path):
        with open(iso_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# This initialization MUST happen before the sidebar is drawn
macro_summary = load_macro_summary()
gdf_h3 = load_h3_gdf()
node_isochrones_precomputed = load_node_isochrones()

# Initialize the data
gdf_districts = load_district_boundaries()

# 2. STATE & CALLBACK
if 'selected_node' not in st.session_state: 
    st.session_state.selected_node = None

active_district_name = None
if st.session_state.selected_node:
    node_attributes = gini_mapping.get(str(st.session_state.selected_node), {})
    active_district_name = node_attributes.get('name', None)

def on_map_click():
    event = st.session_state.get("map_layer")
    if event and "selection" in event and "objects" in event["selection"]:
        for layer_key, objects in event["selection"]["objects"].items():
            if objects:
                clicked_obj = objects[0]
                clicked_str = str(clicked_obj.get('str_id', clicked_obj.get('node_id', '')))
                if clicked_str in G.nodes():
                    st.session_state.selected_node = clicked_str
                    break

# 3. ANALYSIS PARAMETERS & DROPDOWN
with st.sidebar:
    st.markdown(
        """
        <div style="margin-top: -15px; margin-bottom: 16px; padding: 14px; background: linear-gradient(135deg, rgba(15, 20, 28, 0.95), rgba(5, 10, 18, 0.95)); border: 1px solid #00F0FF; border-radius: 8px; box-shadow: 0 0 15px rgba(0, 240, 255, 0.15);">
            <div style="font-size: 9.5px; font-weight: 700; color: #00F0FF; letter-spacing: 1.5px; font-family: monospace; text-transform: uppercase; margin-bottom: 5px;">
                Bauhaus-Universität Weimar // IUDD
            </div>
            <div style="font-size: 1.25rem; font-weight: 900; color: #FFFFFF; letter-spacing: 0.2px; line-height: 1.2; font-family: 'Segoe UI', -apple-system, sans-serif;">
                PEDESTRIAN INEQUITY <span style="color: #00F0FF;">MAPPER</span>
            </div>
            <div style="font-size: 10px; color: #8F9CA7; margin-top: 6px; font-family: monospace; font-weight: 600; display: flex; align-items: center; justify-content: space-between; border-top: 1px solid rgba(0, 240, 255, 0.2); padding-top: 5px;">
                <span>V.3.2</span>
                <span style="color: #66FF99;">● ACTIVE</span>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # ==========================================
    # 1. ANALYSIS PARAMETERS
    # ==========================================
    # Native Streamlit expander (stable, no CSS hacks)
    with st.expander("ANALYSIS PARAMETERS", expanded=True):
        # Enforces the 5-minute increments
        travel_time = st.select_slider(
            "Select Catchment Radius (Minutes)",
            options=[5, 10, 15, 20],
            value=15
        )
        
        # Dropdown logic (Lock to primary 4 categories)
        poi_cols = ['poi_Subsistence', 'poi_Health', 'poi_Education', 'poi_Recreation']
        category_names = [c.replace('poi_', '') for c in poi_cols]
        dropdown_options = ['All Categories'] + category_names
        
        selected_cat = st.selectbox(
            "Highlight Amenity Category", 
            dropdown_options
        )
        
        # Heatmap toggle

        def get_iqm(series):
            s = series.dropna()
            if len(s) == 0:
                return 0.0
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            filtered = s[(s >= q1) & (s <= q3)]
            if len(filtered) == 0:
                return float(s.mean()) # Fallback if all values are identical
            return float(filtered.mean())

        show_heatmap = st.checkbox("Urban Vitality Ratio Heatmap", value=False)
        show_false_positives = st.checkbox("Highlight Euclidean Accessibility Illusions", value=False)
        
        heatmap_data = []
        if show_heatmap:
            # 1. Demographic Selector
            demo_options = {
                "Total Population": "pop_total",
                "Children": "pop_children",
                "Adults": "pop_adults",
                "Elderly": "pop_elderly"
            }
            selected_demo_label = st.selectbox(
                "Select Target Demographic for Per Capita Ratio:", 
                options=list(demo_options.keys())
            )
            selected_demo_col = demo_options[selected_demo_label]

            # Thermal Heatmap Legend in Sidebar (Aligned with UVR Pressure Metric)
            st.markdown("""
            <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 10px 12px; margin-top: 10px; margin-bottom: 15px;">
                <div style="height: 10px; border-radius: 5px; background: linear-gradient(to right, rgb(44,123,182), rgb(171,217,233), rgb(254,224,144), rgb(253,174,97), rgb(215,25,28)); margin-bottom: 6px;"></div>
                <div style="display: flex; justify-content: space-between; font-size: 10px; color: #c9d1d9; font-weight: 500;">
                    <span>Low Pressure</span>
                    <span>Moderate</span>
                    <span>High Inequity Deficit</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Dynamic Urban Vitality Ratio: UVR_c = P_n / (IQM(UV_n) + 1)
            if gdf_h3 is not None and not gdf_h3.empty:
                gdf_temp = gdf_h3.copy()
                pop_col = selected_demo_col if selected_demo_col in gdf_temp.columns else 'pop_total'
                vitality_col = f"iqm_vitality_{travel_time}m" if f"iqm_vitality_{travel_time}m" in gdf_temp.columns else 'iqm_vitality_15m'
                
                # Compute exact ratio: P_n / (IQM(UV_n) + 1) for the user's active travel_time threshold
                gdf_temp['uvr_ratio'] = gdf_temp[pop_col] / (gdf_temp[vitality_col] + 1.0)
                
                # Percentile ranking for residential cells with population > 0
                has_pop = gdf_temp[pop_col] > 0
                gdf_temp['uvr_pct'] = 0.0
                if has_pop.any():
                    gdf_temp.loc[has_pop, 'uvr_pct'] = gdf_temp.loc[has_pop, 'uvr_ratio'].rank(pct=True) * 100
                
                def get_uvr_color(row):
                    if row[pop_col] <= 0:
                        return [100, 100, 100, 25]  # Translucent Dark Gray for unpopulated cells
                    pct = row['uvr_pct']
                    if pct < 20:
                        return [44, 123, 182, 100]    # Blue: Low Pressure (High Vitality relative to pop)
                    elif pct < 40:
                        return [171, 217, 233, 105]   # Light Cyan
                    elif pct < 60:
                        return [254, 224, 144, 110]   # Yellow: Moderate
                    elif pct < 80:
                        return [253, 174, 97, 115]    # Orange: High Pressure
                    else:
                        return [215, 25, 28, 125]     # Red: Peak Inequity Deficit
                
                records = []
                for _, row in gdf_temp.iterrows():
                    geom = row.geometry
                    if geom is not None and not geom.is_empty:
                        coords = [[float(pt[0]), float(pt[1])] for pt in list(geom.exterior.coords)]
                        fill_c = get_uvr_color(row)
                        tip = (
                            f"<b>{selected_demo_label}:</b> {int(round(row[pop_col])):,}<br/>"
                            f"<b>Urban Vitality (IQM {travel_time}m):</b> {row[vitality_col]:.2f}<br/>"
                            f"<b>Socio-Functional Ratio (UVR):</b> {row['uvr_ratio']:.2f}<br/>"
                            f"<b>Inequity Percentile:</b> {row['uvr_pct']:.1f}%"
                        )
                        records.append({
                            'h3_index': str(row['h3_index']),
                            'polygon': coords,
                            'fill_color': fill_c,
                            'tooltip_html': tip
                        })
                heatmap_data = records

reachable_nodes = set()
df_active_edges = gpd.GeoDataFrame()
amenity_summary = pd.DataFrame()
demo_summary = pd.DataFrame()

walkability_score = 0.0
completeness_index = 0.0
spatial_gini = 0.0
urban_vitality = 0.0

iso_data = []
nearest_amenities_data = []
euc_line_records = []
red_euc_line_records = []
net_route_features = []
letter_pin_records = []
raw_letter_pin_records = []
orange_endpoint_records = []

# --- Master Temporal Color Palette ---
TIME_PALETTE = {
        5:  {"pydeck": [215, 25, 28, 255],   "hex": "#D7191C"},
        10: {"pydeck": [253, 174, 97, 255],  "hex": "#FDAE61"},
        15: {"pydeck": [171, 217, 233, 255], "hex": "#ABD9E9"},
        20: {"pydeck": [44, 123, 182, 255],  "hex": "#2C7BB6"}
    }

if st.session_state.selected_node:
    walking_speed_m_min = 72
    max_dist = travel_time * walking_speed_m_min
    time_intervals = [20, 15, 10, 5] 
    
    lengths = nx.single_source_dijkstra_path_length(
        G, 
        st.session_state.selected_node, 
        cutoff=max_dist, 
        weight='length'
    )
    
    reachable_nodes = set(lengths.keys())
    df_active_edges = df_edges[df_edges['u'].astype(str).isin(reachable_nodes) & df_edges['v'].astype(str).isin(reachable_nodes)]
    df_reachable_subset = df_nodes[df_nodes['node_id'].astype(str).isin(reachable_nodes)].copy()
    
    if not df_reachable_subset.empty:
        amenity_totals = df_reachable_subset[poi_cols].sum()
        amenity_summary = pd.DataFrame({'Category': category_names, 'Count': amenity_totals.values})
        amenity_summary['Category_Short'] = amenity_summary['Category'].apply(lambda c: c[0].upper() if c else '')
        
        total_amenities_at_node = df_reachable_subset[poi_cols].sum(axis=1)
        
        node_ids = df_reachable_subset['node_id'].astype(str)
        distances = node_ids.map(lengths).fillna(0)
        beta = 0.002
        decay_factors = np.exp(beta * distances)
        walkability_score = round(np.sum(total_amenities_at_node / decay_factors), 2)
        
        # Calculate Functional Completeness (Boolean Threshold)
        if not amenity_summary.empty:
            fulfilled_categories = (amenity_summary['Count'] > 0).sum()
            total_categories = len(poi_cols)
            completeness_index = round(fulfilled_categories / total_categories, 2)
        else:
            completeness_index = 0.0

        # Calculate Urban Vitality ---
        if not amenity_summary.empty:
            # A. Logarithmic Redundancy (Bonus Factor)
            # Aplicamos np.log1p (log(1+x)) a los conteos agregados, escalado a base 2
            redundancy_array = np.log1p(amenity_summary['Count'].values) / np.log(2)
            bonus_factor = float(redundancy_array.mean())
            
            # B. Synthesis: Urban Vitality Index 
            vitality_score = round(walkability_score * completeness_index * bonus_factor, 2)
        else:
            bonus_factor = 0.0
            vitality_score = 0.0
            
        H3_RES_LOCAL = 9
        if hasattr(h3, 'latlng_to_cell'): 
            df_reachable_subset['h3_index'] = df_reachable_subset.apply(lambda row: h3.latlng_to_cell(row['lat'], row['lon'], H3_RES_LOCAL), axis=1)
        else: 
            df_reachable_subset['h3_index'] = df_reachable_subset.apply(lambda row: h3.geo_to_h3(row['lat'], row['lon'], H3_RES_LOCAL), axis=1)
        
        h3_amenities_local = df_reachable_subset.groupby('h3_index')[poi_cols].sum().sum(axis=1)
        n = len(h3_amenities_local)
        mean_x = np.mean(h3_amenities_local)
        
        if mean_x > 0 and n > 1:
            diff_matrix = np.abs(h3_amenities_local.values[:, None] - h3_amenities_local.values)
            spatial_gini = round(diff_matrix.sum() / (2 * (n**2) * mean_x), 3)

    demo_cols = ['pop_total', 'pop_children', 'pop_adults', 'pop_elderly'] 
    available_demo_cols = [c for c in demo_cols if c in df_nodes.columns]
    
    if available_demo_cols and not df_reachable_subset.empty:
        # Sum all nodes within the 15-minute network corridor as floats first
        demo_totals = df_reachable_subset[available_demo_cols].sum()
        
        # Round to the nearest whole integer only AFTER the mass aggregation is complete
        rounded_population_counts = np.round(demo_totals.values).astype(int)
        
        # Map clean integers to the display categories for the sidebar chart
        # Excluding 'pop_total' from the pie chart to prevent double-counting
        chart_cols = [c for c in available_demo_cols if c != 'pop_total']
        chart_counts = [int(np.round(df_reachable_subset[c].sum())) for c in chart_cols]
        
        demo_summary = pd.DataFrame({
            'Demographic Group': [c.replace('pop_', '').capitalize() for c in chart_cols], 
            'Population': chart_counts
        })

    valid_intervals = [t for t in time_intervals if t <= travel_time]
    
    for t in valid_intervals:
        max_dist_t = t * walking_speed_m_min
        valid_nodes = [n for n, d in lengths.items() if d <= max_dist_t]
        
        if len(valid_nodes) >= 3:
            sub_gdf = df_nodes[df_nodes['node_id'].astype(str).isin(valid_nodes)]
            raw_multipoint = MultiPoint(sub_gdf.geometry.tolist())
            
            # Attempt high-resolution Concave Hull
            try:
                hull = raw_multipoint.concave_hull(ratio=0.3, allow_holes=False)
            except AttributeError:
                hull = raw_multipoint.convex_hull
            
            if not hull.is_empty and hull.geom_type == 'Polygon':
                iso_data.append({
                    "polygon": list(hull.exterior.coords),
                    # Now dynamically reads the Pydeck RGBA array from the master palette
                    "line_color": TIME_PALETTE[t]["pydeck"], 
                    "tooltip_html": f"<b>{t} min Boundary</b>"
                })

    # --- Extract 1st Nearest Amenity Dual Paths (S, H, E, R) ---
    selected_node_str = str(st.session_state.selected_node)
    sel_rows = df_nodes[df_nodes['node_id'].astype(str) == selected_node_str]
    
    if not sel_rows.empty:
        origin_lon, origin_lat = float(sel_rows.iloc[0]['lon']), float(sel_rows.iloc[0]['lat'])
        category_badges = {
            'Subsistence': 'S',
            'Health': 'H',
            'Education': 'E',
            'Recreation': 'R'
        }
        
        for cat_name, badge_char in category_badges.items():
            poi_col = f'poi_{cat_name}'
            if poi_col in df_nodes.columns:
                # Check if the selected node itself contains this amenity
                sel_poi_count = float(sel_rows.iloc[0].get(poi_col, 0)) if not sel_rows.empty else 0
                if sel_poi_count > 0:
                    tooltip_text = f"<b>{badge_char} ({cat_name})</b><br/>Network Walk: 0 m (0.0 min)<br/>Straight Euclidean: 0 m (0.0 min)"
                    orange_endpoint_records.append({'lon': float(origin_lon), 'lat': float(origin_lat), 'tooltip_html': tooltip_text})
                    
                    raw_letter_pin_records.append({
                        'target_id': selected_node_str,
                        'lon': float(origin_lon),
                        'lat': float(origin_lat),
                        'badge': str(badge_char),
                        'category': str(cat_name),
                        'net_dist': 0,
                        'euc_dist': 0,
                        'circuity': 1.0,
                        'tooltip_html': tooltip_text
                    })
                    
                    nearest_amenities_data.append({
                        'Badge': badge_char,
                        'Category': cat_name,
                        'Euclidean_m': 0,
                        'Euclidean_min': 0.0,
                        'Network_m': 0,
                        'Network_min': 0.0,
                        'Circuity': 1.0
                    })
                    continue

                cat_nodes = df_nodes[df_nodes[poi_col] > 0].copy()
                if not cat_nodes.empty:
                    cat_nodes['net_dist'] = cat_nodes['node_id'].astype(str).map(lengths)
                    cat_reachable = cat_nodes.dropna(subset=['net_dist'])
                    
                    if not cat_reachable.empty:
                        target_row = cat_reachable.sort_values('net_dist').iloc[0]
                        target_node_id = str(target_row['node_id'])
                        target_lon, target_lat = float(target_row['lon']), float(target_row['lat'])
                        d_net_m = float(target_row['net_dist'])
                        
                        # Geodesic Euclidean Distance in meters
                        dlat = np.radians(target_lat - origin_lat)
                        dlon = np.radians(target_lon - origin_lon)
                        a = np.sin(dlat/2)**2 + np.cos(np.radians(origin_lat)) * np.cos(np.radians(target_lat)) * np.sin(dlon/2)**2
                        d_euc_m = float(round(6371000 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)), 1))
                        
                        t_euc_min = round(d_euc_m / 72.0, 1)
                        t_net_min = round(d_net_m / 72.0, 1)
                        circuity = round(d_net_m / d_euc_m, 2) if d_euc_m > 0 else 1.0
                        
                        # 1. Dashed straight-line ray (uniform 10m metric dash-gap) & midpoint distance pin
                        mid_lon = (origin_lon + target_lon) / 2.0
                        mid_lat = (origin_lat + target_lat) / 2.0
                        
                        # Generate tight 2.5m metric dash-gap sub-segments (1/4th of previous 10m value)
                        dash_step_m = 2.5
                        n_sub = max(2, int(np.round(d_euc_m / dash_step_m)))
                        if n_sub % 2 != 0:
                            n_sub += 1
                            
                        lons_seg = np.linspace(origin_lon, target_lon, n_sub + 1)
                        lats_seg = np.linspace(origin_lat, target_lat, n_sub + 1)
                        for idx_s in range(0, n_sub, 2):
                            euc_line_records.append({
                                'from': [float(lons_seg[idx_s]), float(lats_seg[idx_s])],
                                'to': [float(lons_seg[idx_s+1]), float(lats_seg[idx_s+1])]
                            })
                        
                        # Target Endpoint receives Amenity Tooltip; Origin receives dot without tooltip
                        tooltip_text = f"<b>{badge_char} ({cat_name})</b><br/>Network Walk: {int(round(d_net_m))} m ({t_net_min} min)<br/>Straight Euclidean: {int(round(d_euc_m))} m ({t_euc_min} min)"
                        orange_endpoint_records.append({'lon': float(target_lon), 'lat': float(target_lat), 'tooltip_html': tooltip_text})
                        orange_endpoint_records.append({'lon': float(origin_lon), 'lat': float(origin_lat)})

                        # 2. Topological shortest path using true curved edge geometries
                        try:
                            shortest_nodes = nx.shortest_path(G, selected_node_str, target_node_id, weight='length')
                            route_coords = []
                            for u_n, v_n in zip(shortest_nodes[:-1], shortest_nodes[1:]):
                                pair = (str(u_n), str(v_n))
                                if pair in edge_geom_dict:
                                    seg_coords = edge_geom_dict[pair]
                                else:
                                    u_info = df_nodes[df_nodes['node_id'].astype(str) == str(u_n)].iloc[0]
                                    v_info = df_nodes[df_nodes['node_id'].astype(str) == str(v_n)].iloc[0]
                                    seg_coords = [(float(u_info['lon']), float(u_info['lat'])), (float(v_info['lon']), float(v_info['lat']))]
                                
                                if not route_coords:
                                    route_coords.extend(seg_coords)
                                else:
                                    if seg_coords[0] == route_coords[-1]:
                                        route_coords.extend(seg_coords[1:])
                                    else:
                                        route_coords.extend(seg_coords)
                            
                            if len(route_coords) >= 2:
                                net_route_features.append({
                                    'type': 'Feature',
                                    'geometry': {'type': 'LineString', 'coordinates': route_coords},
                                    'properties': {'category': cat_name}
                                })
                        except Exception:
                            pass
                        
                        # 3. Iconic Pop-Up Pin Marker Record (S, H, E, R)
                        raw_letter_pin_records.append({
                            'target_id': target_node_id,
                            'lon': float(target_lon),
                            'lat': float(target_lat),
                            'badge': str(badge_char),
                            'category': str(cat_name),
                            'net_dist': int(round(d_net_m)),
                            'euc_dist': int(round(d_euc_m)),
                            'circuity': float(circuity),
                            'tooltip_html': tooltip_text
                        })
                        
                        # 4. Table entry
                        nearest_amenities_data.append({
                            'Badge': badge_char,
                            'Category': cat_name,
                            'Euclidean_m': int(round(d_euc_m)),
                            'Euclidean_min': t_euc_min,
                            'Network_m': int(round(d_net_m)),
                            'Network_min': t_net_min,
                            'Circuity': circuity
                        })

                    # Draw red dashed rays ONLY if 0 POIs of this category are reachable on foot within max_dist, but >=1 POI exists within Euclidean radius
                    cat_reachable_within_max = cat_reachable[cat_reachable['net_dist'] <= max_dist]
                    if cat_reachable_within_max.empty:
                        cat_nodes['euc_dist_m'] = np.sqrt(
                            ((cat_nodes['lon'] - origin_lon) * 111320 * np.cos(np.radians(origin_lat)))**2 +
                            ((cat_nodes['lat'] - origin_lat) * 110574)**2
                        )
                        euc_in_radius = cat_nodes[cat_nodes['euc_dist_m'] <= max_dist]
                        if not euc_in_radius.empty:
                            ill_target = euc_in_radius.sort_values('euc_dist_m').iloc[0]
                            ill_lon, ill_lat = float(ill_target['lon']), float(ill_target['lat'])
                            d_ill_m = float(ill_target['euc_dist_m'])
                            
                            dash_step_m = 2.5
                            n_sub_red = max(2, int(np.round(d_ill_m / dash_step_m)))
                            if n_sub_red % 2 != 0:
                                n_sub_red += 1
                            lons_r = np.linspace(origin_lon, ill_lon, n_sub_red + 1)
                            lats_r = np.linspace(origin_lat, ill_lat, n_sub_red + 1)
                            for idx_s in range(0, n_sub_red, 2):
                                red_euc_line_records.append({
                                    'from': [float(lons_r[idx_s]), float(lats_r[idx_s])],
                                    'to': [float(lons_r[idx_s+1]), float(lats_r[idx_s+1])]
                                })
                        
        # Handle Co-Location for Multi-Amenities on the Same Target Node (Combined Comma-Separated Badges)
        if raw_letter_pin_records:
            # Group by target location (round to 5 decimal places)
            coloc_groups = {}
            for rec in raw_letter_pin_records:
                key = (round(rec['lon'], 5), round(rec['lat'], 5))
                coloc_groups.setdefault(key, []).append(rec)
                
            order_map = {'H': 0, 'E': 1, 'R': 2, 'S': 3}
            for key, rec_list in coloc_groups.items():
                sorted_recs = sorted(rec_list, key=lambda r: order_map.get(r['badge'], 99))
                unique_badges = []
                seen_badges = set()
                for r in sorted_recs:
                    if r['badge'] not in seen_badges:
                        unique_badges.append(r['badge'])
                        seen_badges.add(r['badge'])
                
                merged_badge = ", ".join(unique_badges)
                first_rec = sorted_recs[0]
                combined_tooltips = "<br/>".join([r['tooltip_html'] for r in sorted_recs])
                
                letter_pin_records.append({
                    'target_id': first_rec['target_id'],
                    'lon': float(first_rec['lon']),
                    'lat': float(first_rec['lat']),
                    'badge': merged_badge,
                    'category': ", ".join([r['category'] for r in sorted_recs]),
                    'net_dist': first_rec['net_dist'],
                    'euc_dist': first_rec['euc_dist'],
                    'circuity': first_rec['circuity'],
                    'tooltip_html': combined_tooltips
                })

# 4. VISUAL PREP & RENDER ORDER
df_viz = df_nodes.copy()
# Define the active demographic schema, explicitly isolating the partial cohorts
demo_partials = ['pop_children', 'pop_adults', 'pop_elderly']

# --- District Map Highlighting (Translucent White Surrounding Veil, Clear Active District) ---
def get_district_fill(name):
    if active_district_name == 'All Districts':
        return [0, 0, 0, 0]  # 100% clear interior across city when all districts selected
    elif name == active_district_name:
        return [0, 0, 0, 0]  # 100% clear interior / no background for active district
    else:
        return [255, 255, 255, 38]  # Translucent white background veil (85% transparency / 15% opacity) on surrounding inactive districts

def get_district_line(name):
    if active_district_name == 'All Districts':
        return [100, 115, 130, 90]  # Soft slate boundary lines
    elif name == active_district_name:
        return [0, 240, 255, 255]  # Bold 4px Neon Cyan border for active district
    else:
        return [120, 130, 140, 100]  # Faint grey border for inactive districts

def get_district_line_width(name):
    if name == active_district_name:
        return 4  # Bold 4px stroke for active district
    else:
        return 1  # 1px subtle stroke for non-active districts

gdf_districts['fill_color'] = gdf_districts['name'].apply(get_district_fill)
gdf_districts['line_color'] = gdf_districts['name'].apply(get_district_line)
gdf_districts['line_width'] = gdf_districts['name'].apply(get_district_line_width)

df_edges['line_color'] = [[60, 60, 68, 160]] * len(df_edges)

def get_node_properties(row):
    node_id_str = str(row['node_id'])
    is_sel = node_id_str == st.session_state.selected_node
    is_reach = node_id_str in reachable_nodes
    
    # Evaluate node payload
    has_amenities = is_reach and row[poi_cols].sum() > 0
    has_population = is_reach and any(row.get(c, 0) > 0.1 for c in demo_partials)
    
    is_high = False
    if is_reach:
        if selected_cat == 'All Categories' and has_amenities:
            is_high = True
        elif selected_cat != 'All Categories' and row.get(f'poi_{selected_cat}', 0) > 0:
            is_high = True

    delta_val = row.get(f'delta_walk_{travel_time}m', 0)
    is_false_pos = (delta_val > 0)

    if is_sel:
        radius = 15
        fill = [255, 0, 0, 255] 
        border = [0, 0, 0, 255]
        render_order = 4 
    elif is_false_pos and show_false_positives:
        radius = 12
        render_order = 4
        n_lost = int(round(delta_val))
        if n_lost <= 1:
            fill = [255, 180, 0, 128]     # Soft Amber (50% transparency)
            border = [255, 180, 0, 128]
        elif n_lost == 2:
            fill = [255, 110, 0, 128]     # Vibrant Orange (50% transparency)
            border = [255, 110, 0, 128]
        elif n_lost == 3:
            fill = [255, 30, 0, 128]      # Bright Red (50% transparency)
            border = [255, 30, 0, 128]
        else:
            fill = [160, 0, 40, 128]      # Deep Crimson (50% transparency)
            border = [200, 0, 40, 128]
    elif is_high:
        radius = 10
        fill = [255, 255, 0, 255] 
        border = [0, 0, 0, 255]
        render_order = 3 
    elif has_amenities:
        radius = 5
        fill = [128, 0, 128, 255] 
        border = [0, 0, 0, 255] 
        render_order = 2 
    elif has_population:
        radius = 5
        fill = [44, 123, 182, 200] 
        border = [0, 0, 0, 255]
        render_order = 1
    elif is_reach:
        radius = 5
        fill = [0, 255, 100, 200] 
        border = [0, 0, 0, 255]
        render_order = 1
    else:
        radius = 5
        fill = [100, 100, 100, 150] 
        border = [0, 0, 0, 255]
        render_order = 0 
        
    # Tooltip construction (Strict mathematical parity)
    tooltip_lines = []
    
    if is_false_pos:
        euc_cats = int(round(row.get(f'walk_euc_{travel_time}m', 0)))
        net_cats = int(round(row.get(f'walk_raw_{travel_time}m', 0)))
        diff = euc_cats - net_cats
        tooltip_lines.append(f"Euclidian access: {euc_cats}")
        tooltip_lines.append(f"Topological access: {net_cats}")
        tooltip_lines.append(f"False positives: {diff}")
        tooltip_lines.append("<br/>")
    
    # 1. Add Amenities
    if has_amenities:
        tooltip_lines.append("<b>NODE AMENITIES</b>")
        amenities = [f"• {col.replace('poi_', '')}: {int(row[col])}" 
                     for col in poi_cols if row.get(col, 0) > 0]
        tooltip_lines.extend(amenities)
    
    # 2. Add Demographics
    if has_population:
        if has_amenities:
            tooltip_lines.append("<br/><b>NODE DEMOGRAPHICS</b>")
        else:
            tooltip_lines.append("<b>NODE DEMOGRAPHICS</b>")
            
        # Extract, round, and store the partial groups first
        rounded_partials = {}
        for col in demo_partials:
            raw_val = row.get(col, 0)
            if raw_val > 0.1:  
                clean_name = col.replace('pop_', '').capitalize()
                display_val = int(round(raw_val))
                if display_val > 0:
                    rounded_partials[clean_name] = display_val
        
        # Enforce mathematical parity: Calculate total directly from the visible rounded partials
        if rounded_partials:
            enforced_total = sum(rounded_partials.values())
            tooltip_lines.append(f"• <b>Total: {enforced_total}</b>")
            
            # Append the specific demographic breakdown with indentation for hierarchy
            for name, val in rounded_partials.items():
                tooltip_lines.append(f"&nbsp;&nbsp;&nbsp;- {name}: {val}")
        
    tooltip = "<br/>".join(tooltip_lines) if tooltip_lines else None
    
    return radius, fill, border, render_order, tooltip

props = df_viz.apply(get_node_properties, axis=1, result_type='expand')
df_viz['radius'], df_viz['fill_color'], df_viz['border_color'], df_viz['render_order'], df_viz['tooltip_html'] = [props[i] for i in range(5)]

def build_solid_overlays(df):
    overlay_layers = []
    for order in [2, 3, 4]:
        df_subset = df[df['render_order'] == order]
        if not df_subset.empty:
            overlay_layers.append(
                pdk.Layer(
                    'ScatterplotLayer',
                    data=df_subset.to_dict('records'),
                    get_position='[lon, lat]',
                    get_radius='radius',
                    get_fill_color='fill_color',
                    get_line_color='border_color',
                    filled=True,
                    stroked=True,
                    get_line_width_min_pixels=3,
                    pickable=True,  
                    opacity=1.0,
                    parameters={"depthTest": False} 
                )
            )
    return overlay_layers

# 5. RENDER MAP
# Define PyDeck map view state centered on Weimar nodes
view_state = pdk.ViewState(
    latitude=df_nodes.lat.mean(),
    longitude=df_nodes.lon.mean(),
    zoom=14,
    pitch=0,
    bearing=0
)

base_layers = []

# Layer 0: Municipal Districts (Background Boundary)
base_layers.append(
    pdk.Layer(
        "GeoJsonLayer",
        data=gdf_districts,
        opacity=1,
        stroked=True,
        filled=True,
        extruded=False,
        wireframe=True,
        get_line_color="line_color", # Dynamically reads from dataframe
        get_fill_color="fill_color", # Dynamically reads from dataframe
        get_line_width="line_width", # Dynamically reads 3px for active district, 1px for others
        line_width_min_pixels=1,
        pickable=False,                       # Prevents blocking node click events
        parameters={"depthTest": False}
    )
)

# Layer 1: Base Network (Bottom)
base_layers.append(
    pdk.Layer('GeoJsonLayer', data=df_edges, get_line_color="line_color", get_line_width=1, pickable=False, parameters={"depthTest": False})
)

# Layer 2: H3 Hex Grid (Renders underneath active network and isochrones)
if show_heatmap and heatmap_data:
    base_layers.append(
        pdk.Layer(
            "PolygonLayer",
            data=heatmap_data,
            get_polygon="polygon",
            get_fill_color="fill_color",
            get_line_color=[255, 255, 255, 80],
            get_line_width=1,
            line_width_min_pixels=1,
            pickable=True,
            stroked=True,
            filled=True,
            extruded=False,
            parameters={"depthTest": False}
        )
    )

# Layer 3: Active Reachable Network (100% Opacity, 1px Width)
base_layers.append(
    pdk.Layer('GeoJsonLayer', data=df_active_edges, get_line_color=[0, 255, 100, 255], get_line_width=1, pickable=False, parameters={"depthTest": False})
)

# Layer 4: Isochrone Boundaries (Renders on top of the Hex Grid)
if iso_data:
    base_layers.append(
        pdk.Layer(
            "PolygonLayer",
            data=iso_data,
            get_polygon="polygon",
            get_line_color="line_color",
            get_line_width=4,
            pickable=True, 
            stroked=True,
            filled=False, 
            parameters={"depthTest": False}
        )
    )

df_euc_lines = pd.DataFrame(euc_line_records) if euc_line_records else pd.DataFrame()
df_letter_pins = pd.DataFrame(letter_pin_records) if letter_pin_records else pd.DataFrame()

# Layer 4b: Target Amenity White Circle Pin Base (UNDERNEATH Original Nodes & Amenity Dots)
if not df_letter_pins.empty:
    base_layers.append(
        pdk.Layer(
            'ScatterplotLayer',
            data=df_letter_pins,
            get_position='[lon, lat]',
            get_radius=13,                      # 13 meters constant radius
            get_fill_color=[255, 255, 255, 255],# PURE WHITE Background Fill
            get_line_color=[0, 0, 0, 255],      # SOLID BLACK Rim
            filled=True,
            stroked=True,
            get_line_width_min_pixels=2,
            pickable=False,
            parameters={"depthTest": False}
        )
    )

# Layer 5: Node Scatterplot (Original Nodes)
base_layers.append(
    pdk.Layer(
        'ScatterplotLayer', 
        data=df_viz.to_dict('records'),
        get_position='[lon, lat]', 
        get_radius='radius', 
        get_fill_color='fill_color', 
        get_line_color='border_color',
        filled=True,
        stroked=True, 
        get_line_width_min_pixels=3, 
        pickable=True, 
        id="nodes_layer",
        opacity=1.0,
        parameters={"depthTest": False}
    )
)

solid_overlays = build_solid_overlays(df_viz)

# Topmost Overlays (Appended AFTER solid_overlays to render ON TOP of all active nodes & amenity dots)
topmost_overlays = []

df_orange_dots = pd.DataFrame(orange_endpoint_records) if orange_endpoint_records else pd.DataFrame()

# Layer 6a_casing: White Underlay Outline (GeoJsonLayer)
if net_route_features:
    topmost_overlays.append(
        pdk.Layer(
            'GeoJsonLayer',
            data={'type': 'FeatureCollection', 'features': net_route_features},
            get_line_color=[255, 255, 255, 255], # Solid Pure White Casing Underlay
            get_line_width=6,                    # +2px wider than original 4px layer
            pickable=False,
            parameters={"depthTest": False}
        )
    )

# Layer 6a: Topological Shortest Routes (GeoJsonLayer)
if net_route_features:
    topmost_overlays.append(
        pdk.Layer(
            'GeoJsonLayer',
            data={'type': 'FeatureCollection', 'features': net_route_features},
            get_line_color=[0, 255, 100, 255], # 100% Solid Opacity matching Active Graph Edges
            get_line_width=4,
            pickable=False,
            parameters={"depthTest": False}
        )
    )

df_red_euc_lines = pd.DataFrame(red_euc_line_records) if red_euc_line_records else pd.DataFrame()

# Layer 6b: Uniform Dashed Euclidean Rays (Segmented LineLayer)
if not df_euc_lines.empty:
    topmost_overlays.append(
        pdk.Layer(
            'LineLayer',
            data=df_euc_lines,
            get_source_position='from',
            get_target_position='to',
            get_color=[255, 180, 0, 255], # Bright Dashed Amber
            get_width=3,                  # 3px fine-scale dashed stroke
            pickable=False,
            parameters={"depthTest": False}
        )
    )

# Layer 6b3: Illusory Red Dashed Euclidean Rays (Segmented LineLayer for Unreachable POIs)
if not df_red_euc_lines.empty:
    topmost_overlays.append(
        pdk.Layer(
            'LineLayer',
            data=df_red_euc_lines,
            get_source_position='from',
            get_target_position='to',
            get_color=[255, 50, 50, 255], # Bright Dashed Red for Illusory Rays
            get_width=3,
            pickable=False,
            parameters={"depthTest": False}
        )
    )

# Layer 6b2: Solid Orange Ray End Points (Geographic Meters Scaling, NO Border Rim)
if not df_orange_dots.empty:
    topmost_overlays.append(
        pdk.Layer(
            'ScatterplotLayer',
            data=df_orange_dots,
            get_position='[lon, lat]',
            get_radius=5,                       # 5 meters radius (per user comment)
            get_fill_color=[255, 180, 0, 255],  # Solid Orange/Amber fill
            filled=True,
            stroked=False,                      # NO border rim!
            pickable=True,                      # Active tooltip on orange dots
            parameters={"depthTest": False}
        )
    )

# Layer 6d: Crisp Black Letter Badges S, H, E, R (High-Resolution 120px Font Atlas)
if not df_letter_pins.empty:
    topmost_overlays.append(
        pdk.Layer(
            'TextLayer',
            data=df_letter_pins,
            get_position='[lon, lat]',
            get_text='badge',                  # Plain string 'S', 'H', 'H, S', etc.
            get_size=120,                      # 120px constant font size
            get_color=[0, 0, 0, 255],          # PURE BLACK TEXT
            get_text_anchor='"middle"',
            get_alignment_baseline='"center"',
            font_family='"Arial", "Helvetica", sans-serif',
            font_weight='bold',
            font_settings={"fontSize": 120, "buffer": 8}, # High-resolution font texture atlas for crisp, vector-sharp text!
            pickable=False,
            parameters={"depthTest": False}
        )
    )

final_layers = base_layers + solid_overlays + topmost_overlays

deck = pdk.Deck(
    layers=final_layers,
    initial_view_state=pdk.ViewState(latitude=df_nodes.lat.mean(), longitude=df_nodes.lon.mean(), zoom=14),
    map_style=pdk.map_styles.DARK,
    tooltip={"html": "{tooltip_html}"},
    parameters={"depthTest": False, "blend": True, "blendFunc": [770, 771, 1, 771]}
)

# --- Build Right-Side Card Panel Charts when Node is Selected ---
spider_fig = None
amenity_chart = None
demo_chart = None
current_node_id = str(st.session_state.selected_node) if st.session_state.selected_node else ""

if st.session_state.selected_node:
    node_row = df_nodes[(df_nodes['str_id'] == current_node_id) | (df_nodes['node_id'].astype(str) == current_node_id)] if 'str_id' in df_nodes.columns else df_nodes[df_nodes['node_id'].astype(str) == current_node_id]
    if node_row.empty and 'osmid' in df_nodes.columns:
        node_row = df_nodes[df_nodes['osmid'].astype(str) == current_node_id]

    # 1. Spider Chart
    walk_col = f'walk_raw_{travel_time}m'
    comp_col = f'completeness_{travel_time}m'
    vit_col  = f'urban_vitality_{travel_time}m'
    dens_col = f'amenity_density_{travel_time}m'
    pop_col  = f'pop_catchment_{travel_time}m'
    dist_data_plot = macro_summary.get("districts", {}).get(active_district_name, {})

    node_walk_raw = float(node_row[walk_col].iloc[0]) if (not node_row.empty and walk_col in node_row.columns) else 0.0
    node_comp_raw = float(node_row[comp_col].iloc[0]) if (not node_row.empty and comp_col in node_row.columns) else 0.0
    node_vit_raw  = float(node_row[vit_col].iloc[0]) if (not node_row.empty and vit_col in node_row.columns) else 0.0
    node_dens_raw = float(node_row[dens_col].iloc[0]) if (not node_row.empty and dens_col in node_row.columns) else 0.0
    node_pop_raw  = float(node_row[pop_col].iloc[0]) if (not node_row.empty and pop_col in node_row.columns) else 0.0

    city_walk_mean = macro_summary.get("city", {}).get(f"global_mean_walkability_{travel_time}m", 0.0)
    city_comp_mean = macro_summary.get("city", {}).get(f"global_mean_completeness_{travel_time}m", 0.0)
    city_vit_mean  = macro_summary.get("city", {}).get(f"global_mean_vitality_{travel_time}m", 0.0)
    city_dens_mean = macro_summary.get("city", {}).get(f"global_mean_amenity_density_{travel_time}m", 0.0)
    city_pop_mean  = macro_summary.get("city", {}).get(f"global_mean_population_{travel_time}m", 0.0)

    dist_walk_mean = float(dist_data_plot.get(f"mean_walk_{travel_time}m", 0.0))
    dist_comp_mean = float(dist_data_plot.get(f"mean_comp_{travel_time}m", 0.0))
    dist_vit_mean  = float(dist_data_plot.get(f"mean_vit_{travel_time}m", 0.0))
    dist_dens_mean = float(dist_data_plot.get(f"mean_amenity_density_{travel_time}m", dist_data_plot.get(f"mean_density_{travel_time}m", 0.0)))
    if dist_dens_mean == 0.0 and active_district_name and 'name' in df_nodes.columns and dens_col in df_nodes.columns:
        dist_nodes_subset = df_nodes[df_nodes['name'] == active_district_name]
        if not dist_nodes_subset.empty:
            dist_dens_mean = float(dist_nodes_subset[dens_col].mean())
    dist_pop_mean  = float(dist_data_plot.get(f"mean_pop_{travel_time}m", 0.0))

    max_walk = float(df_nodes[f'walk_raw_{travel_time}m'].max()) if f'walk_raw_{travel_time}m' in df_nodes.columns else 4.0
    max_comp = float(df_nodes[f'completeness_{travel_time}m'].max()) if f'completeness_{travel_time}m' in df_nodes.columns else 1.0
    max_vit  = float(df_nodes[f'urban_vitality_{travel_time}m'].max()) if f'urban_vitality_{travel_time}m' in df_nodes.columns else 400.0
    max_dens = float(df_nodes[f'amenity_density_{travel_time}m'].max()) if f'amenity_density_{travel_time}m' in df_nodes.columns else 250.0
    max_pop  = float(df_nodes[f'pop_catchment_{travel_time}m'].max()) if f'pop_catchment_{travel_time}m' in df_nodes.columns else 20000.0

    max_walk = max_walk if max_walk > 0 else 1.0
    max_comp = max_comp if max_comp > 0 else 1.0
    max_vit  = max_vit if max_vit > 0 else 1.0
    max_dens = max_dens if max_dens > 0 else 1.0
    max_pop  = max_pop if max_pop > 0 else 1.0

    node_norm = [
        min(1.0, max(0.0, node_walk_raw / max_walk)),
        min(1.0, max(0.0, node_comp_raw / max_comp)),
        min(1.0, max(0.0, node_vit_raw / max_vit)),
        min(1.0, max(0.0, node_dens_raw / max_dens)),
        min(1.0, max(0.0, node_pop_raw / max_pop))
    ]

    district_norm = [
        min(1.0, max(0.0, dist_walk_mean / max_walk)),
        min(1.0, max(0.0, dist_comp_mean / max_comp)),
        min(1.0, max(0.0, dist_vit_mean / max_vit)),
        min(1.0, max(0.0, dist_dens_mean / max_dens)),
        min(1.0, max(0.0, dist_pop_mean / max_pop))
    ]

    city_norm = [
        min(1.0, max(0.0, city_walk_mean / max_walk)),
        min(1.0, max(0.0, city_comp_mean / max_comp)),
        min(1.0, max(0.0, city_vit_mean / max_vit)),
        min(1.0, max(0.0, city_dens_mean / max_dens)),
        min(1.0, max(0.0, city_pop_mean / max_pop))
    ]

    code_label_map = {
        'W': '(W) Walkability',
        'POP': '(POP) Population Density',
        'AD': '(AD) Amenity Density',
        'UV': '(UV) Urban Vitality',
        'C': '(C) Completeness'
    }

    categories_codes = ['W', 'C', 'UV', 'AD', 'POP']
    categories_closed = categories_codes + [categories_codes[0]]
    node_norm_closed = node_norm + [node_norm[0]]
    district_norm_closed = district_norm + [district_norm[0]]
    city_norm_closed = city_norm + [city_norm[0]]

    hover_fmt = "<b>%{hovertext}</b><br>Value: %{r:.2f}<extra></extra>"
    node_hover = [code_label_map[c] for c in categories_codes] + [code_label_map[categories_codes[0]]]
    district_hover = node_hover
    city_hover = node_hover

    spider_fig = go.Figure()
    spider_fig.add_trace(go.Scatterpolar(
        r=node_norm_closed,
        theta=categories_closed,
        hovertext=node_hover,
        fill='toself',
        fillcolor="rgba(0, 210, 255, 0.25)",
        line=dict(color="#00d2ff", width=3),
        name='Selected Node',
        hovertemplate=hover_fmt
    ))

    spider_fig.add_trace(go.Scatterpolar(
        r=district_norm_closed,
        theta=categories_closed,
        hovertext=district_hover,
        fill='toself',
        fillcolor='rgba(0, 0, 0, 0)',
        name=f'{active_district_name} Avg',
        line=dict(color='orange', width=2, dash='dash'),
        hovertemplate=hover_fmt
    ))

    spider_fig.add_trace(go.Scatterpolar(
        r=city_norm_closed,
        theta=categories_closed,
        hovertext=city_hover,
        fill='toself',
        fillcolor='rgba(0, 0, 0, 0)',
        line=dict(color='white',  width=2, dash='dash'),
        name='Weimar Avg',
        hovertemplate=hover_fmt
    ))

    spider_fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, 1.15], showticklabels=False),
            angularaxis=dict(rotation=90)
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.38, xanchor="center", x=0.5, font=dict(size=10)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=25, r=25, t=20, b=35),
        height=285
    )

    # 2. Aggregated Amenities Chart
    if not df_reachable_subset.empty and not amenity_summary.empty:
        time_intervals = [5, 10, 15, 20]
        for t in time_intervals:
            if travel_time >= t:
                max_dist_t = t * walking_speed_m_min
                valid_nodes_t = [str(n) for n, d in lengths.items() if d <= max_dist_t]
                if valid_nodes_t:
                    sub_df_t = df_nodes[df_nodes['node_id'].astype(str).isin(valid_nodes_t)]
                    amenity_summary[f'{t} min'] = sub_df_t[poi_cols].sum().values
                else:
                    amenity_summary[f'{t} min'] = 0
        
        amenity_summary['Category_Short'] = amenity_summary['Category'].apply(lambda c: c[0].upper() if c else '')
        sorted_short_categories = amenity_summary.sort_values(by='Count', ascending=False)['Category_Short'].tolist()
        bars = alt.Chart(amenity_summary).mark_bar(opacity=0.4).encode(
            x=alt.X('Category_Short:N', sort=sorted_short_categories, axis=alt.Axis(labelAngle=0, title='Category')),
            y=alt.Y('Count:Q', title='Amenity Count'),
            color=alt.condition(alt.datum.Category == selected_cat, alt.value('yellow'), alt.value('purple')),
            tooltip=['Category:N', 'Count:Q']
        )
        
        melt_vars = [f'{t} min' for t in time_intervals if travel_time >= t]
        legend_order = [f"{t} min" for t in [5, 10, 15, 20]]
        hex_colors = [TIME_PALETTE[t]["hex"] for t in [5, 10, 15, 20]]
        
        if melt_vars:
            df_lines = amenity_summary.melt(id_vars=['Category', 'Category_Short'], value_vars=melt_vars, var_name='Threshold', value_name='Cumulative Count')
            lines = alt.Chart(df_lines).mark_line(point=True).encode(
                x=alt.X('Category_Short:N', sort=sorted_short_categories),
                y=alt.Y('Cumulative Count:Q'),
                color=alt.Color('Threshold:N', sort=legend_order, scale=alt.Scale(domain=legend_order, range=hex_colors), legend=None),
                tooltip=['Category:N', 'Threshold:N', 'Cumulative Count:Q']
            )
            amenity_chart = alt.layer(bars, lines).properties(height=240)
        else:
            amenity_chart = bars.properties(height=240)

    # 3. Catchment Demographics Chart & Legend
    available_demo_cols = [c for c in ['pop_total', 'pop_children', 'pop_adults', 'pop_elderly'] if c in df_nodes.columns]
    demo_legend_html = ""
    if not df_reachable_subset.empty and available_demo_cols:
        demo_totals = df_reachable_subset[available_demo_cols].sum()
        total_catchment_pop = int(np.round(demo_totals.get('pop_total', 0)))
        if total_catchment_pop > 0:
            chart_cols = [c for c in available_demo_cols if c != 'pop_total']
            chart_data = []
            for c in chart_cols:
                val = int(np.round(demo_totals.get(c, 0)))
                chart_data.append({'Group': c.replace('pop_', '').capitalize(), 'Population': val})
            demo_summary = pd.DataFrame(chart_data)
            demo_summary['Percentage'] = (demo_summary['Population'] / total_catchment_pop * 100).round(1)
            demo_summary['Text_Label'] = demo_summary['Percentage'].apply(lambda x: f"{x:.1f}%" if x >= 4 else "")
            demo_order = ["Children", "Adults", "Elderly"]
            
            demo_legend_html = '<div style="display: flex; justify-content: center; gap: 14px; margin-bottom: 6px;"><div style="display: flex; align-items: center; gap: 5px;"><div style="width: 8px; height: 8px; background-color: #3182bd; border-radius: 50%;"></div><span style="font-size: 11px; color: #FFFFFF; font-weight: 500;">Children</span></div><div style="display: flex; align-items: center; gap: 5px;"><div style="width: 8px; height: 8px; background-color: #00D2FF; border-radius: 50%;"></div><span style="font-size: 11px; color: #FFFFFF; font-weight: 500;">Adults</span></div><div style="display: flex; align-items: center; gap: 5px;"><div style="width: 8px; height: 8px; background-color: #ff9999; border-radius: 50%;"></div><span style="font-size: 11px; color: #FFFFFF; font-weight: 500;">Elderly</span></div></div>'
            
            base = alt.Chart(demo_summary).encode(
                theta=alt.Theta("Population:Q", stack=True),
                color=alt.Color("Group:N", sort=demo_order, scale=alt.Scale(domain=["Children", "Adults", "Elderly"], range=["#3182bd", "#00D2FF", "#ff9999"]), legend=None),
                tooltip=["Group:N", "Population:Q", "Percentage:Q"]
            )
            arcs = base.mark_arc(innerRadius=28, outerRadius=52)
            text = base.mark_text(radius=40, color='white', fontSize=10, fontWeight='bold').encode(text="Text_Label:N")
            demo_chart = alt.layer(arcs, text).properties(height=135, padding={"top": 5, "bottom": 5, "left": 0, "right": 0})

# CSS Injection for Right-Side Expander Cards & Panel Scrollbar
st.markdown("""
<style>
/* Right Column Panel Scrollbar Styling */
section:not([data-testid="stSidebar"]) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2)::-webkit-scrollbar {
    width: 5px !important;
}

section:not([data-testid="stSidebar"]) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2)::-webkit-scrollbar-track {
    background: transparent !important;
}

section:not([data-testid="stSidebar"]) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2)::-webkit-scrollbar-thumb {
    background: rgba(0, 240, 255, 0.3) !important;
    border-radius: 4px !important;
}

section:not([data-testid="stSidebar"]) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2)::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 240, 255, 0.6) !important;
}

/* Custom Glassmorphism Card styling */
.glass-card {
    background: rgba(15, 20, 28, 0.94);
    border: 1px solid rgba(0, 240, 255, 0.5);
    border-radius: 10px;
    padding: 14px 14px;
    color: #FAFAFA;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6);
    backdrop-filter: blur(8px);
    margin-bottom: 10px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.glass-card:hover {
    border-color: #00F0FF !important;
}

section:not([data-testid="stSidebar"]) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) div[data-testid="stExpander"] {
    background: rgba(15, 20, 28, 0.94) !important;
    border: 1px solid rgba(0, 240, 255, 0.5) !important;
    border-radius: 10px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6) !important;
    backdrop-filter: blur(8px) !important;
    margin-bottom: 10px !important;
}

section:not([data-testid="stSidebar"]) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) div[data-testid="stExpander"] summary span p {
    color: #00F0FF !important;
    font-weight: bold !important;
    letter-spacing: 0.5px !important;
    font-size: 12px !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    text-transform: uppercase !important;
}
</style>
""", unsafe_allow_html=True)

# Render Main Dashboard Layout
if st.session_state.selected_node:
    col_map, col_cards = st.columns([2.1, 1.0])
    with col_map:
        st.pydeck_chart(deck, key="map_layer", on_select=on_map_click, use_container_width=True, height=720)
    with col_cards:
        # Card 1: Detour Analysis (Original Custom HTML Glassmorphism Card)
        if nearest_amenities_data:
            table_rows_html = ""
            for r in nearest_amenities_data:
                c_val = r['Circuity']
                color = "#66FF99" if c_val < 1.25 else ("#FFCC00" if c_val < 1.5 else "#FF5555")
                detour_pct = int(round((c_val - 1) * 100))
                detour_str = f"+{detour_pct}%" if detour_pct > 0 else "0%"
                table_rows_html += f"<tr style=\"border-bottom: 1px solid rgba(255,255,255,0.1);\"><td style=\"padding: 4px 6px; font-weight: bold; color: #00F0FF; text-align: center;\">{r['Badge']}</td><td style=\"padding: 4px 6px; font-size: 11px;\">{r['Category']}</td><td style=\"padding: 4px 6px; color: #BBB;\">{r['Euclidean_m']} m ({r['Euclidean_min']}m)</td><td style=\"padding: 4px 6px; color: #FFF;\">{r['Network_m']} m ({r['Network_min']}m)</td><td style=\"padding: 4px 6px; color: {color}; font-weight: bold; text-align: center;\">{detour_str}</td></tr>"

            table_html = f"<table style=\"width:100%; font-size: 11px; text-align: left; border-collapse: collapse;\"><tr style=\"border-bottom: 1px solid rgba(255,255,255,0.2); color: #888; font-size: 10px;\"><th style=\"padding: 4px 4px; text-align: center;\">PIN</th><th style=\"padding: 4px 4px;\">CATEGORY</th><th style=\"padding: 4px 4px;\">STRAIGHT</th><th style=\"padding: 4px 4px;\">NETWORK</th><th style=\"padding: 4px 4px; text-align: center;\">DETOUR</th></tr>{table_rows_html}</table>"

            card1_html = f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: bold; color: #00F0FF; letter-spacing: 0.5px; font-size: 12px; text-transform: uppercase;">DETOUR ANALYSIS</span>
                    <span style="font-size: 11px; color: #888; font-weight: 500;">Node #{current_node_id}</span>
                </div>
                {table_html}
            </div>
            """
            st.markdown(card1_html, unsafe_allow_html=True)

        # Card 2: Node Performance vs. Benchmark (Spider Chart inside st.expander)
        if spider_fig:
            with st.expander("NODE PERFORMANCE VS. BENCHMARK", expanded=True):
                st.plotly_chart(spider_fig, use_container_width=True, key=f"spider_chart_{current_node_id}_{travel_time}")

        # Card 3: Aggregated Amenities (inside st.expander)
        if amenity_chart:
            with st.expander("AGGREGATED AMENITIES", expanded=False):
                st.altair_chart(amenity_chart, use_container_width=True)

        # Card 4: Catchment Demographics (inside st.expander)
        if demo_chart:
            with st.expander("CATCHMENT DEMOGRAPHICS", expanded=False):
                if demo_legend_html:
                    st.markdown(demo_legend_html, unsafe_allow_html=True)
                st.altair_chart(demo_chart, use_container_width=True)
else:
    st.pydeck_chart(deck, key="map_layer", on_select=on_map_click, use_container_width=True, height=720)


# 6. SIDEBAR CHART & INFO
with st.sidebar:
    
    # ==========================================
    # 2. CITY-LEVEL DATA
    # ==========================================
    # Set expanded=False to have it collapsed by default, or True to keep it open
    with st.expander("CITY-LEVEL DATA", expanded=False):
        
        col_city1, col_city2 = st.columns(2)
        
        city_pop = macro_summary.get("city", {}).get("total_population", 0)
        
        # Dynamically link the city-wide Gini metric to the active travel_time slider
        city_gini_key = f"global_gini_{travel_time}m"
        city_walk_key = f"global_mean_walkability_{travel_time}m"
        city_comp_key = f"global_mean_completeness_{travel_time}m"
        city_vit_key = f"global_mean_vitality_{travel_time}m"


        city_gini = macro_summary.get("city", {}).get(city_gini_key, 0.0)
        city_walk = macro_summary.get("city", {}).get(city_walk_key, 0.0)
        city_comp = macro_summary.get("city", {}).get(city_comp_key, 0.0)
        city_vit = macro_summary.get("city", {}).get(city_vit_key, 0.0)
        
        # First Row: Population and Spatial Gini
        col_city1, col_city2 = st.columns(2)
        with col_city1:
            st.metric("Total Population", f"{city_pop:,}")
        with col_city2:
            st.metric(
                label=f"Global Spatial Gini ({travel_time}m)", 
                value=f"{city_gini:.3f}", 
                help="Calculates city-wide access to amenities inequality by applying the Gini coefficient. A value closer to 0 indicates an equitable distribution, while 1 reveals spatial concentration."
            )
            
        # Second Row: Walkability and Completeness Baseline
        col_city3, col_city4 = st.columns(2)
        with col_city3:
            st.metric(
                label=f"Global Mean Walkability ({travel_time}m)", 
                value=f"{city_walk:.3f}", 
                help="A distance-decayed score based on the number of accesible amenities of all cities' nodes, where closer amenities are valued higher than distant ones. Calculated using IQR to exclude outliers."
            )
        with col_city4:
            st.metric(
                label=f"Global Mean Completeness ({travel_time}m)", 
                value=f"{city_comp * 100:.0f}%", 
                help="Measures city-wide access to at least one amenity of each amenity-category [Healthcare, Education, Retail and Recreation] within the defined catchment limit. Calculated using IQR to exclude outliers."
            )

        st.metric(
            label=f"Global Mean Urban Vitality ({travel_time}m)", 
            value=f"{city_vit:.3f}", 
            help="City-wide score for the quality of the accessibility by integrating walkability, completeness modified by a redundancy bonus. Calculated using IQR to exclude outliers."
        )
        st.divider()

        # --- CITY WALKABILITY PROFILE (Chart) ---
        with st.expander("City Walkability Profile", expanded=False):
        
            # 1. Build a temporal DataFrame with the active city-wide values
            city_walk_records = []
            time_intervals = [5, 10, 15, 20]
            
            for t in time_intervals:
                if travel_time >= t:
                    # Extract the metric at the city level directly from macro_summary
                    city_walk_key = f"global_mean_walkability_{t}m"
                    val = float(macro_summary.get("city", {}).get(city_walk_key, 0.0))
                    
                    city_walk_records.append({
                        'Threshold': f'{t} min',
                        'Mean Walkability': val,
                        'Minutes': t
                    })
            
            df_city_walk_profile = pd.DataFrame(city_walk_records)
            
            if not df_city_walk_profile.empty:
                
                legend_order = [f"{t} min" for t in time_intervals if travel_time >= t]
                hex_colors = [TIME_PALETTE[t]["hex"] for t in time_intervals if travel_time >= t]
                
                # Layer 1: Continuous line
                base_line = alt.Chart(df_city_walk_profile).mark_line(
                    strokeWidth=3,
                    color='rgba(255, 255, 255, 0.5)'
                ).encode(
                    x=alt.X('Threshold:N', sort=legend_order, title='Catchment Threshold', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Mean Walkability:Q', title='Mean Walkability (City)')
                )
                
                # Layer 2: Colored points
                colored_points = alt.Chart(df_city_walk_profile).mark_circle(
                    size=150,
                    opacity=1
                ).encode(
                    x=alt.X('Threshold:N', sort=legend_order),
                    y=alt.Y('Mean Walkability:Q'),
                    color=alt.Color(
                        'Threshold:N',
                        sort=legend_order,
                        scale=alt.Scale(domain=legend_order, range=hex_colors),
                        legend=None
                    ),
                    tooltip=['Threshold:N', alt.Tooltip('Mean Walkability:Q', format='.2f')]
                )
                
                city_walk_chart = alt.layer(base_line, colored_points).properties(height=280)
                
                # Custom HTML Legend
                legend_html = '<div style="display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 15px; padding-left: 0px;">'
                for t in time_intervals:
                    if travel_time >= t:
                        hex_color = TIME_PALETTE[t]["hex"]
                        legend_html += f'<div style="display: flex; align-items: center; gap: 6px;"><div style="width: 10px; height: 10px; background-color: {hex_color}; border-radius: 50%;"></div><span style="font-size: 0.9rem; color: rgba(255, 255, 255, 0.8); font-weight: 500;">{t} min.</span></div>'
                legend_html += '</div>'
                
                st.markdown(legend_html, unsafe_allow_html=True)
                st.altair_chart(city_walk_chart, use_container_width=True)
                
            else:
                st.warning("No walkability profile data available at the city level.")









        # --- CITY VITALITY PROFILE (Chart) ---
        with st.expander("City Urban Vitality Profile", expanded=False):
        
            # 1. Build a temporal DataFrame with the active city-wide vitality values
            city_vit_records = []
            
            for t in time_intervals:
                if travel_time >= t:
                    # Extract the vitality metric at the city level directly from macro_summary
                    city_vit_key = f"global_mean_vitality_{t}m"
                    val = float(macro_summary.get("city", {}).get(city_vit_key, 0.0))
                    
                    city_vit_records.append({
                        'Threshold': f'{t} min',
                        'Mean Vitality': val,
                        'Minutes': t
                    })
            
            df_city_vit_profile = pd.DataFrame(city_vit_records)
            
            if not df_city_vit_profile.empty:
                
                # Layer 1: Continuous line
                vit_base_line = alt.Chart(df_city_vit_profile).mark_line(
                    strokeWidth=3,
                    color='rgba(255, 255, 255, 0.5)'
                ).encode(
                    x=alt.X('Threshold:N', sort=legend_order, title='Catchment Threshold', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Mean Vitality:Q', title='Mean Global Urban Vitality')
                )
                
                # Layer 2: Colored points
                vit_colored_points = alt.Chart(df_city_vit_profile).mark_circle(
                    size=150,
                    opacity=1
                ).encode(
                    x=alt.X('Threshold:N', sort=legend_order),
                    y=alt.Y('Mean Vitality:Q'),
                    color=alt.Color(
                        'Threshold:N',
                        sort=legend_order,
                        scale=alt.Scale(domain=legend_order, range=hex_colors),
                        legend=None
                    ),
                    tooltip=['Threshold:N', alt.Tooltip('Mean Vitality:Q', format='.2f')]
                )
                
                city_vit_chart = alt.layer(vit_base_line, vit_colored_points).properties(height=280)
                
                # Re-use the Custom HTML Legend built in the walkability block
                st.markdown(legend_html, unsafe_allow_html=True)
                st.altair_chart(city_vit_chart, use_container_width=True)
                
            else:
                st.warning("No urban vitality profile data available at the city level.")











    
    # ==========================================
    # 3. DISTRICT-LEVEL DATA
    # ==========================================
    # Replaced st.container with st.expander to make it collapsible.
    # Kept expanded=True by default so users immediately see data when clicking a node.
    with st.expander("DISTRICT-LEVEL DATA", expanded=False):
        
        dist_data = {}
        display_name = "Select a node on the map"
        
        if active_district_name:
            display_name = active_district_name
            dist_data = macro_summary.get("districts", {}).get(active_district_name, {})

        district_radar_values = []

        if active_district_name and dist_data:
            radar_key = f"radar_values_{travel_time}m"
            district_radar_values = dist_data.get(radar_key, [0.0, 0.0, 0.0])

        st.markdown(f"**Active District:** *{display_name}*")
        
        # Extract metrics safely, defaulting to "---" if no node is selected
        dist_pop = dist_data.get("population", "---")
        
        # Dynamically link the Gini metric to the active travel_time slider
        gini_key = f"gini_density_{travel_time}m"
        dist_gini = dist_data.get(gini_key, "---")
        
        walk_key = f"mean_walk_{travel_time}m"
        dist_walk = dist_data.get(walk_key, "---")

        comp_key = f"mean_comp_{travel_time}m"
        dist_comp = dist_data.get(comp_key, "---")

        vit_key = f"mean_vit_{travel_time}m"
        dist_vit = dist_data.get(vit_key, "---")
        
        # Format the metrics for display
        pop_str = f"{dist_pop:,}" if isinstance(dist_pop, int) else dist_pop
        gini_str = f"{dist_gini:.3f}" if isinstance(dist_gini, float) else dist_gini
        walk_str = f"{dist_walk:.3f}" if isinstance(dist_walk, float) else dist_walk
        
        # UPDATE: Format mean completeness as a percentage
        comp_str = f"{dist_comp * 100:.0f}%" if isinstance(dist_comp, float) else dist_comp
        vit_str = f"{dist_vit:.2f}" if isinstance(dist_vit, float) else dist_vit # Defines vit_str
        
        col_dist1, col_dist2 = st.columns(2)
        with col_dist1:
            st.metric("Population", pop_str)
        with col_dist2:
            st.metric(
                label=f"Spatial Gini ({travel_time}m)", 
                value=gini_str, 
                help="Calculates district-wide access to amenities inequality by applying the Gini coefficient. A value closer to 0 indicates an equitable distribution, while 1 reveals spatial concentration."
            )
            
        col_dist3, col_dist4 = st.columns(2)
        with col_dist3:
            st.metric(
                label=f"Mean Walkability ({travel_time}m)", 
                value=walk_str, 
                help="A distance-decayed score based on the number of accesible amenities of all district´s nodes, where closer amenities are valued higher than distant ones. Calculated using IQR to exclude outliers."
            )
        with col_dist4:
            # METRIC UPDATE: Reflects the mean functional completeness
            st.metric(
                label=f"Mean Completeness ({travel_time}m)", 
                value=comp_str, 
                help="Measures district-wide access to at least one amenity of each amenity-category [Healthcare, Education, Retail and Recreation] within the defined catchment limit. Calculated using IQR to exclude outliers."
            )

        # 2. Calcular y mostrar el Score Relativo (1.0 - 5.0) usando el percentil precalculado
        district_row = gdf_districts[gdf_districts['name'] == active_district_name]
        
        score_val = 1.0 # Valor por defecto
        status_component = st.error 

        if not district_row.empty:
            pct_col = f'mean_vit_pct_{travel_time}m'
            if pct_col in district_row.columns and len(district_row) > 0:
                pct_val = float(district_row[pct_col].iloc[0])
            else:
                pct_val = 0.0
            score_val = 1.0 + (4.0 * pct_val)

            col_vit1, col_vit2 = st.columns(2)

            with col_vit1:
                st.metric(
                label=f"Mean Urban Vitality ({travel_time}m)", 
                value=vit_str, 
                help="District-wide score for the quality of the accessibility by integrating walkability, completeness modified by a redundancy bonus. Calculated using IQR to exclude outliers."
                )

            with col_vit2:
                st.metric(
                    label=f"UV Performance", 
                    value=f"{score_val:.2f}",
                    help="Performance of the urban vitality of the analized district ranked relative to the entire district universe."
                )

        st.divider()

        # --- DISTRICT WALKABILITY PROFILE (Collapsible) ---
        with st.expander("District Walkability Profile", expanded=False):
    
            # 1. Build a temporal DataFrame with the active district's values
            dist_walk_records = []
            time_intervals = [5, 10, 15, 20]
            
            for t in time_intervals:
                if travel_time >= t:
                    # Extraemos la métrica promediada a nivel de distrito
                    col_name = f'mean_walk_{t}m'
                    
                    if col_name in district_row.columns and len(district_row) > 0:
                        val = float(district_row[col_name].iloc[0])
                    else:
                        val = 0.0
                        
                    dist_walk_records.append({
                        'Threshold': f'{t} min',
                        'Mean Walkability': val,
                        'Minutes': t
                    })
            
            df_dist_walk_profile = pd.DataFrame(dist_walk_records)
            
            if not df_dist_walk_profile.empty:
                
                legend_order = [f"{t} min" for t in time_intervals if travel_time >= t]
                hex_colors = [TIME_PALETTE[t]["hex"] for t in time_intervals if travel_time >= t]
                
                # Layer 1: Continuous line
                base_line = alt.Chart(df_dist_walk_profile).mark_line(
                    strokeWidth=3,
                    color='rgba(255, 255, 255, 0.5)'
                ).encode(
                    x=alt.X('Threshold:N', sort=legend_order, title='Catchment Threshold', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Mean Walkability:Q', title='Mean Walkability')
                )
                
                # Layer 2: Colored points
                colored_points = alt.Chart(df_dist_walk_profile).mark_circle(
                    size=150,
                    opacity=1
                ).encode(
                    x=alt.X('Threshold:N', sort=legend_order),
                    y=alt.Y('Mean Walkability:Q'),
                    color=alt.Color(
                        'Threshold:N',
                        sort=legend_order,
                        scale=alt.Scale(domain=legend_order, range=hex_colors),
                        legend=None
                    ),
                    tooltip=['Threshold:N', alt.Tooltip('Mean Walkability:Q', format='.2f')]
                )
                
                dist_walk_chart = alt.layer(base_line, colored_points).properties(height=280)
                
                # Custom HTML Legend
                legend_html = '<div style="display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 15px; padding-left: 0px;">'
                for t in time_intervals:
                    if travel_time >= t:
                        hex_color = TIME_PALETTE[t]["hex"]
                        legend_html += f'<div style="display: flex; align-items: center; gap: 6px;"><div style="width: 10px; height: 10px; background-color: {hex_color}; border-radius: 50%;"></div><span style="font-size: 0.9rem; color: rgba(255, 255, 255, 0.8); font-weight: 500;">{t} min.</span></div>'
                legend_html += '</div>'
                
                st.markdown(legend_html, unsafe_allow_html=True)
                st.altair_chart(dist_walk_chart, use_container_width=True)
                
            else:
                st.warning("No walkability profile data available for this district.")

        # --- DISTRICT URBAN VITALITY PROFILE (Collapsible) ---
        with st.expander("District Urban Vitality Profile", expanded=False):
    
            # 1. Build a temporal DataFrame with the active district's values
            dist_vit_records = []
            time_intervals = [5, 10, 15, 20]
            
            for t in time_intervals:
                if travel_time >= t:
                    # Extraemos la métrica promediada (IQR) a nivel de distrito
                    col_name = f'mean_vit_{t}m'
                    
                    if col_name in district_row.columns and len(district_row) > 0:
                        val = float(district_row[col_name].iloc[0])
                    else:
                        val = 0.0
                        
                    dist_vit_records.append({
                        'Threshold': f'{t} min',
                        'Mean Vitality': val,
                        'Minutes': t
                    })
            
            df_dist_vit_profile = pd.DataFrame(dist_vit_records)
            
            if not df_dist_vit_profile.empty:
                
                legend_order = [f"{t} min" for t in time_intervals if travel_time >= t]
                hex_colors = [TIME_PALETTE[t]["hex"] for t in time_intervals if travel_time >= t]
                
                # Layer 1: Continuous line
                base_line = alt.Chart(df_dist_vit_profile).mark_line(
                    strokeWidth=3,
                    color='rgba(255, 255, 255, 0.5)'
                ).encode(
                    x=alt.X('Threshold:N', sort=legend_order, title='Catchment Threshold', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Mean Vitality:Q', title='Mean Urban Vitality')
                )
                
                # Layer 2: Colored points
                colored_points = alt.Chart(df_dist_vit_profile).mark_circle(
                    size=150,
                    opacity=1
                ).encode(
                    x=alt.X('Threshold:N', sort=legend_order),
                    y=alt.Y('Mean Vitality:Q'),
                    color=alt.Color(
                        'Threshold:N',
                        sort=legend_order,
                        scale=alt.Scale(domain=legend_order, range=hex_colors),
                        legend=None
                    ),
                    tooltip=['Threshold:N', alt.Tooltip('Mean Vitality:Q', format='.2f')]
                )
                
                dist_vit_chart = alt.layer(base_line, colored_points).properties(height=280)
                
                # Custom HTML Legend
                legend_html = '<div style="display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 15px; padding-left: 0px;">'
                for t in time_intervals:
                    if travel_time >= t:
                        hex_color = TIME_PALETTE[t]["hex"]
                        legend_html += f'<div style="display: flex; align-items: center; gap: 6px;"><div style="width: 10px; height: 10px; background-color: {hex_color}; border-radius: 50%;"></div><span style="font-size: 0.9rem; color: rgba(255, 255, 255, 0.8); font-weight: 500;">{t} min.</span></div>'
                legend_html += '</div>'
                
                st.markdown(legend_html, unsafe_allow_html=True)
                st.altair_chart(dist_vit_chart, use_container_width=True)
                
            else:
                st.warning("No urban vitality profile data available for this district.")
        
    # ==========================================
    # 4. NODE-LEVEL DATA
    # ==========================================
    if st.session_state.selected_node:
        current_node_id = str(st.session_state.selected_node)
        node_row = gdf_precomputed[(gdf_precomputed['str_id'] == current_node_id) | (gdf_precomputed['node_id'].astype(str) == current_node_id)] if 'str_id' in gdf_precomputed.columns else gdf_precomputed[gdf_precomputed['node_id'].astype(str) == current_node_id]
        if node_row.empty and 'osmid' in gdf_precomputed.columns:
            node_row = gdf_precomputed[gdf_precomputed['osmid'].astype(str) == current_node_id]

        if not node_row.empty:
            node_walk = float(node_row[f'walk_raw_{travel_time}m'].iloc[0]) if f'walk_raw_{travel_time}m' in node_row.columns else walkability_score
            node_comp = float(node_row[f'completeness_{travel_time}m'].iloc[0]) if f'completeness_{travel_time}m' in node_row.columns else completeness_index
            node_vit = float(node_row[f'urban_vitality_{travel_time}m'].iloc[0]) if f'urban_vitality_{travel_time}m' in node_row.columns else vitality_score
            pct_val = float(node_row[f'urban_vitality_pct_{travel_time}m'].iloc[0]) if f'urban_vitality_pct_{travel_time}m' in node_row.columns else 0.5
            score_continuous = 1.0 + (4.0 * pct_val)
        else:
            node_walk = walkability_score
            node_comp = completeness_index
            node_vit = vitality_score
            score_continuous = 1.0

        with st.expander("NODE-LEVEL DATA", expanded=False):
            # --- 4.1 Walkability & Completeness Indices ---
            col_node1, col_node2 = st.columns(2)
            with col_node1:
                st.metric(
                    label=f"Walkability Score ({travel_time}m)", 
                    value=f"{node_walk:.2f}", 
                    help="A distance-decayed score for the selected node, where closer amenities are valued higher than distant ones."
                )

            with col_node2:
                st.metric(
                    label=f"Completeness Index ({travel_time}m)", 
                    value=f"{node_comp * 100:.0f}%", 
                    help="Measures the analyzed node's access to at least one amenity of each amenity-category [Healthcare, Education, Retail and Recreation] within the defined catchment limit."
                )

            col_vit1, col_vit2 = st.columns(2)
            with col_vit1:
                st.metric(
                    label=f"Urban Vitality ({travel_time}m)", 
                    value=f"{node_vit:.2f}",
                    help="Node score for the quality of the accessibility by integrating walkability, completeness modified by a redundancy bonus."
                )

            with col_vit2:
                st.metric(
                    label=f"UV Performance", 
                    value=f"{score_continuous:.2f}",
                    help="Performance of the urban vitality of the analyzed district ranked relative to the entire node universe."
                )

            # --- WALKABILITY PROFILE (Collapsible) ---
            with st.expander("Walkability Profile", expanded=False):
        
                # 1. Construir un DataFrame temporal con los valores del nodo activo
                walkability_records = []
                time_intervals = [5, 10, 15, 20]
                
                for t in time_intervals:
                    if travel_time >= t:
                        # Ajusta 'walk_raw' al nombre real de tu columna en el GeoDataFrame
                        col_name = f'walk_raw_{t}m' 
                        val = float(node_row[col_name].values[0]) if (not node_row.empty and col_name in node_row.columns) else 0.0
                            
                        walkability_records.append({
                            'Threshold': f'{t} min',
                            'Walkability Score': val,
                            'Minutes': t
                        })
                
                df_walk_profile = pd.DataFrame(walkability_records)
                
                if not df_walk_profile.empty:
                    
                    legend_order = [f"{t} min" for t in time_intervals if travel_time >= t]
                    hex_colors = [TIME_PALETTE[t]["hex"] for t in time_intervals if travel_time >= t]
                    
                    # CAPA 1: Línea continua
                    base_line = alt.Chart(df_walk_profile).mark_line(
                        strokeWidth=3,
                        color='rgba(255, 255, 255, 0.5)'
                    ).encode(
                        x=alt.X('Threshold:N', sort=legend_order, title='Catchment Threshold', axis=alt.Axis(labelAngle=0)),
                        y=alt.Y('Walkability Score:Q', title='Walkability')
                    )
                    
                    # CAPA 2: Puntos coloreados
                    colored_points = alt.Chart(df_walk_profile).mark_circle(
                        size=150,
                        opacity=1
                    ).encode(
                        x=alt.X('Threshold:N', sort=legend_order),
                        y=alt.Y('Walkability Score:Q'),
                        color=alt.Color(
                            'Threshold:N',
                            sort=legend_order,
                            scale=alt.Scale(domain=legend_order, range=hex_colors),
                            legend=None
                        ),
                        tooltip=['Threshold:N', alt.Tooltip('Walkability Score:Q', format='.2f')]
                    )
                    
                    walkability_chart = alt.layer(base_line, colored_points).properties(height=280)
                    
                    # Leyenda HTML personalizada
                    legend_html = '<div style="display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 15px; padding-left: 0px;">'
                    for t in time_intervals:
                        if travel_time >= t:
                            hex_color = TIME_PALETTE[t]["hex"]
                            legend_html += f'<div style="display: flex; align-items: center; gap: 6px;"><div style="width: 10px; height: 10px; background-color: {hex_color}; border-radius: 50%;"></div><span style="font-size: 0.9rem; color: rgba(255, 255, 255, 0.8); font-weight: 500;">{t} min.</span></div>'
                    legend_html += '</div>'
                    
                    st.markdown(legend_html, unsafe_allow_html=True)
                    st.altair_chart(walkability_chart, use_container_width=True)
                    
                else:
                    st.warning("No walkability profile data available for this node.")

            # --- URBAN VITALITY PROFILE (Collapsible) ---
            with st.expander("Urban Vitality Profile", expanded=False):
        
                # 1. Construir un DataFrame temporal con los valores del nodo activo
                vitality_records = []
                time_intervals = [5, 10, 15, 20]
                
                for t in time_intervals:
                    if travel_time >= t:
                        col_name = f'urban_vitality_{t}m'
                        val = float(node_row[col_name].values[0]) if (not node_row.empty and col_name in node_row.columns) else 0.0
                            
                        vitality_records.append({
                            'Threshold': f'{t} min',
                            'Vitality Index': val,
                            'Minutes': t
                        })
                
                df_vitality_profile = pd.DataFrame(vitality_records)
                
                if not df_vitality_profile.empty:
                    
                    legend_order = [f"{t} min" for t in time_intervals if travel_time >= t]
                    hex_colors = [TIME_PALETTE[t]["hex"] for t in time_intervals if travel_time >= t]
                    
                    # CAPA 1: Línea (Se verá solo si hay 2 o más puntos)
                    base_line = alt.Chart(df_vitality_profile).mark_line(
                        strokeWidth=3,
                        color='rgba(255, 255, 255, 0.5)'
                    ).encode(
                        x=alt.X('Threshold:N', sort=legend_order, title='Catchment Threshold', axis=alt.Axis(labelAngle=0)),
                        # CORRECCIÓN: Eje Y dinámico, sin restricción de domain=[0,5]
                        y=alt.Y('Vitality Index:Q', title='Urban Vitality')
                    )
                    
                    # CAPA 2: Puntos (Siempre se verán, incluso si el slider está en 5 min)
                    colored_points = alt.Chart(df_vitality_profile).mark_circle(
                        size=150,
                        opacity=1
                    ).encode(
                        x=alt.X('Threshold:N', sort=legend_order),
                        y=alt.Y('Vitality Index:Q'),
                        color=alt.Color(
                            'Threshold:N',
                            sort=legend_order,
                            scale=alt.Scale(domain=legend_order, range=hex_colors),
                            legend=None
                        ),
                        tooltip=['Threshold:N', alt.Tooltip('Vitality Index:Q', format='.2f')]
                    )
                    
                    vitality_chart = alt.layer(base_line, colored_points).properties(height=280)
                    
                    # Leyenda HTML personalizada
                    legend_html = '<div style="display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 15px; padding-left: 0px;">'
                    for t in time_intervals:
                        if travel_time >= t:
                            hex_color = TIME_PALETTE[t]["hex"]
                            legend_html += f'<div style="display: flex; align-items: center; gap: 6px;"><div style="width: 10px; height: 10px; background-color: {hex_color}; border-radius: 50%;"></div><span style="font-size: 0.9rem; color: rgba(255, 255, 255, 0.8); font-weight: 500;">{t} min.</span></div>'
                    legend_html += '</div>'
                    
                    st.markdown(legend_html, unsafe_allow_html=True)
                    st.altair_chart(vitality_chart, use_container_width=True)
                    
                else:
                    st.warning("No vitality profile data available for this node.")

    else:
        st.info("Waiting for node selection on the map...")