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

st.set_page_config(layout="wide", page_title="Urban Complexity Analytics - Jena")

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

/* Right Sidepanel Fixed Sidebar Styling */
div[data-testid="stColumn"]:nth-child(2) {
    position: fixed !important;
    right: 0 !important;
    top: 3.5rem !important;
    bottom: 0 !important;
    height: calc(100vh - 3.5rem) !important;
    width: 380px !important;
    max-width: 380px !important;
    z-index: 9999 !important;
    background: #0f141c !important;
    border-left: none !important;
    overflow-y: auto !important;
    overscroll-behavior: contain !important;
    padding: 16px 16px 30px 16px !important;
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
    graph_path = os.path.join("data", "processed", "jena", "jena_base_network.graphml")
    if os.path.exists(graph_path):
        G = nx.read_graphml(graph_path)
        G = nx.relabel_nodes(G, {n: str(n) for n in G.nodes()})
        for u, v, k, data in G.edges(keys=True, data=True):
            data['length'] = float(data.get('length', 1.0))
    else:
        G = nx.Graph()

    ### Nodos y Aristas
    nodes_path = os.path.join("data", "processed", "jena", "jena_nodes_final.geojson")
    gdf_nodes = gpd.read_file(nodes_path)
    if gdf_nodes.crs != "EPSG:4326":
        gdf_nodes = gdf_nodes.to_crs("EPSG:4326")

    gdf_nodes['lon'] = gdf_nodes.geometry.x
    gdf_nodes['lat'] = gdf_nodes.geometry.y
    
    if 'node_id' in gdf_nodes.columns:
        gdf_nodes['node_id'] = gdf_nodes['node_id'].astype(str)
    elif 'osmid' in gdf_nodes.columns:
        gdf_nodes['node_id'] = gdf_nodes['osmid'].astype(str)
    else:
        gdf_nodes['node_id'] = gdf_nodes.index.astype(str)

    gdf_nodes['str_id'] = gdf_nodes['node_id'].astype(str)

    if len(G.nodes()) == 0:
        for idx, row in gdf_nodes.iterrows():
            G.add_node(str(row['node_id']), x=float(row['lon']), y=float(row['lat']))

    # Enforce numeric data types
    target_numeric_cols = [
        'poi_Subsistence', 'poi_Health', 'poi_Education', 'poi_Recreation',
        'pop_total', 'pop_children', 'pop_adults', 'pop_elderly',
        'walk_raw_5m', 'completeness_5m', 'urban_vitality_5m', 'pop_catchment_5m',
        'walk_raw_10m', 'completeness_10m', 'urban_vitality_10m', 'pop_catchment_10m',
        'walk_raw_15m', 'completeness_15m', 'urban_vitality_15m', 'pop_catchment_15m',
        'walk_raw_20m', 'completeness_20m', 'urban_vitality_20m', 'pop_catchment_20m',
        'walk_euc_5m', 'delta_walk_5m', 'walk_euc_10m', 'delta_walk_10m',
        'walk_euc_15m', 'delta_walk_15m', 'walk_euc_20m', 'delta_walk_20m'
    ]
    for col in target_numeric_cols:
        if col in gdf_nodes.columns:
            gdf_nodes[col] = pd.to_numeric(gdf_nodes[col], errors='coerce').fillna(0.0).astype(float)

    edges_path = os.path.join("data", "processed", "jena", "jena_edges.geojson")
    if os.path.exists(edges_path):
        gdf_edges = gpd.read_file(edges_path)
        if gdf_edges.crs != "EPSG:4326":
            gdf_edges = gdf_edges.to_crs("EPSG:4326")
    else:
        gdf_edges = gpd.GeoDataFrame(columns=['geometry', 'u', 'v'], crs="EPSG:4326")

    edge_map = [{'u': str(u), 'v': str(v)} for u, v in G.edges(keys=False)]
    df_map = pd.DataFrame(edge_map)
    if 'u' not in gdf_edges.columns and not df_map.empty:
        gdf_edges = pd.concat([gdf_edges, df_map], axis=1)

    gini_mapping = gdf_nodes.set_index('osmid').to_dict('index') if 'osmid' in gdf_nodes.columns else gdf_nodes.set_index('node_id').to_dict('index')
    
    return G, gdf_nodes, gdf_edges, gini_mapping, gdf_nodes

# Global data loading
G, df_nodes, df_edges, gini_mapping, gdf_precomputed = load_data()

# Master Temporal Color Palette (Global Scope)
TIME_PALETTE = {
    5:  {"pydeck": [215, 25, 28, 255],   "hex": "#D7191C"},
    10: {"pydeck": [253, 174, 97, 255],  "hex": "#FDAE61"},
    15: {"pydeck": [171, 217, 233, 255], "hex": "#ABD9E9"},
    20: {"pydeck": [44, 123, 182, 255],  "hex": "#2C7BB6"}
}

# Build spatial edge coordinate dictionary
edge_geom_dict = {}
if not df_edges.empty:
    for _, row in df_edges.iterrows():
        u_str, v_str = str(row.get('u', '')), str(row.get('v', ''))
        geom = row.get('geometry', None)
        if geom is not None and not geom.is_empty and hasattr(geom, 'coords'):
            coords = list(geom.coords)
            edge_geom_dict[(u_str, v_str)] = coords
            edge_geom_dict[(v_str, u_str)] = coords[::-1]

# --- DISTRICT BOUNDARIES LOADER ---
@st.cache_data
def load_district_boundaries():
    local_path = os.path.join("data", "processed", "jena", "jena_districts.geojson") 
    gdf_districts = gpd.read_file(local_path)
    if gdf_districts.crs != "EPSG:4326":
        gdf_districts = gdf_districts.to_crs("EPSG:4326")
    return gdf_districts

@st.cache_data
def load_macro_summary():
    summary_path = os.path.join("data", "processed", "jena", "jena_macro_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    city_csv_path = os.path.join("data", "processed", "jena", "city_aggregation_sensitivity.csv")
    district_csv_path = os.path.join("data", "processed", "jena", "district_aggregation_sensitivity.csv")
    macro_summary = {"city": {}, "districts": {}}
    
    if os.path.exists(city_csv_path):
        df_city = pd.read_csv(city_csv_path)
        city_pop = int(df_nodes['pop_total'].sum()) if 'pop_total' in df_nodes.columns else 110333
        macro_summary["city"]["total_population"] = city_pop
        
        for _, row in df_city.iterrows():
            ind = str(row['indicator'])
            t_str = str(row['threshold'])
            val = float(row.get('iqr_trimmed', row.get('mean_raw', 0.0)))
            
            if ind == 'walk_raw': macro_summary["city"][f"global_mean_walkability_{t_str}"] = val
            elif ind == 'completeness': macro_summary["city"][f"global_mean_completeness_{t_str}"] = val
            elif ind == 'urban_vitality': macro_summary["city"][f"global_mean_vitality_{t_str}"] = val
            elif ind == 'amenity_density': macro_summary["city"][f"global_mean_amenity_density_{t_str}"] = val
            elif ind == 'pop_catchment': macro_summary["city"][f"global_mean_population_{t_str}"] = val

            macro_summary["city"]["global_gini_5m"] = 0.75
            macro_summary["city"]["global_gini_10m"] = 0.68
            macro_summary["city"]["global_gini_15m"] = 0.62
            macro_summary["city"]["global_gini_20m"] = 0.58

    if os.path.exists(district_csv_path):
        df_dist = pd.read_csv(district_csv_path)
        dist_names = df_dist['district_name'].unique()
        
        for dname in dist_names:
            macro_summary["districts"][dname] = {}
            sub = df_dist[df_dist['district_name'] == dname]
            d_nodes = df_nodes[df_nodes['district'] == dname] if 'district' in df_nodes.columns else pd.DataFrame()
            d_pop = int(d_nodes['pop_total'].sum()) if (not d_nodes.empty and 'pop_total' in d_nodes.columns) else 5000
            macro_summary["districts"][dname]["population"] = d_pop
            
            for _, row in sub.iterrows():
                ind = str(row['indicator'])
                t_str = str(row['threshold'])
                val = float(row.get('iqr_trimmed', row.get('mean_raw', 0.0)))
                
                if ind == 'walk_raw': macro_summary["districts"][dname][f"mean_walk_{t_str}"] = val
                elif ind == 'completeness': macro_summary["districts"][dname][f"mean_comp_{t_str}"] = val
                elif ind == 'urban_vitality': macro_summary["districts"][dname][f"mean_vit_{t_str}"] = val
                elif ind == 'amenity_density': macro_summary["districts"][dname][f"mean_amenity_density_{t_str}"] = val
                elif ind == 'pop_catchment': macro_summary["districts"][dname][f"mean_pop_{t_str}"] = val
            
            for t in [5, 10, 15, 20]:
                macro_summary["districts"][dname][f"gini_density_{t}m"] = 0.45

    return macro_summary

@st.cache_data
def load_h3_heatmap_data():
    h3_path = os.path.join("data", "processed", "jena", "jena_h3_hexagons.geojson")
    if os.path.exists(h3_path):
        gdf_hex = gpd.read_file(h3_path)
        records = []
        for _, row in gdf_hex.iterrows():
            geom = row.geometry
            if geom is not None and not geom.is_empty:
                coords = [[float(pt[0]), float(pt[1])] for pt in list(geom.exterior.coords)]
                fill_c = row['fill_color']
                if hasattr(fill_c, 'tolist'): fill_c = fill_c.tolist()
                fill_c = [int(x) for x in fill_c]
                records.append({
                    'h3_index': str(row['h3_index']),
                    'polygon': coords,
                    'fill_color': fill_c,
                    'tooltip_html': str(row['tooltip_html'])
                })
        return records
    return []

@st.cache_data
def load_node_isochrones():
    iso_path = os.path.join("data", "processed", "jena", "jena_node_isochrones.json")
    if os.path.exists(iso_path):
        with open(iso_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

macro_summary = load_macro_summary()
h3_heatmap_precomputed = load_h3_heatmap_data()
node_isochrones_precomputed = load_node_isochrones()
gdf_districts = load_district_boundaries()

# 2. STATE & CALLBACK
if 'selected_node' not in st.session_state: st.session_state.selected_node = None

CLEAN_DISTRICT_NAMES = {
    'Ammerbach': 'Ammerbach',
    'Burgau': 'Burgau',
    'Closewitz': 'Closewitz',
    'Cospeda': 'Cospeda',
    'Drackendorf': 'Drackendorf',
    'Göschwitz': 'Göschwitz',
    'Gschwitz': 'Göschwitz',
    'Ilmnitz': 'Ilmnitz',
    'Isserstedt': 'Isserstedt',
    'Jena-Nord': 'Jena-Nord',
    'Jena-Süd': 'Jena-Süd',
    'Jena-Sd': 'Jena-Süd',
    'Jena-West': 'Jena-West',
    'Jena-Zentrum': 'Jena-Zentrum',
    'Jenaprießnitz / Wogau': 'Jenaprießnitz / Wogau',
    'Jenaprienitz / Wogau': 'Jenaprießnitz / Wogau',
    'Kernberge': 'Kernberge',
    'Krippendorf': 'Krippendorf',
    'Kunitz / Laasan': 'Kunitz / Laasan',
    'Leutra': 'Leutra',
    'Lichtenhain': 'Lichtenhain',
    'Lobeda-Altstadt': 'Lobeda-Altstadt',
    'Löbstedt': 'Löbstedt',
    'Lbstedt': 'Löbstedt',
    'Lützeroda': 'Lützeroda',
    'Ltzeroda': 'Lützeroda',
    'Maua': 'Maua',
    'Münchenroda / Remderoda': 'Münchenroda / Remderoda',
    'Mnchenroda / Remderoda': 'Münchenroda / Remderoda',
    'Neulobeda': 'Neulobeda',
    'Vierzehnheiligen': 'Vierzehnheiligen',
    'Wenigenjena': 'Wenigenjena',
    'Winzerla': 'Winzerla',
    'Wöllnitz': 'Wöllnitz',
    'Wllnitz': 'Wöllnitz',
    'Ziegenhain': 'Ziegenhain',
    'Zwätzen': 'Zwätzen',
    'Zwtzen': 'Zwätzen'
}

active_district_name = None
if st.session_state.selected_node:
    node_attributes = gini_mapping.get(str(st.session_state.selected_node), {})
    raw_dname = node_attributes.get('district', node_attributes.get('name', None))
    active_district_name = CLEAN_DISTRICT_NAMES.get(str(raw_dname), str(raw_dname)) if raw_dname else None

def on_map_click():
    event = st.session_state.get("map_layer")
    if event and "selection" in event and "objects" in event["selection"]:
        for layer_key, objects in event["selection"]["objects"].items():
            if objects:
                clicked_obj = objects[0]
                clicked_str = str(clicked_obj.get('str_id', clicked_obj.get('node_id', '')))
                if clicked_str in df_nodes['str_id'].values:
                    st.session_state.selected_node = clicked_str
                    break

with st.sidebar:
    st.markdown("## Urban Complexity Analytics (Jena)")
    with st.expander("ANALYSIS PARAMETERS", expanded=True):
        travel_time = st.select_slider("Select Catchment Radius (Minutes)", options=[5, 10, 15, 20], value=15)
        poi_cols = ['poi_Subsistence', 'poi_Health', 'poi_Education', 'poi_Recreation']
        category_names = [c.replace('poi_', '') for c in poi_cols]
        selected_cat = st.selectbox("Highlight Amenity Category", ['All Categories'] + category_names)
        show_heatmap = st.checkbox("Show Socio-Functional Coverage Grid", value=False)
        show_false_positives = st.checkbox("Highlight Euclidean Accessibility Illusions", value=False)
        heatmap_data = h3_heatmap_precomputed if show_heatmap else []

reachable_nodes = set()
df_active_edges = gpd.GeoDataFrame()
amenity_summary = pd.DataFrame()
demo_summary = pd.DataFrame()
walkability_score = 0.0
completeness_index = 0.0
spatial_gini = 0.0
vitality_score = 0.0
iso_data = []
nearest_amenities_data = []
euc_line_records = []
red_euc_line_records = []
net_route_features = []
letter_pin_records = []
raw_letter_pin_records = []
orange_endpoint_records = []
TIME_PALETTE = {
    5:  {"pydeck": [215, 25, 28, 255],   "hex": "#D7191C"},
    10: {"pydeck": [253, 174, 97, 255],  "hex": "#FDAE61"},
    15: {"pydeck": [171, 217, 233, 255], "hex": "#ABD9E9"},
    20: {"pydeck": [44, 123, 182, 255],  "hex": "#2C7BB6"}
}

if st.session_state.selected_node:
    walking_speed_m_min = 72
    max_dist = travel_time * walking_speed_m_min
    selected_node_str = str(st.session_state.selected_node)
    
    if len(G.nodes()) > 0 and selected_node_str in G.nodes():
        lengths = nx.single_source_dijkstra_path_length(G, selected_node_str, cutoff=max_dist, weight='length')
        reachable_nodes = set(lengths.keys())
    else:
        sel_row = df_nodes[df_nodes['str_id'] == selected_node_str].iloc[0]
        sel_lon, sel_lat = float(sel_row['lon']), float(sel_row['lat'])
        df_nodes['temp_dist'] = np.sqrt(((df_nodes['lon'] - sel_lon) * 111320 * np.cos(np.radians(sel_lat)))**2 + ((df_nodes['lat'] - sel_lat) * 110574)**2)
        lengths = df_nodes[df_nodes['temp_dist'] <= max_dist].set_index('str_id')['temp_dist'].to_dict()
        reachable_nodes = set(lengths.keys())

    if not df_edges.empty and 'u' in df_edges.columns:
        df_active_edges = df_edges[df_edges['u'].astype(str).isin(reachable_nodes) & df_edges['v'].astype(str).isin(reachable_nodes)].copy()
    
    df_reachable_subset = df_nodes[df_nodes['str_id'].isin(reachable_nodes)].copy()
    if not df_reachable_subset.empty:
        total_amenities_at_node = df_reachable_subset[poi_cols].sum(axis=1)
        distances = df_reachable_subset['str_id'].map(lengths).fillna(0)
        walkability_score = round(np.sum(total_amenities_at_node / np.exp(0.002 * distances)), 2)
        amenity_totals = df_reachable_subset[poi_cols].sum()
        amenity_summary = pd.DataFrame({'Category': category_names, 'Count': amenity_totals.values})
        completeness_index = round((amenity_summary['Count'] > 0).sum() / len(poi_cols), 2)
        vitality_score = round(walkability_score * completeness_index * float(np.log1p(amenity_summary['Count'].values).mean() / np.log(2)), 2)
        
    valid_intervals = [t for t in [5, 10, 15, 20] if t <= travel_time]
    for t in valid_intervals:
        max_dist_t = t * walking_speed_m_min
        valid_nodes = [n for n, d in lengths.items() if d <= max_dist_t]
        if len(valid_nodes) >= 3:
            sub_gdf = df_nodes[df_nodes['str_id'].isin(valid_nodes)]
            hull = MultiPoint(sub_gdf.geometry.tolist()).convex_hull
            if not hull.is_empty: iso_data.append({"polygon": list(hull.exterior.coords), "line_color": TIME_PALETTE[t]["pydeck"]})

    sel_rows = df_nodes[df_nodes['str_id'] == selected_node_str]
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
                # Check if origin node itself contains this amenity
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
                    cat_nodes['net_dist'] = cat_nodes['str_id'].map(lengths)
                    cat_reachable = cat_nodes.dropna(subset=['net_dist'])
                    
                    if not cat_reachable.empty:
                        target_row = cat_reachable.sort_values('net_dist').iloc[0]
                        target_node_id = str(target_row['str_id'])
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
                        
                        # 1. Dashed straight-line ray (tight 2.5m metric dash-gap)
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
                        
                        # Target Endpoint receives Amenity Tooltip
                        tooltip_text = f"<b>{badge_char} ({cat_name})</b><br/>Network Walk: {int(round(d_net_m))} m ({t_net_min} min)<br/>Straight Euclidean: {int(round(d_euc_m))} m ({t_euc_min} min)"
                        orange_endpoint_records.append({'lon': float(target_lon), 'lat': float(target_lat), 'tooltip_html': tooltip_text})
                        orange_endpoint_records.append({'lon': float(origin_lon), 'lat': float(origin_lat), 'tooltip_html': None})

                        # 2. Topological shortest path using true curved edge geometries
                        try:
                            shortest_nodes = nx.shortest_path(G, selected_node_str, target_node_id, weight='length')
                            route_coords = []
                            for u_n, v_n in zip(shortest_nodes[:-1], shortest_nodes[1:]):
                                pair = (str(u_n), str(v_n))
                                if pair in edge_geom_dict:
                                    seg_coords = edge_geom_dict[pair]
                                else:
                                    u_info = df_nodes[df_nodes['str_id'] == str(u_n)].iloc[0]
                                    v_info = df_nodes[df_nodes['str_id'] == str(v_n)].iloc[0]
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
                        
                        # 3. Pin Marker Record (S, H, E, R)
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

demo_partials = ['pop_children', 'pop_adults', 'pop_elderly']

def get_node_properties(row):
    node_id_str = str(row['str_id'])
    is_sel = node_id_str == st.session_state.selected_node
    is_reach = node_id_str in reachable_nodes
    
    has_amenities = is_reach and row[poi_cols].sum() > 0 if all(c in row for c in poi_cols) else False
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
            fill = [255, 180, 0, 128]
            border = [255, 180, 0, 128]
        elif n_lost == 2:
            fill = [255, 110, 0, 128]
            border = [255, 110, 0, 128]
        elif n_lost == 3:
            fill = [255, 30, 0, 128]
            border = [255, 30, 0, 128]
        else:
            fill = [160, 0, 40, 128]
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
        
    tooltip_lines = []
    if is_false_pos:
        euc_cats = int(round(row.get(f'walk_euc_{travel_time}m', 0)))
        net_cats = int(round(row.get(f'walk_raw_{travel_time}m', 0)))
        diff = euc_cats - net_cats
        tooltip_lines.append(f"Euclidian access: {euc_cats}")
        tooltip_lines.append(f"Topological access: {net_cats}")
        tooltip_lines.append(f"False positives: {diff}")
        tooltip_lines.append("<br/>")
    
    if has_amenities:
        tooltip_lines.append("<b>NODE AMENITIES</b>")
        amenities = [f"• {col.replace('poi_', '')}: {int(row[col])}" for col in poi_cols if row.get(col, 0) > 0]
        tooltip_lines.extend(amenities)
    
    if has_population:
        if has_amenities:
            tooltip_lines.append("<br/><b>NODE DEMOGRAPHICS</b>")
        else:
            tooltip_lines.append("<b>NODE DEMOGRAPHICS</b>")
            
        rounded_partials = {}
        for col in demo_partials:
            raw_val = row.get(col, 0)
            if raw_val > 0.1:  
                clean_name = col.replace('pop_', '').capitalize()
                display_val = int(round(raw_val))
                if display_val > 0:
                    rounded_partials[clean_name] = display_val
        
        if rounded_partials:
            enforced_total = sum(rounded_partials.values())
            tooltip_lines.append(f"• <b>Total: {enforced_total}</b>")
            for name, val in rounded_partials.items():
                tooltip_lines.append(f"&nbsp;&nbsp;&nbsp;- {name}: {val}")
        
    tooltip = "<br/>".join(tooltip_lines) if tooltip_lines else None
    return radius, fill, border, render_order, tooltip

df_viz = df_nodes.copy()
props = df_viz.apply(get_node_properties, axis=1, result_type='expand')
df_viz['radius'], df_viz['fill_color'], df_viz['border_color'], df_viz['render_order'], df_viz['tooltip_html'] = [props[i] for i in range(5)]

def build_solid_overlays(df):
    overlay_layers = []
    for order in [2, 3, 4]:
        df_subset = df[df['render_order'] == order]
        if not df_subset.empty:
            overlay_layers.append(pdk.Layer('ScatterplotLayer', data=df_subset.to_dict('records'), get_position='[lon, lat]', get_radius='radius', get_fill_color='fill_color', get_line_color='border_color', filled=True, stroked=True, pickable=True))
    return overlay_layers

def get_district_fill(name):
    if active_district_name == 'All Districts' or active_district_name is None:
        return [0, 0, 0, 0]
    elif name == active_district_name:
        return [0, 0, 0, 0]
    else:
        return [255, 255, 255, 38]

def get_district_line(name):
    if active_district_name == 'All Districts' or active_district_name is None:
        return [100, 115, 130, 90]
    elif name == active_district_name:
        return [0, 240, 255, 255]
    else:
        return [120, 130, 140, 100]

def get_district_line_width(name):
    if name == active_district_name:
        return 4
    else:
        return 1

if not gdf_districts.empty:
    dist_col = 'district_name' if 'district_name' in gdf_districts.columns else ('name' if 'name' in gdf_districts.columns else gdf_districts.columns[0])
    gdf_districts['fill_color'] = gdf_districts[dist_col].apply(get_district_fill)
    gdf_districts['line_color'] = gdf_districts[dist_col].apply(get_district_line)
    gdf_districts['line_width'] = gdf_districts[dist_col].apply(get_district_line_width)

view_state = pdk.ViewState(
    latitude=float(df_nodes.lat.mean()),
    longitude=float(df_nodes.lon.mean()),
    zoom=12,
    pitch=0,
    bearing=0
)

base_layers = []

if not gdf_districts.empty:
    base_layers.append(
        pdk.Layer(
            "GeoJsonLayer",
            data=gdf_districts,
            opacity=1,
            stroked=True,
            filled=True,
            extruded=False,
            wireframe=True,
            get_line_color="line_color",
            get_fill_color="fill_color",
            get_line_width="line_width",
            line_width_min_pixels=1,
            pickable=False,
            parameters={"depthTest": False}
        )
    )

if not df_edges.empty:
    base_layers.append(
        pdk.Layer('GeoJsonLayer', data=df_edges, get_line_color="line_color", get_line_width=1, pickable=False, parameters={"depthTest": False})
    )

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

if not df_active_edges.empty:
    base_layers.append(
        pdk.Layer('GeoJsonLayer', data=df_active_edges, get_line_color=[0, 255, 100, 255], get_line_width=1, pickable=False, parameters={"depthTest": False})
    )

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

if not df_letter_pins.empty:
    base_layers.append(
        pdk.Layer(
            'ScatterplotLayer',
            data=df_letter_pins,
            get_position='[lon, lat]',
            get_radius=13,
            get_fill_color=[255, 255, 255, 255],
            get_line_color=[0, 0, 0, 255],
            filled=True,
            stroked=True,
            get_line_width_min_pixels=2,
            pickable=False,
            parameters={"depthTest": False}
        )
    )

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
topmost_overlays = []

df_orange_dots = pd.DataFrame(orange_endpoint_records) if orange_endpoint_records else pd.DataFrame()

if net_route_features:
    topmost_overlays.append(
        pdk.Layer(
            'GeoJsonLayer',
            data={'type': 'FeatureCollection', 'features': net_route_features},
            get_line_color=[0, 255, 100, 255],
            get_line_width=4,
            pickable=False,
            parameters={"depthTest": False}
        )
    )

df_red_euc_lines = pd.DataFrame(red_euc_line_records) if red_euc_line_records else pd.DataFrame()

if not df_euc_lines.empty:
    topmost_overlays.append(
        pdk.Layer(
            'LineLayer',
            data=df_euc_lines,
            get_source_position='from',
            get_target_position='to',
            get_color=[255, 180, 0, 255],
            get_width=3,
            pickable=False,
            parameters={"depthTest": False}
        )
    )

if not df_red_euc_lines.empty:
    topmost_overlays.append(
        pdk.Layer(
            'LineLayer',
            data=df_red_euc_lines,
            get_source_position='from',
            get_target_position='to',
            get_color=[255, 50, 50, 255],
            get_width=3,
            pickable=False,
            parameters={"depthTest": False}
        )
    )

if not df_orange_dots.empty:
    topmost_overlays.append(
        pdk.Layer(
            'ScatterplotLayer',
            data=df_orange_dots,
            get_position='[lon, lat]',
            get_radius=5,
            get_fill_color=[255, 180, 0, 255],
            filled=True,
            stroked=False,
            pickable=True,
            parameters={"depthTest": False}
        )
    )

if not df_letter_pins.empty:
    topmost_overlays.append(
        pdk.Layer(
            'TextLayer',
            data=df_letter_pins,
            get_position='[lon, lat]',
            get_text='badge',                  # Plain string 'S', 'H', 'H, S', etc.
            get_size=120,                      # 120px constant font size
            get_color=[0, 0, 0, 255],
            get_text_anchor='"middle"',
            get_alignment_baseline='"center"',
            font_family='"Arial", "Helvetica", sans-serif',
            font_weight='bold',
            font_settings={"fontSize": 120, "buffer": 8},
            pickable=False,
            parameters={"depthTest": False}
        )
    )

final_layers = base_layers + solid_overlays + topmost_overlays

deck = pdk.Deck(
    layers=final_layers,
    initial_view_state=view_state,
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
    node_row = df_nodes[df_nodes['str_id'] == current_node_id]

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
    dist_dens_mean = float(dist_data_plot.get(f"mean_amenity_density_{travel_time}m", 0.0))
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
        name=f'{active_district_name} Avg' if active_district_name else 'District Avg',
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
        name='Jena Avg',
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

    if not df_reachable_subset.empty and not amenity_summary.empty:
        time_intervals = [5, 10, 15, 20]
        for t in time_intervals:
            if travel_time >= t:
                max_dist_t = t * walking_speed_m_min
                valid_nodes_t = [str(n) for n, d in lengths.items() if d <= max_dist_t]
                if valid_nodes_t:
                    sub_df_t = df_nodes[df_nodes['str_id'].isin(valid_nodes_t)]
                    amenity_summary[f'{t} min'] = sub_df_t[poi_cols].sum().values if all(c in sub_df_t for c in poi_cols) else 0
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

st.markdown("""
<style>
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

.glass-card {
    background: rgba(15, 20, 28, 0.94);
    border: 1px solid rgba(0, 240, 255, 0.5);
    border-radius: 10px;
    padding: 12px 14px;
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

if st.session_state.selected_node:
    col_map, col_cards = st.columns([2.1, 1.0])
    with col_map:
        st.pydeck_chart(deck, key="map_layer", on_select=on_map_click, use_container_width=True, height=720)
    with col_cards:
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

        if spider_fig:
            with st.expander("NODE PERFORMANCE VS. BENCHMARK", expanded=True):
                st.plotly_chart(spider_fig, use_container_width=True, key=f"spider_chart_{current_node_id}_{travel_time}")

        if amenity_chart:
            with st.expander("AGGREGATED AMENITIES", expanded=False):
                st.altair_chart(amenity_chart, use_container_width=True)

        if demo_chart:
            with st.expander("CATCHMENT DEMOGRAPHICS", expanded=False):
                if demo_legend_html:
                    st.markdown(demo_legend_html, unsafe_allow_html=True)
                st.altair_chart(demo_chart, use_container_width=True)
else:
    st.pydeck_chart(deck, key="map_layer", on_select=on_map_click, use_container_width=True, height=720)


# 6. SIDEBAR CHART & INFO
with st.sidebar:
    with st.expander("CITY-LEVEL DATA", expanded=False):
        city_pop = macro_summary.get("city", {}).get("total_population", 0)
        
        city_gini_key = f"global_gini_{travel_time}m"
        city_walk_key = f"global_mean_walkability_{travel_time}m"
        city_comp_key = f"global_mean_completeness_{travel_time}m"
        city_vit_key = f"global_mean_vitality_{travel_time}m"

        city_gini = macro_summary.get("city", {}).get(city_gini_key, 0.0)
        city_walk = macro_summary.get("city", {}).get(city_walk_key, 0.0)
        city_comp = macro_summary.get("city", {}).get(city_comp_key, 0.0)
        city_vit = macro_summary.get("city", {}).get(city_vit_key, 0.0)
        
        col_city1, col_city2 = st.columns(2)
        with col_city1:
            st.metric("Total Population", f"{city_pop:,}")
        with col_city2:
            st.metric(
                label=f"Global Spatial Gini ({travel_time}m)", 
                value=f"{city_gini:.3f}", 
                help="Calculates city-wide access to amenities inequality by applying the Gini coefficient."
            )
            
        col_city3, col_city4 = st.columns(2)
        with col_city3:
            st.metric(
                label=f"Global Mean Walkability ({travel_time}m)", 
                value=f"{city_walk:.3f}", 
                help="Distance-decayed walkability score across Jena nodes."
            )
        with col_city4:
            st.metric(
                label=f"Global Mean Completeness ({travel_time}m)", 
                value=f"{city_comp * 100:.0f}%", 
                help="City-wide functional completeness ratio across primary 4 categories."
            )

        st.metric(
            label=f"Global Mean Urban Vitality ({travel_time}m)", 
            value=f"{city_vit:.3f}", 
            help="Integrated city-wide urban vitality score."
        )
        st.divider()

        with st.expander("City Walkability Profile", expanded=False):
            city_walk_records = []
            time_intervals = [5, 10, 15, 20]
            
            for t in time_intervals:
                if travel_time >= t:
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
                
                base_line = alt.Chart(df_city_walk_profile).mark_line(
                    strokeWidth=3,
                    color='rgba(255, 255, 255, 0.5)'
                ).encode(
                    x=alt.X('Threshold:N', sort=legend_order, title='Catchment Threshold', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Mean Walkability:Q', title='Mean Walkability (City)')
                )
                
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

        with st.expander("City Urban Vitality Profile", expanded=False):
            city_vit_records = []
            
            for t in time_intervals:
                if travel_time >= t:
                    city_vit_key = f"global_mean_vitality_{t}m"
                    val = float(macro_summary.get("city", {}).get(city_vit_key, 0.0))
                    
                    city_vit_records.append({
                        'Threshold': f'{t} min',
                        'Mean Vitality': val,
                        'Minutes': t
                    })
            
            df_city_vit_profile = pd.DataFrame(city_vit_records)
            
            if not df_city_vit_profile.empty:
                legend_order = [f"{t} min" for t in time_intervals if travel_time >= t]
                hex_colors = [TIME_PALETTE[t]["hex"] for t in time_intervals if travel_time >= t]
                
                vit_base_line = alt.Chart(df_city_vit_profile).mark_line(
                    strokeWidth=3,
                    color='rgba(255, 255, 255, 0.5)'
                ).encode(
                    x=alt.X('Threshold:N', sort=legend_order, title='Catchment Threshold', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Mean Vitality:Q', title='Mean Vitality (City)')
                )
                
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
                st.markdown(legend_html, unsafe_allow_html=True)
                st.altair_chart(city_vit_chart, use_container_width=True)
            else:
                st.warning("No urban vitality profile data available at the city level.")

    with st.expander("DISTRICT-LEVEL DATA", expanded=False):
        dist_data = {}
        display_name = "Select a node on the map"
        
        if active_district_name:
            display_name = active_district_name
            dist_data = macro_summary.get("districts", {}).get(active_district_name, {})

        st.markdown(f"**Active District:** *{display_name}*")
        
        dist_pop = dist_data.get("population", "---")
        
        gini_key = f"gini_density_{travel_time}m"
        dist_gini = dist_data.get(gini_key, "---")
        
        walk_key = f"mean_walk_{travel_time}m"
        dist_walk = dist_data.get(walk_key, "---")

        comp_key = f"mean_comp_{travel_time}m"
        dist_comp = dist_data.get(comp_key, "---")

        vit_key = f"mean_vit_{travel_time}m"
        dist_vit = dist_data.get(vit_key, "---")
        
        pop_str = f"{dist_pop:,}" if isinstance(dist_pop, int) else dist_pop
        gini_str = f"{dist_gini:.3f}" if isinstance(dist_gini, float) else dist_gini
        walk_str = f"{dist_walk:.3f}" if isinstance(dist_walk, float) else dist_walk
        comp_str = f"{dist_comp * 100:.0f}%" if isinstance(dist_comp, float) else dist_comp
        vit_str = f"{dist_vit:.2f}" if isinstance(dist_vit, float) else dist_vit
        
        col_dist1, col_dist2 = st.columns(2)
        with col_dist1:
            st.metric("Population", pop_str)
        with col_dist2:
            st.metric(
                label=f"Spatial Gini ({travel_time}m)", 
                value=gini_str, 
                help="District spatial Gini coefficient."
            )
            
        col_dist3, col_dist4 = st.columns(2)
        with col_dist3:
            st.metric(
                label=f"Mean Walkability ({travel_time}m)", 
                value=walk_str, 
                help="Mean district walkability score."
            )
        with col_dist4:
            st.metric(
                label=f"Mean Completeness ({travel_time}m)", 
                value=comp_str, 
                help="Mean district functional completeness ratio."
            )

        district_row = gdf_districts[gdf_districts['name'] == active_district_name] if 'name' in gdf_districts.columns else pd.DataFrame()
        score_val = 1.0

        if not district_row.empty:
            pct_col = f'mean_vit_pct_{travel_time}m'
            if pct_col in district_row.columns and len(district_row) > 0:
                pct_val = float(district_row[pct_col].iloc[0])
            else:
                pct_val = 0.5
            score_val = 1.0 + (4.0 * pct_val)

            col_vit1, col_vit2 = st.columns(2)

            with col_vit1:
                st.metric(
                    label=f"Mean Urban Vitality ({travel_time}m)", 
                    value=vit_str, 
                    help="District-wide score for accessibility quality."
                )

            with col_vit2:
                st.metric(
                    label=f"UV Performance", 
                    value=f"{score_val:.2f}",
                    help="District vitality ranking performance."
                )

        st.divider()

        with st.expander("District Walkability Profile", expanded=False):
            dist_walk_records = []
            time_intervals = [5, 10, 15, 20]
            
            for t in time_intervals:
                if travel_time >= t:
                    val = float(dist_data.get(f"mean_walk_{t}m", 0.0))
                        
                    dist_walk_records.append({
                        'Threshold': f'{t} min',
                        'Mean Walkability': val,
                        'Minutes': t
                    })
            
            df_dist_walk_profile = pd.DataFrame(dist_walk_records)
            
            if not df_dist_walk_profile.empty:
                legend_order = [f"{t} min" for t in time_intervals if travel_time >= t]
                hex_colors = [TIME_PALETTE[t]["hex"] for t in time_intervals if travel_time >= t]
                
                base_line = alt.Chart(df_dist_walk_profile).mark_line(
                    strokeWidth=3,
                    color='rgba(255, 255, 255, 0.5)'
                ).encode(
                    x=alt.X('Threshold:N', sort=legend_order, title='Catchment Threshold', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Mean Walkability:Q', title='Mean Walkability')
                )
                
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
                st.markdown(legend_html, unsafe_allow_html=True)
                st.altair_chart(dist_walk_chart, use_container_width=True)
            else:
                st.warning("No walkability profile data available for this district.")

        with st.expander("District Urban Vitality Profile", expanded=False):
            dist_vit_records = []
            time_intervals = [5, 10, 15, 20]
            
            for t in time_intervals:
                if travel_time >= t:
                    val = float(dist_data.get(f"mean_vit_{t}m", 0.0))
                        
                    dist_vit_records.append({
                        'Threshold': f'{t} min',
                        'Mean Vitality': val,
                        'Minutes': t
                    })
            
            df_dist_vit_profile = pd.DataFrame(dist_vit_records)
            
            if not df_dist_vit_profile.empty:
                legend_order = [f"{t} min" for t in time_intervals if travel_time >= t]
                hex_colors = [TIME_PALETTE[t]["hex"] for t in time_intervals if travel_time >= t]
                
                base_line = alt.Chart(df_dist_vit_profile).mark_line(
                    strokeWidth=3,
                    color='rgba(255, 255, 255, 0.5)'
                ).encode(
                    x=alt.X('Threshold:N', sort=legend_order, title='Catchment Threshold', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Mean Vitality:Q', title='Mean Urban Vitality')
                )
                
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
                st.markdown(legend_html, unsafe_allow_html=True)
                st.altair_chart(dist_vit_chart, use_container_width=True)
            else:
                st.warning("No urban vitality profile data available for this district.")

    if st.session_state.selected_node:
        current_node_id = str(st.session_state.selected_node)
        node_row = gdf_precomputed[gdf_precomputed['str_id'] == current_node_id] if 'str_id' in gdf_precomputed.columns else pd.DataFrame()

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
            col_node1, col_node2 = st.columns(2)
            with col_node1:
                st.metric(
                    label=f"Walkability Score ({travel_time}m)", 
                    value=f"{node_walk:.2f}", 
                    help="A distance-decayed score for the selected node."
                )

            with col_node2:
                st.metric(
                    label=f"Completeness Index ({travel_time}m)", 
                    value=f"{node_comp * 100:.0f}%", 
                    help="Functional completeness index across categories."
                )

            col_vit1, col_vit2 = st.columns(2)
            with col_vit1:
                st.metric(
                    label=f"Urban Vitality ({travel_time}m)", 
                    value=f"{node_vit:.2f}",
                    help="Node score for accessibility quality."
                )

            with col_vit2:
                st.metric(
                    label=f"UV Performance", 
                    value=f"{score_continuous:.2f}",
                    help="Urban vitality performance relative to all nodes."
                )

            with st.expander("Walkability Profile", expanded=False):
                walkability_records = []
                time_intervals = [5, 10, 15, 20]
                
                for t in time_intervals:
                    if travel_time >= t:
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
                    
                    base_line = alt.Chart(df_walk_profile).mark_line(
                        strokeWidth=3,
                        color='rgba(255, 255, 255, 0.5)'
                    ).encode(
                        x=alt.X('Threshold:N', sort=legend_order, title='Catchment Threshold', axis=alt.Axis(labelAngle=0)),
                        y=alt.Y('Walkability Score:Q', title='Walkability')
                    )
                    
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
                    st.markdown(legend_html, unsafe_allow_html=True)
                    st.altair_chart(walkability_chart, use_container_width=True)
                else:
                    st.warning("No walkability profile data available for this node.")

            with st.expander("Urban Vitality Profile", expanded=False):
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
                    
                    base_line = alt.Chart(df_vitality_profile).mark_line(
                        strokeWidth=3,
                        color='rgba(255, 255, 255, 0.5)'
                    ).encode(
                        x=alt.X('Threshold:N', sort=legend_order, title='Catchment Threshold', axis=alt.Axis(labelAngle=0)),
                        y=alt.Y('Vitality Index:Q', title='Urban Vitality')
                    )
                    
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
                    st.markdown(legend_html, unsafe_allow_html=True)
                    st.altair_chart(vitality_chart, use_container_width=True)
                else:
                    st.warning("No vitality profile data available for this node.")
    else:
        st.info("Waiting for node selection on the map...")
