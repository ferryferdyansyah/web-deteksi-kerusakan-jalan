# map_tab.py - CLEANED VERSION
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import numpy as np
import geopandas as gpd
import zipfile
import os
import tempfile
from shapely.geometry import Point
import json

def calculate_zoom_from_bounds(bounds):
    """Calculate appropriate zoom level from bounds"""
    # bounds = [minx, miny, maxx, maxy]
    lat_range = bounds[3] - bounds[1]  # max_lat - min_lat
    lon_range = bounds[2] - bounds[0]  # max_lon - min_lon
    
    # Use the maximum range to determine zoom
    max_range = max(lat_range, lon_range)
    
    # Zoom level mapping based on coordinate range
    if max_range <= 0.001:  # Very small area
        return 18
    elif max_range <= 0.005:
        return 16
    elif max_range <= 0.01:
        return 15
    elif max_range <= 0.05:
        return 17
    elif max_range <= 0.1:
        return 12
    elif max_range <= 0.5:
        return 10
    elif max_range <= 1.0:
        return 9
    elif max_range <= 2.0:
        return 8
    elif max_range <= 5.0:
        return 7
    else:
        return 6

def load_shapefile(uploaded_file):
    """Load shapefile from uploaded zip file"""
    try:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract zip file
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find .shp file
            shp_files = [f for f in os.listdir(temp_dir) if f.endswith('.shp')]
            if not shp_files:
                return None, "Tidak ditemukan file .shp dalam zip"
            
            shp_path = os.path.join(temp_dir, shp_files[0])
            
            # Read shapefile
            gdf = gpd.read_file(shp_path)
            
            # Check if GeoDataFrame is empty
            if len(gdf) == 0:
                return None, "Shapefile kosong atau tidak memiliki data"
            
            # Check if geometry is valid
            if gdf.geometry.isnull().all():
                return None, "Shapefile tidak memiliki geometri yang valid"
            
            # Ensure CRS is WGS84
            if gdf.crs is None:
                gdf = gdf.set_crs('EPSG:4326')
                st.warning("CRS tidak ditemukan, menggunakan WGS84 (EPSG:4326)")
            elif gdf.crs.to_epsg() != 4326:
                original_crs = gdf.crs.to_epsg()
                gdf = gdf.to_crs('EPSG:4326')
                st.info(f"CRS dikonversi dari EPSG:{original_crs} ke WGS84 (EPSG:4326)")
            
            return gdf, None
            
    except Exception as e:
        return None, f"Error loading shapefile: {str(e)}"

def check_points_in_aoi(gps_locations, aoi_gdf):
    """Check which GPS points are inside the AOI"""
    if aoi_gdf is None or gps_locations is None:
        return gps_locations
    
    try:
        # Create points from GPS locations
        points = [Point(loc['longitude'], loc['latitude']) for loc in gps_locations]
        
        # Check which points are within AOI
        for i, (point, loc) in enumerate(zip(points, gps_locations)):
            within_aoi = any(aoi_gdf.geometry.contains(point))
            gps_locations[i]['within_aoi'] = within_aoi
            
        return gps_locations
        
    except Exception as e:
        st.error(f"Error checking points in AOI: {str(e)}")
        return gps_locations

def add_aoi_to_map(folium_map, aoi_gdf):
    """Add AOI polygons to the folium map"""
    if aoi_gdf is None:
        return folium_map
    
    try:
        # Add AOI polygons to map
        for idx, row in aoi_gdf.iterrows():
            try:
                # Get geometry bounds for debugging
                bounds = row.geometry.bounds
                
                # Convert geometry to GeoJSON
                if hasattr(row.geometry, '__geo_interface__'):
                    geom = row.geometry.__geo_interface__
                else:
                    # Alternative method using json
                    geom = json.loads(row.geometry.to_json())
                
                # Create popup content for AOI
                aoi_popup = f"""
                <div style="font-family: Arial, sans-serif; max-width: 250px;">
                    <h4 style="color: #1f77b4; margin: 0 0 10px 0;">Area of Interest {idx + 1}</h4>
                """
                
                # Add attribute information if available
                for col in aoi_gdf.columns:
                    if col != 'geometry' and pd.notna(row[col]):
                        aoi_popup += f"<div style='margin: 2px 0;'><strong>{col}:</strong> {row[col]}</div>"
                
                # Add geometry info
                geom_type = row.geometry.geom_type
                if geom_type == 'Polygon':
                    area_approx = row.geometry.area * 111000 * 111000  # Rough conversion to m²
                    aoi_popup += f"<div style='margin: 2px 0;'><strong>Type:</strong> {geom_type}</div>"
                    aoi_popup += f"<div style='margin: 2px 0;'><strong>Area (approx):</strong> {area_approx:.0f} m²</div>"
                
                aoi_popup += "</div>"
                
                # Add polygon to map with distinct styling
                folium.GeoJson(
                    geom,
                    popup=folium.Popup(aoi_popup, max_width=300),
                    tooltip=f"AOI {idx + 1} ({geom_type})",
                    style_function=lambda feature, idx=idx: {
                        'fillColor': '#ff7800' if idx % 2 == 0 else '#1f77b4',
                        'color': '#ff7800' if idx % 2 == 0 else '#1f77b4',
                        'weight': 3,
                        'fillOpacity': 0.1,
                        'opacity': 0.8,
                        'dashArray': '10, 5'
                    }
                ).add_to(folium_map)
                
            except Exception as e:
                st.error(f"Error adding AOI {idx + 1} to map: {str(e)}")
                continue
                
        return folium_map
        
    except Exception as e:
        st.error(f"Error processing AOI data: {str(e)}")
        return folium_map

def create_detection_map(locations_data, aoi_gdf=None):
    """Create a map with detection markers and detailed popups"""
    default_lat, default_lon = -7.7956, 110.3695
    
    # Determine map center and zoom - prioritize AOI
    if aoi_gdf is not None and not aoi_gdf.empty:
        # Use AOI bounds for center and zoom
        bounds = aoi_gdf.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        zoom_level = calculate_zoom_from_bounds(bounds)
    elif locations_data:
        # Filter locations that have GPS data
        gps_locations = [loc for loc in locations_data if 'latitude' in loc and 'longitude' in loc]
        
        if gps_locations:
            # Calculate center point for multiple locations
            lats = [loc['latitude'] for loc in gps_locations]
            lons = [loc['longitude'] for loc in gps_locations]
            center_lat = np.mean(lats)
            center_lon = np.mean(lons)
            
            # Calculate zoom based on GPS data spread
            if len(gps_locations) == 1:
                zoom_level = 15
            else:
                lat_range = max(lats) - min(lats)
                lon_range = max(lons) - min(lons)
                gps_bounds = [min(lons), min(lats), max(lons), max(lats)]
                zoom_level = calculate_zoom_from_bounds(gps_bounds)
        else:
            center_lat, center_lon = default_lat, default_lon
            zoom_level = 12
    else:
        # Default location (Yogyakarta)
        center_lat, center_lon = default_lat, default_lon
        zoom_level = 12

    # Create the map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_level)
    
    # Add AOI layer first (so it appears behind markers)
    if aoi_gdf is not None:
        m = add_aoi_to_map(m, aoi_gdf)

    # Handle case with no location data
    if not locations_data:
        if aoi_gdf is None:
            folium.Marker(
                [center_lat, center_lon], 
                popup="Yogyakarta (Default Location)", 
                icon=folium.Icon(color='gray', icon='info-sign')
            ).add_to(m)
        return m

    # Filter locations that have GPS data
    gps_locations = [loc for loc in locations_data if 'latitude' in loc and 'longitude' in loc]
    
    if not gps_locations:
        if aoi_gdf is None:
            folium.Marker(
                [center_lat, center_lon], 
                popup="No GPS Data Available", 
                icon=folium.Icon(color='gray', icon='info-sign')
            ).add_to(m)
        return m

    # Check if points are within AOI
    if aoi_gdf is not None:
        gps_locations = check_points_in_aoi(gps_locations, aoi_gdf)

    # Add markers for each location with detailed popups
    for idx, loc in enumerate(gps_locations):
        lat = loc['latitude']
        lon = loc['longitude']
        
        # Create detailed popup content
        popup_html = f"""
        <div style="width: 300px; font-family: Arial, sans-serif;">
            <h4 style="margin: 0 0 10px 0; color: #d32f2f;">Detection: {loc.get('filename', 'Unknown File')}</h4>
            
            <div style="margin-bottom: 8px;">
                <strong>Coordinates:</strong><br>
                Lat: {lat:.6f}<br>
                Lon: {lon:.6f}
            </div>
            
            <div style="margin-bottom: 8px;">
                <strong>Detections:</strong> {loc.get('detections', 0)} objects
            </div>
            
            <div style="margin-bottom: 8px;">
                <strong>Image Size:</strong> {loc.get('image_size', 'N/A')}
            </div>
            
            <div style="margin-bottom: 8px;">
                <strong>Process Time:</strong><br>{loc.get('timestamp', 'N/A')}
            </div>
        """
        
        # Add AOI status if AOI is loaded
        if aoi_gdf is not None:
            within_aoi_status = "Inside AOI" if loc.get('within_aoi', False) else "Outside AOI"
            popup_html += f"""
            <div style="margin-bottom: 8px;">
                <strong>AOI Status:</strong> {within_aoi_status}
            </div>
            """
        
        # Add detection details if available
        if 'detection_details' in loc and loc['detection_details']:
            popup_html += """
            <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid #ccc;">
                <strong>Detection Details:</strong><br>
            """
            for i, detail in enumerate(loc['detection_details'][:3]):  # Show max 3 detections in popup
                confidence_percent = detail['confidence'] * 100
                popup_html += f"""
                <div style="margin: 4px 0; font-size: 11px;">
                    {i+1}. <strong>{detail['class_name']}</strong><br>
                    &nbsp;&nbsp;&nbsp;Confidence: {confidence_percent:.1f}%<br>
                    &nbsp;&nbsp;&nbsp;BBox: {detail['bbox_formatted']}
                </div>
                """
            
            if len(loc['detection_details']) > 3:
                remaining = len(loc['detection_details']) - 3
                popup_html += f"<div style='font-size: 11px; color: #666;'>... and {remaining} more detections</div>"
        
        # Add GPS metadata if available
        if loc.get('altitude') != 'N/A' and loc.get('altitude'):
            popup_html += f"""
            <div style="margin-top: 8px;">
                <strong>Altitude:</strong> {loc.get('altitude')}
            </div>
            """
        
        if loc.get('timestamp') != 'N/A' and 'gps_timestamp' in loc:
            popup_html += f"""
            <div style="margin-top: 4px;">
                <strong>GPS Time:</strong> {loc.get('gps_timestamp')}
            </div>
            """
        
        popup_html += "</div>"
        
        # Choose marker color based on number of detections and AOI status
        detections_count = loc.get('detections', 0)
        within_aoi = loc.get('within_aoi', True)  # Default True if no AOI
        
        if not within_aoi:
            # Gray for points outside AOI
            marker_color = 'gray'
            icon_name = 'map-marker'
        elif detections_count == 0:
            marker_color = 'green'
            icon_name = 'ok-sign'
        elif detections_count <= 2:
            marker_color = 'orange'
            icon_name = 'warning-sign'
        else:
            marker_color = 'red'
            icon_name = 'exclamation-sign'
        
        # Add marker to map
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"Image: {loc.get('filename', 'Unknown')} - {detections_count} detections",
            icon=folium.Icon(
                color=marker_color, 
                icon=icon_name,
                prefix='glyphicon'
            )
        ).add_to(m)
        
        # Add a circle to highlight the area
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            popup=f"Area {idx + 1}",
            color=marker_color,
            fill=True,
            fillColor=marker_color,
            fillOpacity=0.3,
            weight=2
        ).add_to(m)

    return m

def map_tab():
    """Main map tab function - this is what gets imported"""
    
    # AOI Upload Section
    st.markdown("### Area of Interest (AOI)")
    uploaded_shp = st.file_uploader(
        "Upload AOI Shapefile (.zip containing .shp, .shx, .dbf files)",
        type=['zip'],
        key='shapefile_uploader',
        help="Upload zip file containing shapefile (.shp, .shx, .dbf, .prj)"
    )
    
    # Buttons below file uploader
    col1, col2, col3 = st.columns([1, 1, 2])
    
    aoi_gdf = None
    
    if uploaded_shp is not None:
        with col1:
            if st.button("Load Shapefile", use_container_width=True):
                with st.spinner("Loading shapefile..."):
                    aoi_gdf, error = load_shapefile(uploaded_shp)
                    if error:
                        st.error(f"Error: {error}")
                    else:
                        st.success("Shapefile loaded successfully!")
                        st.session_state.aoi_gdf = aoi_gdf
    
    # Use previously loaded AOI if available
    if 'aoi_gdf' in st.session_state:
        aoi_gdf = st.session_state.aoi_gdf
        with col2:
            if st.button("Remove AOI", use_container_width=True):
                if 'aoi_gdf' in st.session_state:
                    del st.session_state.aoi_gdf
                    st.success("AOI removed successfully!")
                    st.rerun()

    # Always show map, with or without location data
    st.markdown("### Interactive Map")
    
    if st.session_state.get('location_data'):
        # Filter locations with GPS data
        gps_locations = [loc for loc in st.session_state.location_data if 'latitude' in loc and 'longitude' in loc]
        total_locations = len(st.session_state.location_data)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Images", total_locations)
        with col2:
            st.metric("With GPS", len(gps_locations))
        with col3:
            total_detections = sum([loc.get('detections', 0) for loc in gps_locations])
            st.metric("Total Detections", total_detections)
        with col4:
            if aoi_gdf is not None:
                # Count points within AOI
                gps_locations_checked = check_points_in_aoi(gps_locations, aoi_gdf)
                within_aoi_count = sum([1 for loc in gps_locations_checked if loc.get('within_aoi', False)])
                st.metric("Inside AOI", within_aoi_count)
            else:
                st.metric("AOI Status", "None")

        # Control buttons
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Refresh Map", use_container_width=True):
                st.rerun()

        # Display map
        with st.container():
            # Determine map height based on available data
            map_height = 800 if (aoi_gdf is not None or gps_locations) else 400
            
            if gps_locations or aoi_gdf is not None:
                if gps_locations:
                    st.success(f"Displaying {len(gps_locations)} locations with GPS data")
                if aoi_gdf is not None:
                    st.info(f"AOI with {len(aoi_gdf)} features displayed")
                
                detection_map = create_detection_map(gps_locations, aoi_gdf)
                folium_static(detection_map, width=None, height=map_height)

                # Legend
                st.markdown("""
                #### Legend
                **GPS Markers:**
                - **Green**: No damage detections
                - **Orange**: 1-2 damage detections  
                - **Red**: 3+ damage detections
                - **Gray**: Outside AOI
                
                **AOI:**
                - **Orange/Blue**: Area of Interest (dashed polygon outline)
                """)

                # Detailed location table
                if gps_locations:
                    st.markdown("#### Location and Detection Details")
                    
                    # Create detailed dataframe
                    detailed_data = []
                    for loc in gps_locations:
                        detection_summary = []
                        if 'detection_details' in loc and loc['detection_details']:
                            for detail in loc['detection_details']:
                                detection_summary.append(f"{detail['class_name']} ({detail['confidence']:.2f})")
                        
                        row_data = {
                            'File': loc['filename'],
                            'Latitude': f"{loc['latitude']:.6f}",
                            'Longitude': f"{loc['longitude']:.6f}",
                            'Detections': loc['detections'],
                            'Detection Details': '; '.join(detection_summary) if detection_summary else 'None',
                            'Image Size': loc.get('image_size', 'N/A'),
                            'Process Time': loc['timestamp'],
                            'Altitude': loc.get('altitude', 'N/A')
                        }
                        
                        # Add AOI status if AOI is loaded
                        if aoi_gdf is not None:
                            # Check AOI status for this location
                            checked_locations = check_points_in_aoi([loc], aoi_gdf)
                            within_aoi = checked_locations[0].get('within_aoi', False)
                            row_data['AOI Status'] = 'Inside AOI' if within_aoi else 'Outside AOI'
                        
                        detailed_data.append(row_data)
                    
                    location_df = pd.DataFrame(detailed_data)
                    st.dataframe(location_df, use_container_width=True)
                    
                    # Download option for GPS data
                    if st.button("Download GPS Data (CSV)", use_container_width=True):
                        csv_data = location_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv_data,
                            file_name=f"gps_detection_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            else:
                st.warning("No GPS data found in processed images.")
                st.info("Make sure uploaded images have GPS metadata (usually from smartphone photos with GPS enabled)")
                
                # Show default map with height 400 (no data available)
                default_map = create_detection_map([], aoi_gdf)
                folium_static(default_map, width=None, height=400)
    else:
        # Determine map height: 800 if AOI exists, otherwise 400
        map_height = 800 if aoi_gdf is not None else 400
        
        st.info("No location data available yet. Upload and process images or videos first.")
        
        # Show default map centered on Yogyakarta or AOI
        default_map = create_detection_map([], aoi_gdf)
        folium_static(default_map, width=None, height=map_height)
        
        if aoi_gdf is not None:
            st.success(f"Displaying AOI with {len(aoi_gdf)} features")
        
        st.markdown("""
        #### How to Use the Map Feature:
        1. Upload AOI shapefile (.zip) if needed
        2. Upload images with GPS data (EXIF metadata)
        3. Click **"Process All Images"** on the Image tab
        4. Return to this tab to view location map
        5. Markers will show locations with detection details
        """)

# Ensure the function is available at module level
__all__ = ['map_tab']