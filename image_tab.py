# image_tab.py - COMPLETE FIX
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from collections import Counter
import io
import zipfile
from datetime import datetime

# Fungsi untuk mengekstrak GPS dari EXIF data (existing function)
def extract_gps_from_image(image_path_or_pil):
    """Ekstrak koordinat GPS dari metadata gambar"""
    try:
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil)
        else:
            image = image_path_or_pil
            
        exif_data = image._getexif()
        
        if exif_data is None:
            return None, None
            
        gps_info = {}
        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            if tag_name == "GPSInfo":
                for gps_tag in value:
                    sub_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                    gps_info[sub_tag_name] = value[gps_tag]
                break
        
        if not gps_info:
            return None, exif_data
            
        # Konversi koordinat GPS
        def convert_to_degrees(value):
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                d, m, s = value[0], value[1], value[2]
                # Handle IFDRational objects
                if hasattr(d, 'numerator'):
                    d = float(d.numerator) / float(d.denominator)
                if hasattr(m, 'numerator'):
                    m = float(m.numerator) / float(m.denominator)
                if hasattr(s, 'numerator'):
                    s = float(s.numerator) / float(s.denominator)
                return float(d) + (float(m) / 60.0) + (float(s) / 3600.0)
            return 0.0
        
        lat = convert_to_degrees(gps_info.get('GPSLatitude', [0, 0, 0]))
        lon = convert_to_degrees(gps_info.get('GPSLongitude', [0, 0, 0]))
        
        # Cek orientasi (N/S, E/W)
        if gps_info.get('GPSLatitudeRef') == 'S':
            lat = -lat
        if gps_info.get('GPSLongitudeRef') == 'W':
            lon = -lon
            
        # Handle altitude
        altitude = gps_info.get('GPSAltitude', 'N/A')
        if altitude != 'N/A' and hasattr(altitude, 'numerator'):
            altitude = f"{float(altitude.numerator) / float(altitude.denominator):.1f}m"
        elif altitude != 'N/A':
            altitude = f"{float(altitude):.1f}m"
            
        # Handle timestamp
        timestamp = gps_info.get('GPSTimeStamp', 'N/A')
        if timestamp != 'N/A' and isinstance(timestamp, (list, tuple)):
            try:
                # Convert IFDRational to float for time components
                time_parts = []
                for t in timestamp:
                    if hasattr(t, 'numerator'):
                        time_parts.append(float(t.numerator) / float(t.denominator))
                    else:
                        time_parts.append(float(t))
                timestamp = f"{int(time_parts[0]):02d}:{int(time_parts[1]):02d}:{int(time_parts[2]):02d}"
            except:
                timestamp = str(timestamp)
            
        # Validasi koordinat
        if -90 <= lat <= 90 and -180 <= lon <= 180 and (lat != 0 or lon != 0):
            return {
                'latitude': float(lat), 
                'longitude': float(lon),
                'altitude': str(altitude),
                'timestamp': str(timestamp)
            }, exif_data
    except Exception as e:
        st.error(f"Error extracting GPS: {str(e)}")
    
    return None, None

# *** CRITICAL ADDITION: Function to save location data to session state ***
def save_location_data_to_session(damaged_images):
    """Save location data to session state for map display"""
    location_data = []
    
    for img_data in damaged_images:
        location_entry = {
            'filename': img_data['name'],
            'detections': img_data['damage_count'],
            'detection_details': img_data['detections'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'image_size': f"{img_data['image'].shape[1]}x{img_data['image'].shape[0]}"
        }
        
        # Add GPS data if available
        if img_data['gps']:
            location_entry.update({
                'latitude': img_data['gps']['latitude'],
                'longitude': img_data['gps']['longitude'],
                'altitude': img_data['gps']['altitude'],
                'gps_timestamp': img_data['gps']['timestamp']
            })
        
        location_data.append(location_entry)
    
    # Save to session state
    st.session_state.location_data = location_data
    
    # Confirmation message
    gps_count = len([loc for loc in location_data if 'latitude' in loc])
    st.success(f"✅ Location data saved: {len(location_data)} total images, {gps_count} with GPS coordinates")

def create_detection_txt(img_data):
    """Create comprehensive detection data as text format"""
    lines = []
    
    # Header information
    lines.append("# Road Damage Detection Results")
    lines.append(f"# Image: {img_data['name']}")
    lines.append(f"# Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"# Total Detections: {img_data['damage_count']}")
    lines.append("")
    
    # GPS Information (if available)
    if img_data['gps']:
        lines.append("# GPS COORDINATES")
        lines.append(f"GPS_LATITUDE: {img_data['gps']['latitude']:.6f}")
        lines.append(f"GPS_LONGITUDE: {img_data['gps']['longitude']:.6f}")
        lines.append(f"GPS_ALTITUDE: {img_data['gps']['altitude']}")
        lines.append(f"GPS_TIMESTAMP: {img_data['gps']['timestamp']}")
        lines.append("")
    
    # Detection Details (YOLO format + additional info)
    lines.append("# DETECTION DETAILS")
    lines.append("# Format: class_name confidence x1 y1 x2 y2 center_x center_y")
    
    for detection in img_data['detections']:
        x1, y1, x2, y2 = detection['bbox']
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        line = f"{detection['class_name']} {detection['confidence']:.6f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {center_x:.1f} {center_y:.1f}"
        lines.append(line)
    
    lines.append("")
    
    # Summary by damage type
    damage_summary = Counter([d['class_name'] for d in img_data['detections']])
    lines.append("# DAMAGE SUMMARY")
    for damage_type, count in damage_summary.items():
        lines.append(f"{damage_type}: {count}")
    
    return '\n'.join(lines)

def create_download_button(img_data):
    """Create download button for individual image and data"""
    try:
        # Prepare annotated image (without EXIF)
        annotated_pil = Image.fromarray(img_data['image'])
        
        # Save image to bytes
        img_buffer = io.BytesIO()
        annotated_pil.save(img_buffer, format='JPEG', quality=95)
        img_buffer.seek(0)
        
        # Create detection data text
        detection_txt = create_detection_txt(img_data)
        
        # Create ZIP file with image + txt
        zip_buffer = io.BytesIO()
        name_without_ext = img_data['name'].rsplit('.', 1)[0]
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add annotated image
            zip_file.writestr(f"{name_without_ext}_analyzed.jpg", img_buffer.getvalue())
            
            # Add detection data
            zip_file.writestr(f"{name_without_ext}_detections.txt", detection_txt.encode('utf-8'))
        
        zip_buffer.seek(0)
        
        # Download button
        st.download_button(
            label="⬇️ Download Hasil",
            data=zip_buffer.getvalue(),
            file_name=f"{name_without_ext}_analysis.zip",
            mime="application/zip",
            key=f"download_{img_data['name']}"
        )
        
    except Exception as e:
        st.error(f"Error creating download: {str(e)}")

def create_batch_download(damaged_images):
    """Create batch download as ZIP file with images and detection data"""
    try:
        zip_buffer = io.BytesIO()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Create summary report
            summary_lines = []
            summary_lines.append("# BATCH ROAD DAMAGE ANALYSIS REPORT")
            summary_lines.append(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            summary_lines.append(f"# Total Images Analyzed: {len(damaged_images)}")
            summary_lines.append("")
            
            total_damages = sum([img['damage_count'] for img in damaged_images])
            summary_lines.append(f"# OVERALL STATISTICS")
            summary_lines.append(f"TOTAL_DAMAGE_OBJECTS: {total_damages}")
            
            # Count damage types across all images
            all_damage_types = Counter()
            for img_data in damaged_images:
                for detection in img_data['detections']:
                    all_damage_types[detection['class_name']] += 1
            
            summary_lines.append("# DAMAGE TYPE DISTRIBUTION")
            for damage_type, count in all_damage_types.items():
                summary_lines.append(f"{damage_type}: {count}")
            
            summary_lines.append("")
            summary_lines.append("# IMAGE-WISE BREAKDOWN")
            
            for img_data in damaged_images:
                name_without_ext = img_data['name'].rsplit('.', 1)[0]
                
                # Add annotated image
                annotated_pil = Image.fromarray(img_data['image'])
                img_buffer = io.BytesIO()
                annotated_pil.save(img_buffer, format='JPEG', quality=95)
                zip_file.writestr(f"images/{name_without_ext}_analyzed.jpg", img_buffer.getvalue())
                
                # Add individual detection data
                detection_txt = create_detection_txt(img_data)
                zip_file.writestr(f"detections/{name_without_ext}_detections.txt", detection_txt.encode('utf-8'))
                
                # Add to summary
                summary_lines.append(f"## {img_data['name']}")
                summary_lines.append(f"DAMAGE_COUNT: {img_data['damage_count']}")
                if img_data['gps']:
                    summary_lines.append(f"GPS_COORDINATES: {img_data['gps']['latitude']:.6f}, {img_data['gps']['longitude']:.6f}")
                summary_lines.append("")
            
            # Add summary report
            zip_file.writestr(f"ANALYSIS_SUMMARY_{timestamp}.txt", '\n'.join(summary_lines).encode('utf-8'))
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
        
    except Exception as e:
        st.error(f"Error creating batch download: {str(e)}")
        return None

def get_detection_details(result):
    """Ekstrak detail deteksi dari hasil YOLO"""
    detection_details = []
    
    if result.boxes is not None and len(result.boxes) > 0:
        for i, box in enumerate(result.boxes):
            # Get class name
            class_id = int(box.cls[0])
            class_name = result.names[class_id] if result.names else f"Class_{class_id}"
            
            # Get confidence
            confidence = float(box.conf[0])
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            detection_details.append({
                'class_name': class_name,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2],
                'bbox_formatted': f"({int(x1)}, {int(y1)}) - ({int(x2)}, {int(y2)})"
            })
    
    return detection_details

def display_damage_summary(damage_stats, total_images, damaged_images_count):
    """Tampilkan ringkasan kerusakan dalam format yang rapi"""
    st.markdown("## 📊 Ringkasan Hasil Analisis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Gambar", total_images)
    with col2:
        st.metric("Gambar Berkerusakan", damaged_images_count)
    with col3:
        st.metric("Gambar Normal", total_images - damaged_images_count)
    with col4:
        st.metric("Total Objek Kerusakan", sum(damage_stats.values()))
    
    if damage_stats:
        st.markdown("### 🔍 Detail Jenis Kerusakan")
        
        # Create damage type summary table
        damage_types = {
            'D00': 'Retak Longitudinal',
            'D10': 'Retak Transversal', 
            'D20': 'Retak Kulit Buaya',
            'D40': 'Lubang (Pothole)'
        }
        
        cols = st.columns(len(damage_types))
        for idx, (damage_code, damage_name) in enumerate(damage_types.items()):
            with cols[idx]:
                count = damage_stats.get(damage_code, 0)
                color = "🔴" if count > 0 else "⚪"
                st.metric(
                    label=f"{color} {damage_code}",
                    value=count,
                    help=damage_name
                )

def display_detection_preview(img_data, use_expander=True):
    """Display detection data preview in a formatted way"""
    detection_text = create_detection_txt(img_data)
    
    if use_expander:
        with st.expander("📄 Preview Data Deteksi", expanded=False):
            st.code(detection_text, language="text")
    else:
        st.markdown("**📄 Data Deteksi:**")
        st.code(detection_text, language="text")

def image_tab(model, confidence_threshold):
    uploaded_images = st.file_uploader(
        "Pilih gambar (bisa lebih dari satu)",
        type=["jpg", "jpeg", "png"],
        key="image_uploader",
        label_visibility="collapsed",
        accept_multiple_files=True
    )

    if uploaded_images:
        st.success(f"✅ {len(uploaded_images)} gambar berhasil diupload!")

        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🚀 Analisis Kerusakan Jalan", type="primary", use_container_width=True):
                st.session_state.process_all_images = True
        with col2:
            batch_size = st.selectbox("Batch Size", [1, 2, 3, 4], index=1)

        if st.session_state.get('process_all_images', False):
            # Initialize tracking variables
            damage_stats = Counter()  # Count each damage type
            damaged_images = []  # Store only damaged images info
            total_objects_detected = 0
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_batches = (len(uploaded_images) + batch_size - 1) // batch_size
            
            # Process images in batches
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(uploaded_images))
                current_batch = uploaded_images[start_idx:end_idx]
                
                status_text.write(f"🔄 Memproses batch {batch_idx + 1}/{total_batches}...")
                
                # Prepare batch data
                batch_images, batch_names, batch_pil_images = [], [], []
                for uploaded_image in current_batch:
                    uploaded_image.seek(0)  # Reset stream position
                    pil_image = Image.open(uploaded_image)
                    
                    img_array = np.array(pil_image)
                    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    batch_images.append(img_cv2)
                    batch_names.append(uploaded_image.name)
                    batch_pil_images.append(pil_image)
                
                # Run detection
                results = model(batch_images, conf=confidence_threshold) if len(batch_images) > 1 else [model(batch_images[0], conf=confidence_threshold)[0]]
                
                # Process results
                for img_idx, (img_cv2, result, img_name, uploaded_image, pil_img) in enumerate(zip(batch_images, results, batch_names, current_batch, batch_pil_images)):
                    detection_details = get_detection_details(result)
                    
                    # Only process images with detections
                    if detection_details:
                        # Count damage types
                        for detail in detection_details:
                            damage_type = detail['class_name']
                            damage_stats[damage_type] += 1
                            total_objects_detected += 1
                        
                        # Get GPS data
                        uploaded_image.seek(0)
                        gps_data, exif_data = extract_gps_from_image(Image.open(uploaded_image))
                        
                        # Create annotated image
                        annotated = result.plot()
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        
                        # Store damaged image info
                        damaged_images.append({
                            'name': img_name,
                            'image': annotated_rgb,
                            'original': cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB),
                            'detections': detection_details,
                            'gps': gps_data,
                            'damage_count': len(detection_details)
                        })
                
                # Update progress
                progress_bar.progress((batch_idx + 1) / total_batches)
            
            # Clear status
            status_text.empty()
            progress_bar.empty()
            
            # *** CRITICAL FIX: Save location data to session state ***
            if damaged_images:
                save_location_data_to_session(damaged_images)
            
            # Display summary
            display_damage_summary(damage_stats, len(uploaded_images), len(damaged_images))
            
            # Display damaged images only
            if damaged_images:
                st.markdown("## 🚨 Gambar dengan Kerusakan Terdeteksi")
                
                # Add batch download button
                if len(damaged_images) > 1:
                    st.markdown("### 📦 Download Semua Hasil")
                    st.info("💡 **Format download batch**: Gambar hasil analisis + file deteksi (.txt) berisi koordinat GPS dan bounding box")
                    
                    batch_download_data = create_batch_download(damaged_images)
                    if batch_download_data:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.download_button(
                            label=f"⬇️ Download Batch ({len(damaged_images)} gambar + data)",
                            data=batch_download_data,
                            file_name=f"road_damage_batch_{timestamp}.zip",
                            mime="application/zip",
                            type="primary"
                        )
                    st.markdown("---")
                
                for idx, img_data in enumerate(damaged_images):
                    with st.expander(f"📷 {img_data['name']} - {img_data['damage_count']} kerusakan", expanded=True):
                        # Create columns for info and download button
                        col1, col2 = st.columns([4, 1])
                        
                        with col1:
                            # GPS info if available
                            if img_data['gps']:
                                st.info(f"📍 **Lokasi GPS**: {img_data['gps']['latitude']:.6f}, {img_data['gps']['longitude']:.6f}")
                            else:
                                st.warning("📍 **GPS**: Tidak tersedia")
                            
                            # Detection details
                            st.write("🔍 **Kerusakan terdeteksi:**")
                            for i, detail in enumerate(img_data['detections']):
                                confidence_color = "🟢" if detail['confidence'] > 0.8 else "🟡" if detail['confidence'] > 0.6 else "🟠"
                                st.write(f"   {i+1}. {confidence_color} **{detail['class_name']}** (Confidence: {detail['confidence']:.2f})")
                        
                        with col2:
                            # Download button for individual image + data
                            create_download_button(img_data)
                        
                        # Display image
                        st.image(img_data['image'], use_container_width=True)
                        
                        # Show detection data preview (without nested expander)
                        display_detection_preview(img_data, use_expander=False)
                
                st.success(f"✅ Analisis selesai! Ditemukan {len(damaged_images)} gambar dengan kerusakan dari {len(uploaded_images)} total gambar.")
            else:
                st.success("✅ Analisis selesai! Tidak ada kerusakan jalan yang terdeteksi pada gambar yang diupload.")
            
            # Reset processing flag
            st.session_state.process_all_images = False
            
        else:
            # Quick preview of uploaded images (simplified)
            st.markdown("### 🖼️ Preview Gambar")
            st.info(f"Siap menganalisis {len(uploaded_images)} gambar. Klik tombol 'Analisis Kerusakan Jalan' untuk memulai.")