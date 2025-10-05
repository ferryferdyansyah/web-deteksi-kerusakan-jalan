# video_tab.py
import streamlit as st
import tempfile
import cv2
import os
import time
from datetime import datetime

def parse_gpx_file(file):
    # Dummy parser
    return []

def parse_kml_file(file):
    return []

def interpolate_gps_for_video(gps_points, duration, total_frames):
    return [gps_points[0]] * total_frames if gps_points else []

def video_tab(model, confidence_threshold):
    uploaded_video = st.file_uploader("Pilih video", type=["mp4", "mov", "avi"], key="video_uploader", label_visibility="collapsed")

    st.subheader("📍 GPS Tracking (Opsional)")
    gps_file_col1, gps_file_col2 = st.columns(2)
    with gps_file_col1:
        uploaded_gpx = st.file_uploader("Upload file GPX", type=["gpx"], key="gpx_uploader")
    with gps_file_col2:
        uploaded_kml = st.file_uploader("Upload file KML", type=["kml"], key="kml_uploader")

    if uploaded_video:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("⚡ Kontrol Pemrosesan")
            if st.button("🚀 Mulai Proses Video", type="primary"):
                st.session_state.process_video = True
            if st.button("⏹️ Stop Proses"):
                st.session_state.process_video = False
        with col2:
            st.subheader("📊 Statistik Real-time")
            frame_count_placeholder = st.empty()
            detection_count_placeholder = st.empty()
            gps_info_placeholder = st.empty()

        st.subheader("🎬 Live Detection Feed")
        video_placeholder = st.empty()

        if st.session_state.get('process_video', False):
            gps_points = []
            if uploaded_gpx:
                gps_points = parse_gpx_file(uploaded_gpx)
            elif uploaded_kml:
                gps_points = parse_kml_file(uploaded_kml)

            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            tfile.close()
            cap = cv2.VideoCapture(tfile.name)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            duration = total_frames / fps if fps > 0 else 0

            st.info(f"📹 Info Video: {total_frames} frames, {fps} FPS, {duration:.1f}s")

            interpolated_gps = interpolate_gps_for_video(gps_points, duration, total_frames)
            video_progress = st.progress(0)

            frame_num = 0
            total_detections = 0
            frame_detections_list = []

            while cap.isOpened() and st.session_state.get('process_video', False):
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame, conf=confidence_threshold)[0]
                annotated = results.plot()
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                detections = len(results.boxes) if results.boxes else 0
                total_detections += detections
                frame_detections_list.append(detections)

                gps_info = None
                if interpolated_gps and frame_num < len(interpolated_gps):
                    gps_info = interpolated_gps[frame_num]

                video_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)
                frame_count_placeholder.metric("🎞️ Frame", f"{frame_num + 1}/{total_frames}")
                detection_count_placeholder.metric("🎯 Total Deteksi", total_detections)

                if gps_info:
                    gps_info_placeholder.info(f"📍 GPS: Lat {gps_info['latitude']:.6f}, Lon {gps_info['longitude']:.6f}")

                frame_num += 1
                video_progress.progress(frame_num / total_frames)
                time.sleep(0.03)

            cap.release()
            os.unlink(tfile.name)
            st.success("✅ Video selesai diproses!")
            st.balloons()
