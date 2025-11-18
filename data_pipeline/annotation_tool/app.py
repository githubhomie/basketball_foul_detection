# annotation_tool/app.py
import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from utils.s3_loader import get_clip_frames
from utils.annotation_io import (
    save_annotation, load_annotation, get_annotated_clips,
    export_annotations_to_csv, get_annotation_stats
)

# Page config - Must be first Streamlit command
st.set_page_config(
    page_title="NBA Foul Annotator",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force sidebar to always be visible - CRITICAL CSS
st.markdown("""
<style>
    /* FORCE SIDEBAR OPEN - Multiple selectors for maximum compatibility */
    section[data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        width: 21rem !important;
        min-width: 21rem !important;
        transform: none !important;
        transition: none !important;
    }
    section[data-testid="stSidebar"][aria-hidden="true"] {
        display: block !important;
        visibility: visible !important;
    }
    section[data-testid="stSidebar"] > div {
        display: block !important;
        visibility: visible !important;
    }
    /* Hide collapse button completely */
    button[kind="header"][data-testid="baseButton-header"],
    button[data-testid="collapsedControl"],
    [data-testid="collapsedControl"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }
    /* Remove any margins/padding that create white space */
    .main .block-container {
        padding-left: 2rem !important;
        max-width: 100% !important;
    }
</style>
<script>
    // JavaScript fallback to force sidebar open
    setTimeout(function() {
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            sidebar.style.display = 'block';
            sidebar.style.visibility = 'visible';
            sidebar.style.transform = 'none';
            sidebar.removeAttribute('aria-hidden');
        }
    }, 100);
</script>
""", unsafe_allow_html=True)

# Clean centered layout
st.markdown("""
<style>
    /* Hide Streamlit header and menu */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    .block-container {
        padding: 2rem 3rem !important;
        padding-top: 1rem !important;
        max-width: 1400px !important;
        margin: 0 auto !important;
    }
    .stButton > button {
        font-size: 0.85rem;
        padding: 0.4rem 0.8rem;
        font-weight: 600;
    }
    /* Hide fullscreen button on images */
    button[title="View fullscreen"],
    button[data-testid="StyledFullScreenButton"] {
        visibility: hidden !important;
        display: none !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    img:hover button {
        display: none !important;
    }
    .video-container {
        position: relative;
        margin: 0;
        padding: 0;
    }
    .video-overlay-top {
        position: absolute;
        top: 10px;
        left: 10px;
        z-index: 10;
        background: rgba(0, 0, 0, 0.7);
        padding: 8px 12px;
        border-radius: 6px;
        color: white;
        font-size: 0.85rem;
        line-height: 1.4;
    }
    .frame-number-overlay {
        position: absolute;
        top: 10px;
        right: 10px;
        z-index: 10;
        background: rgba(0, 0, 0, 0.7);
        padding: 8px 16px;
        border-radius: 6px;
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .foul-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.75rem;
        margin-right: 6px;
    }
    .shooting-foul { background: #90EE90; color: #000; }
    .charging { background: #FFB6C1; color: #000; }
    .personal-foul { background: #87CEEB; color: #000; }
    .loose-ball { background: #DDA0DD; color: #000; }
    .offensive-foul { background: #F0E68C; color: #000; }
    .frame-container {
        border: 4px solid #ddd;
        border-radius: 8px;
        padding: 0;
        background: #000;
        margin: 0;
        overflow: hidden;
    }
    .selected-frame {
        border: 6px solid #FF4B4B !important;
    }
    .controls-compact {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .slider-container {
        position: absolute;
        bottom: 20px;
        left: 20px;
        right: 20px;
        z-index: 10;
        background: rgba(0, 0, 0, 0.6);
        padding: 10px 15px;
        border-radius: 8px;
    }
    div[data-testid="stSlider"] {
        padding: 0 !important;
        border: none !important;
    }
    div[data-testid="stSlider"] label {
        display: none !important;
    }
    div[data-testid="stSlider"] > div {
        border: none !important;
        padding: 0 !important;
    }
    /* Remove gray lines around slider */
    .stSlider {
        border: none !important;
    }
    hr {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'csv_loaded' not in st.session_state:
    st.session_state.csv_loaded = False
if 'current_clip_idx' not in st.session_state:
    st.session_state.current_clip_idx = 0
if 'selected_frame' not in st.session_state:
    st.session_state.selected_frame = None
if 'annotator_name' not in st.session_state:
    st.session_state.annotator_name = "default"
if 'session_start_time' not in st.session_state:
    st.session_state.session_start_time = time.time()
if 'session_annotations' not in st.session_state:
    st.session_state.session_annotations = 0
if 'filter_foul_type' not in st.session_state:
    st.session_state.filter_foul_type = "All"
if 'show_annotated' not in st.session_state:
    st.session_state.show_annotated = False
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 13  # Start at ideal frame
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'load_all_frames' not in st.session_state:
    st.session_state.load_all_frames = False
if 'clip_history' not in st.session_state:
    st.session_state.clip_history = []

@st.cache_data
def load_csv_from_s3():
    """Load CSV from S3"""
    import boto3
    from io import StringIO
    import os

    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION')
    )

    bucket = os.getenv('S3_BUCKET', 'nba-foul-dataset-oh')
    key = 'metadata/dataset.csv'

    obj = s3.get_object(Bucket=bucket, Key=key)
    csv_string = obj['Body'].read().decode('utf-8')
    return StringIO(csv_string)

def load_csv_data(csv_source):
    """Load and process CSV data"""
    df = pd.read_csv(csv_source, dtype={'game_id': str})

    # Group by clip (game_id + event_num)
    clips = []
    for (game_id, event_num), clip_df in df.groupby(['game_id', 'event_num']):
        first_row = clip_df.iloc[0]
        clips.append({
            'game_id': game_id,
            'event_num': event_num,
            'df': clip_df.sort_values('frame_index'),
            'description': first_row['description'],
            'fouler_name': first_row.get('fouler_name', 'Unknown'),
            'fouler_team': first_row.get('fouler_team', ''),
            'fouled_player_name': first_row.get('fouled_player_name', 'Unknown'),
            'fouled_team': first_row.get('fouled_team', ''),
            'foul_type': first_row.get('foul_type', 'unknown'),
            'period': first_row.get('period', ''),
            'game_clock': first_row.get('game_clock', ''),
            'season': first_row.get('season', '')
        })

    return clips

def filter_clips(clips):
    """Filter clips based on annotation status and foul type"""
    annotated = get_annotated_clips()

    filtered = []
    for clip in clips:
        # Filter by annotation status
        is_annotated = (clip['game_id'], clip['event_num']) in annotated
        if not st.session_state.show_annotated and is_annotated:
            continue

        # Filter by foul type
        if st.session_state.filter_foul_type != "All":
            if clip['foul_type'] != st.session_state.filter_foul_type:
                continue

        filtered.append(clip)

    return filtered

def navigate_to_clip(idx, add_to_history=True):
    """Navigate to specific clip index"""
    if add_to_history and st.session_state.current_clip_idx != idx:
        # Add current clip to history before navigating away
        st.session_state.clip_history.append(st.session_state.current_clip_idx)
        # Keep history limited to last 10 clips
        if len(st.session_state.clip_history) > 10:
            st.session_state.clip_history.pop(0)
    st.session_state.current_clip_idx = max(0, idx)
    st.session_state.selected_frame = None
    st.session_state.current_frame = 13  # Reset to ideal frame
    st.session_state.is_playing = False
    st.session_state.load_all_frames = False  # Reset to default 8-22 range

def next_clip():
    """Move to next clip"""
    navigate_to_clip(st.session_state.current_clip_idx + 1)

def prev_clip():
    """Move to previous clip"""
    navigate_to_clip(st.session_state.current_clip_idx - 1)

def back_clip():
    """Go back to previous clip in history (undo)"""
    if st.session_state.clip_history:
        prev_idx = st.session_state.clip_history.pop()
        navigate_to_clip(prev_idx, add_to_history=False)
        st.rerun()

def save_and_next(clip, frame_idx, notes=""):
    """Save annotation and move to next clip"""
    save_annotation(
        clip['game_id'],
        clip['event_num'],
        frame_idx,
        annotator=st.session_state.annotator_name,
        notes=notes
    )
    st.session_state.session_annotations += 1
    next_clip()
    st.rerun()

def skip_clip():
    """Skip current clip without annotating"""
    next_clip()
    st.rerun()

def get_session_stats():
    """Calculate session statistics"""
    elapsed = time.time() - st.session_state.session_start_time
    if st.session_state.session_annotations > 0:
        avg_time = elapsed / st.session_state.session_annotations
        clips_per_min = 60 / avg_time if avg_time > 0 else 0
    else:
        avg_time = 0
        clips_per_min = 0

    return {
        'elapsed': timedelta(seconds=int(elapsed)),
        'annotations': st.session_state.session_annotations,
        'avg_time': avg_time,
        'clips_per_min': clips_per_min
    }

# Sidebar - Configuration
with st.sidebar:
    st.header("📁 Dataset")

    # Auto-load CSV from S3
    if not st.session_state.csv_loaded:
        try:
            with st.spinner("Loading dataset from S3..."):
                csv_source = load_csv_from_s3()
                all_clips = load_csv_data(csv_source)
                st.session_state.all_clips = all_clips
                st.session_state.csv_loaded = True
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            st.stop()

    if st.session_state.csv_loaded:
        all_clips = st.session_state.all_clips

        # Get stats
        stats = get_annotation_stats()
        total_clips = len(all_clips)
        annotated_count = stats['total_annotated']
        remaining = total_clips - annotated_count

        st.success(f"✅ Loaded {total_clips} clips")

        # Progress metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Annotated", annotated_count)
        with col2:
            st.metric("Remaining", remaining)

        if total_clips > 0:
            pct = (annotated_count / total_clips) * 100
            st.progress(annotated_count / total_clips, text=f"{pct:.1f}% Complete")

        st.markdown("---")

        # Filters
        st.subheader("🔍 Filters")

        st.session_state.show_annotated = st.checkbox(
            "Show annotated clips",
            value=st.session_state.show_annotated
        )

        # Get unique foul types
        foul_types = sorted(set(clip['foul_type'] for clip in all_clips))
        st.session_state.filter_foul_type = st.selectbox(
            "Foul Type",
            ["All"] + foul_types,
            index=0
        )

        # Apply filters
        st.session_state.clips = filter_clips(all_clips)
        st.info(f"Showing {len(st.session_state.clips)} clips")

        st.markdown("---")

        # Annotator name
        st.subheader("👤 Annotator")
        st.session_state.annotator_name = st.text_input(
            "Name",
            value=st.session_state.annotator_name
        )

        st.markdown("---")

        # Session stats
        st.subheader("📊 Session Stats")
        session = get_session_stats()
        st.write(f"⏱️ Time: {session['elapsed']}")
        st.write(f"✅ Annotated: {session['annotations']}")
        if session['clips_per_min'] > 0:
            st.write(f"⚡ Rate: {session['clips_per_min']:.1f}/min")
            st.write(f"📈 Avg: {session['avg_time']:.1f}s/clip")

        st.markdown("---")

        # Export
        if st.button("📤 Export Annotations"):
            st.info("Export feature coming soon - annotations are saved to S3")

# Main annotation interface
if not st.session_state.csv_loaded:
    st.title("🏀 NBA Foul Frame Annotator")
    st.info("👈 Upload your CSV file from the sidebar to begin")

    st.markdown("""
    ### How to Use

    1. **Upload CSV** - Load your collected clips dataset
    2. **Review all 30 frames** - Displayed in a 5×6 grid
    3. **Click the frame** where the foul contact occurs
    4. **Confirm in preview** - Large version shows below
    5. **Save & Next** - Move to next clip

    ### Features
    - ✨ All 30 frames visible at once
    - 🖱️ Click to select (no keyboard needed)
    - 🔍 Large preview of selected frame
    - 📊 Real-time progress tracking
    - 🏷️ Filter by foul type
    - ⏭️ Skip unclear clips
    - 💾 Auto-save annotations

    ### Goal
    Annotate ~1,200 clips to build training dataset for foul detection model.
    """)

else:
    clips = st.session_state.clips

    # Check if done
    if st.session_state.current_clip_idx >= len(clips):
        st.success("🎉 All clips in current filter annotated!")
        st.balloons()

        stats = get_annotation_stats()
        st.write(f"### Total Progress: {stats['total_annotated']} clips annotated")

        if st.button("🔄 Reset to First Clip"):
            navigate_to_clip(0)
            st.rerun()
        st.stop()

    # Get current clip
    current_clip = clips[st.session_state.current_clip_idx]
    clip_df = current_clip['df']

    # Check if already annotated
    existing_annotation = load_annotation(current_clip['game_id'], current_clip['event_num'])

    # Load all 30 frames
    all_frames = get_clip_frames(clip_df)

    if not all_frames:
        st.error("❌ Failed to load frames")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⏮️ Previous"):
                prev_clip()
        with col2:
            if st.button("⏭️ Skip"):
                skip_clip()
        st.stop()

    # Load frames based on toggle
    if st.session_state.load_all_frames:
        playback_frames = all_frames  # All 30 frames (0-29)
    else:
        playback_frames = [(idx, img) for idx, img in all_frames if 8 <= idx <= 22]  # Middle 15 frames

    # Get current frame image
    current_frame_data = next((img for idx, img in playback_frames if idx == st.session_state.current_frame), None)

    if current_frame_data is None and playback_frames:
        # Reset to ideal frame if out of range
        st.session_state.current_frame = 13
        current_frame_data = next((img for idx, img in playback_frames if idx == 13), playback_frames[0][1])

    # Prepare info
    foul_type = current_clip['foul_type']
    foul_class = foul_type.replace('_', '-')
    pct = ((st.session_state.current_clip_idx + 1) / len(clips)) * 100

    # Header with info and action buttons
    col1, col2, col3, col4, col5, col6 = st.columns([2.5, 1.2, 1.2, 0.8, 0.8, 0.8])
    with col1:
        st.markdown(
            f"<div style='font-size: 0.85rem; color: #666; margin-bottom: 0; padding-top: 0.3rem;'>"
            f"<span class='foul-badge {foul_class}'>{foul_type.upper()}</span>"
            f"{current_clip['fouler_name']} → {current_clip['fouled_player_name']} | "
            f"{st.session_state.current_clip_idx + 1}/{len(clips)} ({pct:.0f}%)"
            f"{'  ⚠️ Was: ' + str(existing_annotation['foul_frame']) if existing_annotation else ''}"
            f"</div>",
            unsafe_allow_html=True
        )
    with col2:
        if st.button(
            f"✓ SELECT {st.session_state.current_frame}",
            width="stretch",
            type="primary",
            key="select_btn"
        ):
            st.session_state.selected_frame = st.session_state.current_frame
            st.rerun()
    with col3:
        if st.session_state.selected_frame is not None:
            if st.button(
                f"✅ SAVE {st.session_state.selected_frame}",
                width="stretch",
                type="primary",
                key="save_btn"
            ):
                save_and_next(current_clip, st.session_state.selected_frame)
        else:
            st.button("Select first", width="stretch", disabled=True, key="save_btn_disabled")
    with col4:
        if st.button("🚩 Flag", width="stretch", key="flag_btn"):
            save_annotation(
                current_clip['game_id'],
                current_clip['event_num'],
                -1,
                annotator=st.session_state.annotator_name,
                notes="Flagged as bad/unclear clip"
            )
            st.session_state.session_annotations += 1
            next_clip()
            st.rerun()
    with col5:
        btn_text = "8-22" if st.session_state.load_all_frames else "All 30"
        if st.button(btn_text, width="stretch", key="load_all_btn"):
            st.session_state.load_all_frames = not st.session_state.load_all_frames
            st.rerun()
    with col6:
        can_go_back = len(st.session_state.clip_history) > 0
        if st.button("↶ Back", width="stretch", key="back_btn", disabled=not can_go_back, help="Undo - go back to previous clip"):
            back_clip()

    # Scrubber directly below header
    st.markdown("<div class='controls-compact' style='margin-top: 0.3rem; margin-bottom: 0.3rem;'>", unsafe_allow_html=True)

    # Adjust slider range based on load_all_frames
    if st.session_state.load_all_frames:
        min_val, max_val = 0, 29
    else:
        min_val, max_val = 8, 22

    # Clamp current frame to valid range
    if st.session_state.current_frame < min_val or st.session_state.current_frame > max_val:
        st.session_state.current_frame = max(min_val, min(max_val, st.session_state.current_frame))

    new_frame = st.slider(
        "Frame",
        min_value=min_val,
        max_value=max_val,
        value=st.session_state.current_frame,
        key=f"frame_slider_{st.session_state.current_clip_idx}",
        label_visibility="collapsed"
    )
    if new_frame != st.session_state.current_frame:
        st.session_state.current_frame = new_frame
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Video container
    is_selected = st.session_state.selected_frame == st.session_state.current_frame
    container_class = "frame-container selected-frame" if is_selected else "frame-container"

    st.markdown(f"<div class='{container_class}'>", unsafe_allow_html=True)
    if current_frame_data:
        st.image(current_frame_data, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    # Navigation controls at bottom
    st.markdown("<div style='margin-top: 1rem; text-align: center;'>", unsafe_allow_html=True)
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 0.8, 0.8, 1.5, 1])

    with nav_col2:
        if st.button("← Prev", key="nav_left", use_container_width=True):
            if st.session_state.current_frame > min_val:
                st.session_state.current_frame -= 1
                st.rerun()

    with nav_col3:
        if st.button("Next →", key="nav_right", use_container_width=True):
            if st.session_state.current_frame < max_val:
                st.session_state.current_frame += 1
                st.rerun()

    with nav_col4:
        if st.button(f"Save Frame {st.session_state.current_frame}", key="nav_save", type="primary", use_container_width=True):
            save_and_next(current_clip, st.session_state.current_frame)

    st.markdown("</div>", unsafe_allow_html=True)
