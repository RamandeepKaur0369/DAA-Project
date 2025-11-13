"""
Music Analysis & Order Statistics Finder with Song Remix/Mashup Feature
Main Streamlit Application - Enhanced UI Version with Working Navigation
"""
import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import time
from scipy import signal
import soundfile as sf
import io

# ============================================================================
# CUSTOM CSS FOR STYLING
# ============================================================================

def apply_custom_styles():
    st.markdown("""
        <style>
        /* Background image for main app */
        .stApp {
            background-image: url('https://thumbs.dreamstime.com/b/vibrant-dynamic-image-featuring-shiny-gold-musical-notes-flowing-against-black-background-creating-luxurious-315022193.jpg');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        
        /* Add overlay for better readability */
        .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            z-index: -1;
        }
        
        /* Main content styling - clean and simple */
        .main .block-container {
            background: rgba(0, 0, 0, 0.8);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 2px 8px rgba(255, 215, 0, 0.3);
        }
        
        /* White text for all content */
        .stApp, .stApp p, .stApp span, .stApp div, .stApp label, 
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
        .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown div {
            color: #ffffff !important;
        }
        
        /* Title styling */
        h1 {
            color: #ffd700 !important;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
            font-weight: 800 !important;
        }
        
        /* Headers styling */
        h2, h3, h4 {
            color: #ffffff !important;
            text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8);
        }
        
        /* Sidebar styling with background image */
        [data-testid="stSidebar"] {
            position: relative;
            border-right: 2px solid rgba(255, 215, 0, 0.3);
        }
        
        [data-testid="stSidebar"]::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: url('https://thumbs.dreamstime.com/b/vibrant-dynamic-image-featuring-shiny-gold-musical-notes-flowing-against-black-background-creating-luxurious-315022193.jpg');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            z-index: 0;
        }
        
        [data-testid="stSidebar"]::after {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(180deg, rgba(0, 0, 0, 0.85) 0%, rgba(20, 20, 20, 0.85) 100%);
            backdrop-filter: blur(5px);
            z-index: 1;
        }
        
        [data-testid="stSidebar"] > div {
            position: relative;
            z-index: 2;
        }
        
        [data-testid="stSidebar"] * {
            color: #ffffff !important;
        }
        
        /* Sidebar button styling */
        [data-testid="stSidebar"] button {
            background: linear-gradient(135deg, rgba(255, 215, 0, 0.8) 0%, rgba(218, 165, 32, 0.8) 100%);
            color: black !important;
            border: 2px solid rgba(255, 215, 0, 0.4);
            border-radius: 12px;
            padding: 15px;
            margin: 10px 0;
            transition: all 0.3s ease;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(255, 215, 0, 0.3);
        }
        
        [data-testid="stSidebar"] button:hover {
            background: linear-gradient(135deg, rgba(255, 215, 0, 1) 0%, rgba(218, 165, 32, 1) 100%);
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(255, 215, 0, 0.5);
        }
        
        /* Active sidebar button */
        [data-testid="stSidebar"] button[kind="primary"] {
            background: linear-gradient(135deg, #ffd700 0%, #ffaa00 100%) !important;
            box-shadow: 0 6px 20px rgba(255, 215, 0, 0.6) !important;
            transform: scale(1.05);
            color: black !important;
        }
        
        /* Button styling - Main buttons */
        .stButton > button {
            background: linear-gradient(135deg, #ffd700 0%, #daa520 100%);
            color: black !important;
            border: none;
            border-radius: 10px;
            padding: 12px 24px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(255, 215, 0, 0.4);
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #daa520 0%, #ffd700 100%);
            box-shadow: 0 6px 20px rgba(255, 215, 0, 0.6);
            transform: translateY(-2px);
        }
        
        /* Metric styling */
        [data-testid="stMetricValue"] {
            font-size: 28px;
            font-weight: 700;
            color: #ffd700 !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: #ffffff !important;
        }
        
        [data-testid="stMetricDelta"] {
            color: #ffd700 !important;
        }
        
        /* Enhanced visibility for images and plots */
        .stImage, [data-testid="stImage"] {
            border-radius: 10px;
            box-shadow: 0 8px 25px rgba(255, 215, 0, 0.3);
            border: 3px solid rgba(255, 215, 0, 0.5);
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background: linear-gradient(90deg, #ffd700 0%, #daa520 100%);
        }
        
        /* Info boxes */
        .stAlert {
            border-radius: 10px;
            border-left: 5px solid #ffd700;
            background: rgba(255, 215, 0, 0.1);
            color: #ffffff !important;
        }
        
        /* File uploader */
        [data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 10px;
            padding: 15px;
            border: 2px dashed #ffd700;
        }
        
        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] p,
        [data-testid="stFileUploader"] div,
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] button {
            color: #000000 !important;
        }
        
        [data-testid="stFileUploader"] .uploadedFileName {
            color: #000000 !important;
        }
        
        /* File uploader drag and drop text */
        [data-testid="stFileUploadDropzone"] label,
        [data-testid="stFileUploadDropzone"] span,
        [data-testid="stFileUploadDropzone"] small {
            color: #000000 !important;
        }
        
        /* Download button special styling */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #ffd700 0%, #b8860b 100%);
            color: black !important;
            border: none;
            border-radius: 10px;
            padding: 12px 24px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(255, 215, 0, 0.4);
        }
        
        .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #b8860b 0%, #ffd700 100%);
            box-shadow: 0 6px 20px rgba(255, 215, 0, 0.6);
            transform: translateY(-2px);
        }
        
        /* Slider styling */
        .stSlider label {
            color: #ffffff !important;
        }
        
        /* Radio button styling */
        .stRadio label {
            color: #ffffff !important;
        }
        
        /* Checkbox styling */
        .stCheckbox label {
            color: #ffffff !important;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            color: #ffffff !important;
            background: rgba(0, 0, 0, 0.6);
        }
        
        /* Table styling */
        table {
            color: #ffffff !important;
            background: rgba(0, 0, 0, 0.6) !important;
        }
        
        thead tr th {
            color: #ffd700 !important;
            background: rgba(0, 0, 0, 0.8) !important;
        }
        
        tbody tr td {
            color: #ffffff !important;
            background: rgba(0, 0, 0, 0.6) !important;
        }
        </style>
    """, unsafe_allow_html=True)

# ============================================================================
# SIDEBAR FEATURE NAVIGATION
# ============================================================================

def create_feature_sidebar():
    st.sidebar.markdown("### 🎵 **Quick Navigation**")
    
    # Initialize session state for tab selection
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
    
    features = [
        ("📊", "Algorithm Comparison", "Compare performance of different sorting algorithms"),
        ("🎵", "Audio Analysis", "Analyze your music files in depth"),
        ("🔇", "Noise Reduction", "Remove noise from audio tracks"),
        ("🎧", "Song Remix/Mashup", "Create amazing song mashups"),
        ("📈", "Order Statistics", "Interactive statistics demo")
    ]
    
    for idx, (icon, title, desc) in enumerate(features):
        button_type = "primary" if st.session_state.active_tab == idx else "secondary"
        if st.sidebar.button(
            f"{icon} {title}",
            key=f"nav_btn_{idx}",
            help=desc,
            use_container_width=True,
            type=button_type
        ):
            st.session_state.active_tab = idx
            st.rerun()
        
        st.sidebar.markdown(f"""
            <div style="margin: 5px 0; padding: 5px; opacity: 0.7; font-size: 12px;">
                {desc}
            </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        <div style="text-align: center; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 10px; margin-top: 20px;">
            <p style="font-size: 14px; margin: 5px 0;"><b>💡 Pro Tips</b></p>
            <p style="font-size: 12px; opacity: 0.8;">• Use WAV files for best quality</p>
            <p style="font-size: 12px; opacity: 0.8;">• Keep files under 60 seconds</p>
            <p style="font-size: 12px; opacity: 0.8;">• Match tempos for better mashups</p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# ORDER STATISTICS ALGORITHMS
# ============================================================================

def quickselect(arr, k):
    """Deterministic QuickSelect to find k-th smallest element"""
    if len(arr) == 1:
        return arr[0]
    
    pivot = float(np.median([arr[0], arr[len(arr)//2], arr[-1]]))
    
    lows = [x for x in arr if x < pivot]
    highs = [x for x in arr if x > pivot]
    pivots = [x for x in arr if x == pivot]
    
    if k < len(lows):
        return quickselect(lows, k)
    elif k < len(lows) + len(pivots):
        return pivots[0]
    else:
        return quickselect(highs, k - len(lows) - len(pivots))


def randomized_select(arr, k):
    """Randomized QuickSelect for k-th smallest element"""
    if len(arr) == 1:
        return arr[0]
    
    pivot = arr[int(np.random.randint(len(arr)))]
    
    lows = [x for x in arr if x < pivot]
    highs = [x for x in arr if x > pivot]
    pivots = [x for x in arr if x == pivot]
    
    if k < len(lows):
        return randomized_select(lows, k)
    elif k < len(lows) + len(pivots):
        return pivots[0]
    else:
        return randomized_select(highs, k - len(lows) - len(pivots))


def find_kth_element(arr, k, method='quickselect'):
    """Find k-th smallest element using specified method"""
    start_time = time.time()
    
    arr_copy = arr.tolist() if isinstance(arr, np.ndarray) else arr.copy()
    
    try:
        if method == 'quickselect':
            result = quickselect(arr_copy, k)
        elif method == 'randomized':
            result = randomized_select(arr_copy, k)
        elif method == 'sort':
            sorted_arr = sorted(arr_copy)
            result = sorted_arr[k]
        else:
            result = float(np.partition(arr, k)[k])
    except Exception as e:
        st.error(f"Error in {method}: {str(e)}")
        result = float(np.median(arr))
    
    elapsed_time = time.time() - start_time
    return result, elapsed_time


# ============================================================================
# AUDIO ANALYSIS FUNCTIONS
# ============================================================================

def calculate_median_statistics(audio_data, sr):
    """Calculate various median-based audio statistics"""
    stats = {}
    
    stats['median_amplitude'] = float(np.median(np.abs(audio_data)))
    rms = librosa.feature.rms(y=audio_data)[0]
    stats['median_energy'] = float(np.median(rms))
    
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
    stats['median_brightness'] = float(np.median(spectral_centroid))
    
    zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
    stats['median_zcr'] = float(np.median(zcr))
    
    tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
    stats['tempo'] = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    stats['beat_count'] = int(len(beats))
    
    return stats


def median_filter_denoise(audio_data, kernel_size=5):
    """Apply median filter for noise reduction"""
    return signal.medfilt(audio_data, kernel_size=kernel_size)


def adaptive_noise_reduction(audio_data, sr):
    """Advanced noise reduction using order statistics"""
    noise_threshold = float(np.percentile(np.abs(audio_data), 25))
    
    D = librosa.stft(audio_data)
    magnitude, phase = np.abs(D), np.angle(D)
    median_mag = np.median(magnitude, axis=1, keepdims=True)
    magnitude_clean = np.where(magnitude > median_mag * 1.5, magnitude, magnitude * 0.3)
    
    D_clean = magnitude_clean * np.exp(1j * phase)
    audio_clean = librosa.istft(D_clean)
    
    return audio_clean


def analyze_audio_segments(audio_data, sr, segment_duration=1.0):
    """Analyze audio in segments using order statistics"""
    segment_length = int(segment_duration * sr)
    num_segments = len(audio_data) // segment_length
    
    segment_stats = []
    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        segment = audio_data[start:end]
        
        stats = {
            'segment': i,
            'median_amp': float(np.median(np.abs(segment))),
            'max_amp': float(np.max(np.abs(segment))),
            'energy': float(np.sum(segment**2))
        }
        segment_stats.append(stats)
    
    return segment_stats


# ============================================================================
# SONG REMIX/MASHUP FUNCTIONS
# ============================================================================

def time_stretch(audio, rate):
    """Time stretch audio without changing pitch"""
    return librosa.effects.time_stretch(audio, rate=rate)


def pitch_shift(audio, sr, n_steps):
    """Shift pitch without changing tempo"""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def match_tempo(audio1, sr1, audio2, sr2):
    """Match tempo of audio2 to audio1"""
    tempo1, _ = librosa.beat.beat_track(y=audio1, sr=sr1)
    tempo2, _ = librosa.beat.beat_track(y=audio2, sr=sr2)
    
    tempo1 = float(tempo1) if np.isscalar(tempo1) else float(tempo1[0])
    tempo2 = float(tempo2) if np.isscalar(tempo2) else float(tempo2[0])
    
    if tempo2 > 0:
        rate = tempo1 / tempo2
        audio2_stretched = time_stretch(audio2, rate)
        return audio2_stretched, rate
    return audio2, 1.0


def mix_audio(audio1, audio2, mix_ratio=0.5):
    """Mix two audio tracks with given ratio"""
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]
    
    mixed = audio1 * mix_ratio + audio2 * (1 - mix_ratio)
    
    max_val = float(np.max(np.abs(mixed)))
    if max_val > 1.0:
        mixed = mixed / max_val
    
    return mixed


def crossfade_audio(audio1, audio2, crossfade_duration=2.0, sr=22050):
    """Crossfade between two audio tracks"""
    crossfade_samples = int(crossfade_duration * sr)
    
    if len(audio2) < crossfade_samples:
        crossfade_samples = len(audio2)
    
    fade_out = np.linspace(1, 0, crossfade_samples)
    fade_in = np.linspace(0, 1, crossfade_samples)
    
    audio1_end = audio1[:-crossfade_samples]
    audio1_fade = audio1[-crossfade_samples:] * fade_out
    audio2_fade = audio2[:crossfade_samples] * fade_in
    audio2_rest = audio2[crossfade_samples:]
    
    crossfaded_section = audio1_fade + audio2_fade
    result = np.concatenate([audio1_end, crossfaded_section, audio2_rest])
    
    return result


def create_mashup(audio1, audio2, sr1, sr2, 
                 tempo_match=True, 
                 pitch_shift_steps=0, 
                 mix_ratio=0.5,
                 use_crossfade=False,
                 crossfade_duration=2.0):
    """Create a mashup from two audio tracks"""
    
    if sr1 != sr2:
        audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=sr1)
        sr2 = sr1
    
    if tempo_match:
        audio2, rate = match_tempo(audio1, sr1, audio2, sr2)
        st.info(f"✅ Tempo matched: Stretch rate = {rate:.2f}x")
    
    if pitch_shift_steps != 0:
        audio2 = pitch_shift(audio2, sr1, pitch_shift_steps)
        st.info(f"✅ Pitch shifted: {pitch_shift_steps:+d} semitones")
    
    if use_crossfade:
        mashup = crossfade_audio(audio1, audio2, crossfade_duration, sr1)
        st.info(f"✅ Crossfade applied: {crossfade_duration}s")
    else:
        mashup = mix_audio(audio1, audio2, mix_ratio)
        st.info(f"✅ Mixed: {mix_ratio*100:.0f}% Song 1, {(1-mix_ratio)*100:.0f}% Song 2")
    
    return mashup, sr1


# ============================================================================
# TAB CONTENT FUNCTIONS
# ============================================================================

def show_algorithm_comparison():
    st.header("📊 Algorithm Performance Comparison")
    st.markdown("Compare different methods for finding k-th smallest element")
    
    col1, col2 = st.columns(2)
    
    with col1:
        array_size = st.slider("Array Size", 100, 50000, 10000, step=100)
        k_percentile = st.slider("k (percentile)", 0, 100, 50)
    
    with col2:
        num_trials = st.slider("Number of Trials", 1, 10, 5)
    
    if st.button("🚀 Run Algorithm Comparison", type="primary", use_container_width=True):
        k = int(array_size * k_percentile / 100)
        
        methods = ['quickselect', 'randomized', 'sort', 'numpy_partition']
        results = {method: [] for method in methods}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for trial in range(num_trials):
            status_text.text(f"Running trial {trial + 1}/{num_trials}...")
            test_array = np.random.randn(array_size)
            
            for method in methods:
                _, elapsed_time = find_kth_element(test_array, k, method)
                results[method].append(elapsed_time * 1000)
            
            progress_bar.progress((trial + 1) / num_trials)
        
        status_text.text("✅ Analysis complete!")
        
        st.subheader("📊 Performance Results (milliseconds)")
        
        perf_data = []
        for method in methods:
            avg_time = float(np.mean(results[method]))
            std_time = float(np.std(results[method]))
            perf_data.append({
                'Method': method.replace('_', ' ').title(),
                'Avg Time (ms)': f"{avg_time:.4f}",
                'Std Dev': f"{std_time:.4f}",
                'Min': f"{min(results[method]):.4f}",
                'Max': f"{max(results[method]):.4f}"
            })
        
        st.table(perf_data)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(0.9)
        
        positions = np.arange(len(methods))
        means = [float(np.mean(results[m])) for m in methods]
        stds = [float(np.std(results[m])) for m in methods]
        
        bars = ax.bar(positions, means, yerr=stds, capsize=5, alpha=0.8, 
                     color=['#667eea', '#764ba2', '#f093fb', '#f5576c'],
                     edgecolor='black', linewidth=1.5)
        
        ax.set_xticks(positions)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in methods], fontsize=11, fontweight='bold')
        ax.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
        ax.set_title(f'Algorithm Performance Comparison (n={array_size}, k={k})', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8f9fa')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def show_audio_analysis():
    st.header("🎵 Audio File Analysis")
    st.markdown("Upload an audio file to analyze its properties using order statistics")
    
    uploaded_file = st.file_uploader("📁 Upload Audio File", type=['wav', 'mp3', 'ogg', 'flac'], key='audio_analysis')
    
    if uploaded_file is not None:
        try:
            with st.spinner("🎵 Loading audio file..."):
                audio_data, sr = librosa.load(uploaded_file, sr=None)
            duration = len(audio_data) / sr
            
            st.success(f"✅ Loaded audio: {duration:.2f}s @ {sr}Hz")
            
            with st.spinner("📊 Analyzing audio..."):
                stats = calculate_median_statistics(audio_data, sr)
            
            st.markdown("### 📈 Audio Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Median Amplitude", f"{stats['median_amplitude']:.4f}", delta="Level")
            with col2:
                st.metric("Median Energy", f"{stats['median_energy']:.4f}", delta="Power")
            with col3:
                st.metric("Tempo (BPM)", f"{stats['tempo']:.1f}", delta="Rhythm")
            with col4:
                st.metric("Beats Detected", stats['beat_count'], delta="Count")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Median Brightness", f"{stats['median_brightness']:.0f} Hz", delta="Frequency")
            with col2:
                st.metric("Median ZCR", f"{stats['median_zcr']:.4f}", delta="Texture")
            
            st.markdown("### 🌊 Waveform Analysis")
            fig, ax = plt.subplots(figsize=(14, 5))
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(0.95)
            
            times = np.arange(len(audio_data)) / sr
            ax.plot(times, audio_data, linewidth=0.5, alpha=0.8, color='#667eea')
            ax.axhline(y=stats['median_amplitude'], color='#f5576c', linestyle='--', 
                      linewidth=2, label='Median Amplitude', alpha=0.8)
            ax.axhline(y=-stats['median_amplitude'], color='#f5576c', linestyle='--', 
                      linewidth=2, alpha=0.8)
            ax.fill_between(times, audio_data, alpha=0.3, color='#667eea')
            
            ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
            ax.set_title('Audio Waveform with Median Statistics', fontsize=14, fontweight='bold', pad=15)
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(alpha=0.3, linestyle='--')
            ax.set_facecolor('#f8f9fa')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.markdown("### 📊 Segment Analysis (1-second windows)")
            segment_stats = analyze_audio_segments(audio_data, sr, 1.0)
            
            if segment_stats:
                fig, ax = plt.subplots(figsize=(14, 5))
                fig.patch.set_facecolor('white')
                fig.patch.set_alpha(0.95)
                
                segments = [s['segment'] for s in segment_stats]
                median_amps = [s['median_amp'] for s in segment_stats]
                
                ax.plot(segments, median_amps, marker='o', linewidth=2.5, 
                       markersize=6, color='#764ba2', markerfacecolor='#f093fb',
                       markeredgewidth=2, markeredgecolor='#764ba2')
                ax.fill_between(segments, median_amps, alpha=0.3, color='#764ba2')
                
                ax.set_xlabel('Segment Number', fontsize=12, fontweight='bold')
                ax.set_ylabel('Median Amplitude', fontsize=12, fontweight='bold')
                ax.set_title('Median Amplitude per Segment', fontsize=14, fontweight='bold', pad=15)
                ax.grid(alpha=0.3, linestyle='--')
                ax.set_facecolor('#f8f9fa')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
        except Exception as e:
            st.error(f"❌ Error loading audio: {str(e)}")


def show_noise_reduction():
    st.header("🔇 Noise Reduction using Order Statistics")
    st.markdown("Apply advanced noise reduction techniques to your audio files")
    
    uploaded_file_nr = st.file_uploader("📁 Upload Noisy Audio", type=['wav', 'mp3', 'ogg'], key='noise_reduction')
    
    if uploaded_file_nr is not None:
        try:
            with st.spinner("🎵 Loading audio..."):
                audio_data, sr = librosa.load(uploaded_file_nr, sr=None)
            
            st.success(f"✅ Audio loaded: {len(audio_data)/sr:.2f}s @ {sr}Hz")
            
            col1, col2 = st.columns(2)
            
            with col1:
                filter_method = st.radio("🔧 Denoising Method", 
                                        ["Median Filter", "Adaptive (Spectral)"],
                                        help="Choose noise reduction algorithm")
                
            with col2:
                if filter_method == "Median Filter":
                    kernel_size = st.slider("Kernel Size", 3, 21, 5, step=2,
                                          help="Larger = more smoothing")
                else:
                    kernel_size = 5
            
            if st.button("🚀 Apply Noise Reduction", type="primary", use_container_width=True):
                with st.spinner("⚙️ Processing audio..."):
                    if filter_method == "Median Filter":
                        audio_clean = median_filter_denoise(audio_data, kernel_size)
                    else:
                        audio_clean = adaptive_noise_reduction(audio_data, sr)
                
                st.success("✅ Noise reduction complete!")
                
                st.markdown("### 🎨 Visual Comparison")
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
                fig.patch.set_facecolor('white')
                fig.patch.set_alpha(0.95)
                
                times = np.arange(len(audio_data)) / sr
                
                # Original
                ax1.plot(times, audio_data, linewidth=0.5, alpha=0.8, color='#ef4444')
                ax1.fill_between(times, audio_data, alpha=0.3, color='#ef4444')
                ax1.set_title('🔴 Original (Noisy)', fontsize=13, fontweight='bold', pad=10)
                ax1.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
                ax1.grid(alpha=0.3, linestyle='--')
                ax1.set_facecolor('#fef2f2')
                
                # Cleaned
                times_clean = np.arange(len(audio_clean)) / sr
                ax2.plot(times_clean, audio_clean, linewidth=0.5, alpha=0.8, color='#22c55e')
                ax2.fill_between(times_clean, audio_clean, alpha=0.3, color='#22c55e')
                ax2.set_title('🟢 Cleaned', fontsize=13, fontweight='bold', pad=10)
                ax2.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
                ax2.grid(alpha=0.3, linestyle='--')
                ax2.set_facecolor('#f0fdf4')
                
                # Noise
                if len(audio_clean) == len(audio_data):
                    noise = audio_data - audio_clean
                    ax3.plot(times, noise, linewidth=0.5, alpha=0.8, color='#f97316')
                    ax3.fill_between(times, noise, alpha=0.3, color='#f97316')
                    ax3.set_title('🟠 Removed Noise', fontsize=13, fontweight='bold', pad=10)
                    ax3.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
                    ax3.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
                    ax3.grid(alpha=0.3, linestyle='--')
                    ax3.set_facecolor('#fff7ed')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.markdown("### 📊 Statistical Comparison")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🔴 Original Statistics")
                    st.metric("Median Amplitude", f"{float(np.median(np.abs(audio_data))):.4f}")
                    st.metric("Max Amplitude", f"{float(np.max(np.abs(audio_data))):.4f}")
                    st.metric("Std Deviation", f"{float(np.std(audio_data)):.4f}")
                
                with col2:
                    st.markdown("#### 🟢 Cleaned Statistics")
                    st.metric("Median Amplitude", f"{float(np.median(np.abs(audio_clean))):.4f}")
                    st.metric("Max Amplitude", f"{float(np.max(np.abs(audio_clean))):.4f}")
                    st.metric("Std Deviation", f"{float(np.std(audio_clean)):.4f}")
                
                # Audio Player and Download Section
                st.markdown("---")
                st.markdown("### 🎧 Listen & Download Cleaned Audio")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🔴 Original Audio")
                    buffer_original = io.BytesIO()
                    sf.write(buffer_original, audio_data, sr, format='WAV')
                    buffer_original.seek(0)
                    st.audio(buffer_original, format='audio/wav')
                
                with col2:
                    st.markdown("#### 🟢 Cleaned Audio")
                    buffer_clean = io.BytesIO()
                    sf.write(buffer_clean, audio_clean, sr, format='WAV')
                    buffer_clean.seek(0)
                    st.audio(buffer_clean, format='audio/wav')
                
                st.markdown("---")
                
                # Download buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    buffer_download = io.BytesIO()
                    sf.write(buffer_download, audio_clean, sr, format='WAV')
                    buffer_download.seek(0)
                    
                    st.download_button(
                        label="⬇️ Download Cleaned Audio (WAV)",
                        data=buffer_download,
                        file_name="cleaned_audio.wav",
                        mime="audio/wav",
                        use_container_width=True
                    )
                
                with col2:
                    buffer_download_mp3 = io.BytesIO()
                    sf.write(buffer_download_mp3, audio_clean, sr, format='WAV')
                    buffer_download_mp3.seek(0)
                    
                    st.download_button(
                        label="⬇️ Download Cleaned Audio (Backup)",
                        data=buffer_download_mp3,
                        file_name="cleaned_audio_backup.wav",
                        mime="audio/wav",
                        use_container_width=True
                    )
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def show_song_remix():
    st.header("🎧 Song Remix & Mashup Creator")
    st.markdown("**Create amazing mashups by mixing two songs together!**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎵 Song 1 (Base Track)")
        song1_file = st.file_uploader("Upload First Song", type=['wav', 'mp3', 'ogg', 'flac'], key='song1')
        
    with col2:
        st.markdown("#### 🎵 Song 2 (Overlay Track)")
        song2_file = st.file_uploader("Upload Second Song", type=['wav', 'mp3', 'ogg', 'flac'], key='song2')
    
    if song1_file is not None and song2_file is not None:
        try:
            with st.spinner("🎵 Loading songs..."):
                audio1, sr1 = librosa.load(song1_file, sr=None, duration=60)
                audio2, sr2 = librosa.load(song2_file, sr=None, duration=60)
            
            st.success(f"✅ Song 1: {len(audio1)/sr1:.1f}s @ {sr1}Hz | Song 2: {len(audio2)/sr2:.1f}s @ {sr2}Hz")
            
            # Display tempo info
            tempo1, _ = librosa.beat.beat_track(y=audio1, sr=sr1)
            tempo2, _ = librosa.beat.beat_track(y=audio2, sr=sr2)
            
            tempo1 = float(tempo1) if np.isscalar(tempo1) else float(tempo1[0])
            tempo2 = float(tempo2) if np.isscalar(tempo2) else float(tempo2[0])
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"🎵 Song 1 Tempo: {tempo1:.1f} BPM")
            with col2:
                st.info(f"🎵 Song 2 Tempo: {tempo2:.1f} BPM")
            
            st.markdown("---")
            st.markdown("### 🎛️ Remix Controls")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                tempo_match = st.checkbox("🎼 Match Tempo", value=True, 
                                         help="Automatically match tempo of Song 2 to Song 1")
                
                mix_mode = st.radio("🎚️ Mix Mode", 
                                   ["Blend (Mix)", "Crossfade"],
                                   help="Blend: Mix both songs | Crossfade: Transition between songs")
            
            with col2:
                if mix_mode == "Blend (Mix)":
                    mix_ratio = st.slider("Mix Ratio (Song 1 %)", 0, 100, 50, 
                                        help="0% = Only Song 2, 100% = Only Song 1") / 100
                else:
                    crossfade_duration = st.slider("Crossfade Duration (s)", 1.0, 10.0, 2.0, 0.5)
                
                pitch_shift_steps = st.slider("🎹 Pitch Shift (Song 2)", -12, 12, 0, 1,
                                             help="Semitones: -12 = octave down, +12 = octave up")
            
            with col3:
                st.markdown("**✨ Quick Presets:**")
                if st.button("🎸 Rock Mashup", use_container_width=True):
                    st.session_state['preset'] = 'rock'
                if st.button("🎹 Electronic Mix", use_container_width=True):
                    st.session_state['preset'] = 'electronic'
                if st.button("🎺 Jazz Blend", use_container_width=True):
                    st.session_state['preset'] = 'jazz'
            
            if st.button("🎵 Create Mashup", type="primary", use_container_width=True):
                with st.spinner("🎨 Creating your mashup..."):
                    # Create mashup
                    if mix_mode == "Blend (Mix)":
                        mashup, sr_out = create_mashup(
                            audio1, audio2, sr1, sr2,
                            tempo_match=tempo_match,
                            pitch_shift_steps=pitch_shift_steps,
                            mix_ratio=mix_ratio,
                            use_crossfade=False
                        )
                    else:
                        mashup, sr_out = create_mashup(
                            audio1, audio2, sr1, sr2,
                            tempo_match=tempo_match,
                            pitch_shift_steps=pitch_shift_steps,
                            use_crossfade=True,
                            crossfade_duration=crossfade_duration
                        )

                    st.success("🎉 Mashup created successfully!")

                    # Visualizations
                    st.markdown("### 🎨 Mashup Visualization")

                    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
                    fig.patch.set_facecolor('white')
                    fig.patch.set_alpha(0.95)

                    # Song 1
                    times1 = np.arange(len(audio1)) / sr1
                    ax1.plot(times1, audio1, linewidth=0.6, alpha=0.8, color='#3b82f6')
                    ax1.fill_between(times1, audio1, alpha=0.3, color='#3b82f6')
                    ax1.set_title('🎵 Song 1 (Base Track)', fontsize=13, fontweight='bold', pad=10)
                    ax1.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
                    ax1.grid(alpha=0.3, linestyle='--')
                    ax1.set_facecolor('#eff6ff')
                    ax1.set_xlim(0, max(times1))

                    # Song 2
                    times2 = np.arange(len(audio2)) / sr2
                    ax2.plot(times2, audio2, linewidth=0.6, alpha=0.8, color='#10b981')
                    ax2.fill_between(times2, audio2, alpha=0.3, color='#10b981')
                    ax2.set_title('🎵 Song 2 (Overlay Track)', fontsize=13, fontweight='bold', pad=10)
                    ax2.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
                    ax2.grid(alpha=0.3, linestyle='--')
                    ax2.set_facecolor('#ecfdf5')
                    ax2.set_xlim(0, max(times2))

                    # Mashup
                    times_mashup = np.arange(len(mashup)) / sr_out
                    ax3.plot(times_mashup, mashup, linewidth=0.6, alpha=0.8, color='#ec4899')
                    ax3.fill_between(times_mashup, mashup, alpha=0.3, color='#ec4899')
                    ax3.set_title('🎉 Mashup Result', fontsize=13, fontweight='bold', pad=10)
                    ax3.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
                    ax3.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
                    ax3.grid(alpha=0.3, linestyle='--')
                    ax3.set_facecolor('#fdf2f8')
                    ax3.set_xlim(0, max(times_mashup))

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    # Statistics
                    st.markdown("### 📈 Mashup Statistics")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("⏱️ Duration", f"{len(mashup)/sr_out:.2f}s")
                        st.metric("🎚️ Sample Rate", f"{sr_out} Hz")

                    with col2:
                        st.metric("📊 Median Amplitude", f"{float(np.median(np.abs(mashup))):.4f}")
                        st.metric("📈 Max Amplitude", f"{float(np.max(np.abs(mashup))):.4f}")

                    with col3:
                        mashup_tempo, _ = librosa.beat.beat_track(y=mashup, sr=sr_out)
                        mashup_tempo = float(mashup_tempo) if np.isscalar(mashup_tempo) else float(mashup_tempo[0])
                        st.metric("🎵 Estimated Tempo", f"{mashup_tempo:.1f} BPM")
                        st.metric("📉 Peak-to-Peak", f"{float(np.ptp(mashup)):.4f}")

                    # Download and playback
                    st.markdown("### 💾 Download & Listen")
                    buffer = io.BytesIO()
                    sf.write(buffer, mashup, sr_out, format='WAV')
                    buffer.seek(0)

                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            label="⬇️ Download Mashup (WAV)",
                            data=buffer,
                            file_name="mashup_output.wav",
                            mime="audio/wav",
                            use_container_width=True
                        )

                    buffer.seek(0)
                    with col2:
                        st.audio(buffer, format='audio/wav')
            
        except Exception as e:
            st.error(f"❌ Error creating mashup: {str(e)}")
            st.info("💡 Tip: Use smaller files (under 60s) for faster processing")
    
    else:
        st.info("👆 Upload two audio files to create a mashup!")
        
        with st.expander("ℹ️ How to Create Amazing Mashups"):
            st.markdown("""
            ### 🎵 Tips for Great Mashups:
            
            **1. Choose Compatible Songs:**
            - Songs with similar tempo work best
            - Songs in same/related keys sound better
            
            **2. Tempo Matching:**
            - Enable "Match Tempo" to sync beats automatically
            - App stretches Song 2 to match Song 1
            
            **3. Pitch Shifting:**
            - Match key signatures with pitch shift
            - ±12 semitones = 1 octave
            - Small adjustments (±2-5) often sound best
            
            **4. Mix Modes:**
            - **Blend**: Mix both songs throughout
            - **Crossfade**: Smooth transition from one song to another
            """)


def show_order_statistics():
    st.header("📈 Interactive Order Statistics Demonstration")
    st.markdown("Visualize how different order statistics algorithms work on random data")
    
    demo_size = st.slider("📊 Dataset Size", 10, 1000, 100)
    
    if st.button("🎲 Generate Random Data", type="primary", use_container_width=True):
        data = np.random.randn(demo_size)
        
        median = float(np.median(data))
        q1 = float(np.percentile(data, 25))
        q3 = float(np.percentile(data, 75))
        iqr = q3 - q1
        
        st.markdown("### 📊 Statistical Measures")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Median", f"{median:.3f}", delta="50th percentile")
        with col2:
            st.metric("Q1 (25%)", f"{q1:.3f}", delta="Lower quartile")
        with col3:
            st.metric("Q3 (75%)", f"{q3:.3f}", delta="Upper quartile")
        with col4:
            st.metric("IQR", f"{iqr:.3f}", delta="Spread")
        
        st.markdown("### 📊 Data Visualization")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(0.95)
        
        # Histogram
        n, bins, patches = ax1.hist(data, bins=30, alpha=0.7, color='#667eea', edgecolor='black', linewidth=1.2)
        
        for i, patch in enumerate(patches):
            bin_center = (bins[i] + bins[i+1]) / 2
            if bin_center < q1:
                patch.set_facecolor('#3b82f6')
            elif bin_center < median:
                patch.set_facecolor('#10b981')
            elif bin_center < q3:
                patch.set_facecolor('#f59e0b')
            else:
                patch.set_facecolor('#ef4444')
        
        ax1.axvline(median, color='red', linestyle='--', linewidth=2.5, label='Median', alpha=0.8)
        ax1.axvline(q1, color='orange', linestyle='--', linewidth=2.5, label='Q1', alpha=0.8)
        ax1.axvline(q3, color='green', linestyle='--', linewidth=2.5, label='Q3', alpha=0.8)
        ax1.set_xlabel('Value', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Distribution with Order Statistics', fontsize=13, fontweight='bold', pad=15)
        ax1.legend(fontsize=10, loc='upper right')
        ax1.grid(alpha=0.3, linestyle='--')
        ax1.set_facecolor('#f8f9fa')
        
        # Box plot
        bp = ax2.boxplot(data, vert=True, patch_artist=True,
                        boxprops=dict(facecolor='#667eea', alpha=0.7, linewidth=2),
                        medianprops=dict(color='#ef4444', linewidth=3),
                        whiskerprops=dict(linewidth=2, color='#1e3a8a'),
                        capprops=dict(linewidth=2, color='#1e3a8a'),
                        flierprops=dict(marker='o', markerfacecolor='#f59e0b', markersize=8, alpha=0.5))
        
        ax2.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax2.set_title('Box Plot (Order Statistics Visualization)', fontsize=13, fontweight='bold', pad=15)
        ax2.grid(alpha=0.3, linestyle='--', axis='y')
        ax2.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("### ⚡ Algorithm Performance for Finding Median")
        
        k = len(data) // 2
        methods = ['quickselect', 'randomized', 'sort', 'numpy_partition']
        times = []
        
        for method in methods:
            _, elapsed = find_kth_element(data, k, method)
            times.append(elapsed * 1000)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(0.95)
            
            colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c']
            bars = ax.barh([m.replace('_', ' ').title() for m in methods], times, 
                          color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            for i, (bar, time) in enumerate(zip(bars, times)):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'  {time:.6f} ms', 
                       va='center', fontweight='bold', fontsize=10)
            
            ax.set_xlabel('Time (milliseconds)', fontsize=12, fontweight='bold')
            ax.set_title('Algorithm Performance Comparison', fontsize=13, fontweight='bold', pad=15)
            ax.grid(alpha=0.3, linestyle='--', axis='x')
            ax.set_facecolor('#f8f9fa')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("#### ⏱️ Timing Results")
            for method, time_ms in zip(methods, times):
                st.metric(
                    method.replace('_', ' ').title(),
                    f"{time_ms:.6f} ms",
                    delta=f"{((time_ms - min(times))/min(times)*100):.1f}% slower" if time_ms != min(times) else "Fastest ⚡"
                )


# ============================================================================
# STREAMLIT APP MAIN
# ============================================================================

def main():
    st.set_page_config(page_title="Music Analysis & Remix Tool", layout="wide", page_icon="🎵")
    
    apply_custom_styles()
    create_feature_sidebar()
    
    st.title("🎵 Music Analysis & Order Statistics with Remix/Mashup")
    st.markdown("**Analyze audio, compare algorithms, reduce noise, and create amazing remixes!**")
    
    # Get active tab from session state
    active_tab = st.session_state.get('active_tab', 0)
    
    # Show content based on active tab
    if active_tab == 0:
        show_algorithm_comparison()
    elif active_tab == 1:
        show_audio_analysis()
    elif active_tab == 2:
        show_noise_reduction()
    elif active_tab == 3:
        show_song_remix()
    elif active_tab == 4:
        show_order_statistics()
    
    # Footer
    st.markdown("---")
    st.markdown("### 💡 Quick Tips")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎵 Audio Analysis**
        - Upload MP3/WAV files
        - View median statistics
        - Analyze tempo & beats
        - Export results
        """)
    
    with col2:
        st.markdown("""
        **🎧 Remix/Mashup**
        - Mix two songs together
        - Match tempo automatically
        - Apply pitch shift & effects
        - Download creations
        """)
    
    with col3:
        st.markdown("""
        **📊 Algorithms**
        - Compare performance
        - Test order statistics
        - Visualize results
        - Learn concepts
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(255, 215, 0, 0.1) 0%, rgba(218, 165, 32, 0.1) 100%); border-radius: 10px; border: 2px solid rgba(255, 215, 0, 0.3);">
        <p style="font-size: 16px; font-weight: 600; color: #ffd700; margin: 0;">
            Built with ❤️ using Streamlit, Librosa, NumPy & SciPy
        </p>
        <p style="font-size: 12px; color: #ffffff; margin-top: 5px;">
            🎵 Transform your music • 🔬 Analyze with precision • 🎨 Create masterpieces
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
