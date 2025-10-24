"""
Music Analysis & Order Statistics Finder with Song Remix/Mashup Feature
Main Streamlit Application
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
# ORDER STATISTICS ALGORITHMS
# ============================================================================

def quickselect(arr, k):
    """Deterministic QuickSelect to find k-th smallest element"""
    if len(arr) == 1:
        return arr[0]
    
    pivot = np.median([arr[0], arr[len(arr)//2], arr[-1]])
    
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
    
    pivot = arr[np.random.randint(len(arr))]
    
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
    
    if method == 'quickselect':
        result = quickselect(arr.copy(), k)
    elif method == 'randomized':
        result = randomized_select(arr.copy(), k)
    elif method == 'sort':
        sorted_arr = np.sort(arr)
        result = sorted_arr[k]
    else:
        result = np.partition(arr, k)[k]
    
    elapsed_time = time.time() - start_time
    return result, elapsed_time


# ============================================================================
# AUDIO ANALYSIS FUNCTIONS
# ============================================================================

def calculate_median_statistics(audio_data, sr):
    """Calculate various median-based audio statistics"""
    stats = {}
    
    # Median amplitude
    stats['median_amplitude'] = np.median(np.abs(audio_data))
    
    # RMS energy
    rms = librosa.feature.rms(y=audio_data)[0]
    stats['median_energy'] = np.median(rms)
    
    # Spectral centroid (brightness)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
    stats['median_brightness'] = np.median(spectral_centroid)
    
    # Zero crossing rate (noisiness)
    zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
    stats['median_zcr'] = np.median(zcr)
    
    # Tempo
    tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
    stats['tempo'] = tempo
    stats['beat_count'] = len(beats)
    
    return stats


def median_filter_denoise(audio_data, kernel_size=5):
    """Apply median filter for noise reduction"""
    return signal.medfilt(audio_data, kernel_size=kernel_size)


def adaptive_noise_reduction(audio_data, sr):
    """Advanced noise reduction using order statistics"""
    noise_threshold = np.percentile(np.abs(audio_data), 25)
    mask = np.abs(audio_data) > noise_threshold * 2
    
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
            'median_amp': np.median(np.abs(segment)),
            'max_amp': np.max(np.abs(segment)),
            'energy': np.sum(segment**2)
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
    
    if tempo2 > 0:
        rate = tempo1 / tempo2
        audio2_stretched = time_stretch(audio2, rate)
        return audio2_stretched, rate
    return audio2, 1.0


def mix_audio(audio1, audio2, mix_ratio=0.5):
    """Mix two audio tracks with given ratio"""
    # Match lengths
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]
    
    # Mix with ratio
    mixed = audio1 * mix_ratio + audio2 * (1 - mix_ratio)
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val
    
    return mixed


def crossfade_audio(audio1, audio2, crossfade_duration=2.0, sr=22050):
    """Crossfade between two audio tracks"""
    crossfade_samples = int(crossfade_duration * sr)
    
    # Ensure audio2 is long enough
    if len(audio2) < crossfade_samples:
        crossfade_samples = len(audio2)
    
    # Create fade curves
    fade_out = np.linspace(1, 0, crossfade_samples)
    fade_in = np.linspace(0, 1, crossfade_samples)
    
    # Apply crossfade
    audio1_end = audio1[:-crossfade_samples]
    audio1_fade = audio1[-crossfade_samples:] * fade_out
    audio2_fade = audio2[:crossfade_samples] * fade_in
    audio2_rest = audio2[crossfade_samples:]
    
    # Combine
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
    
    # Resample if sample rates don't match
    if sr1 != sr2:
        audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=sr1)
        sr2 = sr1
    
    # Match tempo if requested
    if tempo_match:
        audio2, rate = match_tempo(audio1, sr1, audio2, sr2)
        st.info(f"Tempo matched: Stretch rate = {rate:.2f}x")
    
    # Pitch shift if requested
    if pitch_shift_steps != 0:
        audio2 = pitch_shift(audio2, sr1, pitch_shift_steps)
        st.info(f"Pitch shifted: {pitch_shift_steps:+d} semitones")
    
    # Mix or crossfade
    if use_crossfade:
        mashup = crossfade_audio(audio1, audio2, crossfade_duration, sr1)
        st.info(f"Crossfade applied: {crossfade_duration}s")
    else:
        mashup = mix_audio(audio1, audio2, mix_ratio)
        st.info(f"Mixed: {mix_ratio*100:.0f}% Song 1, {(1-mix_ratio)*100:.0f}% Song 2")
    
    return mashup, sr1


def apply_audio_effects(audio, sr, effect_type, **kwargs):
    """Apply various audio effects"""
    
    if effect_type == "reverb":
        # Simple reverb using convolution with impulse response
        impulse = np.random.randn(int(sr * 0.1))
        impulse = impulse / np.max(np.abs(impulse))
        audio_with_effect = signal.convolve(audio, impulse, mode='same')
        audio_with_effect = audio_with_effect[:len(audio)]
        
    elif effect_type == "echo":
        delay = kwargs.get('delay', 0.3)
        decay = kwargs.get('decay', 0.5)
        delay_samples = int(delay * sr)
        
        echo = np.zeros(len(audio) + delay_samples)
        echo[:len(audio)] = audio
        echo[delay_samples:] += audio * decay
        audio_with_effect = echo[:len(audio)]
        
    elif effect_type == "speed_up":
        rate = kwargs.get('rate', 1.2)
        audio_with_effect = time_stretch(audio, rate)
        
    elif effect_type == "slow_down":
        rate = kwargs.get('rate', 0.8)
        audio_with_effect = time_stretch(audio, rate)
        
    elif effect_type == "bass_boost":
        # Simple bass boost using low-pass emphasis
        nyquist = sr / 2
        cutoff = 200 / nyquist
        b, a = signal.butter(2, cutoff, btype='low')
        bass = signal.filtfilt(b, a, audio)
        audio_with_effect = audio + bass * 0.5
        
    else:
        audio_with_effect = audio
    
    # Normalize
    max_val = np.max(np.abs(audio_with_effect))
    if max_val > 1.0:
        audio_with_effect = audio_with_effect / max_val
    
    return audio_with_effect


# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(page_title="Music Analysis & Remix Tool", layout="wide", page_icon="🎵")
    
    st.title("🎵 Music Analysis & Order Statistics with Remix/Mashup")
    st.markdown("**Analyze audio, compare algorithms, reduce noise, and create amazing remixes!**")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    st.sidebar.info(
        "**Features:**\n\n"
        "📊 Algorithm Comparison\n\n"
        "🎵 Audio Analysis\n\n"
        "🔇 Noise Reduction\n\n"
        "🎧 Song Remix/Mashup\n\n"
        "📈 Order Statistics Demo"
    )
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Algorithm Comparison", 
        "🎵 Audio Analysis", 
        "🔇 Noise Reduction",
        "🎧 Song Remix/Mashup",
        "📈 Order Statistics Demo"
    ])
    
    # ========================================================================
    # TAB 1: Algorithm Comparison
    # ========================================================================
    with tab1:
        st.header("Algorithm Performance Comparison")
        st.markdown("Compare different methods for finding k-th smallest element")
        
        col1, col2 = st.columns(2)
        
        with col1:
            array_size = st.slider("Array Size", 100, 100000, 10000, step=100)
            k_percentile = st.slider("k (percentile)", 0, 100, 50)
        
        with col2:
            num_trials = st.slider("Number of Trials", 1, 10, 5)
        
        if st.button("Run Algorithm Comparison", type="primary"):
            k = int(array_size * k_percentile / 100)
            
            methods = ['quickselect', 'randomized', 'sort', 'numpy_partition']
            results = {method: [] for method in methods}
            
            progress_bar = st.progress(0)
            
            for trial in range(num_trials):
                test_array = np.random.randn(array_size)
                
                for method in methods:
                    _, elapsed_time = find_kth_element(test_array, k, method)
                    results[method].append(elapsed_time * 1000)
                
                progress_bar.progress((trial + 1) / num_trials)
            
            st.subheader("Performance Results (milliseconds)")
            
            perf_data = []
            for method in methods:
                avg_time = np.mean(results[method])
                std_time = np.std(results[method])
                perf_data.append({
                    'Method': method.replace('_', ' ').title(),
                    'Avg Time (ms)': f"{avg_time:.4f}",
                    'Std Dev': f"{std_time:.4f}",
                    'Min': f"{min(results[method]):.4f}",
                    'Max': f"{max(results[method]):.4f}"
                })
            
            st.table(perf_data)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            positions = np.arange(len(methods))
            means = [np.mean(results[m]) for m in methods]
            stds = [np.std(results[m]) for m in methods]
            
            ax.bar(positions, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue')
            ax.set_xticks(positions)
            ax.set_xticklabels([m.replace('_', ' ').title() for m in methods])
            ax.set_ylabel('Time (ms)')
            ax.set_title(f'Algorithm Performance (n={array_size}, k={k})')
            ax.grid(axis='y', alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
    
    # ========================================================================
    # TAB 2: Audio Analysis
    # ========================================================================
    with tab2:
        st.header("Audio File Analysis")
        
        uploaded_file = st.file_uploader("Upload Audio File (WAV, MP3, etc.)", type=['wav', 'mp3', 'ogg', 'flac'])
        
        if uploaded_file is not None:
            try:
                audio_data, sr = librosa.load(uploaded_file, sr=None)
                duration = len(audio_data) / sr
                
                st.success(f"✅ Loaded audio: {duration:.2f}s @ {sr}Hz")
                
                with st.spinner("Analyzing audio..."):
                    stats = calculate_median_statistics(audio_data, sr)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Median Amplitude", f"{stats['median_amplitude']:.4f}")
                with col2:
                    st.metric("Median Energy", f"{stats['median_energy']:.4f}")
                with col3:
                    st.metric("Tempo (BPM)", f"{stats['tempo']:.1f}")
                with col4:
                    st.metric("Beats Detected", stats['beat_count'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Median Brightness", f"{stats['median_brightness']:.0f} Hz")
                with col2:
                    st.metric("Median ZCR", f"{stats['median_zcr']:.4f}")
                
                st.subheader("Waveform Analysis")
                fig, ax = plt.subplots(figsize=(12, 4))
                
                times = np.arange(len(audio_data)) / sr
                ax.plot(times, audio_data, linewidth=0.5, alpha=0.7)
                ax.axhline(y=stats['median_amplitude'], color='r', linestyle='--', label='Median Amplitude')
                ax.axhline(y=-stats['median_amplitude'], color='r', linestyle='--')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.set_title('Waveform')
                ax.legend()
                ax.grid(alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
                
                st.subheader("Segment Analysis (1-second windows)")
                segment_stats = analyze_audio_segments(audio_data, sr, 1.0)
                
                if segment_stats:
                    fig, ax = plt.subplots(figsize=(12, 4))
                    segments = [s['segment'] for s in segment_stats]
                    median_amps = [s['median_amp'] for s in segment_stats]
                    
                    ax.plot(segments, median_amps, marker='o', linewidth=2)
                    ax.set_xlabel('Segment Number')
                    ax.set_ylabel('Median Amplitude')
                    ax.set_title('Median Amplitude per Segment')
                    ax.grid(alpha=0.3)
                    
                    st.pyplot(fig)
                    plt.close()
                
            except Exception as e:
                st.error(f"Error loading audio: {str(e)}")
    
    # ========================================================================
    # TAB 3: Noise Reduction
    # ========================================================================
    with tab3:
        st.header("Noise Reduction using Order Statistics")
        
        uploaded_file_nr = st.file_uploader("Upload Noisy Audio", type=['wav', 'mp3', 'ogg'], key='noise_reduction')
        
        if uploaded_file_nr is not None:
            try:
                audio_data, sr = librosa.load(uploaded_file_nr, sr=None)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    filter_method = st.radio("Denoising Method", 
                                            ["Median Filter", "Adaptive (Spectral)"])
                    
                with col2:
                    if filter_method == "Median Filter":
                        kernel_size = st.slider("Median Filter Kernel Size", 3, 21, 5, step=2)
                    else:
                        kernel_size = 5
                
                if st.button("Apply Noise Reduction", type="primary"):
                    with st.spinner("Processing..."):
                        if filter_method == "Median Filter":
                            audio_clean = median_filter_denoise(audio_data, kernel_size)
                        else:
                            audio_clean = adaptive_noise_reduction(audio_data, sr)
                    
                    st.success("✅ Noise reduction complete!")
                    
                    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9))
                    
                    times = np.arange(len(audio_data)) / sr
                    
                    ax1.plot(times, audio_data, linewidth=0.5, alpha=0.7, color='red')
                    ax1.set_title('Original (Noisy)')
                    ax1.set_ylabel('Amplitude')
                    ax1.grid(alpha=0.3)
                    
                    times_clean = np.arange(len(audio_clean)) / sr
                    ax2.plot(times_clean, audio_clean, linewidth=0.5, alpha=0.7, color='green')
                    ax2.set_title('Cleaned')
                    ax2.set_ylabel('Amplitude')
                    ax2.grid(alpha=0.3)
                    
                    if len(audio_clean) == len(audio_data):
                        noise = audio_data - audio_clean
                        ax3.plot(times, noise, linewidth=0.5, alpha=0.7, color='orange')
                        ax3.set_title('Removed Noise')
                        ax3.set_xlabel('Time (s)')
                        ax3.set_ylabel('Amplitude')
                        ax3.grid(alpha=0.3)
                    
                    st.pyplot(fig)
                    plt.close()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Original Statistics")
                        st.write(f"Median Amplitude: {np.median(np.abs(audio_data)):.4f}")
                        st.write(f"Max Amplitude: {np.max(np.abs(audio_data)):.4f}")
                        st.write(f"Std Deviation: {np.std(audio_data):.4f}")
                    
                    with col2:
                        st.subheader("Cleaned Statistics")
                        st.write(f"Median Amplitude: {np.median(np.abs(audio_clean)):.4f}")
                        st.write(f"Max Amplitude: {np.max(np.abs(audio_clean)):.4f}")
                        st.write(f"Std Deviation: {np.std(audio_clean):.4f}")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # ========================================================================
    # TAB 4: Song Remix/Mashup
    # ========================================================================
    with tab4:
        st.header("🎧 Song Remix & Mashup Creator")
        st.markdown("Create amazing mashups by mixing two songs together!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎵 Song 1 (Base Track)")
            song1_file = st.file_uploader("Upload First Song", type=['wav', 'mp3', 'ogg', 'flac'], key='song1')
            
        with col2:
            st.subheader("🎵 Song 2 (Overlay Track)")
            song2_file = st.file_uploader("Upload Second Song", type=['wav', 'mp3', 'ogg', 'flac'], key='song2')
        
        if song1_file is not None and song2_file is not None:
            try:
                with st.spinner("Loading songs..."):
                    audio1, sr1 = librosa.load(song1_file, sr=None, duration=60)  # Limit to 60s for demo
                    audio2, sr2 = librosa.load(song2_file, sr=None, duration=60)
                
                st.success(f"✅ Song 1: {len(audio1)/sr1:.1f}s @ {sr1}Hz | Song 2: {len(audio2)/sr2:.1f}s @ {sr2}Hz")
                
                # Display tempo info
                tempo1, _ = librosa.beat.beat_track(y=audio1, sr=sr1)
                tempo2, _ = librosa.beat.beat_track(y=audio2, sr=sr2)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"🎵 Song 1 Tempo: {tempo1:.1f} BPM")
                with col2:
                    st.info(f"🎵 Song 2 Tempo: {tempo2:.1f} BPM")
                
                st.markdown("---")
                st.subheader("🎛️ Remix Controls")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    tempo_match = st.checkbox("Match Tempo", value=True, 
                                             help="Automatically match tempo of Song 2 to Song 1")
                    
                    mix_mode = st.radio("Mix Mode", 
                                       ["Blend (Mix)", "Crossfade"],
                                       help="Blend: Mix both songs together | Crossfade: Transition from Song 1 to Song 2")
                
                with col2:
                    if mix_mode == "Blend (Mix)":
                        mix_ratio = st.slider("Mix Ratio (Song 1 %)", 0, 100, 50, 
                                            help="0% = Only Song 2, 100% = Only Song 1") / 100
                    else:
                        crossfade_duration = st.slider("Crossfade Duration (seconds)", 1.0, 10.0, 2.0, 0.5)
                    
                    pitch_shift_steps = st.slider("Pitch Shift (Song 2)", -12, 12, 0, 1,
                                                 help="Shift pitch in semitones (-12 = 1 octave down, +12 = 1 octave up)")
                
                with col3:
                    st.markdown("**Apply Effects to Mashup:**")
                    apply_effect = st.checkbox("Apply Audio Effect")
                    
                    if apply_effect:
                        effect_type = st.selectbox("Effect Type", 
                                                   ["reverb", "echo", "bass_boost", "speed_up", "slow_down"])
                
                if st.button("🎵 Create Mashup", type="primary", use_container_width=True):
                    with st.spinner("Creating your mashup... 🎵"):
                        # Create mashup according to selected options
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

                        # Apply effects if selected
                        if apply_effect:
                            mashup = apply_audio_effects(mashup, sr_out, effect_type)
                            st.success(f"✨ Applied {effect_type} effect!")

                        st.success("🎉 Mashup created successfully!")

                        # Visualizations
                        st.subheader("� Mashup Visualization")

                        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))

                        # Song 1 waveform
                        times1 = np.arange(len(audio1)) / sr1
                        ax1.plot(times1, audio1, linewidth=0.5, alpha=0.7, color='blue')
                        ax1.set_title('Song 1 (Base Track)', fontsize=12, fontweight='bold')
                        ax1.set_ylabel('Amplitude')
                        ax1.grid(alpha=0.3)
                        ax1.set_xlim(0, max(times1))

                        # Song 2 waveform
                        times2 = np.arange(len(audio2)) / sr2
                        ax2.plot(times2, audio2, linewidth=0.5, alpha=0.7, color='green')
                        ax2.set_title('Song 2 (Overlay Track)', fontsize=12, fontweight='bold')
                        ax2.set_ylabel('Amplitude')
                        ax2.grid(alpha=0.3)
                        ax2.set_xlim(0, max(times2))

                        # Mashup waveform
                        times_mashup = np.arange(len(mashup)) / sr_out
                        ax3.plot(times_mashup, mashup, linewidth=0.5, alpha=0.7, color='red')
                        ax3.set_title('Mashup Result 🎵', fontsize=12, fontweight='bold')
                        ax3.set_xlabel('Time (seconds)')
                        ax3.set_ylabel('Amplitude')
                        ax3.grid(alpha=0.3)
                        ax3.set_xlim(0, max(times_mashup))

                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                        # Statistics
                        st.subheader("📈 Mashup Statistics")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Duration", f"{len(mashup)/sr_out:.2f}s")
                            st.metric("Sample Rate", f"{sr_out} Hz")

                        with col2:
                            st.metric("Median Amplitude", f"{np.median(np.abs(mashup)):.4f}")
                            st.metric("Max Amplitude", f"{np.max(np.abs(mashup)):.4f}")

                        with col3:
                            mashup_tempo, _ = librosa.beat.beat_track(y=mashup, sr=sr_out)
                            st.metric("Estimated Tempo", f"{mashup_tempo:.1f} BPM")
                            st.metric("Peak-to-Peak", f"{np.ptp(mashup):.4f}")

                        # Download and playback
                        st.subheader("💾 Download Your Mashup")
                        buffer = io.BytesIO()
                        sf.write(buffer, mashup, sr_out, format='WAV')
                        buffer.seek(0)

                        st.download_button(
                            label="⬇️ Download Mashup (WAV)",
                            data=buffer,
                            file_name="mashup_output.wav",
                            mime="audio/wav",
                            use_container_width=True
                        )

                        st.subheader("🔊 Listen to Your Mashup")
                        st.audio(buffer, format='audio/wav')
                
            except Exception as e:
                st.error(f"Error creating mashup: {str(e)}")
                st.info("💡 Tip: Try with smaller audio files (under 60 seconds) for faster processing")
        
        else:
            st.info("👆 Please upload two audio files to create a mashup!")
            
            with st.expander("ℹ️ How to Create Amazing Mashups"):
                st.markdown("""
                ### 🎵 Tips for Great Mashups:
                
                1. **Choose Compatible Songs:**
                   - Songs with similar tempo work best
                   - Songs in the same or related keys sound better
                
                2. **Tempo Matching:**
                   - Enable "Match Tempo" to automatically sync beats
                   - The app will stretch/compress Song 2 to match Song 1
                
                3. **Pitch Shifting:**
                   - Use pitch shift to match key signatures
                   - +/- 12 semitones = 1 octave up/down
                   - Small adjustments (±2-5) often sound best
                
                4. **Mix Modes:**
                   - **Blend**: Mix both songs throughout
                   - **Crossfade**: Smooth transition from one to another
                
                5. **Effects:**
                   - Reverb: Adds space and depth
                   - Echo: Creates repeating delays
                   - Bass Boost: Enhances low frequencies
                   - Speed Up/Slow Down: Changes tempo
                
                ### 🎹 Example Combinations:
                - Pop + EDM = Energetic mashup
                - Rock + Hip-Hop = Fusion style
                - Classical + Electronic = Modern remix
                """)
    
    # ========================================================================
    # TAB 5: Order Statistics Demo
    # ========================================================================
    with tab5:
        st.header("Interactive Order Statistics Demonstration")
        
        st.markdown("Visualize how different order statistics algorithms work")
        
        demo_size = st.slider("Dataset Size", 10, 1000, 100)
        
        if st.button("Generate Random Data", type="primary"):
            data = np.random.randn(demo_size)
            
            median = np.median(data)
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Median", f"{median:.3f}")
            with col2:
                st.metric("Q1 (25%)", f"{q1:.3f}")
            with col3:
                st.metric("Q3 (75%)", f"{q3:.3f}")
            with col4:
                st.metric("IQR", f"{iqr:.3f}")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            ax1.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.axvline(median, color='red', linestyle='--', linewidth=2, label='Median')
            ax1.axvline(q1, color='orange', linestyle='--', linewidth=2, label='Q1')
            ax1.axvline(q3, color='green', linestyle='--', linewidth=2, label='Q3')
            ax1.set_xlabel('Value')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Distribution with Order Statistics')
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            ax2.boxplot(data, vert=True)
            ax2.set_ylabel('Value')
            ax2.set_title('Box Plot (Order Statistics Visualization)')
            ax2.grid(alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
            
            st.subheader("Algorithm Performance for Finding Median")
            
            k = len(data) // 2
            methods = ['quickselect', 'randomized', 'sort', 'numpy_partition']
            times = []
            
            for method in methods:
                _, elapsed = find_kth_element(data, k, method)
                times.append(elapsed * 1000)
            
            perf_df = {
                'Method': [m.replace('_', ' ').title() for m in methods],
                'Time (ms)': [f"{t:.6f}" for t in times]
            }
            
            st.table(perf_df)
    
    # Footer
    st.markdown("---")
    st.markdown("**💡 Quick Tips:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎵 Audio Analysis**
        - Upload MP3/WAV files
        - View median statistics
        - Analyze tempo & beats
        """)
    
    with col2:
        st.markdown("""
        **🎧 Remix/Mashup**
        - Mix two songs together
        - Match tempo automatically
        - Apply pitch shift & effects
        """)
    
    with col3:
        st.markdown("""
        **📊 Algorithms**
        - Compare performance
        - Test order statistics
        - Visualize results
        """)
    
    st.markdown("---")
    st.markdown("*Built with ❤️ using Streamlit, Librosa, NumPy & SciPy*")


if __name__ == "__main__":
    main()