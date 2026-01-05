import librosa
import numpy as np
import matplotlib.pyplot as plt
import traceback
import sys

def safe_import():
    """Safely import module"""
    try:
        print("Importing APU module...")
        import apu_final_atomic_multi_frequencies_decay_v4 as apu_module
        print("APU module imported successfully")
        return apu_module
    except Exception as e:
        print(f"Import failed: {e}")
        traceback.print_exc()
        sys.exit(1)

def load_audio_safe():
    """Safely load audio file"""
    try:
        print("Loading audio file...")
        y, sr = librosa.load('cup_sound.wav', sr=None)
        print(f"Audio loaded successfully: sample_rate={sr}Hz, length={len(y)} samples")
        return y, sr
    except Exception as e:
        print(f"Audio loading failed: {e}")
        # Use simulated data if audio file not found
        print("Using simulated data instead...")
        sr = 22050
        t = np.linspace(0, 2, sr*2)
        # Simulate frequency components
        y = 0.5 * np.sin(2*np.pi*622*t) + 0.3 * np.sin(2*np.pi*735*t) + 0.2 * np.sin(2*np.pi*5041*t)
        y += 0.1 * np.random.randn(len(t))
        return y, sr

def extract_features(y, sr):
    """Extract audio features"""
    try:
        print("Extracting spectral features...")
        # Use windowed processing
        n_fft = 2048
        hop_length = 512
        D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        
        # Get frequency axis
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        # Find frequencies with highest energy
        avg_spectrum = np.mean(D, axis=1)
        top_indices = np.argsort(avg_spectrum)[-50:]
        
        # Filter low-frequency noise
        clean_indices = [i for i in top_indices if freqs[i] > 200][:8]
        
        CUP_FREQS = freqs[clean_indices].tolist()
        CUP_AMPS = (avg_spectrum[clean_indices] / np.max(avg_spectrum[clean_indices])).tolist()
        
        # Get phase information
        phase_data = np.angle(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        CUP_PHASES = [np.mean(phase_data[i, :]) for i in clean_indices]
        
        print(">>> Audio features extracted successfully:")
        for i, (f, a, p) in enumerate(zip(CUP_FREQS, CUP_AMPS, CUP_PHASES)):
            print(f"  Component {i+1}: {f:.2f} Hz (strength: {a:.4f}, phase: {p:.2f})")
        
        return CUP_FREQS, CUP_AMPS, CUP_PHASES
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        # Use default frequencies
        print("Using default frequencies...")
        return [622.0, 735.0, 5041.0, 575.0, 729.0, 1939.0, 736.0, 246.0], \
               [0.96, 0.97, 0.97, 0.98, 0.98, 0.99, 0.99, 1.00], \
               [0.0] * 8

def simulate_decay_process(apu, memory_state, target_freqs, target_amps, target_phases):
    """Simulate the decay process of resonance in the APU system"""
    steps_active = 50   # Signal active phase
    steps_decay = 150   # Decay phase after signal stops (simulating 2-3 seconds)
    
    coherence_history = []
    apu.reset()  # Start with clean field
    
    # 1. Active excitation phase
    for t in range(steps_active):
        apu.emit_spectrum(128, 128, target_freqs, target_amps, target_phases)
        apu.step()
        c = np.abs(np.sum(apu.field * np.conj(memory_state)))
        coherence_history.append(c)
        
    # 2. Decay phase after signal stops
    # We stop emit_spectrum, only run apu.step()
    for t in range(steps_decay):
        apu.step()
        c = np.abs(np.sum(apu.field * np.conj(memory_state)))
        coherence_history.append(c)
        
    return coherence_history

def main():
    """Main function"""
    print("=== APU Audio Fingerprint Recognition System ===")
    
    # 1. Safely import module
    apu_module = safe_import()
    
    # 2. Create APU instance
    print("\nCreating APU kernel instance...")
    try:
        apu = apu_module.APU_Kernel_Critical(resolution=(256, 256))
        print(f"APU instance created successfully, resolution: {apu.res}")
    except Exception as e:
        print(f"Failed to create APU instance: {e}")
        traceback.print_exc()
        return
    
    # 3. Load audio and extract features
    y, sr = load_audio_safe()
    cup_freqs_hz, cup_amps, cup_phases = extract_features(y, sr)
    
    # 4. Map to APU space (0-1 range)
    print("\nMapping frequencies to APU space...")
    max_hz = 6000.0  # Assume maximum frequency is 6000Hz
    apu_freqs = [f / max_hz for f in cup_freqs_hz]
    
    # Normalize amplitudes
    apu_amps = [a / max(cup_amps) for a in cup_amps] if cup_amps else [1.0] * len(apu_freqs)
    
    print("Mapped APU parameters:")
    for i, (f_hz, f_apu, a) in enumerate(zip(cup_freqs_hz, apu_freqs, apu_amps)):
        print(f"  Component {i+1}: {f_hz:.1f} Hz -> {f_apu:.4f} (strength: {a:.4f})")
    
    # 5. Training phase: form memory
    print("\n>>> Locking audio fingerprint into holographic standing wave field...")
    try:
        apu.reset()
        for step in range(50):
            success = apu.emit_spectrum(128, 128, apu_freqs, apu_amps, cup_phases)
            if not success:
                print(f"Step {step} emission failed")
            apu.step()
            if step % 10 == 0:
                print(f"  Training progress: {step+1}/50")
        
        # Save memory state
        CUP_MEMORY = apu.field.copy()
        memory_energy = np.abs(CUP_MEMORY).sum()
        print(f"Memory field created, total energy: {memory_energy:.2f}")
    except Exception as e:
        print(f"Training phase failed: {e}")
        traceback.print_exc()
        return
    
    # 6. Run decay simulation to visualize resonance behavior
    print("\n>>> Running decay process simulation...")
    try:
        # Create separate APU instances for each simulation
        apu_match = apu_module.APU_Kernel_Critical(resolution=(256, 256))
        apu_mismatch = apu_module.APU_Kernel_Critical(resolution=(256, 256))
        
        # Simulate matched resonance
        history_match = simulate_decay_process(apu_match, CUP_MEMORY, apu_freqs, apu_amps, cup_phases)
        
        # Simulate mismatched resonance (5% frequency shift)
        mismatched_freqs = [f * 1.05 for f in apu_freqs]
        history_wrong = simulate_decay_process(apu_mismatch, CUP_MEMORY, mismatched_freqs, apu_amps, cup_phases)
        
        # Plot decay comparison
        plt.figure(figsize=(12, 6))
        plt.axvline(x=50, color='gray', linestyle='--', label='Signal Cut-off (Subway Stopped)')
        plt.plot(history_match, color='#2ecc71', linewidth=3, label='Matched Resonance (High Q)')
        plt.plot(history_wrong, color='#e74c3c', linewidth=2, label='Mismatched Noise (Low Q)')
        
        plt.title('Temporal Evolution of Coherence: The "Subway Cup" Decay Effect', fontsize=14)
        plt.xlabel('Time Steps (Arbitrary Units)', fontsize=12)
        plt.ylabel('Holographic Coherence Score', fontsize=12)
        plt.fill_between(range(len(history_match)), history_match, color='#2ecc71', alpha=0.1)
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.savefig('decay_process_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Calculate decay metrics
        match_decay = history_match[50] / max(history_match[:50]) if max(history_match[:50]) > 0 else 0
        wrong_decay = history_wrong[50] / max(history_wrong[:50]) if max(history_wrong[:50]) > 0 else 0
        
        print(f"Resonance decay rate (matched): {match_decay:.4f}")
        print(f"Resonance decay rate (mismatched): {wrong_decay:.4f}")
        print(f"Decay ratio (difference factor): {match_decay/wrong_decay:.2f}x slower decay for matched resonance")
        
    except Exception as e:
        print(f"Decay simulation failed: {e}")
        traceback.print_exc()
    
    # 7. Verification function for quick tests
    def verify(test_freqs, label, use_phases=None):
        """Verification function"""
        try:
            apu.reset()
            
            # Inject test signal
            phases = use_phases if use_phases is not None else cup_phases
            success = apu.emit_spectrum(128, 128, test_freqs, apu_amps, phases)
            if not success:
                print(f"{label}: Signal injection failed")
                return 0.0
            
            # Evolve for several steps
            for _ in range(10):
                apu.step()
            
            # Calculate coherence
            coherence = np.abs(np.sum(apu.field * np.conj(CUP_MEMORY)))
            print(f"{label} coherence score: {coherence:.6f}")
            return coherence
        except Exception as e:
            print(f"{label} verification failed: {e}")
            return 0.0
    
    # 8. Execute quick verification tests
    print("\n>>> Starting quick verification tests:")
    
    # Test A: Original audio (perfect match)
    score_a = verify(apu_freqs, "[Sample A: Original Audio]")
    
    # Test B: Slightly shifted frequencies (simulate different object)
    fake_freqs = [f * 1.02 for f in apu_freqs]  # 2% frequency shift
    score_b = verify(fake_freqs, "[Sample B: Slight Difference (2% shift)]")
    
    # Test C: Random frequencies (simulate noise)
    np.random.seed(42)  # Fixed random seed for reproducibility
    noise_freqs = [np.random.random() * 0.8 for _ in range(len(apu_freqs))]
    score_c = verify(noise_freqs, "[Sample C: Random Noise]")
    
    # 9. Display results
    print("\n" + "="*50)
    print("=== VERIFICATION RESULTS ===")
    print(f"Original audio coherence: {score_a:.6f}")
    print(f"Slight difference coherence: {score_b:.6f}")
    print(f"Random noise coherence: {score_c:.6f}")
    
    if score_a > 0 and score_b > 0:
        margin = (score_a - score_b) / score_a * 100
        print(f"\n>>> APU discrimination improvement: {margin:.2f}%")
        
        if score_a > score_b > score_c:
            print("✅ VERIFICATION SUCCESS: APU successfully recognized specific audio fingerprint!")
        else:
            print("⚠️  Verification not ideal, parameter adjustment may be needed")
    
    # 10. Advanced visualization
    try:
        print("\nGenerating comprehensive visualization...")
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # Spectrum plot
        axes[0, 0].plot(cup_freqs_hz, cup_amps, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_title('Audio Spectral Features')
        axes[0, 0].set_xlabel('Frequency (Hz)')
        axes[0, 0].set_ylabel('Relative Strength')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Coherence comparison (bar chart)
        labels = ['Original Audio', '2% Shift', 'Random Noise']
        scores = [score_a, score_b, score_c]
        colors = ['green', 'orange', 'red']
        
        bars = axes[0, 1].bar(labels, scores, color=colors, alpha=0.7)
        axes[0, 1].set_title('Coherence Score Comparison')
        axes[0, 1].set_ylabel('Coherence')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{score:.4f}', ha='center', va='bottom')
        
        # Memory field magnitude
        memory_mag = np.abs(CUP_MEMORY)
        im1 = axes[0, 2].imshow(memory_mag, cmap='hot', aspect='auto')
        axes[0, 2].set_title('Memory Field Magnitude')
        axes[0, 2].set_xlabel('X')
        axes[0, 2].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0, 2])
        
        # Phase distribution
        memory_phase = np.angle(CUP_MEMORY)
        im2 = axes[1, 0].imshow(memory_phase, cmap='hsv', aspect='auto', vmin=-np.pi, vmax=np.pi)
        axes[1, 0].set_title('Memory Field Phase')
        axes[1, 0].set_xlabel('X')
        axes[1, 0].set_ylabel('Y')
        plt.colorbar(im2, ax=axes[1, 0])
        
        # Decay process comparison (if available)
        if 'history_match' in locals() and 'history_wrong' in locals():
            axes[1, 1].axvline(x=50, color='gray', linestyle='--', alpha=0.5)
            axes[1, 1].plot(history_match, color='#2ecc71', linewidth=2, label='Matched')
            axes[1, 1].plot(history_wrong, color='#e74c3c', linewidth=2, label='Mismatched')
            axes[1, 1].set_title('Resonance Decay Comparison')
            axes[1, 1].set_xlabel('Time Steps')
            axes[1, 1].set_ylabel('Coherence')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Frequency shift impact
        axes[1, 2].plot([0, 1, 2, 5, 10], 
                       [score_a, score_b, verify([f*1.01 for f in apu_freqs], "", cup_phases), 
                        verify([f*1.05 for f in apu_freqs], "", cup_phases),
                        verify([f*1.10 for f in apu_freqs], "", cup_phases)],
                       'ro-', linewidth=2, markersize=8)
        axes[1, 2].set_title('Coherence vs Frequency Shift')
        axes[1, 2].set_xlabel('Frequency Shift (%)')
        axes[1, 2].set_ylabel('Coherence Score')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('apu_comprehensive_results.png', dpi=150, bbox_inches='tight')
        print("Visualization saved as 'apu_comprehensive_results.png'")
        plt.show()
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Program execution error: {e}")
        traceback.print_exc()

