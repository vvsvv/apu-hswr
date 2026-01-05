import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Critical Kernel: Maintain 0.986 net gain (non-explosive, only resonance)
# ==========================================
class APU_Kernel_Critical:
    def __init__(self, resolution=(256, 256)):
        self.res = resolution
        self.field = np.zeros(resolution, dtype=np.complex128)
        
        # Core physics parameters: 0.85 * 1.16 = 0.986 (< 1.0)
        self.decay = 0.85 
        self.persistence_factor = 1.16 
        
        cx, cy = resolution[0] // 2, resolution[1] // 2
        size = 40
        self.resonator_mask = np.zeros(resolution, dtype=bool)
        self.resonator_mask[cy-size:cy+size, cx-size:cx+size] = True
        self.processor_mask = np.zeros(resolution, dtype=bool)
        self.processor_mask[cy-10:cy+10, cx-10:cx+10] = True

    def emit_spectrum(self, x, y, frequencies, amplitudes, phases):
        """Emit spectrum to specified location"""
        try:
            y_grid, x_grid = np.meshgrid(
                np.arange(self.res[0]), 
                np.arange(self.res[1]), 
                indexing='ij'
            )
            dist = np.sqrt((x_grid - x)**2 + (y_grid - y)**2)
            combined_wave = np.zeros_like(self.field)
            
            for f, a, p in zip(frequencies, amplitudes, phases):
                combined_wave += a * np.exp(1j * (f * dist + p))
            
            self.field += combined_wave
            return True
        except Exception as e:
            print(f"Error in emit_spectrum: {e}")
            return False

    def step(self):
        """Execute one time evolution step"""
        try:
            # 1. Energy decay
            self.field *= self.decay
            # 2. Resonator gain
            self.field[self.resonator_mask] *= self.persistence_factor
            # 3. Weak nonlinear activation
            mag = np.abs(self.field)
            threshold = 0.05
            mask = mag > threshold
            self.field[mask] = np.tanh(mag[mask]) * (self.field[mask]/(mag[mask]+1e-8))
            return True
        except Exception as e:
            print(f"Error in step: {e}")
            return False

    def reset(self):
        """Reset field to zero state"""
        self.field = np.zeros(self.res, dtype=np.complex128)
        return self

    def get_field_magnitude(self):
        """Get field magnitude"""
        return np.abs(self.field)

# ==========================================
# 2. Utility functions for decay simulation
# ==========================================
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

# ==========================================
# 3. Spatial encoding test function
# ==========================================
def run_spatial_test(apu_instance, freqs, amps, phases, memory_state, label):
    """Run spatial encoding test with given parameters"""
    # Reset to clean field
    apu_instance.reset()
    
    # Simulate injecting spectral components from different positions
    for i, (f, a, p) in enumerate(zip(freqs, amps, phases)):
        offset = i * 5 
        apu_instance.emit_spectrum(128 + offset, 128 + offset, [f], [a], [p])
    
    mags = []
    for _ in range(100):
        apu_instance.step()
        coherence = np.abs(np.sum(apu_instance.field * np.conj(memory_state)))
        mags.append(coherence)
        
    print(f"--- {label} --- Max Coherence: {max(mags):.6f}")
    return mags

# ==========================================
# Execute example code when running this file directly
# ==========================================
if __name__ == "__main__":
    print("=== Running APU Kernel Test ===")
    
    # Test parameters
    FREQS_A = [0.2, 0.35, 0.5, 0.65, 0.8]
    AMPS_A  = [1.0, 0.8, 0.6, 0.4, 0.2]
    PHASES_A = [0.0, 1.1, 2.3, 0.5, 3.0]
    FREQS_B = [f * 1.05 for f in FREQS_A]  # 5% frequency shift

    apu = APU_Kernel_Critical()

    # --- PHASE 1: Training (form memory M) ---
    print(">>> Phase 1: Training Memory A...")
    for _ in range(50):
        apu.emit_spectrum(128, 128, FREQS_A, AMPS_A, PHASES_A)
        apu.step()
    MEMORY_STATE = apu.field.copy()

    # --- Run spatial encoding tests ---
    results_correct = run_spatial_test(apu, FREQS_A, AMPS_A, PHASES_A, MEMORY_STATE, "STATION A (MATCH)")
    results_drift = run_spatial_test(apu, FREQS_B, AMPS_A, PHASES_A, MEMORY_STATE, "STATION B (SHIFTED)")
    results_noise = run_spatial_test(apu, [np.random.random()*0.8 for _ in range(5)], 
                                     [1.0]*5, [0.0]*5, MEMORY_STATE, "RANDOM NOISE")

    # --- Visualization ---
    plt.figure(figsize=(10, 6))
    plt.plot(results_correct, label="Station A (Correct Echo)", color='green', linewidth=2)
    plt.plot(results_drift, label="Station B (De-phasing Decay)", color='orange')
    plt.plot(results_noise, label="Noise (Rapid Collapse)", color='red', linestyle='--')
    plt.title("APU Criticality Response: Memory Echo vs. Entropy")
    plt.xlabel("Time Steps (After Impulse)")
    plt.ylabel("Internal Energy Magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

