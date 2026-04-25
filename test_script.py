from analysis_unified import run_synthetic_brownian_vector_corr_test
import analysis_pipeline.velocity_spectrum

# Create synthetic dataset
result_dict = run_synthetic_brownian_vector_corr_test(frame_count=20, average_window_frames=5, particle_count=20, seed=7, base_dir='data')
config_dict = result_dict['config']
state = result_dict['state']

class Map(dict):
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = Map(value)
    def __getattr__(self, item):
        return self.get(item)
    def __setattr__(self, key, value):
        self[key] = value

config = Map(config_dict)
config.velocity_spectrum.multi_frame_average = True
config.velocity_spectrum.multi_frame_count = 2

spectrum_df = analysis_pipeline.velocity_spectrum.run_velocity_spectrum_core(config, state)

print(f"Spectrum row count: {len(spectrum_df)}")
