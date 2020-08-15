
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Ignore a bunch of deprecation warnings
import warnings
warnings.filterwarnings("ignore")

import copy
import os
import time

import crepe
#import ddsp
#import ddsp.training
import training
from training.models import Autoencoder
#from ddsp.colab.colab_utils import (download, play, record, specplot, upload,
#                                   DEFAULT_SAMPLE_RATE)
import gin
#from google.colab import files
import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import spectral_ops
import eval_util
# Helper Functions
sample_rate = 16000  


print('Done!')


import librosa
import os
cnt = 0
audio_file_directory =  '/Users/rtchen/Downloads/nsynth-test/audio/bass_electronic'
def find_model_dir(dir_name):
  # Iterate through directories until model directory is found
  for root, dirs, filenames in os.walk(dir_name):
    for filename in filenames:
      if filename.endswith(".gin") and not filename.startswith("."):
        model_dir = root
        break
  return model_dir 
def shift_ld(audio_features, ld_shift=0.0):
  """Shift loudness by a number of ocatves."""
  audio_features['loudness_db'] += ld_shift
  return audio_features


def shift_f0(audio_features, f0_octave_shift=0.0):
  """Shift f0 by a number of ocatves."""
  audio_features['f0_hz'] *= 2.0 ** (f0_octave_shift)
  audio_features['f0_hz'] = np.clip(audio_features['f0_hz'], 
                                    0.0, 
                                    librosa.midi_to_hz(110.0))
  return audio_features


def mask_by_confidence(audio_features, confidence_level=0.1):
  """For the violin model, the masking causes fast dips in loudness. 
  This quick transient is interpreted by the model as the "plunk" sound.
  """
  mask_idx = audio_features['f0_confidence'] < confidence_level
  audio_features['f0_hz'][mask_idx] = 0.0
  # audio_features['loudness_db'][mask_idx] = -ddsp.spectral_ops.LD_RANGE
  return audio_features


def smooth_loudness(audio_features, filter_size=3):
  """Smooth loudness with a box filter."""
  smoothing_filter = np.ones([filter_size]) / float(filter_size)
  audio_features['loudness_db'] = np.convolve(audio_features['loudness_db'], 
                                           smoothing_filter, 
                                           mode='same')
  return audio_features



accu_harmonics = np.zeros((1000,5))
for filename in os.listdir(audio_file_directory):
    print(filename) 
    cnt = cnt+1
    if cnt >1:
        break
    audio,sr = librosa.load(os.path.join(audio_file_directory,filename), sr=sample_rate)
    audio = audio[np.newaxis, :]
    print(audio.shape)
    print('\nExtracting audio features...')


    spectral_ops.reset_crepe()


    start_time = time.time()
    audio_features = eval_util.compute_audio_features(audio)
    audio_features['loudness_db'] = audio_features['loudness_db'].astype(np.float32)
    audio_features_mod = None
    print('Audio features took %.1f seconds' % (time.time() - start_time))

    model = 'Violin' #@param ['Violin', 'Flute', 'Flute2', 'Trumpet', 'Tenor_Saxophone','Upload your own (checkpoint folder as .zip)']
    MODEL = model


    gin_file = os.path.join('.', 'operative_config-0.gin')

    with gin.unlock_config():
        gin.parse_config_file(gin_file, skip_unknown=True)
 

    ckpt_files = [f for f in tf.io.gfile.listdir('.') if 'ckpt' in f]
    ckpt_name = ckpt_files[0].split('.')[0]
    ckpt = os.path.join('.', ckpt_name)


    time_steps_train = gin.query_parameter('DefaultPreprocessor.time_steps')
    n_samples_train = gin.query_parameter('Additive.n_samples')
    hop_size = int(n_samples_train / time_steps_train)

    time_steps = int(audio.shape[1] / hop_size)
    n_samples = time_steps * hop_size


    gin_params = [
        'Additive.n_samples = {}'.format(n_samples),
        'FilteredNoise.n_samples = {}'.format(n_samples),
        'DefaultPreprocessor.time_steps = {}'.format(time_steps),
    ]

    with gin.unlock_config():
        gin.parse_config(gin_params)



    for key in ['f0_hz', 'f0_confidence', 'loudness_db']:
        audio_features[key] = audio_features[key][:time_steps]
        audio_features['audio'] = audio_features['audio'][:, :n_samples]



    model = Autoencoder()
    model.restore(ckpt)


    start_time = time.time()
    _ = model(audio_features, training=False)
    print('Restoring model took %.1f seconds' % (time.time() - start_time))
    auto_adjust = True #@param{type:"boolean"}


    f0_octave_shift =  0 #@param {type:"slider", min:-2, max:2, step:1}
    f0_confidence_threshold =  0 #@param {type:"slider", min:0.0, max:1.0, step:0.05}
    loudness_db_shift = 0 #@param {type:"slider", min:-20, max:20, step:1}



    audio_features_mod = {k: v.copy() for k, v in audio_features.items()}
    if MODEL in ['Violin', 'Flute', 'Flute2', 'Trumpet', 'Saxophone', 'Tenor_Saxophone']:
    # Adjust the peak loudness.
        l = audio_features['loudness_db']
        model_ld_avg_max = {
           'Violin': -34.0,
           'Flute': -45.0,
           'Flute2': -44.0,
           'Trumpet': -52.3,
           'Tenor_Saxophone': -31.2
           }[MODEL]
        ld_max = np.max(audio_features['loudness_db'])
        ld_diff_max = model_ld_avg_max - ld_max
        audio_features_mod = shift_ld(audio_features_mod, ld_diff_max)

        # Further adjust the average loudness above a threshold.
        l = audio_features_mod['loudness_db']
        model_ld_mean = {
            'Violin': -44.0,
            'Flute': -51.0,
            'Flute2': -53.0,
            'Trumpet': -69.2,
            'Tenor_Saxophone': -50.8
        }[MODEL]
        ld_thresh = -70.0
        ld_mean = np.mean(l[l > ld_thresh])
        ld_diff_mean = model_ld_mean - ld_mean
        audio_features_mod = shift_ld(audio_features_mod, ld_diff_mean)

    
        model_p_mean = {
           'Violin': 73.0,
           'Flute': 81.0,
           'Flute2': 74.0,
           'Trumpet': 65.8,
           'Tenor_Saxophone': 57.8
        }[MODEL]
        p = librosa.hz_to_midi(audio_features['f0_hz'])
        p[p == -np.inf] = 0.0
        p_mean = p[l > ld_thresh].mean()
        p_diff = model_p_mean - p_mean
        p_diff_octave = p_diff / 12.0
        round_fn = np.floor if p_diff_octave > 1.5 else np.ceil
        p_diff_octave = round_fn(p_diff_octave)
        audio_features_mod = shift_f0(audio_features_mod, p_diff_octave)

    else:
        print('\nUser uploaded model: disabling auto-adjust.')

  
    audio_features_mod = shift_ld(audio_features_mod, loudness_db_shift)
    audio_features_mod = shift_f0(audio_features_mod, f0_octave_shift)
    audio_features_mod = mask_by_confidence(audio_features_mod, f0_confidence_threshold)
    
    af = audio_features if audio_features_mod is None else audio_features_mod

    #af = audio_features
    start_time = time.time()
    audio_gen = model(af, training=False)
    print('Prediction took %.1f seconds' % (time.time() - start_time))
    control_dict = model.get_controls(af)
    top_harmonics = control_dict['harmonic_distribution'][0, :, 0:5]
    accu_harmonics = accu_harmonics+top_harmonics[0:1000,:]
    plt.clf()
    plt.plot(top_harmonics)
    plt.xlabel('time(frames)')
    plt.ylabel('amplitude(unnormalized)')
    plt.title(filename.strip('.wav'))
    plt.savefig(os.path.join(audio_file_directory,filename.strip('.wav')))
accu_harmonics = accu_harmonics/(cnt-1)
'''
plt.clf()
plt.plot(top_harmonics)
plt.xlabel('time(frames)')
plt.ylabel('amplitude(unnormalized)')
plt.title('0average')
plt.show()
'''
audio_gen = model(af, training=False)
D = librosa.stft(np.array(audio_gen[0]),n_fft = 65535,hop_length =int(65535))
#D = np.abs(D)
#print(D.shape)
D=librosa.amplitude_to_db(D,ref=np.max)

Hz_size = D.shape[0]
max_freqs_value = np.ones(30)*-100
max_freqs = np.zeros(30)
freqs = librosa.core.fft_frequencies(sr = 16000, n_fft = 65535)
for i in range(30):
    for j in range(Hz_size):
        flag = 0
        if max_freqs_value[i] >= D[j][0]:
            continue
        for k in range(i):
            if abs(max_freqs[k]-freqs[j]) <=250:
               flag = 1
               break
        if flag == 1:
            continue
        max_freqs[i] = freqs[j]
        max_freqs_value[i] = D[j][0]

print(max_freqs)
print(max_freqs_value)
print(max_freqs/float(max_freqs[0]))
print(max_freqs_value/(float(max_freqs_value[0]*0.1)))
import matplotlib.pyplot as plt
import librosa.display
librosa.display.specshow(librosa.amplitude_to_db(D,
                                                 ref=np.max),
                          y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()











