import os

# Define the musical styles
genre = [
    # 'action',
    # 'adventure',
    'arcade'
    # 'horror'
]

styles = [
    # [
    #     'data/action/batman',
    #     'data/action/doom'
    # ],
    # [
    #     'data/adventure/blade_runner',
    #     'data/adventure/indiana_jones',
    #     'data/adventure/myst'
    # ],
    [
        # 'data/arcade/blox',
        # 'data/arcade/burning_monkey',
        'data/arcade/mario'
    ]
    # [
    #     'data/horror/blood',
    #     'data/horror/house_of_the_dead'
    # ]
]

NUM_STYLES = sum(len(s) for s in styles)

NUM_INSTRUMENTS = 19
MAX_INSTRUMENTS_PER_SONG = 3
FS = 5.25

# MIDI Resolution
DEFAULT_RES = 96
MIDI_MAX_NOTES = 128
MAX_VELOCITY = 127

# Number of octaves supported
NUM_OCTAVES = 4
OCTAVE = 8

# Min and max note (in MIDI note number)
MIN_NOTE = 36
# MAX_NOTE = MIN_NOTE + NUM_OCTAVES * OCTAVE
MAX_NOTE = 70
NUM_NOTES_INSTRUMENT = MAX_NOTE - MIN_NOTE
NUM_NOTES = (NUM_INSTRUMENTS + 1) * NUM_NOTES_INSTRUMENT

# Number of beats in a bar
BEATS_PER_BAR = 4
# Notes per quarter note
NOTES_PER_BEAT = 4
# The quickest note is a half-note
# NOTES_PER_BAR = NOTES_PER_BEAT * BEATS_PER_BAR
NOTE_TIME_STEPS = 1
NOTES_PER_BAR = NOTES_PER_BEAT * BEATS_PER_BAR * NOTE_TIME_STEPS

# Training parameters
BARS = 4
BATCH_SIZE = 16
SEQ_LEN = BARS * NOTES_PER_BAR

# Hyper Parameters
OCTAVE_UNITS = 64
STYLE_UNITS = 64
NOTE_UNITS = 2
TIME_AXIS_UNITS = 256
NOTE_AXIS_UNITS = 128

TIME_AXIS_LAYERS = 2
NOTE_AXIS_LAYERS = 2

# Move file save location
OUT_DIR = 'out'
MODEL_DIR = os.path.join(OUT_DIR, 'models')
MODEL_FILE = os.path.join(OUT_DIR, 'model.h5')
SAMPLES_DIR = os.path.join(OUT_DIR, 'samples')
CACHE_DIR = os.path.join(OUT_DIR, 'cache')
