"""
Handles MIDI file loading
"""
import pretty_midi
import pretty_midi as pm
import numpy as np
from constants import *
import midi
from copy import deepcopy
import mido
import os
import glob
import tensorflow as tf


def midi_encode(note_seq, resolution=NOTES_PER_BEAT, step=1):
    """
    Takes a piano roll and encodes it into MIDI pattern
    """
    # Instantiate a MIDI Pattern (contains a list of tracks)
    pattern = midi.Pattern()
    pattern.resolution = resolution
    # Instantiate a MIDI Track (contains a list of MIDI events)
    track = midi.Track()
    # Append the track to the pattern
    pattern.append(track)

    play = note_seq[:, :, 0]
    replay = note_seq[:, :, 1]
    volume = note_seq[:, :, 2]

    # The current pattern being played
    current = np.zeros_like(play[0])
    # Absolute tick of last event
    last_event_tick = 0
    # Amount of NOOP ticks
    noop_ticks = 0

    for tick, data in enumerate(play):
        data = np.array(data)

        if not np.array_equal(current, data):# or np.any(replay[tick]):
            noop_ticks = 0

            for index, next_volume in np.ndenumerate(data):
                if next_volume > 0 and current[index] == 0:
                    # Was off, but now turned on
                    evt = midi.NoteOnEvent(
                        tick=(tick - last_event_tick) * step,
                        velocity=int(volume[tick][index[0]] * MAX_VELOCITY),
                        pitch=index[0]
                    )
                    track.append(evt)
                    last_event_tick = tick
                elif current[index] > 0 and next_volume == 0:
                    # Was on, but now turned off
                    evt = midi.NoteOffEvent(
                        tick=(tick - last_event_tick) * step,
                        pitch=index[0]
                    )
                    track.append(evt)
                    last_event_tick = tick

                elif current[index] > 0 and next_volume > 0 and replay[tick][index[0]] > 0:
                    # Handle replay
                    evt_off = midi.NoteOffEvent(
                        tick=(tick- last_event_tick) * step,
                        pitch=index[0]
                    )
                    track.append(evt_off)
                    evt_on = midi.NoteOnEvent(
                        tick=0,
                        velocity=int(volume[tick][index[0]] * MAX_VELOCITY),
                        pitch=index[0]
                    )
                    track.append(evt_on)
                    last_event_tick = tick

        else:
            noop_ticks += 1

        current = data

    tick += 1

    # Turn off all remaining on notes
    for index, vol in np.ndenumerate(current):
        if vol > 0:
            # Was on, but now turned off
            evt = midi.NoteOffEvent(
                tick=(tick - last_event_tick) * step,
                pitch=index[0]
            )
            track.append(evt)
            last_event_tick = tick
            noop_ticks = 0

    # Add the end of track event, append it to the track
    eot = midi.EndOfTrackEvent(tick=noop_ticks)
    track.append(eot)

    return pattern


def midi_decode_instrument(pattern,
                           instrument,
                           classes=MIDI_MAX_NOTES,
                           step=None):
    """
    Takes a MIDI pattern and decodes it into a piano roll.
    """
    if step is None:
        step = pattern.ticks_per_beat // NOTES_PER_BEAT

    # Extract all tracks at highest resolution
    merged_replay = None
    merged_volume = None

    for track in pattern.tracks:
        # The downsampled sequences
        replay_sequence = []
        volume_sequence = []

        # Raw sequences
        replay_buffer = [np.zeros((classes,))]
        volume_buffer = [np.zeros((classes,))]

        for i, event in enumerate(track):
            if 'channel' not in event.dict():
                continue
            # Duplicate the last note pattern to wait for next event
            for _ in range(event.time):
                replay_buffer.append(np.zeros(classes))
                volume_buffer.append(np.copy(volume_buffer[-1]))

                # Buffer & downscale sequence
                if len(volume_buffer) > step:
                    # Take the min
                    replay_any = np.minimum(np.sum(replay_buffer[:-1], axis=0), 1)
                    replay_sequence.append(replay_any)

                    # Determine volume by max
                    volume_sum = np.amax(volume_buffer[:-1], axis=0)
                    volume_sequence.append(volume_sum)

                    # Keep the last one (discard things in the middle)
                    replay_buffer = replay_buffer[-1:]
                    volume_buffer = volume_buffer[-1:]

            if event.dict()['type'] == 'end_of_track':
                break

            # Modify the last note pattern
            # TODO: Nu stiu daca e bine asa dar merge.
            #   Poate trebuie sa fin cont si de channel petnru fiecare event
            if event.dict()['type'] == 'note_on' and event.dict()['channel'] == instrument:
                if event.dict()['velocity'] == 0:
                    pitch = event.dict()['note']
                    volume_buffer[-1][pitch] = 0
                else:
                    pitch = event.dict()['note']
                    velocity = event.dict()['velocity']
                    volume_buffer[-1][pitch] = velocity / MAX_VELOCITY

                    # Check for replay_buffer, which is true if the current note was previously played and needs to be replayed
                    if len(volume_buffer) > 1 and volume_buffer[-2][pitch] > 0 and volume_buffer[-1][pitch] > 0:
                        replay_buffer[-1][pitch] = 1
                        # Override current volume with previous volume
                        volume_buffer[-1][pitch] = volume_buffer[-2][pitch]

            if event.dict()['type'] == 'note_off' and event.dict()['channel'] == instrument:
                pitch = event.dict()['note']
                volume_buffer[-1][pitch] = 0

        # Add the remaining
        replay_any = np.minimum(np.sum(replay_buffer, axis=0), 1)
        replay_sequence.append(replay_any)
        volume_sequence.append(volume_buffer[0])

        replay_sequence = np.array(replay_sequence)
        volume_sequence = np.array(volume_sequence)
        assert len(volume_sequence) == len(replay_sequence)

        if merged_volume is None:
            merged_replay = replay_sequence
            merged_volume = volume_sequence
        else:
            # Merge into a single track, padding with zeros of needed
            if len(volume_sequence) > len(merged_volume):
                # Swap variables such that merged_notes is always at least
                # as large as play_sequence
                tmp = replay_sequence
                replay_sequence = merged_replay
                merged_replay = tmp

                tmp = volume_sequence
                volume_sequence = merged_volume
                merged_volume = tmp

            assert len(merged_volume) >= len(volume_sequence)

            diff = len(merged_volume) - len(volume_sequence)
            merged_replay += np.pad(replay_sequence, ((0, diff), (0, 0)), 'constant')
            merged_volume += np.pad(volume_sequence, ((0, diff), (0, 0)), 'constant')

    merged = np.stack([np.ceil(merged_volume), merged_replay, merged_volume], axis=2)
    # Prevent stacking duplicate notes to exceed one.
    merged = np.minimum(merged, 1)
    return merged


def load_midi(fname):
    p = mido.MidiFile(fname)
    cache_path = os.path.join(CACHE_DIR, fname + '.npy')
    try:
        note_seq = np.load(cache_path)
    except Exception as e:
        # Perform caching
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        note_seq = midi_decode_v1(p)
        np.save(cache_path, note_seq)

    assert len(note_seq.shape) == 3, note_seq.shape
    assert note_seq.shape[1] == MIDI_MAX_NOTES, note_seq.shape
    assert note_seq.shape[2] == 3, note_seq.shape
    assert (note_seq >= 0).all()
    assert (note_seq <= 1).all()
    return note_seq


def load_midi_v2(fname):

    p = pm.PrettyMIDI(fname)
    cache_path = os.path.join(CACHE_DIR, fname + '.npy')
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    note_seq = midi_decode_v2(p)
    # np.save(cache_path, note_seq)
    # try:
    #     note_seq = np.load(cache_path)
    # except Exception:
    #     # Perform caching
    #     os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    #
    #     note_seq = midi_decode_v2(p)
    #     print(note_seq[1].shape)
    #     np.save(cache_path, note_seq)

    return note_seq


def compute_drum_piano_roll(instrument, piano_roll, fs):

    roll = deepcopy(piano_roll)
    for note in instrument.notes:
        # Should interpolate
        roll[note.pitch, int(note.start * fs):int(note.end * fs)] += note.velocity

    return roll


def midi_decode_v2(p):

    # Compute piano rolls for every instrument
    # Remove duplicated instruments and keep only the one with max notes length
    instruments = {}
    sorted_instruments = sorted(p.instruments, key=lambda x: len(x.notes), reverse=True)
    for instrument in sorted_instruments:
        if instrument.program not in instruments:
            instruments[instrument.program] = instrument
    # for instrument in instruments.values():
    #     print(instrument, pretty_midi.program_to_instrument_name(instrument.program), len(instrument.notes))
    # TODO: compute for drums separately because pretty midi can't
    # TODO: determine the frequency for a 16th note? (but if the fs is higher then no replay matrix is needed since the
    #   pauses between consecutive notes are captured)
    piano_rolls = [] # [[instrument, t_play, t_volume]]
    for instrument in instruments.values():
        print(pm.program_to_instrument_name(instrument.program), instrument.is_drum,
              instrument.get_piano_roll(FS).shape)
        if instrument.is_drum:
            piano_rolls.append([instrument, compute_drum_piano_roll(instrument, instrument.get_piano_roll(FS), FS)])
        else:
            piano_rolls.append([instrument, instrument.get_piano_roll(FS)])

    # Pad the smaller piano rolls with zeros so that all instruments have the same time_steps
    pitches, max_time_steps = sorted([piano_roll[1].shape for piano_roll in piano_rolls], key=lambda x: -x[1])[0]
    for piano_roll in piano_rolls:
        padding = np.zeros((pitches, max_time_steps - piano_roll[1].shape[1]))
        piano_roll[1] = np.concatenate((piano_roll[1], padding), axis=1)

        # Compute the 'play' and 'velocity' matrices
        # TODO: should I leave the piano roll as it is right now with the velocity in it?
        t_volume = deepcopy(piano_roll[1])
        normalize_velocity = lambda v: v / MAX_VELOCITY
        vfunc = np.vectorize(normalize_velocity)
        t_volume = vfunc(t_volume)

        t_play = piano_roll[1]
        t_play[t_play == 100] = 1

        piano_roll.append(t_volume)

    # Compute final array with instrument dimension
    # TODO: what should be the instrument encoding for drums? for now in NUM_INSTRUMENTS
    drum_roll_play = np.zeros((pitches, max_time_steps))
    drum_roll_volume = np.zeros((pitches, max_time_steps))
    final = np.zeros((max_time_steps, (NUM_INSTRUMENTS + 1) * pitches, 2)) # + drum dimension
    for piano_roll in piano_rolls:
        instrument = piano_roll[0]
        t_play = piano_roll[1]
        t_volume = piano_roll[2]

        # If there are more drums then put them in a single channel
        if instrument.is_drum:
            drum_roll_play = drum_roll_play + t_play
            drum_roll_volume = drum_roll_volume + t_volume
        else:
            # print(np.stack([t_volume.T, t_play.T], axis=2).shape)
            # print(t_volume.flatten('F')[20*128:21*128])
            # print(instrument.program)
            # print(t_volume.shape) # (128, 383)
            # print(t_volume.flatten('F').shape)
            # print(t_volume[:, 20])
            # print(final[:, :, 0].shape)
            # print(final[:, instrument.program * pitches:(instrument.program + 1) * pitches, 0].shape) # (383, 128)
            final[:, instrument.program * pitches:(instrument.program + 1) * pitches, 0] = t_play.T
            final[:, instrument.program * pitches:(instrument.program + 1) * pitches, 1] = t_volume.T
            #final[:, instrument.program] = np.stack([t_volume.T, t_play.T], axis=2) # 37

    # Limit the notes in any case there are more drums
    drum_roll_play[drum_roll_play > 1] = 1
    drum_roll_volume[drum_roll_volume > 1] = 1
    final[:, NUM_INSTRUMENTS * pitches:(NUM_INSTRUMENTS + 1) * pitches, 0] = drum_roll_play.T
    final[:, NUM_INSTRUMENTS * pitches:(NUM_INSTRUMENTS + 1) * pitches, 1] = drum_roll_volume.T
    # print(final[20, 37*pitches:38*pitches, 0])
    # drum_roll = np.stack([drum_roll_volume.T, drum_roll_play.T], axis=2)
    # final[:, NUM_INSTRUMENTS] = drum_roll
    #print(final[NUM_INSTRUMENTS, :, max_time_steps - 100, 1])

    beat_duration = p.get_beats()[1] - p.get_beats()[0]

    return p.get_beats(), final


def midi_decode_v1(pattern,
                   classes=MIDI_MAX_NOTES,
                   step=None):

    channel_to_program = {}
    for i, track in enumerate(pattern.tracks):
        print('Track {}: {}'.format(i, track.name))
        for msg in track:
            msg_dict = msg.dict()
            if msg_dict['type'] == 'program_change':
                channel_to_program[msg_dict['channel']] = msg_dict['program']

    if 9 not in channel_to_program:
        channel_to_program[9] = 118
    print(channel_to_program)
    instruments = []
    for channel, program in channel_to_program.items():
        decoded = midi_decode_instrument(pattern, channel, classes=classes, step=step)
        instruments.append((program, decoded))
        print(pm.program_to_instrument_name(program), decoded.shape)
    print(instruments[5][1][0, :, 2])

    time_steps, pitches, dims = instruments[0][1].shape
    final = np.zeros((NUM_INSTRUMENTS, time_steps, pitches, dims))
    for instrument in instruments:
        final[instrument[0]] = instrument[1]
    print(final.shape)

    return final


def transpose_keys(filename, out_dir):

    one_track_midi = pm.PrettyMIDI(filename)
    tempo = pm.get_tempo_changes()[1][0]
    for i in range(12):
        midi_transposed = pm.PrettyMIDI(initial_tempo=tempo)
        midi_transposed.time_signature_changes = deepcopy(one_track_midi.time_signature_changes)
        midi_transposed.key_signature_changes = [pm.KeySignature(i, 0)]
        midi_transposed.instruments = deepcopy(one_track_midi.instruments)

        file_name = out_dir + "/" + filename.split("/")[-1][:-4] + "_" + pm.key_number_to_key_name(i).replace(" ", "_") + ".mid"
        print(file_name)
        f = open(file_name, "w")
        f.close()
        midi_transposed.write(file_name)


def are_same_notes(f1, f2):

    p1 = pm.PrettyMIDI(f1)
    p2 = pm.PrettyMIDI(f2)

    for instrument_idx in range(len(p1.instruments)):
        for i in range(len(p1.instruments[instrument_idx].notes)):
            if p1.instruments[instrument_idx].notes[i].pitch != p2.instruments[instrument_idx].notes[i].pitch:
                return False
    return True


def delete_same_files(dir_name):

    transposed_files = glob.glob(dir_name + "/*.mid")
    to_delete = {}

    for t1 in transposed_files:
        for t2 in transposed_files:
            if t1 != t2:
                # Check if they are the same
                if are_same_notes(t1, t2):
                    if t1 not in to_delete:
                        to_delete[t1] = [t2]
                    else:
                        to_delete[t1].append(t2)

    for key, val in to_delete.items():
        if glob.glob(key):
            for file in val:
                os.remove(file)

def one_hot(i, nb_classes):
    arr = np.zeros((nb_classes,))
    arr[i] = 1
    return arr

if __name__ == '__main__':
    # Test
    # pp = mido.MidiFile("out/test_in.mid")
    # midi_decode_v1(pp)
    piece = pm.PrettyMIDI("out/test_in.mid")
    beats, decoded = midi_decode_v2(piece)
    print(beats[1], decoded.shape)
    #p = midi_encode(midi_decode(p))
    #midi.write_midifile("out/test_out.mid", p)

    # pitch_class_matrix = np.array([one_hot(n % OCTAVE, OCTAVE) for n in range(NUM_NOTES_INSTRUMENT)])  # notes, octaves
    pitch_class_matrix = np.array([np.tile(one_hot(n % OCTAVE, OCTAVE), NUM_INSTRUMENTS + 1) for n in range(NUM_NOTES_INSTRUMENT)])  # notes, octaves
    pitch_class_matrix = tf.constant(pitch_class_matrix, dtype='float32')
    print(pitch_class_matrix)
    pitch_class_matrix = tf.reshape(pitch_class_matrix, [1, 1, NUM_NOTES, OCTAVE])
    print(pitch_class_matrix)

    decoded = np.asarray([decoded])
    # decoded = decoded[:, :16*5, :128, :]
    print(decoded.shape)
    octaves_t = []
    for i in range(OCTAVE):
        d = decoded[:, :, i::OCTAVE, 0]
        # print(d.shape[-1] % OCTAVE)
        if d.shape[-1] % (OCTAVE - 1) != 0:
            # print(np.zeros((decoded.shape[0], decoded.shape[1], (OCTAVE - 1) - decoded[:, :, i::OCTAVE, 0].shape[-1])).shape)
            # print(i, OCTAVE, "padded", np.append(d, np.zeros((decoded.shape[0], decoded.shape[1], (OCTAVE - 1) - (d.shape[-1] % OCTAVE))), axis=2).shape)
            octaves_t.append(np.append(d, np.zeros((d.shape[0], d.shape[1], (OCTAVE - 1) - (d.shape[-1] % OCTAVE))), axis=2))
        else:
            # print(i ,OCTAVE, decoded[:, :, i::OCTAVE, 0].shape)
            octaves_t.append(d)

    # print(tf.constant(np.asarray([decoded[:, i::OCTAVE, 0] for i in range(OCTAVE)]), dtype='float32'))
    print("octaves", np.asarray(octaves_t).shape)
    bins_t = tf.reduce_sum(octaves_t, axis=3)
    print(bins_t.shape)
    bins_t = tf.tile(bins_t, [NUM_NOTES // OCTAVE, 1, 1])
    print(bins_t.shape)
    bins_t = tf.reshape(bins_t, [decoded.shape[0], decoded.shape[1], NUM_NOTES, 1])
    print(bins_t.shape)

