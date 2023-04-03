"""
Handles MIDI file loading
"""
import pretty_midi as pm
import numpy as np
from constants import *
import midi
from copy import deepcopy


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


def midi_decode(pattern,
                pm,
                classes=MIDI_MAX_NOTES,
                step=None):
    """
    Takes a MIDI pattern and decodes it into a piano roll.
    """
    if step is None:
        # Resolution is TICKS_PER_BEAT
        step = pm.resolution // NOTES_PER_BEAT

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
            # Duplicate the last note pattern to wait for next event
            for _ in range(event.tick):
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

            if isinstance(event, midi.EndOfTrackEvent):
                break

            # Modify the last note pattern
            if isinstance(event, midi.NoteOnEvent):
                pitch, velocity = event.data
                volume_buffer[-1][pitch] = velocity / MAX_VELOCITY

                # Check for replay_buffer, which is true if the current note was previously played and needs to be replayed
                if len(volume_buffer) > 1 and volume_buffer[-2][pitch] > 0 and volume_buffer[-1][pitch] > 0:
                    replay_buffer[-1][pitch] = 1
                    # Override current volume with previous volume
                    volume_buffer[-1][pitch] = volume_buffer[-2][pitch]

            if isinstance(event, midi.NoteOffEvent):
                pitch, velocity = event.data
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
    p = midi.read_midifile(fname)
    cache_path = os.path.join(CACHE_DIR, fname + '.npy')
    try:
        note_seq = np.load(cache_path)
    except Exception as e:
        # Perform caching
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        note_seq = midi_decode(p)
        np.save(cache_path, note_seq)

    assert len(note_seq.shape) == 3, note_seq.shape
    assert note_seq.shape[1] == MIDI_MAX_NOTES, note_seq.shape
    assert note_seq.shape[2] == 3, note_seq.shape
    assert (note_seq >= 0).all()
    assert (note_seq <= 1).all()
    return note_seq


def compute_drum_piano_roll(instrument, piano_roll, fs):

    roll = deepcopy(piano_roll)
    for note in instrument.notes:
        # Should interpolate
        roll[note.pitch, int(note.start * fs):int(note.end * fs)] += note.velocity

    return roll


def midi_decode_v2(p):

    # Compute piano rolls for every instrument
    # TODO: compute for drums separately because pretty midi can't
    # TODO: determine the frequency for a 16th note? (but if the fs is higher then no replay matrix is needed since the
    #   pauses between consecutive notes are captured)
    fs = 420
    piano_rolls = [] # [[instrument, t_play, t_volume]]
    for instrument in p.instruments:
        print(pm.program_to_instrument_name(instrument.program), instrument.is_drum,
              instrument.get_piano_roll(fs).shape)
        if instrument.is_drum:
            piano_rolls.append([instrument, compute_drum_piano_roll(instrument, instrument.get_piano_roll(fs), fs)])
        else:
            piano_rolls.append([instrument, instrument.get_piano_roll(fs)])

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
    final = np.zeros((NUM_INSTRUMENTS + 1, pitches, 2 * max_time_steps)) # + drum dimension
    for piano_roll in piano_rolls:
        instrument = piano_roll[0]
        t_play = piano_roll[1]
        t_volume = piano_roll[2]

        if instrument.is_drum:
            final[NUM_INSTRUMENTS] = np.concatenate((t_play, t_volume), axis=1)
            pass
        else:
            final[instrument.program] = np.concatenate((t_play, t_volume), axis=1)

    #print(final[NUM_INSTRUMENTS, :, max_time_steps - 1])

    return final


if __name__ == '__main__':
    # Test
    # p = midi.read_midifile("out/test_in.mid")
    # print(p)
    p = pm.PrettyMIDI("out/test_in.mid")
    print(midi_decode_v2(p).shape)
    #p = midi_encode(midi_decode(p))
    #midi.write_midifile("out/test_out.mid", p)
