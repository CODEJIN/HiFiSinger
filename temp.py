# import numpy as np
# import mido, os, pickle, yaml, hgtk, argparse, math
# from tqdm import tqdm
# import matplotlib.pyplot as plt 

# paths = []
# for root, _, files in os.walk('E:/Pattern/Sing/NAMS_MIDI_WAV'):
#     for file in files:
#         if os.path.splitext(file)[1] != '.wav':
#             continue
#         wav_Path = os.path.join(root, file).replace('//', '/')
#         midi_Path = wav_Path.replace('Vox.wav', 'Midi.mid')
#         paths.append((wav_Path, midi_Path))

# notes = []

# for index, (wav_Path, midi_Path) in enumerate(paths):
#     mid = mido.MidiFile(midi_Path, charset='CP949')
#     music = []
#     current_Lyric = ''
#     previous_Used = 0
#     absolute_Position = 0
#     for message in mid:
#         if not message.type in ['note_off']:   # Removing end maker.
#             continue
        
#         notes.append(message.note)


# print(min(notes), max(notes))
# plt.hist(notes, 41)
# plt.show()

# import pickle

# a = pickle.load(open("E:/48K.KO_Music/Train/NAMS/000/NAMS.S_000.P_00016.pickle", 'rb'))
# b = pickle.load(open("E:/48K.KO_Music/Train/NAMS/001/NAMS.S_001.P_00011.pickle", 'rb'))

# list(zip(a['Text'], a['Note']))
# list(zip(b['Text'], b['Note']))




# import mido, os, pickle, yaml, hgtk, argparse, math

# for root, _, files in os.walk('C:/Users/Heejo/Desktop/작업 파일/'):
#     for file in files:
#         if os.path.splitext(file)[1] != '.mid':
#             continue
#         path = os.path.join(root, file).replace('//', '/')
#         mid = mido.MidiFile(path, charset='CP949')
#         for message in mid:
#             if message.type == 'lyrics' and (not hgtk.checker.is_hangul(message.text.strip()) or message.text.strip() == ''):
#                 print(file)
#                 break
                



import mido, os, pickle, yaml, hgtk, argparse, math

# for root, _, files in os.walk('C:/Users/Heejo/Desktop/작업 파일/'):
#     for file in files:
#         if os.path.splitext(file)[1] != '.mid':
#             continue
#         wav_Path = os.path.join(root, file).replace('//', '/')
#         midi_Path = wav_Path.replace('Vox.wav', 'Midi.mid')
#         paths.append((wav_Path, midi_Path))

mid = mido.MidiFile('C:/Users/Heejo/Desktop/작업 파일/76. 00488 물들어 MIDI.mid', charset='CP949')
music = []
current_Lyric = ''
previous_Used = 0
absolute_Position = 0
for message in mid:
    if message.type == 'lyrics':
        print(message)