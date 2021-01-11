import numpy as np
import mido, os, pickle, yaml, hgtk, argparse, math
from tqdm import tqdm
from argparse import Namespace  # for type

from Audio import Audio_Prep, Mel_Generate
from yin import pitch_calc
from Arg_Parser import Recursive_Parse

def Decompose(syllable):
    onset, nucleus, coda = hgtk.letter.decompose(syllable)
    coda += '_'

    return onset, nucleus, coda

def Pattern_Generate(
    hyper_paramters: Namespace,
    dataset_path: str
    ):
    min_Duration, max_Duration = math.inf, -math.inf
    min_Note, max_Note = math.inf, -math.inf

    paths = []
    for root, _, files in os.walk(dataset_path):
        for file in sorted(files):
            if os.path.splitext(file)[1] != '.wav':
                continue
            wav_Path = os.path.join(root, file).replace('\\', '/')
            midi_Path = wav_Path.replace('Vox.wav', 'Midi.mid')
            paths.append((wav_Path, midi_Path))

    for index, (wav_Path, midi_Path) in enumerate(paths):
        mid = mido.MidiFile(midi_Path, charset='CP949')
        midi_Length = sum([msg.time for msg in mid if msg.type != 'marker']) * hyper_paramters.Sound.Sample_Rate
        music = []
        current_Lyric = ''
        previous_Used = 0
        absolute_Position = 0
        wrong_Midi = False
        for message in mid:
            if not message.type in ['note_on', 'lyrics', 'note_off']:   # Removing end maker.
                continue
            if message.type == 'note_on':
                duration = int(message.time * hyper_paramters.Sound.Sample_Rate) + previous_Used
                previous_Used = duration % hyper_paramters.Sound.Frame_Shift
                duration = duration // hyper_paramters.Sound.Frame_Shift
                music.append((absolute_Position, duration, '<X>', 0))
                absolute_Position += duration
            elif message.type == 'lyrics':
                current_Lyric = Decompose(message.text.strip())
            elif message.type == 'note_off':
                duration = int(message.time * hyper_paramters.Sound.Sample_Rate) + previous_Used
                previous_Used = duration % hyper_paramters.Sound.Frame_Shift
                duration = duration // hyper_paramters.Sound.Frame_Shift
                music.append((absolute_Position, 2, current_Lyric[0], message.note))   # Onset
                absolute_Position += 2
                music.append((absolute_Position, duration - 4, current_Lyric[1], message.note))  # excepting onset and coda length
                absolute_Position += duration - 4
                music.append((absolute_Position, 2, current_Lyric[2], message.note))   # Coda
                absolute_Position += 2

                if duration - 4 < 2:    # I want nucleus is also longer than 2.
                    print('\nToo short note. This data is skipped. Please check it: {}, {}'.format(wav_Path, midi_Path))
                    wrong_Midi = True
                    break
        if wrong_Midi:
            continue

        audio = Audio_Prep(wav_Path, hyper_paramters.Sound.Sample_Rate)[:int(midi_Length)]  # trimming additional silence of end point
        mel = Mel_Generate(
            audio,
            sample_rate= hyper_paramters.Sound.Sample_Rate,
            num_mel= hyper_paramters.Sound.Mel_Dim,
            num_frequency= hyper_paramters.Sound.Spectrogram_Dim,
            window_length= hyper_paramters.Sound.Frame_Length,
            hop_length= hyper_paramters.Sound.Frame_Shift,
            mel_fmin= hyper_paramters.Sound.Mel_F_Min,
            mel_fmax= hyper_paramters.Sound.Mel_F_Max,
            max_abs_value= hyper_paramters.Sound.Max_Abs_Mel
            )[:absolute_Position]   # Usually, 1 or 2 step of mel is cut.

        pitch = pitch_calc(
            sig= audio,
            sr= hyper_paramters.Sound.Sample_Rate,
            w_len= hyper_paramters.Sound.Frame_Length,
            w_step= hyper_paramters.Sound.Frame_Shift,
            f0_min= hyper_paramters.Sound.F0_Min,
            f0_max= hyper_paramters.Sound.F0_Max,
            confidence_threshold= hyper_paramters.Sound.Confidence_Threshold,
            gaussian_smoothing_sigma = hyper_paramters.Sound.Gaussian_Smoothing_Sigma
            )[:absolute_Position] / hyper_paramters.Sound.F0_Max

        silence = np.where(np.mean(mel, axis=1) < -3.5, np.zeros_like(np.mean(mel, axis=1)), np.ones_like(np.mean(mel, axis=1)))

        pattern_Index = 0
        for start_Index in tqdm(range(len(music)), desc= os.path.basename(wav_Path)):
            for end_Index in range(start_Index + 1, len(music), 3):
                music_Sample = music[start_Index:end_Index]
                sample_Length = music_Sample[-1][0] + music_Sample[-1][1] - music_Sample[0][0]
                if sample_Length < hyper_paramters.Min_Duration:
                    continue
                elif sample_Length > hyper_paramters.Max_Duration:
                    break

                audio_Sample = audio[music_Sample[0][0] * hyper_paramters.Sound.Frame_Shift:(music_Sample[-1][0] + music_Sample[-1][1]) * hyper_paramters.Sound.Frame_Shift]
                mel_Sample = mel[music_Sample[0][0]:music_Sample[-1][0] + music_Sample[-1][1]]
                silence_Sample = silence[music_Sample[0][0]:music_Sample[-1][0] + music_Sample[-1][1]]
                pitch_Sample = pitch[music_Sample[0][0]:music_Sample[-1][0] + music_Sample[-1][1]]

                _, duration_Sample, text_Sample, Note_Sample = zip(*music_Sample)

                pattern = {
                    'Audio': audio_Sample.astype(np.float32),
                    'Mel': mel_Sample.astype(np.float32),
                    'Silence': silence_Sample.astype(np.uint8),
                    'Pitch': pitch_Sample.astype(np.float32),
                    'Duration': duration_Sample,
                    'Text': text_Sample,
                    'Note': Note_Sample,
                    'Singer': 'Female_0',
                    'Dataset': 'NAMS',
                    }

                pattern_Path = os.path.join(
                    hyper_paramters.Train.Train_Pattern.Path if np.random.rand() > 0.001 else hyper_paramters.Train.Eval_Pattern.Path,
                    'NAMS',
                    '{:03d}'.format(index),
                    'NAMS.S_{:03d}.P_{:05d}.pickle'.format(index, pattern_Index)
                    ).replace('\\', '/')
                os.makedirs(os.path.dirname(pattern_Path), exist_ok= True)
                pickle.dump(
                    pattern,
                    open(pattern_Path, 'wb'),
                    protocol=4
                    )
                pattern_Index += 1

                min_Duration, max_Duration = min(sample_Length, min_Duration), max(sample_Length, max_Duration)
        min_Note, max_Note = min(list(zip(*music))[3] + (min_Note,)), max(list(zip(*music))[3] + (max_Note,))

    print('Duration range: {} - {}'.format(min_Duration, max_Duration))
    print('Note range: {} - {}'.format(min_Note, max_Note))

def Token_Dict_Generate(hyper_parameters: Namespace):
    tokens = \
        list(hgtk.letter.CHO) + \
        list(hgtk.letter.JOONG) + \
        ['{}_'.format(x) for x in hgtk.letter.JONG]
    
    os.makedirs(os.path.dirname(hyper_parameters.Token_Path), exist_ok= True)
    yaml.dump(
        {token: index for index, token in enumerate(['<S>', '<E>', '<X>'] + sorted(tokens))},
        open(hyper_parameters.Token_Path, 'w')
        )


def Metadata_Generate(
    hyper_parameters: Namespace,
    eval: bool= False
    ):
    pattern_Path = hyper_parameters.Train.Eval_Pattern.Path if eval else hyper_parameters.Train.Train_Pattern.Path
    metadata_File = hyper_parameters.Train.Eval_Pattern.Metadata_File if eval else hyper_parameters.Train.Train_Pattern.Metadata_File

    new_Metadata_Dict = {
        'Spectrogram_Dim': hyper_parameters.Sound.Spectrogram_Dim,
        'Mel_Dim': hyper_parameters.Sound.Mel_Dim,
        'Frame_Shift': hyper_parameters.Sound.Frame_Shift,
        'Frame_Length': hyper_parameters.Sound.Frame_Length,
        'Sample_Rate': hyper_parameters.Sound.Sample_Rate,
        'Max_Abs_Mel': hyper_parameters.Sound.Max_Abs_Mel,
        'Mel_F_Min': hyper_parameters.Sound.Mel_F_Min,
        'Mel_F_Max': hyper_parameters.Sound.Mel_F_Max,
        'File_List': [],
        'Audio_Length_Dict': {},
        'Mel_Length_Dict': {},
        'Music_Length_Dict': {},
        }

    files_TQDM = tqdm(
        total= sum([len(files) for root, _, files in os.walk(pattern_Path)]),
        desc= 'Eval_Pattern' if eval else 'Train_Pattern'
        )

    for root, _, files in os.walk(pattern_Path):
        for file in files:
            with open(os.path.join(root, file).replace("\\", "/"), "rb") as f:
                pattern_Dict = pickle.load(f)
            file = os.path.join(root, file).replace("\\", "/").replace(pattern_Path, '').lstrip('/')
            try:
                if not all([
                    key in pattern_Dict.keys()
                    for key in ('Audio', 'Mel', 'Silence', 'Pitch', 'Duration', 'Text', 'Note', 'Singer', 'Dataset')
                    ]):
                    continue
                new_Metadata_Dict['Audio_Length_Dict'][file] = pattern_Dict['Audio'].shape[0]
                new_Metadata_Dict['Mel_Length_Dict'][file] = pattern_Dict['Mel'].shape[0]
                new_Metadata_Dict['Music_Length_Dict'][file] = len(pattern_Dict['Duration'])
                new_Metadata_Dict['File_List'].append(file)
            except:
                print('File \'{}\' is not correct pattern file. This file is ignored.'.format(file))
            files_TQDM.update(1)

    with open(os.path.join(pattern_Path, metadata_File.upper()).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_Metadata_Dict, f, protocol= 4)

    print('Metadata generate done.')



if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-d", "--dataset_path", required= True)
    argParser.add_argument("-hp", "--hyper_paramters", required= True)
    args = argParser.parse_args()

    hp = Recursive_Parse(yaml.load(
        open(args.hyper_paramters, encoding='utf-8'),
        Loader=yaml.Loader
        ))

    Token_Dict_Generate(hyper_parameters= hp)
    Pattern_Generate(hyper_paramters= hp, dataset_path= args.dataset_path)
    Metadata_Generate(hp, False)
    Metadata_Generate(hp, True)

    # python Pattern_Generator.py -hp Hyper_Parameters.yaml -d E:/Kor_Music_Confidential