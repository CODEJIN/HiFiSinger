# If there is no duration in pattern dict, you must add the duration information
# Please use 'Get_Duration.py' in Pitchtron repository

import torch
import numpy as np
import pickle, os
from random import randint
from multiprocessing import Manager

def Text_to_Token(notes, token_dict):
    return [
        (abs_Duration, duration, token_dict[text], note)
        for abs_Duration, duration, text, note in notes
        ]

def Mel_Stack(mels, max_abs_mel):
    max_Mel_Length = max([mel.shape[0] for mel in mels])
    mels = np.stack(
        [np.pad(mel, [[0, max_Mel_Length - mel.shape[0]], [0, 0]], constant_values= -max_abs_mel) for mel in mels],
        axis= 0
        )

    return mels

def Silence_Stack(silences):
    max_Silences_Length = max([silence.shape[0] for silence in silences])
    silences = np.stack(
        [np.pad(silence, [0, max_Silences_Length - silence.shape[0]], constant_values= 0.0) for silence in silences],
        axis= 0
        )

    return silences

def Pitch_Stack(pitches):
    max_Pitch_Length = max([pitch.shape[0] for pitch in pitches])
    pitches = np.stack(
        [np.pad(pitch, [0, max_Pitch_Length - pitch.shape[0]], constant_values= 0.0) for pitch in pitches],
        axis= 0
        )

    return pitches


def Duration_Correct(durations):
    '''
    IMPORTANT:
    The last value of each duration must be '0'.
    This value will be changed to the padding value to correct length difference.
    '''
    sum_Durations = np.sum(durations, axis= 1)
    durations[:, -1] = np.max(sum_Durations) - sum_Durations

    return durations


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pattern_path: str,
        Metadata_file: str,
        token_dict: dict,
        accumulated_dataset_epoch: int= 1,
        use_cache: bool= False
        ):
        super(Dataset, self).__init__()
        self.pattern_Path = pattern_path
        self.token_Dict = token_dict
        self.use_cache = use_cache

        self.metadata_Path = os.path.join(pattern_path, Metadata_file).replace('\\', '/')
        metadata_Dict = pickle.load(open(self.metadata_Path, 'rb'))
        self.patterns = metadata_Dict['File_List']
        self.base_Length = len(self.patterns)
        self.patterns *= accumulated_dataset_epoch
        
        self.cache_Dict = Manager().dict()

    def __getitem__(self, idx: int):
        if (idx % self.base_Length) in self.cache_Dict.keys():
            return self.cache_Dict[self.metadata_Path, idx % self.base_Length]

        path = os.path.join(self.pattern_Path, self.patterns[idx]).replace('\\', '/')
        pattern_Dict = pickle.load(open(path, 'rb'))
        pattern = Text_to_Token(pattern_Dict['Note'], self.token_Dict), pattern_Dict['Mel'], pattern_Dict['Silence'], pattern_Dict['Pitch']
        if self.use_cache:
            self.cache_Dict[self.metadata_Path, idx % self.base_Length] = pattern
        
        return pattern

    def __len__(self):
        return len(self.patterns)

class Inference_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_dict: dict,
        pattern_paths: list= ['./Inference_for_Training/Example.txt'],
        use_cache: bool= False
        ):
        super(Inference_Dataset, self).__init__()
        self.token_Dict = token_dict
        self.use_cache = use_cache

        self.patterns = []
        for path in pattern_paths:
            notes = [
                (None, int(line.strip().split('\t')[0]), line.strip().split('\t')[1], int(line.strip().split('\t')[2]))
                for line in open(path, 'r', encoding= 'utf-8').readlines()[1:]
                ]
            self.patterns.append((notes, path))

        self.cache_Dict = Manager().dict()

    def __getitem__(self, idx: int):
        if idx in self.cache_Dict.keys():
            return self.cache_Dict['Inference', idx]

        notes, path = self.patterns[idx]
        pattern = Text_to_Token(notes, self.token_Dict), os.path.splitext(os.path.basename(path))[0]

        if self.use_cache:
            self.cache_Dict['Inference', idx] = pattern
 
        return pattern

    def __len__(self):
        return len(self.patterns)


class Collater:
    def __init__(
        self,
        token_dict: dict,
        token_length: int,
        max_mel_length: int,
        max_abs_mel: float,
        max_duration: int
        ):
        self.token_Dict = token_dict
        self.token_Length = token_length
        self.max_Mel_Length = max_mel_length
        self.max_ABS_Mel = max_abs_mel
        self.max_Duration = max_duration

    def __call__(self, batch: list):
        durations, tokens, notes, mels, silences, pitches = [], [], [], [], [], []
        for music, mel, silence, pitch in batch:
            for _ in range(10): # If pattern generating failed until 10 times, skipped.
                offset = randint(0, len(music) - self.token_Length)
                music_Sample = music[offset:offset + self.token_Length]
                absolute_duration, duration_Sample, token_Sample, note_Sample = zip(*music_Sample)
                mel_Sample = mel[absolute_duration[0]:absolute_duration[-1] + duration_Sample[-1]]
                silence_Sample = silence[absolute_duration[0]:absolute_duration[-1] + duration_Sample[-1]]
                pitch_Sample = pitch[absolute_duration[0]:absolute_duration[-1] + duration_Sample[-1]]
                if all([
                    mel_Sample.shape[0] < self.max_Mel_Length,
                    mel_Sample.shape[0] - min([x.shape[0] for x in mels + [mel_Sample]]) < self.max_Duration, #padding also must be shorter than max duration.
                    max([x.shape[0] for x in mels + [mel_Sample]]) - mel_Sample.shape[0] < self.max_Duration, #padding also must be shorter than max duration.
                    np.max(duration_Sample) < self.max_Duration
                    ]):
                    durations.append(np.array(duration_Sample + (0,), dtype=np.float32))
                    tokens.append(np.array(token_Sample + (self.token_Dict['<E>'],), dtype=np.float32))
                    notes.append(np.array(note_Sample + (0,), dtype=np.float32))
                    mels.append(mel_Sample)
                    silences.append(silence_Sample)
                    pitches.append(pitch_Sample)
                    break

        mel_Lengths = [mel.shape[0] for mel in mels]

        durations = Duration_Correct(np.stack(durations, axis= 0))
        tokens = np.stack(tokens, axis= 0)
        notes = np.stack(notes, axis= 0)
        mels = Mel_Stack(mels, self.max_ABS_Mel)
        silences = Silence_Stack(silences)
        pitches = Pitch_Stack(pitches)

        durations = torch.LongTensor(durations)   # [Batch, Time]
        tokens = torch.LongTensor(tokens)   # [Batch, Time]
        notes = torch.LongTensor(notes)   # [Batch, Time]
        mels = torch.FloatTensor(mels).transpose(2, 1)   # [Batch, Mel_dim, Time]
        mel_Lengths = torch.LongTensor(mel_Lengths)   # [Batch]
        silences = torch.FloatTensor(silences)   # [Batch, Time]
        pitches = torch.FloatTensor(pitches)   # [Batch, Time]

        return durations, tokens, notes, mels, mel_Lengths, silences, pitches

class Inference_Collater:
    def __init__(
        self,
        token_dict: dict,
        max_abs_mel: float,
        max_duration: int
        ):
        self.token_Dict = token_dict
        self.max_ABS_Mel = max_abs_mel
        self.max_Duration = max_duration
         
    def __call__(self, batch: list):
        durations, tokens, notes, labels = [], [], [], []
        for note, label in batch:            
            _, duration, token, note = zip(*note)
            durations.append(np.array(duration + (0,), dtype=np.float32))
            tokens.append(np.array(token + (self.token_Dict['<E>'],), dtype=np.float32))
            notes.append(np.array(note + (0,), dtype=np.float32))
            labels.append(label)

        durations = Duration_Correct(np.stack(durations, axis= 0))
        tokens = np.stack(tokens, axis= 0)
        notes = np.stack(notes, axis= 0)

        if np.max(durations) > self.max_Duration:
            raise ValueError('There is some notes which have longer duration({}) than max duration({}).'.format(np.max(durations), self.max_Duration))

        durations = torch.LongTensor(durations)   # [Batch, Time]
        tokens = torch.LongTensor(tokens)   # [Batch, Time]
        notes = torch.LongTensor(notes)   # [Batch, Time]

        return durations, tokens, notes, labels

if __name__ == "__main__":
    import yaml
    from Arg_Parser import Recursive_Parse
    hp = Recursive_Parse(yaml.load(
        open('Hyper_Parameters.yaml', encoding='utf-8'),
        Loader=yaml.Loader
        ))
    token_Dict = yaml.load(open(hp.Token_Path), Loader=yaml.Loader)
    dataset = Dataset(
        pattern_path= hp.Train.Train_Pattern.Path,
        Metadata_file= hp.Train.Train_Pattern.Metadata_File,
        token_dict= token_Dict,
        accumulated_dataset_epoch= hp.Train.Train_Pattern.Accumulated_Dataset_Epoch,
        )
    collater = Collater(
        token_dict= token_Dict,
        token_length= hp.Train.Token_Length,
        max_mel_length= hp.Train.Max_Mel_Length,
        max_abs_mel= hp.Sound.Max_Abs_Mel
        )
    dataLoader = torch.utils.data.DataLoader(
        dataset= dataset,
        collate_fn= collater,
        sampler= torch.utils.data.RandomSampler(dataset),
        batch_size= hp.Train.Batch_Size,
        num_workers= hp.Train.Num_Workers,
        pin_memory= True
        )

    print(next(iter(dataLoader))[0])
    
    
    inference_Dataset = Inference_Dataset(
        token_dict= token_Dict,
        pattern_paths= ['./Inference_for_Training/Example.txt'],
        use_cache= False
        )
    inference_Collater = Inference_Collater(
        token_dict= token_Dict,
        max_abs_mel= hp.Sound.Max_Abs_Mel
        )
    inference_DataLoader = torch.utils.data.DataLoader(
        dataset= inference_Dataset,
        collate_fn= inference_Collater,
        sampler= torch.utils.data.SequentialSampler(inference_Dataset),
        batch_size= hp.Train.Batch_Size,
        num_workers= hp.Inference_Batch_Size or hp.Train.Num_Workers,
        pin_memory= True
        )    
    print(next(iter(inference_DataLoader)))
    assert False