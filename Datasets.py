# If there is no duration in pattern dict, you must add the duration information
# Please use 'Get_Duration.py' in Pitchtron repository

import torch
import numpy as np
import pickle, os
from random import randint
from multiprocessing import Manager

def Text_to_Token(text: list, token_dict: dict):
    return [token_dict[x] for x in text]

def Mel_Stack(mels: list, max_abs_mel: float):
    max_Mel_Length = max([mel.shape[0] for mel in mels])
    mels = np.stack(
        [np.pad(mel, [[0, max_Mel_Length - mel.shape[0]], [0, 0]], constant_values= -max_abs_mel) for mel in mels],
        axis= 0
        )

    return mels

def Silence_Stack(silences: list):
    max_Silences_Length = max([silence.shape[0] for silence in silences])
    silences = np.stack(
        [np.pad(silence, [0, max_Silences_Length - silence.shape[0]], constant_values= 0.0) for silence in silences],
        axis= 0
        )

    return silences

def Pitch_Stack(pitches: list):
    max_Pitch_Length = max([pitch.shape[0] for pitch in pitches])
    pitches = np.stack(
        [np.pad(pitch, [0, max_Pitch_Length - pitch.shape[0]], constant_values= 0.0) for pitch in pitches],
        axis= 0
        )

    return pitches


def Duration_Stack(durations: list):
    '''
    The length of durations becomes +1 for padding value of each duration.
    '''
    max_Duration = max([np.sum(duration) for duration in durations])
    max_Duration_Length = max([len(duration) for duration in durations]) + 1    # 1 is for padding duration(max - sum).
    
    durations = np.stack(
        [np.pad(duration, [0, max_Duration_Length - len(duration)], constant_values= 0) for duration in durations],
        axis= 0
        )
    durations[:, -1] = max_Duration - np.sum(durations, axis= 1)   # To fit the time after sample
        
    return durations

def Token_Stack(tokens: list, token_dict: dict):
    '''
    The length of tokens becomes +1 for padding value of each duration.
    '''    
    max_Token_Length = max([len(token) for token in tokens]) + 1    # 1 is for padding '<X>'
    
    tokens = np.stack(
        [np.pad(token, [0, max_Token_Length - len(token)], constant_values= token_dict['<X>']) for token in tokens],
        axis= 0
        )
        
    return tokens

def Note_Stack(notes: list):
    '''
    The length of notes becomes +1 for padding value of each duration.
    '''    
    max_Note_Length = max([len(note) for note in notes]) + 1    # 1 is for padding '<X>'
    
    notes = np.stack(
        [np.pad(note, [0, max_Note_Length - len(note)], constant_values= 0) for note in notes],
        axis= 0
        )
        
    return notes


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

        pattern = pattern_Dict['Duration'], Text_to_Token(pattern_Dict['Text'], self.token_Dict), pattern_Dict['Note'], pattern_Dict['Mel'], pattern_Dict['Silence'], pattern_Dict['Pitch']
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
            music = [
                (int(line.strip().split('\t')[0]), line.strip().split('\t')[1], int(line.strip().split('\t')[2]))
                for line in open(path, 'r', encoding= 'utf-8').readlines()[1:]
                ]
            duration, text, note = zip(*music)
            self.patterns.append((duration, text, note, path))

        self.cache_Dict = Manager().dict()

    def __getitem__(self, idx: int):
        if idx in self.cache_Dict.keys():
            return self.cache_Dict['Inference', idx]

        duration, text, note, path = self.patterns[idx]
        pattern = duration, Text_to_Token(text, self.token_Dict), note, os.path.splitext(os.path.basename(path))[0]

        if self.use_cache:
            self.cache_Dict['Inference', idx] = pattern
 
        return pattern

    def __len__(self):
        return len(self.patterns)


class Collater:
    def __init__(
        self,
        token_dict: dict,
        max_abs_mel: float
        ):
        self.token_Dict = token_dict
        self.max_ABS_Mel = max_abs_mel

    def __call__(self, batch: list):
        durations, tokens, notes, mels, silences, pitches = zip(*batch)
        
        token_Lengths = [len(token) + 1 for token in tokens]
        mel_Lengths = [mel.shape[0] for mel in mels]

        durations = Duration_Stack(durations)
        tokens = Token_Stack(tokens, self.token_Dict)
        notes = Note_Stack(notes)
        mels = Mel_Stack(mels, self.max_ABS_Mel)
        silences = Silence_Stack(silences)
        pitches = Pitch_Stack(pitches)

        durations = torch.LongTensor(durations)   # [Batch, Time]
        tokens = torch.LongTensor(tokens)   # [Batch, Time]
        token_Lengths = torch.LongTensor(token_Lengths) # [Batch]
        notes = torch.LongTensor(notes)   # [Batch, Time]
        mels = torch.FloatTensor(mels).transpose(2, 1)   # [Batch, Mel_dim, Time]
        mel_Lengths = torch.LongTensor(mel_Lengths)   # [Batch]
        silences = torch.FloatTensor(silences)   # [Batch, Time]
        pitches = torch.FloatTensor(pitches)   # [Batch, Time]

        return durations, tokens, notes, token_Lengths, mels, silences, pitches, mel_Lengths

class Inference_Collater:
    def __init__(
        self,
        token_dict: dict,
        max_abs_mel: float
        ):
        self.token_Dict = token_dict
        self.max_ABS_Mel = max_abs_mel
         
    def __call__(self, batch: list):
        durations, tokens, notes, labels = zip(*batch)

        token_Lengths = [len(token) + 1 for token in tokens]

        durations = Duration_Stack(durations)
        tokens = Token_Stack(tokens, self.token_Dict)
        notes = Note_Stack(notes)
        
        durations = torch.LongTensor(durations)   # [Batch, Time]
        tokens = torch.LongTensor(tokens)   # [Batch, Time]
        token_Lengths = torch.LongTensor(token_Lengths) # [Batch]
        notes = torch.LongTensor(notes)   # [Batch, Time]

        return durations, tokens, notes, token_Lengths, labels

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