import torch
import math
from argparse import Namespace  # for type

class HifiSinger(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super(HifiSinger, self).__init__()

        self.hp = hyper_parameters

        self.layer_Dict = torch.nn.ModuleDict()
        self.layer_Dict['Encoder'] = Encoder(self.hp)
        self.layer_Dict['Duration_Predictor'] = Duration_Predictor(self.hp)
        self.layer_Dict['Decoder'] = Decoder(self.hp)
        
    def forward(
        self,
        durations,
        tokens,
        notes,
        token_lengths= None,  # token_length == duration_length == note_length
        ):
        encoder_Masks = None
        if not token_lengths is None:
            encoder_Masks = self.Mask_Generate(
                lengths= token_lengths,
                max_lengths= tokens.size(1)
                )
        
        encodings = self.layer_Dict['Encoder'](
            tokens= tokens,
            durations= durations,
            notes= notes,
            masks= encoder_Masks
            )
        
        encodings, predicted_Durations = self.layer_Dict['Duration_Predictor'](
            encodings= encodings,
            durations= durations
            )

        decoder_Masks = self.Mask_Generate(
            lengths= durations[:, :-1].sum(dim= 1),
            max_lengths= durations[0].sum()
            )

        predicted_Mels, predicted_Silences, predicted_Pitches = self.layer_Dict['Decoder'](
            encodings= encodings,
            masks= decoder_Masks
            )
        
        predicted_Pitches = predicted_Pitches + torch.stack([
            note.repeat_interleave(duration) / self.hp.Max_Note
            for note, duration in zip(notes, durations)
            ], dim= 0)
        
        predicted_Mels.data.masked_fill_(decoder_Masks.unsqueeze(1), -self.hp.Sound.Max_Abs_Mel)
        predicted_Silences.data.masked_fill_(decoder_Masks, 0.0)   # 0.0 -> Silence, 1.0 -> Voice
        predicted_Pitches.data.masked_fill_(decoder_Masks, 0.0)

        return predicted_Mels, torch.sigmoid(predicted_Silences), predicted_Pitches, predicted_Durations

    def Mask_Generate(self, lengths, max_lengths= None):
        '''
        lengths: [Batch]
        '''
        sequence = torch.arange(max_lengths or torch.max(lengths))[None, :].to(lengths.device)
        return sequence >= lengths[:, None]    # [Batch, Time]


class Encoder(torch.nn.Module): 
    def __init__(self, hyper_parameters: Namespace):
        super(Encoder, self).__init__()
        self.hp = hyper_parameters
        
        self.layer_Dict = torch.nn.ModuleDict()
        self.layer_Dict['Phoneme_Embedding'] = torch.nn.Embedding(
            num_embeddings= self.hp.Tokens,
            embedding_dim= self.hp.Encoder.Size,
            )
        self.layer_Dict['Duration_Embedding'] = torch.nn.Embedding(
            num_embeddings= self.hp.Max_Duration,
            embedding_dim= self.hp.Encoder.Size,
            )
        self.layer_Dict['Note_Embedding'] = torch.nn.Embedding(
            num_embeddings= self.hp.Max_Note,
            embedding_dim= self.hp.Encoder.Size,
            )

        self.layer_Dict['Positional_Embedding'] = Sinusoidal_Positional_Embedding(
            channels= self.hp.Encoder.Size,
            dropout= 0.0
            )

        for index in range(self.hp.Encoder.FFT_Block.Stacks):
            self.layer_Dict['FFT_Block_{}'.format(index)] = FFT_Block(
                in_channels= self.hp.Encoder.Size,
                heads= self.hp.Encoder.FFT_Block.Heads,
                dropout_rate= self.hp.Encoder.FFT_Block.Dropout_Rate,
                ff_in_kernel_size= self.hp.Encoder.FFT_Block.FeedForward.In_Kernel_Size,
                ff_out_kernel_size= self.hp.Encoder.FFT_Block.FeedForward.Out_Kernel_Size,
                ff_channels= self.hp.Encoder.FFT_Block.FeedForward.Channels,
                )

    def forward(
        self,
        tokens: torch.LongTensor,
        durations: torch.LongTensor,
        notes: torch.LongTensor,
        masks: torch.BoolTensor= None
        ):
        '''
        x: [Batch, Time]
        lengths: [Batch]
        '''
        tokens = self.layer_Dict['Phoneme_Embedding'](tokens).transpose(2, 1)     # [Batch, Channels, Time]
        durations = self.layer_Dict['Duration_Embedding'](durations).transpose(2, 1)     # [Batch, Channels, Time]
        notes = self.layer_Dict['Note_Embedding'](notes).transpose(2, 1)     # [Batch, Channels, Time]

        x = self.layer_Dict['Positional_Embedding'](tokens + durations + notes)
        for index in range(self.hp.Encoder.FFT_Block.Stacks):
            x = self.layer_Dict['FFT_Block_{}'.format(index)](x, masks)
            
        return x    # [Batch, Channels, Time]

class Duration_Predictor(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super(Duration_Predictor, self).__init__()
        self.hp = hyper_parameters

        self.layer_Dict = torch.nn.ModuleDict()

        previous_Channels = self.hp.Encoder.Size
        for index, (kernel_Size, channels) in enumerate(zip(
            self.hp.Duration_Predictor.Conv.Kernel_Size,
            self.hp.Duration_Predictor.Conv.Channels
            )):
            self.layer_Dict['Conv_{}'.format(index)] = Conv1d(
                in_channels= previous_Channels,
                out_channels= channels,
                kernel_size= kernel_Size,
                padding= (kernel_Size - 1) // 2,
                w_init_gain= 'relu'
                )
            self.layer_Dict['LayerNorm_{}'.format(index)] = torch.nn.LayerNorm(
                normalized_shape= channels
                )
            self.layer_Dict['ReLU_{}'.format(index)] = torch.nn.ReLU()
            self.layer_Dict['Dropout_{}'.format(index)] = torch.nn.Dropout(
                p= self.hp.Duration_Predictor.Conv.Dropout_Rate
                )
            previous_Channels = channels

        self.layer_Dict['Projection'] = torch.nn.Sequential()
        self.layer_Dict['Projection'].add_module('Conv', Conv1d(
            in_channels= previous_Channels,
            out_channels= 1,
            kernel_size= 1,
            w_init_gain= 'relu'
            ))
        self.layer_Dict['Projection'].add_module('ReLU', torch.nn.ReLU())

    def forward(
        self,
        encodings: torch.FloatTensor,
        durations: torch.LongTensor= None
        ):
        x = encodings
        for index in range(len(self.hp.Duration_Predictor.Conv.Kernel_Size)):
            x = self.layer_Dict['Conv_{}'.format(index)](x)
            x = self.layer_Dict['LayerNorm_{}'.format(index)](x.transpose(2, 1)).transpose(2, 1)
            x = self.layer_Dict['ReLU_{}'.format(index)](x)
            x = self.layer_Dict['Dropout_{}'.format(index)](x)
        predicted_Durations = self.layer_Dict['Projection'](x)

        if durations is None:
            durations = predicted_Durations.ceil().long().clamp(0, self.hp.Max_Duration)
            durations = torch.stack([
                (torch.ones_like(duration) if duration.sum() == 0 else duration)
                for duration in durations
                ], dim= 0)

            max_Durations = torch.max(torch.cat([duration.sum(dim= 0, keepdim= True) + 1 for duration in durations]))
            if max_Durations > self.hp.Max_Duration:  # I assume this means failing
                durations = torch.ones_like(predicted_Durations).long()
            else:
                durations = torch.cat([
                    durations[:, :-1], durations[:, -1:] + max_Durations - durations.sum(dim= 1, keepdim= True)
                    ], dim= 1)

        x = torch.stack([
            encoding.repeat_interleave(duration, dim= 1)
            for encoding, duration in zip(encodings, durations)
            ], dim= 0)

        return x, predicted_Durations.squeeze(1)

class Decoder(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super(Decoder, self).__init__()
        self.hp = hyper_parameters

        self.layer_Dict = torch.nn.ModuleDict()

        self.layer_Dict['Positional_Embedding'] = Sinusoidal_Positional_Embedding(
            channels= self.hp.Encoder.Size
            )

        self.layer_Dict['FFT_Block'] = torch.nn.Sequential()
        for index in range(self.hp.Decoder.FFT_Block.Stacks):
            self.layer_Dict['FFT_Block_{}'.format(index)] = FFT_Block(
                in_channels= self.hp.Encoder.Size,
                heads= self.hp.Decoder.FFT_Block.Heads,
                dropout_rate= self.hp.Decoder.FFT_Block.Dropout_Rate,
                ff_in_kernel_size= self.hp.Decoder.FFT_Block.FeedForward.In_Kernel_Size,
                ff_out_kernel_size= self.hp.Decoder.FFT_Block.FeedForward.Out_Kernel_Size,
                ff_channels= self.hp.Decoder.FFT_Block.FeedForward.Channels,
                )

        self.layer_Dict['Projection'] = Conv1d(
            in_channels= self.hp.Encoder.Size,
            out_channels= self.hp.Sound.Mel_Dim + 1 + 1,
            kernel_size= 1,
            w_init_gain= 'linear'
            )

    def forward(
        self,
        encodings: torch.FloatTensor,
        masks: torch.BoolTensor
        ):
        x = encodings
        x = self.layer_Dict['Positional_Embedding'](x)
        for index in range(self.hp.Encoder.FFT_Block.Stacks):
            x = self.layer_Dict['FFT_Block_{}'.format(index)](x, masks= masks)
        x = self.layer_Dict['Projection'](x)

        mels, silences, notes = torch.split(
            x,
            split_size_or_sections= [self.hp.Sound.Mel_Dim, 1, 1],
            dim= 1
            )

        return mels, silences.squeeze(1), notes.squeeze(1)

class FFT_Block(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        heads: int,
        dropout_rate: float,
        ff_in_kernel_size: int,
        ff_out_kernel_size: int,
        ff_channels: int
        ):
        super(FFT_Block, self).__init__()

        self.layer_Dict = torch.nn.ModuleDict()
        self.layer_Dict['Multihead_Attention'] = torch.nn.MultiheadAttention(
            embed_dim= in_channels,
            num_heads= heads
            )
        self.layer_Dict['LayerNorm_0'] = torch.nn.LayerNorm(
            normalized_shape= in_channels
            )
        self.layer_Dict['Dropout'] = torch.nn.Dropout(p= dropout_rate)
        self.layer_Dict['Conv'] = torch.nn.Sequential()
        self.layer_Dict['Conv'].add_module('Conv_0', Conv1d(
            in_channels= in_channels,
            out_channels= ff_channels,
            kernel_size= ff_in_kernel_size,
            padding= (ff_in_kernel_size - 1) // 2,
            w_init_gain= 'relu'
            ))
        self.layer_Dict['Conv'].add_module('ReLU', torch.nn.ReLU())
        self.layer_Dict['Conv'].add_module('Conv_1', Conv1d(
            in_channels= ff_channels,
            out_channels= in_channels,
            kernel_size= ff_out_kernel_size,
            padding= (ff_out_kernel_size - 1) // 2,
            w_init_gain= 'linear'
            ))
        self.layer_Dict['Conv'].add_module('Dropout', torch.nn.Dropout(p= dropout_rate))
        self.layer_Dict['LayerNorm_1'] = torch.nn.LayerNorm(
            normalized_shape= in_channels
            )
        
    def forward(self, x: torch.FloatTensor, masks: torch.BoolTensor= None):
        '''
        x: [Batch, Channels, Time]
        '''
        x = self.layer_Dict['Multihead_Attention'](
            query= x.permute(2, 0, 1),
            key= x.permute(2, 0, 1),
            value= x.permute(2, 0, 1),
            key_padding_mask= masks
            )[0].permute(1, 2, 0) + x
        x = self.layer_Dict['LayerNorm_0'](x.transpose(2, 1)).transpose(2, 1)
        x = self.layer_Dict['Dropout'](x)

        if not masks is None:
            x *= torch.logical_not(masks).unsqueeze(1).float()

        x = self.layer_Dict['Conv'](x) + x
        x = self.layer_Dict['LayerNorm_1'](x.transpose(2, 1)).transpose(2, 1)
        
        if not masks is None:
            x *= torch.logical_not(masks).unsqueeze(1).float()

        return x

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class Sinusoidal_Positional_Embedding(torch.nn.Module):
    def __init__(self, channels, dropout=0.1, max_len=5000):
        super(Sinusoidal_Positional_Embedding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, channels)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2).float() * (-math.log(10000.0) / channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(2, 1)  #[Batch, Channels, Time]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :, :x.size(2)]
        return self.dropout(x)

class Conv1d(torch.nn.Conv1d):
    def __init__(self, w_init_gain= 'relu', *args, **kwargs):
        self.w_init_gain = w_init_gain
        super(Conv1d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        gains = self.w_init_gain
        if isinstance(gains, str) or isinstance(gains, float):
            gains = [gains]
        
        weights = torch.chunk(self.weight, len(gains), dim= 0)
        for gain, weight in zip(gains, weights):
            if gain == 'zero':
                torch.nn.init.zeros_(weight)
            elif gain in ['relu', 'leaky_relu']:
                torch.nn.init.kaiming_uniform_(weight, nonlinearity= gain)
            else:
                if type(gain) == str:
                    gain = torch.nn.init.calculate_gain(gain)
                torch.nn.init.xavier_uniform_(weight, gain= gain)

        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)

if __name__ == "__main__":
    import yaml
    from Arg_Parser import Recursive_Parse
    hp = Recursive_Parse(yaml.load(
        open('Hyper_Parameters.yaml', encoding='utf-8'),
        Loader=yaml.Loader
        ))

    from Datasets import Dataset, Collater
    token_Dict = yaml.load(open(hp.Token_Path), Loader=yaml.Loader)
    dataset = Dataset(
        pattern_path= hp.Train.Train_Pattern.Path,
        Metadata_file= hp.Train.Train_Pattern.Metadata_File,
        token_dict= token_Dict,
        accumulated_dataset_epoch= hp.Train.Train_Pattern.Accumulated_Dataset_Epoch,
        )
    collater = Collater(
        token_dict= token_Dict,
        max_mel_length= hp.Train.Max_Mel_Length
        )
    dataLoader = torch.utils.data.DataLoader(
        dataset= dataset,
        collate_fn= collater,
        sampler= torch.utils.data.RandomSampler(dataset),
        batch_size= hp.Train.Batch_Size,
        num_workers= hp.Train.Num_Workers,
        pin_memory= True
        )
    
    durations, tokens, notes, mels, mel_Lengths = next(iter(dataLoader))

    model = HifiSinger(hp)
    predicted_Mels, predicted_Silences, predicted_Pitches, predicted_Durations = model(
        tokens= tokens,
        durations= durations,
        notes= notes,
        token_lengths= None,  # token_length == duration_length == note_length
        )
