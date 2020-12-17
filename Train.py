import torch
import numpy as np
import logging, yaml, os, sys, argparse, time, math
from tqdm import tqdm
from collections import defaultdict
from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from random import sample
import torch.multiprocessing as mp

from Modules import HifiSinger, Silence_Loss, Note_Loss
from Datasets import Dataset, Inference_Dataset, Collater, Inference_Collater
from Radam import RAdam
from Noam_Scheduler import Modified_Noam_Scheduler
from Logger import Logger
from Arg_Parser import Recursive_Parse

#Unicode problem for Korean
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.family"] = 'NanumGothic'

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )

try:
    from apex import amp
    is_AMP_Exist = True
except:
    logging.info('There is no apex modules in the environment. Mixed precision does not work.')
    is_AMP_Exist = False

class Trainer:
    def __init__(self, hp_path, steps= 0, gpu_id= 0):
        self.hp_Path = hp_path
        self.gpu_id = gpu_id
        
        self.hp = Recursive_Parse(yaml.load(
            open(self.hp_Path, encoding='utf-8'),
            Loader=yaml.Loader
            ))
        if not is_AMP_Exist:
            self.hp.Use_Mixed_Precision = False

        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(gpu_id))
            torch.backends.cudnn.benchmark = True
            torch.cuda.set_device(0)

        self.steps = steps

        self.Datset_Generate()
        self.Model_Generate()

        self.scalar_Dict = {
            'Train': defaultdict(float),
            'Evaluation': defaultdict(float),
            }

        self.writer_Dict = {
            'Train': Logger(os.path.join(self.hp.Log_Path, 'Train')),
            'Evaluation': Logger(os.path.join(self.hp.Log_Path, 'Evaluation')),
            }
        
        self.Load_Checkpoint()

    def Datset_Generate(self):
        token_Dict = yaml.load(open(self.hp.Token_Path), Loader=yaml.Loader)

        train_Dataset = Dataset(
            pattern_path= self.hp.Train.Train_Pattern.Path,
            Metadata_file= self.hp.Train.Train_Pattern.Metadata_File,
            token_dict= token_Dict,
            accumulated_dataset_epoch= self.hp.Train.Train_Pattern.Accumulated_Dataset_Epoch,
            use_cache = self.hp.Train.Use_Pattern_Cache
            )
        eval_Dataset = Dataset(
            pattern_path= self.hp.Train.Eval_Pattern.Path,
            Metadata_file= self.hp.Train.Eval_Pattern.Metadata_File,
            token_dict= token_Dict,
            use_cache = self.hp.Train.Use_Pattern_Cache
            )
        inference_Dataset = Inference_Dataset(
            token_dict= token_Dict,
            pattern_paths= ['./Inference_for_Training/Example.txt'],
            use_cache= False
            )

        if self.gpu_id == 0:
            logging.info('The number of train patterns = {}.'.format(train_Dataset.base_Length))
            logging.info('The number of development patterns = {}.'.format(eval_Dataset.base_Length))
            logging.info('The number of inference patterns = {}.'.format(len(inference_Dataset)))

        collater = Collater(
            token_dict= token_Dict,
            token_length= hp.Train.Token_Length,
            max_mel_length= hp.Train.Max_Mel_Length,
            max_abs_mel= hp.Sound.Max_Abs_Mel
            )
        inference_Collater = Inference_Collater(
            token_dict= token_Dict,
            max_abs_mel= self.hp.Sound.Max_Abs_Mel
            )

        self.dataLoader_Dict = {}
        self.dataLoader_Dict['Train'] = torch.utils.data.DataLoader(
            dataset= train_Dataset,
            sampler= torch.utils.data.DistributedSampler(train_Dataset, shuffle= True) \
                     if self.hp.Use_Multi_GPU else \
                     torch.utils.data.RandomSampler(train_Dataset),
            collate_fn= collater,
            batch_size= self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataLoader_Dict['Eval'] = torch.utils.data.DataLoader(
            dataset= eval_Dataset,
            sampler= torch.utils.data.RandomSampler(eval_Dataset),
            collate_fn= collater,
            batch_size= self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataLoader_Dict['Inference'] = torch.utils.data.DataLoader(
            dataset= inference_Dataset,
            sampler= torch.utils.data.SequentialSampler(inference_Dataset),
            collate_fn= inference_Collater,
            batch_size= self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )

    def Model_Generate(self):
        if self.hp.Use_Multi_GPU:
            self.model = torch.nn.parallel.DistributedDataParallel(
                HifiSinger(self.hp).to(self.device),
                device_ids=[self.gpu_id]
                )
        else:
            self.model = HifiSinger(self.hp).to(self.device)

        self.criterion_Dict = {
            'Mean_Absolute_Error': torch.nn.L1Loss(reduction= 'none').to(self.device),
            'Mean_Squared_Error': torch.nn.MSELoss().to(self.device),
            'Silence_Loss`': Silence_Loss().to(self.device),
            'Note_Loss': Note_Loss().to(self.device)
            }
        self.optimizer = RAdam(
            params= self.model.parameters(),
            lr= self.hp.Train.Learning_Rate.Initial,
            betas=(self.hp.Train.ADAM.Beta1, self.hp.Train.ADAM.Beta2),
            eps= self.hp.Train.ADAM.Epsilon,
            weight_decay= self.hp.Train.Weight_Decay
            )
        self.scheduler = Modified_Noam_Scheduler(
            optimizer= self.optimizer,
            base= self.hp.Train.Learning_Rate.Base
            )
        
        self.vocoder = None
        if not self.hp.Vocoder_Path is None:
            self.vocoder = torch.jit.load(self.hp.Vocoder_Path).to(self.device)

        if self.hp.Use_Mixed_Precision:
            self.model, self.optimizer = amp.initialize(
                models=self.model,
                optimizers=self.optimizer
                )
        if self.gpu_id == 0:
            logging.info(self.model)

    def Train_Step(self, durations, tokens, notes, mels, mel_lengths):
        loss_Dict = {}

        durations = durations.to(self.device, non_blocking=True)
        tokens = tokens.to(self.device, non_blocking=True)
        notes = notes.to(self.device, non_blocking=True)
        mels = mels.to(self.device, non_blocking=True)
        mel_lengths = mel_lengths.to(self.device, non_blocking=True)

        predicted_Mels, predicted_Silences, predicted_Notes, predicted_Durations = self.model(
            durations= durations,
            tokens= tokens,
            notes= notes
            )

        loss_Dict['Mel'] = self.criterion_Dict['Mean_Absolute_Error'](predicted_Mels, mels)
        loss_Dict['Mel'] = loss_Dict['Mel'].sum(dim= 2).mean(dim=1) / mel_lengths.float()
        loss_Dict['Mel'] = loss_Dict['Mel'].mean()
        loss_Dict['Silence'] = self.criterion_Dict['Silence_Loss'](predicted_Silences, notes)
        loss_Dict['Note'] = self.criterion_Dict['Note_Loss'](predicted_Notes, notes)
        loss_Dict['Predicted_Duration'] = self.criterion_Dict['MSE'](predicted_Durations, durations)
        loss_Dict['Total'] = loss_Dict['Mel'] + loss_Dict['Silence'] + loss_Dict['Note'] + loss_Dict['Predicted_Duration']

        self.optimizer.zero_grad()
        if self.hp.Use_Mixed_Precision:
            with amp.scale_loss(loss_Dict['Total'], self.optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters= amp.master_params(self.optimizer),
                max_norm= self.hp.Train.Gradient_Norm
                )
        else:
            loss_Dict['Total'].backward()
            torch.nn.utils.clip_grad_norm_(
                parameters= self.model.parameters(),
                max_norm= self.hp.Train.Gradient_Norm
                )
        torch.cuda.synchronize()

        self.optimizer.step()
        self.scheduler.step()
        torch.cuda.synchronize()
        self.steps += 1
        self.tqdm.update(1)

        for tag, loss in loss_Dict.items():
            self.scalar_Dict['Train']['Loss/{}'.format(tag)] += loss

    def Train_Epoch(self):
        for durations, tokens, notes, mels, mel_lengths in self.dataLoader_Dict['Train']:
            self.Train_Step(durations, tokens, notes, mels, mel_lengths)
            
            if self.steps % self.hp.Train.Checkpoint_Save_Interval == 0:
                self.Save_Checkpoint()

            if self.steps % self.hp.Train.Logging_Interval == 0:
                self.scalar_Dict['Train'] = {
                    tag: loss / self.hp.Train.Logging_Interval
                    for tag, loss in self.scalar_Dict['Train'].items()
                    }
                self.scalar_Dict['Train']['Learning_Rate'] = self.scheduler.get_last_lr()
                self.writer_Dict['Train'].add_scalar_dict(self.scalar_Dict['Train'], self.steps)
                self.scalar_Dict['Train'] = defaultdict(float)

            if self.steps % self.hp.Train.Evaluation_Interval == 0:
                self.Evaluation_Epoch()

            if self.steps % self.hp.Train.Inference_Interval == 0:
                self.Inference_Epoch()
            
            if self.steps >= self.hp.Train.Max_Step:
                return

    @torch.no_grad()
    def Evaluation_Step(self, durations, tokens, notes, mels, mel_lengths):
        loss_Dict = {}
        
        durations = durations.to(self.device, non_blocking=True)
        tokens = tokens.to(self.device, non_blocking=True)
        notes = notes.to(self.device, non_blocking=True)
        mels = mels.to(self.device, non_blocking=True)
        mel_lengths = mel_lengths.to(self.device, non_blocking=True)

        predicted_Mels, predicted_Silences, predicted_Notes, predicted_Durations = self.model(
            durations= durations,
            tokens= tokens,
            notes= notes
            )

        loss_Dict['Mel'] = self.criterion_Dict['Mean_Absolute_Error'](predicted_Mels, mels)
        loss_Dict['Mel'] = loss_Dict['Mel'].sum(dim= 2).mean(dim=1) / mel_lengths.float()
        loss_Dict['Mel'] = loss_Dict['Mel'].mean()
        loss_Dict['Silence'] = self.criterion_Dict['Silence_Loss'](predicted_Silences, notes)
        loss_Dict['Note'] = self.criterion_Dict['Note_Loss'](predicted_Notes, notes)
        loss_Dict['Predicted_Duration'] = self.criterion_Dict['MSE'](predicted_Durations, durations)
        loss_Dict['Total'] = loss_Dict['Mel'] + loss_Dict['Silence'] + loss_Dict['Note'] + loss_Dict['Predicted_Duration']

        for tag, loss in loss_Dict.items():
            self.scalar_Dict['Evaluation']['Loss/{}'.format(tag)] += loss.cpu()

        return predicted_Mels, predicted_Silences, predicted_Notes, predicted_Durations

    def Evaluation_Epoch(self):
        if self.gpu_id != 0:
            return

        logging.info('(Steps: {}) Start evaluation in GPU {}.'.format(self.steps, self.gpu_id))

        self.model.eval()

        for step, (durations, tokens, notes, mels, mel_lengths) in tqdm(
            enumerate(self.dataLoader_Dict['Eval'], 1),
            desc='[Evaluation]',
            total= math.ceil(len(self.dataLoader_Dict['Eval'].dataset) / self.hp.Train.Batch_Size)
            ):
            predicted_Mels, predicted_Silences, predicted_Notes, predicted_Durations = self.Evaluation_Step(durations, tokens, notes, mels, mel_lengths)

        self.scalar_Dict['Evaluation'] = {
            tag: loss / step
            for tag, loss in self.scalar_Dict['Evaluation'].items()
            }
        self.writer_Dict['Evaluation'].add_scalar_dict(self.scalar_Dict['Evaluation'], self.steps)
        self.writer_Dict['Evaluation'].add_histogram_model(self.model, self.steps, delete_keywords=['layer_Dict', 'layer'])
        self.scalar_Dict['Evaluation'] = defaultdict(float)

        note = notes[-1].repeat_interleave(durations[-1])
        silence = torch.where(note, torch.ones_like(note), torch.zeros_like(note)).float()

        duration = durations[-1, :token_Lengths[-1]]
        duration = torch.arange(duration.size(0)).to(duration.device).repeat_interleave(duration).cpu().numpy()
        predicted_Duration = predicted_Durations[-1, :token_Lengths[-1]].ceil().long().clamp(0, self.hp.Max_Duration)
        predicted_Duration = torch.arange(predicted_Duration.size(0)).to(predicted_Duration.device).repeat_interleave(predicted_Duration).cpu().numpy()        
        image_Dict = {
            'Mel/Target': (mels[-1, :mel_lengths[-1]].cpu().numpy(), None),
            'Mel/Prediction': (predicted_Mels[-1, :mel_lengths[-1]].cpu().numpy(), None),
            'Silence/Target': (silence.cpu().numpy(), None),
            'Silence/Prediction': (predicted_Silences[-1].cpu().numpy(), None),
            'Note/Target': (note.cpu().numpy(), None),
            'Note/Prediction': (predicted_Notes[-1].cpu().numpy(), None),
            'Duration/Target': (durations[-1].cpu().numpy(), None),
            'Duration/Prediction': (predicted_Durations[-1].cpu().numpy(), None),
            }
        self.writer_Dict['Evaluation'].add_image_dict(image_Dict, self.steps)

        self.model.train()

    @torch.no_grad()
    def Inference_Step(self, durations, tokens, notes, labels, start_index= 0, tag_step= False):
        durations = durations.to(self.device, non_blocking=True)
        tokens = tokens.to(self.device, non_blocking=True)
        notes = notes.to(self.device, non_blocking=True)

        predicted_Mels, predicted_Silences, predicted_Notes, predicted_Durations = self.model(
            durations= durations,
            tokens= tokens,
            notes= notes
            )
        
        files = []
        for index, label in enumerate(labels):
            tags = []
            if tag_step: tags.append('Step-{}'.format(self.steps))
            tags.append(label)
            tags.append('IDX_{}'.format(index + start_index))
            files.append('.'.join(tags))

        os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG').replace('\\', '/'), exist_ok= True)
        os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'NPY', 'Mel').replace('\\', '/'), exist_ok= True)
        for mel, silence, note, duration, label, file in zip(
            predicted_Mels.cpu(),
            predicted_Silences.cpu(),
            predicted_Notes.cpu(),
            predicted_Durations.cpu(),
            labels,
            files
            ):
            title = 'Note infomation: {}'.format(label)
            new_Figure = plt.figure(figsize=(20, 5 * 4), dpi=100)
            plt.subplot2grid((4, 1), (0, 0))
            plt.imshow(mel, aspect='auto', origin='lower')
            plt.title('Mel    {}'.format(title))
            plt.colorbar()
            plt.subplot2grid((4, 1), (1, 0))
            plt.plot(silence)
            plt.margins(x= 0)
            plt.title('Silence    {}'.format(title))
            plt.colorbar()
            plt.subplot2grid((4, 1), (2, 0))
            plt.plot(note)
            plt.margins(x= 0)
            plt.title('Note    {}'.format(title))
            plt.colorbar()
            duration = duration.ceil().long().clamp(0, self.hp.Max_Duration)
            duration = torch.arange(duration.size(0)).repeat_interleave(duration)
            plt.subplot2grid((4, 1), (3, 0))
            plt.plot(duration)
            plt.margins(x= 0)
            plt.title('Duration    {}'.format(title))
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG', '{}.png'.format(file)).replace('\\', '/'))
            plt.close(new_Figure)
            
            np.save(
                os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'NPY', 'Mel', file).replace('\\', '/'),
                mel.T,
                allow_pickle= False
                )

        # This part may be changed depending on the vocoder used.
        if not self.vocoder is None:
            os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'Wav').replace('\\', '/'), exist_ok= True)
            for mel, file in zip(predicted_Mels, files):
                mel = mel.unsqueeze(0)
                x = torch.randn(size=(mel.size(0), self.hp.Sound.Frame_Shift * mel.size(2))).to(mel.device)
                mel = torch.nn.functional.pad(mel, (2,2), 'reflect')
                wav = self.vocoder(x, mel).cpu().numpy()[0]
                wavfile.write(
                    filename= os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'Wav', '{}.wav'.format(file)).replace('\\', '/'),
                    data= (np.clip(wav, -1.0 + 1e-7, 1.0 - 1e-7) * 32767.5).astype(np.int16),
                    rate= self.hp.Sound.Sample_Rate
                    )
            
    def Inference_Epoch(self):
        if self.gpu_id != 0:
            return

        logging.info('(Steps: {}) Start inference in GPU {}.'.format(self.steps, self.gpu_id))

        self.model.eval()

        for step, (durations, tokens, notes, labels) in tqdm(
            enumerate(self.dataLoader_Dict['Inference']),
            desc='[Inference]',
            total= math.ceil(len(self.dataLoader_Dict['Inference'].dataset) / (self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size))
            ):
            self.Inference_Step(durations, tokens, notes, labels, start_index= step * (self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size))

        self.model.train()

    def Load_Checkpoint(self):
        if self.steps == 0:
            paths = [
                os.path.join(root, file).replace('\\', '/')
                for root, _, files in os.walk(self.hp.Checkpoint_Path)
                for file in files
                if os.path.splitext(file)[1] == '.pt'
                ]
            if len(paths) > 0:
                path = max(paths, key = os.path.getctime)
            else:
                return  # Initial training
        else:
            path = os.path.join(self.hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))

        state_Dict = torch.load(path, map_location= 'cpu')
        
        if self.hp.Use_Multi_GPU:
            self.model.module.load_state_dict(state_Dict['Model'])
        else:
            self.model.load_state_dict(state_Dict['Model'])
        self.optimizer.load_state_dict(state_Dict['Optimizer'])
        self.scheduler.load_state_dict(state_Dict['Scheduler'])
        self.steps = state_Dict['Steps']

        if self.hp.Use_Mixed_Precision:
            if not 'AMP' in state_Dict.keys():
                logging.info('No AMP state dict is in the checkpoint. Model regards this checkpoint is trained without mixed precision.')
            else:                
                amp.load_state_dict(state_Dict['AMP'])

        logging.info('Checkpoint loaded at {} steps in GPU {}.'.format(self.steps, self.gpu_id))

    def Save_Checkpoint(self):
        if self.gpu_id != 0:
            return

        os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)

        state_Dict = {
            'Model': self.model.module.state_dict() if self.hp.Use_Multi_GPU else self.model.state_dict(),
            'Optimizer': self.optimizer.state_dict(),
            'Scheduler': self.scheduler.state_dict(),
            'Steps': self.steps
            }
        if self.hp.Use_Mixed_Precision:
            state_Dict['AMP'] = amp.state_dict()

        torch.save(
            state_Dict,
            os.path.join(self.hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))
            )

        logging.info('Checkpoint saved at {} steps.'.format(self.steps))

    def Train(self):
        hp_Path = os.path.join(self.hp.Checkpoint_Path, 'Hyper_Parameters.yaml').replace('\\', '/')
        if not os.path.exists(hp_Path):
            from shutil import copyfile
            os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)
            copyfile(self.hp_Path, hp_Path)

        if self.steps == 0:
            self.Evaluation_Epoch()

        if self.hp.Train.Initial_Inference:
            self.Inference_Epoch()

        self.tqdm = tqdm(
            initial= self.steps,
            total= self.hp.Train.Max_Step,
            desc='[Training]'
            )

        while self.steps < self.hp.Train.Max_Step:
            try:
                self.Train_Epoch()
            except KeyboardInterrupt:
                self.Save_Checkpoint()
                exit(1)
            
        self.tqdm.close()
        logging.info('Finished training.')


def Worker(gpu, hp_path, steps):
    torch.distributed.init_process_group(
        backend= 'nccl',
        init_method='tcp://127.0.0.1:54321',
        world_size= torch.cuda.device_count(),
        rank= gpu
        )

    new_Trainer = Trainer(hp_path= hp_path, steps= steps, gpu_id= gpu)
    new_Trainer.Train()

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-hp', '--hyper_parameters', required= True, type= str)
    argParser.add_argument('-s', '--steps', default= 0, type= int)    
    args = argParser.parse_args()
    
    hp = Recursive_Parse(yaml.load(
        open(args.hyper_parameters, encoding='utf-8'),
        Loader=yaml.Loader
        ))
    os.environ['CUDA_VISIBLE_DEVICES'] = hp.Device

    if hp.Use_Multi_GPU:
        mp.spawn(
            Worker,
            nprocs= torch.cuda.device_count(),
            args= (args.hyper_parameters, args.steps)
            )
    else:
        new_Trainer = Trainer(hp_path= args.hyper_parameters, steps= args.steps, gpu_id= 0)
        new_Trainer.Train()