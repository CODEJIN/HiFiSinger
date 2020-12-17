import torch
import numpy as np
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib

class Logger(SummaryWriter):
    def add_scalar_dict(self, scalar_dict, global_step= None, walltime= None):
        for tag, scalar in scalar_dict.items():
            self.add_scalar(
                tag= tag,
                scalar_value= scalar,
                global_step= global_step,
                walltime= walltime
                )
        self.flush()

    def add_image_dict(self, image_dict, global_step, walltime= None):
        for tag, (data, limit) in image_dict.items():
            fig= plt.figure(figsize=(10, 5), dpi= 100)
            if data.ndim == 1:
                plt.imshow([[0]], aspect='auto', origin='lower', cmap= matplotlib.colors.ListedColormap(['white']))
                plt.plot(data)
                plt.margins(x= 0)
                if not limit is None:
                    plt.ylim(*limit)
            elif data.ndim == 2:
                plt.imshow(data, aspect='auto', origin='lower')
                if not limit is None:
                    plt.clim(*limit)
            plt.colorbar()
            plt.title(tag)
            plt.tight_layout()
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            self.add_image(tag= tag, img_tensor= data, global_step= global_step, walltime= walltime, dataformats= 'HWC')
        self.flush()

    def add_histogram_model(self, model, global_step=None, bins='tensorflow', walltime=None, max_bins=None, delete_keywords= []):
        for tag, parameter in model.named_parameters():
            x = tag
            tag = '/'.join([x for x in tag.split('.') if not x in delete_keywords])

            self.add_histogram(
                tag= tag,
                values= parameter.data.cpu().numpy(),
                global_step= global_step,
                bins= bins,
                walltime= walltime,
                max_bins= max_bins
                )
            self.flush()