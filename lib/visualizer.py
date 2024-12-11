""" This file contains Visualizer class based on Facebook's visdom.

Returns:
    Visualizer(): Visualizer class to display plots and images
"""

##
import os
import time
import numpy as np
import torchvision.utils as vutils

##
class Visualizer():
    """ Visualizer wrapper based on Visdom.

    Returns:
        Visualizer: Class file.
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, opt):
        self.display_id = opt.display_id
        self.win_size = 256
        self.name = opt.name
        self.opt = opt
        self.vis = None  # 초기화
        if self.opt.display:
            import visdom
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port)
            assert self.vis.check_connection(), "Visdom 서버 연결에 실패했습니다. 서버를 실행하세요."

        # Dictionaries for plotting data and results.
        self.plot_data = None
        self.plot_res = None

        # Path to train and test directories.
        self.img_dir = os.path.join(opt.outf, opt.name, 'train', 'images')
        self.tst_img_dir = os.path.join(opt.outf, opt.name, 'test', 'images')
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        if not os.path.exists(self.tst_img_dir):
            os.makedirs(self.tst_img_dir)

        # Log file.
        self.log_name = os.path.join(opt.outf, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    @staticmethod
    def normalize(inp):
        """Normalize the tensor

        Args:
            inp ([FloatTensor]): Input tensor

        Returns:
            [FloatTensor]: Normalized tensor.
        """
        return (inp - inp.min()) / (inp.max() - inp.min() + 1e-5)

    def plot_current_errors(self, epoch, counter_ratio, errors):
        """Plot current errors.

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            errors (OrderedDict): Error for the current epoch.
        """
        if not hasattr(self, 'plot_data') or self.plot_data is None:
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'Loss'
            },
            win=5
        )

    def plot_performance(self, epoch, counter_ratio, performance):
        """ Plot performance

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            performance (OrderedDict): Performance for the current epoch.
        """
        if not hasattr(self, 'plot_res') or self.plot_res is None:
            self.plot_res = {'X': [], 'Y': [], 'legend': list(performance.keys())}
        self.plot_res['X'].append(epoch + counter_ratio)
        self.plot_res['Y'].append([performance[k] for k in self.plot_res['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_res['X'])] * len(self.plot_res['legend']), 1),
            Y=np.array(self.plot_res['Y']),
            opts={
                'title': self.name + 'Performance Metrics',
                'legend': self.plot_res['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'Stats'
            },
            win=6
        )

    def print_current_errors(self, epoch, errors):
        """ Print current errors.

        Args:
            epoch (int): Current epoch.
            errors (OrderedDict): Error for the current epoch.
        """
        message = '   Loss: [%d/%d] ' % (epoch, self.opt.niter)
        for key, val in errors.items():
            message += '%s: %.3f ' % (key, val)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_current_performance(self, performance, best):
        """ Print current performance results.

        Args:
            performance ([OrderedDict]): Performance of the model
            best ([int]): Best performance.
        """
        message = '   '
        for key, val in performance.items():
            message += '%s: %.3f ' % (key, val)
        message += 'max ' + self.opt.metric + ': %.3f' % best

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def display_current_images(self, reals, fakes, fixed, masked=None):
        """ Display current images. """
        reals = self.normalize(reals.cpu().numpy())
        fakes = self.normalize(fakes.cpu().numpy())
        fixed = self.normalize(fixed.cpu().numpy())

        self.vis.images(reals, win=1, opts={'title': 'Reals'})
        self.vis.images(fakes, win=2, opts={'title': 'Fakes'})
        self.vis.images(fixed, win=3, opts={'title': 'Fixed'})

        if masked is not None:
            masked = self.normalize(masked.cpu().numpy())
            self.vis.images(masked, win=4, opts={'title': 'Masked'})


    def save_current_images(self, epoch, reals, fakes, fixed, masked=None):
        """
        Args:
            epoch ([int])            : Current epoch
            reals ([FloatTensor])    : Real Image
            fakes ([FloatTensor])    : Fake Image
            fixed ([FloatTensor])    : Fixed Fake Image
            masked (Tensor, optional): Contextual Prediction mask.
        """
        # 각 이미지 유형에 대한 디렉토리 설정
        reals_dir = os.path.join(self.img_dir, "reals")
        fakes_dir = os.path.join(self.img_dir, "fakes")
        fixed_dir = os.path.join(self.img_dir, "fixed")
        masked_dir = os.path.join(self.img_dir, "masked_inputs")

        # 디렉토리 생성 (존재하지 않는 경우)
        os.makedirs(reals_dir, exist_ok=True)
        os.makedirs(fakes_dir, exist_ok=True)
        os.makedirs(fixed_dir, exist_ok=True)
        os.makedirs(masked_dir, exist_ok=True)

        # 이미지 저장
        vutils.save_image(reals, f'{reals_dir}/reals_{epoch+1:03d}.png', normalize=True)
        vutils.save_image(fakes, f'{fakes_dir}/fakes_{epoch+1:03d}.png', normalize=True)
        vutils.save_image(fixed, f'{fixed_dir}/fixed_fakes_{epoch+1:03d}.png', normalize=True)
        
        # Contextual Prediction 마스크 저장
        if masked is not None:
            vutils.save_image(masked, f"{masked_dir}/mask_{epoch+1:03d}.png", normalize=True)