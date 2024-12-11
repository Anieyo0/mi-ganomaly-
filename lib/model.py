"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from networks import NetG, NetD, weights_init
from visualizer import Visualizer
from loss import l2_loss
from evaluate import evaluate
import torch.nn.functional as F

class BaseModel():
    """ Base Model for ganomaly
    """
    def __init__(self, opt, dataloader):
        ##
        # Seed for deterministic behavior
        self.seed(opt.manualseed)

        # Initalize variables.
        self.opt = opt
        self.visualizer = Visualizer(opt)
        self.dataloader = dataloader
        self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")
        
    
    ##
    def set_input(self, input:torch.Tensor):
        """ Set input and ground truth

        Args:
            input (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            self.gt.resize_(input[1].size()).copy_(input[1])
            self.label.resize_(input[1].size())

            # Copy the first batch as the fixed input.
            if self.total_steps == self.opt.batchsize:
                self.fixed_input.resize_(input[0].size()).copy_(input[0])
                

    ##
    def seed(self, seed_value):
        """ Seed 
        
        Arguments:
            seed_value {int} -- [description]
        """
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True

    ##
    def get_errors(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """

        errors = OrderedDict([
            ('err_d', self.err_d.item()),
            ('err_g', self.err_g.item()),
            ('err_g_adv', self.err_g_adv.item()),
            ('err_g_con', self.err_g_con.item()),
            ('err_g_enc', self.err_g_enc.item()),
            ('err_g_rec', self.err_g_rec.item())])

        return errors

    ##
    def get_current_images(self):
        """ Returns current images with optional mask. """
        reals = self.input.data
        fakes = self.fake.data
        fixed = self.netg(self.fixed_input)[0].data
        
        # Contextual Prediction이 활성화된 경우, 마스크된 입력 추가
        if self.opt.use_context_pred:
            masked = self.masked_input.data
            return reals, fakes, fixed, masked
        else:
            return reals, fakes, fixed, None  # 네 번째 값은 None으로 반환

    ##
    def save_weights(self, epoch):
        """Save netG and netD weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """

        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir): os.makedirs(weight_dir)

        torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},
               f'{weight_dir}/epoch{epoch + 1}_netG.pth')
        torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()},
               f'{weight_dir}/epoch{epoch + 1}_netD.pth')

    ##
    def train_one_epoch(self):
        """ Train the model for one epoch. """
        self.netg.train()
        epoch_iter = 0
        epoch_loss_g, epoch_loss_d = 0, 0  # 에폭 손실 초기화
        first_batch_saved = False  # 첫 번째 배치 저장 여부를 나타내는 플래그

        for data in tqdm(self.dataloader['train'], leave=False, total=len(self.dataloader['train'])):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            self.set_input(data)
            self.optimize_params()

            # 손실 계산
            epoch_loss_g += self.err_g.item()
            epoch_loss_d += self.err_d.item()

            # 첫 번째 배치의 이미지를 저장
            if not first_batch_saved:
                reals, fakes, fixed, masked = self.get_current_images()
                # visualizer의 save_current_images 호출
                self.visualizer.save_current_images(self.epoch, reals, fakes, fixed, masked)
                first_batch_saved = True  # 첫 번째 배치 저장 후 플래그 설정

        # 에폭 종료 후 display 수행
        errors = self.get_errors()
        if self.opt.display:
            counter_ratio = float(epoch_iter) / len(self.dataloader['train'].dataset)
            self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)
            reals, fakes, fixed, masked = self.get_current_images()
            self.visualizer.display_current_images(reals, fakes, fixed, masked)

        # 에폭 평균 손실 계산
        avg_loss_g = epoch_loss_g / len(self.dataloader['train'])
        avg_loss_d = epoch_loss_d / len(self.dataloader['train'])
        
        print(f"Epoch [{self.epoch + 1}/{self.opt.niter}], Generator Loss: {avg_loss_g:.4f}, Discriminator Loss: {avg_loss_d:.4f}")
        return avg_loss_g, avg_loss_d

    ##
    def train(self):
        """ Train the model """
        self.total_steps = 0
        best_auc = 0

        print(f">> Training model {self.name}.")
        for self.epoch in range(self.opt.iter, self.opt.niter):
            # 한 에폭 동안 학습
            avg_loss_g, avg_loss_d = self.train_one_epoch()
            
            # 테스트 후 AUC 계산
            res = self.test()
            if res[self.opt.metric] > best_auc:
                best_auc = res[self.opt.metric]
                self.save_weights(self.epoch)

            # 에폭 성능 출력
            print(f"Epoch [{self.epoch + 1}/{self.opt.niter}] - Avg Generator Loss: {avg_loss_g:.4f}, Avg Discriminator Loss: {avg_loss_d:.4f}, Best AUC: {best_auc:.4f}")
            
            # 시각화 업데이트
            self.visualizer.print_current_performance(res, best_auc)
        print(f">> Training model {self.name}.[Done]")

    ##
    def test(self):
        """ Test GANomaly model.

        Args:
            dataloader ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights:
                path = "./output/{}/{}/train/weights/netG.pth".format(self.name.lower(), self.opt.dataset)
                pretrained_dict = torch.load(path)['state_dict']

                try:
                    self.netg.load_state_dict(pretrained_dict)
                except IOError:
                    raise IOError("netG weights not found")
                print('   Loaded weights.')

            self.opt.phase = 'test'

            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.long,    device=self.device)
            self.latent_i  = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)
            self.latent_o  = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)
            self.latent_orig = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)

            # print("   Testing model %s." % self.name)
            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            for i, data in enumerate(self.dataloader['test'], 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.set_input(data)
                self.fake, self.reconstructed_from_latent, self.latent_orig, latent_i, latent_o = self.netg(self.input)

                error = torch.mean(torch.pow((latent_i-self.latent_orig), 2), dim=1)
                time_o = time.time()

                self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = self.gt.reshape(error.size(0))
                self.latent_i [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_i.reshape(error.size(0), self.opt.nz)
                self.latent_o [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_o.reshape(error.size(0), self.opt.nz)

                self.times.append(time_o - time_i)

                # Save test images.
                if self.opt.save_test_images:
                    dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    if not os.path.isdir(dst):
                        os.makedirs(dst)
                    real, fake, fixed, masked = self.get_current_images()
                    vutils.save_image(real, '%s/real_%03d.png' % (dst, i+1), normalize=True)
                    vutils.save_image(fake, '%s/fake_%03d.png' % (dst, i+1), normalize=True)

            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))
            # auc, eer = roc(self.gt_labels, self.an_scores)
            auc = evaluate(self.gt_labels, self.an_scores, metric=self.opt.metric)
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), (self.opt.metric, auc)])

            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.dataloader['test'].dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)
            return performance

##
class Ganomaly(BaseModel):
    """GANomaly Class
    """

    @property
    def name(self): return 'Ganomaly'

    def __init__(self, opt, dataloader):
        super(Ganomaly, self).__init__(opt, dataloader)

        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0

        ##
        # Create and initialize networks.
        self.netg = NetG(self.opt).to(self.device)
        self.netd = NetD(self.opt).to(self.device)
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)
        
        ##
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(self.opt.resume['netg_path'])['epoch']
            self.netg.load_state_dict(torch.load(self.opt.resume['netg_path'])['state_dict'])
            self.netd.load_state_dict(torch.load(self.opt.resume['netd_path'])['state_dict'])
            print("\tLoaded pre-trained weights.\n")
    

        self.l_adv = l2_loss
        self.l_con = nn.L1Loss()
        self.l_enc = l2_loss
        self.l_bce = nn.BCELoss()

        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt    = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        # [배치 크기]로 정의
        self.real_label = torch.ones(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.fake_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)


        ##
        # Setup optimizer
        if self.opt.isTrain:
            self.netg.train()
            self.netd.train()
            self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            
            
    # input 이미지에 대한 마스킹 함수
    def apply_context_mask(self, x):
        """Apply a random mask to each image in the batch with varying positions and sizes."""
        B, C, H, W = x.size()
        mask = torch.ones_like(x)

        # Loop through each image in the batch
        for i in range(B):
            # Generate random size for the mask
            mask_size = torch.randint(
                int(self.opt.context_mask_size / 2), 
                self.opt.context_mask_size + 1, 
                (1,)
            ).item()

            # Generate random positions for the mask
            h_start = torch.randint(0, H - mask_size + 1, (1,)).item()
            w_start = torch.randint(0, W - mask_size + 1, (1,)).item()

            # Apply the mask to the current image
            mask[i, :, h_start:h_start+mask_size, w_start:w_start+mask_size] = 0

        masked_input = x * mask
        return masked_input, mask


    ##
    def forward_g(self):
        """ Forward propagate through netG
        """
        if self.opt.use_context_pred:
            # Generate masked input
            self.masked_input, self.mask = self.apply_context_mask(self.input)
            (self.fake, self.reconstructed_from_latent, self.latent_orig, self.latent_i, self.latent_o) = self.netg(self.input, masked_input=self.masked_input)
        else:
            # 마스킹 비활성화일 경우 None 설정
            self.fake, self.latent_i, self.latent_o = self.netg(self.input, masked_input=None)

    ##
    def forward_d(self):
        """ Forward propagate through netD """
        self.pred_real, self.feat_real = self.netd(self.input)
        self.pred_fake, self.feat_fake = self.netd(self.fake.detach())

        # Check if pred_real is of shape [batch_size]
        self.pred_real = self.pred_real.view(-1)  # [batch_size] 형태로 조정
        self.pred_fake = self.pred_fake.view(-1)  # [batch_size] 형태로 조정

    
    ''' 수정 전
    ##
    def backward_g(self):
        """Calculate Generator loss"""
        # 기존 손실
        self.err_g_adv = self.l_adv(self.netd(self.input)[1], self.netd(self.fake)[1])
        self.err_g_con = self.l_con(self.fake, self.input)
        self.err_g_enc = self.l_enc(self.latent_o, self.latent_i)

        # 총 손실
        self.err_g = self.err_g_adv * self.opt.w_adv + \
                    self.err_g_con * self.opt.w_con + \
                    self.err_g_enc * self.opt.w_enc
        self.err_g.backward(retain_graph=True)
    '''
    
    def backward_g(self):
        """Calculate Generator loss"""
        # 기존 손실
        self.err_g_adv = self.l_adv(self.netd(self.input)[1], self.netd(self.fake)[1])
        self.err_g_con = self.l_con(self.fake, self.input)
        
        # Encoder Loss 구성 요소
        # latent_o: 마스킹 이미지 잠재 벡터 / latent_i: 원본 이미지 잠재 벡터
        # 1. 마스킹된 이미지와 마스킹된 이미지 사이의 latent 차이
        enc_loss_masked = self.l_enc(self.latent_o, self.netg.encoder2(self.netg.decoder(self.latent_o)))

        # 2. 원본 이미지와 복원된 원본 이미지 사이의 latent 차이
        reconstructed_image = self.netg.decoder(self.latent_i)
        latent_orig_recon = self.netg.encoder2(reconstructed_image)
        enc_loss_orig = self.l_enc(latent_orig_recon, self.latent_i)

        # `encoder_loss_ratio`를 사용하여 두 loss의 가중치 결정
        weight_masked = self.opt.encoder_loss_ratio
        weight_orig = 1 - weight_masked
        self.err_g_enc = (
            enc_loss_masked * weight_masked +
            enc_loss_orig * weight_orig
        )
        
        # Reconstruction Loss 계산
        self.err_g_rec = self.l_con(self.reconstructed_from_latent, self.input)

        # 총 Generator Loss
        self.err_g = self.err_g_adv * self.opt.w_adv + \
                    self.err_g_con * self.opt.w_con + \
                    self.err_g_enc * self.opt.w_enc + \
                    self.err_g_rec * self.opt.w_rec
        self.err_g.backward(retain_graph=True)
        
    
    '''
    ##
    def backward_d(self): # 컬러
        """ Backpropagate through netD"""
        # Real - Fake Loss
        self.err_d_real = self.l_bce(self.pred_real, self.real_label)
        self.err_d_fake = self.l_bce(self.pred_fake, self.fake_label)

        # NetD Loss & Backward-Pass
        self.err_d = (self.err_d_real + self.err_d_fake) * 0.5
        self.err_d.backward()
    '''
        
    def backward_d(self): # 흑백
        """ Backpropagate through netD """
        # Real - Fake Loss
        batch_size = self.real_label.size(0)  # 실제 배치 크기 가져오기

        # pred_real과 pred_fake 크기 조정
        self.pred_real = self.pred_real.view(batch_size, -1).mean(dim=1)
        self.pred_fake = self.pred_fake.view(batch_size, -1).mean(dim=1)

        # BCELoss 계산
        self.err_d_real = self.l_bce(self.pred_real, self.real_label)
        self.err_d_fake = self.l_bce(self.pred_fake, self.fake_label)

        # NetD Loss & Backward-Pass
        self.err_d = (self.err_d_real + self.err_d_fake) * 0.5
        self.err_d.backward()


    ##
    def reinit_d(self):
        """ Re-initialize the weights of netD
        """
        self.netd.apply(weights_init)
        print('   Reloading net d')

    def optimize_params(self):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        self.forward_g()
        self.forward_d()

        # Backward-pass
        # netg
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()

        # netd
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()
        if self.err_d.item() < 1e-5: self.reinit_d()


