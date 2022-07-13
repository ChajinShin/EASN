import os
import skimage.io as io
import torch
from piq import psnr, multi_scale_ssim
from skimage.util import img_as_ubyte
from solver import Solver
from .src.dataset import get_dataloader
from .src.wrapper import Wrapper
from .src.utils import ElapsedTimeProcess, Logger, MovingAverageMeter, ProcessBar, write_bitstream


class Policy(Solver):
    def __init__(self, opt, dev):
        super(Policy, self).__init__(opt, dev)
        self.set()

    def set(self):
        # model options
        self.experiment_name = self.opt.ExperimentName
        self.opt = getattr(self.opt, self.opt.Model)

        # prepare ckpt folder and result folder
        os.makedirs(f"{self.experiment_name}/ckpt", exist_ok=True)
        os.makedirs(f"{self.experiment_name}/results", exist_ok=True)

        # settings
        self.logger = Logger(os.path.join(self.experiment_name, 'log.txt'))

        # network settings
        self.model = Wrapper(self.opt, self.dev)

        # load pretrained parameters
        self.current_epoch = 1
        if os.path.exists(self.opt.model.params):
            self.current_epoch = self.model.load_parameters()

        print("Setting completed")

    def execute(self):
        if self.opt.mode == 'train':
            print("Training mode")
            self.train()
        elif self.opt.mode == 'test':
            print("Test mode")
            self.test()
        else:
            raise ValueError('mode option is only available for "train" or "test"')

    def train(self):
        # get data loader
        train_data_loader, val_data_loader = get_dataloader(self.opt, is_train=True)

        # settings
        ETA_meter = ElapsedTimeProcess(self.opt.train.epoch, len(train_data_loader))
        avg_meter = MovingAverageMeter()

        for self.current_epoch in range(self.current_epoch, self.opt.train.epoch + 1):  # start from epoch 1 not 0
            self.model.train()
            
            # ------------- one epoch --------------------------
            for step, img in enumerate(train_data_loader):
                # every print step,
                if ((step + 1) % self.opt.print_step) == 0:
                    ETA_meter.start()

                log = self.model.fit(img)
                avg_meter.update(log)
                
                # every print step,
                if ((step + 1) % self.opt.print_step) == 0:
                    # get last lr
                    lr = self.model.get_lr_value()

                    # print log
                    log = avg_meter.get_value()

                    msg = f'Epoch:  {self.current_epoch}/{self.opt.train.epoch}    Step: {step + 1}/{len(train_data_loader)}  |  '
                    for key, value in log.items():
                        msg += f'{key}: {value:.5f}   |   '
                    msg += f'lr: {lr:.2e}  |   '

                    # elapsed time
                    eta = ETA_meter.end(self.current_epoch, step)
                    msg += eta
                    self.logger(msg)

            # ---------------- after finishing one epoch,  --------------------
            # save parameter
            self.save_param()

            # evaluation
            val_loss = self.val(val_data_loader)
            
            # scheduler step
            self.model.lr_scheduler_step(val_loss)
        
    def update(self):
        self.model.update()

    def val(self, val_data_loader):
        print("\n\nStart evaluation")
        avg_meter = MovingAverageMeter()
        process_bar = ProcessBar(len(val_data_loader))

        self.model.eval()
        for img in val_data_loader:
            log = self.model.val(img)
            avg_meter.update(log)
            process_bar.step()
        log = avg_meter.get_value()

        msg = "\nEvaluation Results:\n"
        for name, value in log.items():
            msg += f"{name}: {value:.5f}  "
        msg += "\n\n"
        self.logger(msg)
        return log['loss']
    
    def test(self):
        torch.backends.cudnn.deterministic = True
        torch.set_num_threads(1)
        
        # relocate model to cpu device, update cdf table, change to eval mode
        self.model.test_mode()

        # get dataloader
        test_dataloader = get_dataloader(self.opt, False)

        process_bar = ProcessBar(len(test_dataloader))
        
        avg_psnr = 0.0
        avg_msssim = 0.0
        avg_bpp = 0
        for batch in test_dataloader:
            file_name, img = batch

            img_np = img[0].permute(1, 2, 0).contiguous().numpy()
            io.imsave(f"{self.experiment_name}/results/{file_name[0]}_gt.png", img_as_ubyte(img_np), check_contrast=False)

            # prediction
            output = self.model.test(img.to(self.dev))
            pred = output['x_hat']
            bitstream_dict = output['bitstream_dict']

            # save bit-stream
            file_loc = f'{self.experiment_name}/results/{file_name[0]}.txt'
            with open(file_loc, 'wb') as f:
                write_bitstream(f, bitstream_dict)

            # calculate bpp
            file_size = os.path.getsize(file_loc) * 8   # byte to bit
            total_pixel = pred.size(0) * pred.size(2) * pred.size(3)
            bpp = file_size / total_pixel
            avg_bpp += bpp

            # calculate PSNR, SSIM
            pred = pred.clamp(0.0, 1.0).cpu()
            avg_psnr += psnr(pred, img, data_range=1.0).item()
            avg_msssim += multi_scale_ssim(pred, img, data_range=1.0).item()

            # save process
            pred = pred.squeeze().permute(1, 2, 0).contiguous().cpu().numpy()
            io.imsave(f"{self.experiment_name}/results/{file_name[0]}.png", img_as_ubyte(pred), check_contrast=False)
            process_bar.step()
        avg_bpp /= len(test_dataloader)
        avg_psnr /= len(test_dataloader)
        avg_msssim /= len(test_dataloader)

        msg = f"\nTest Results:  \nPSNR:  {avg_psnr:.4f}  |  MS-SSIM:  {avg_msssim:.4f}  |  BPP:  {avg_bpp:.4f}"
        self.logger(msg)

    def save_param(self):
        param_loc = os.path.join(self.experiment_name, f'ckpt/parameters_{self.current_epoch:03d}_epoch.pth')
        self.model.save_parameters(self.current_epoch, param_loc)

