import torch
import os
from .Losses import RateDistortionLoss
from .network import JA_GAS


class Wrapper(object):
    def __init__(self, opt, dev):
        self.opt = opt
        self.dev = dev

        self.compression_net = JA_GAS(self.opt.model.N, self.opt.model.M).to(dev)

        # ------------------------------------------------------------------------------------------
        # optimizer settings
        parameters = {n for n, p in self.compression_net.named_parameters() if not n.endswith(".quantiles") and p.requires_grad}
        aux_parameters = {n for n, p in self.compression_net.named_parameters() if n.endswith(".quantiles") and p.requires_grad}

        # Make sure we don't have an intersection of parameters
        params_dict = dict(self.compression_net.named_parameters())
        inter_params = parameters & aux_parameters
        union_params = parameters | aux_parameters

        assert len(inter_params) == 0
        assert len(union_params) - len(params_dict.keys()) == 0

        self.optimizer = torch.optim.Adam((params_dict[n] for n in sorted(parameters)), lr=float(opt.train.lr))
        self.aux_optimizer = torch.optim.Adam((params_dict[n] for n in sorted(aux_parameters)), lr=float(opt.train.aux_lr))

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min",
                                                                        threshold=float(opt.train.threshold),
                                                                        factor=float(opt.train.factor),
                                                                        patience=int(opt.train.patience))
        # --------------------------------------------------------------------------------------------------------

        self.clip_max_norm = opt.train.clip_max_norm
        self.distortion_rate_loss = RateDistortionLoss(distortion=opt.train.distortion, lmbda=float(opt.train.lmbda))

    def load_parameters(self):
        if os.path.exists(self.opt.model.params):
            state_dict = torch.load(self.opt.model.params, map_location=self.dev)
            self.compression_net.load_state_dict(state_dict['compression_net'])
            if self.opt.train.load_optimizer:
                self.optimizer.load_state_dict(state_dict['optimizer'])
                self.aux_optimizer.load_state_dict(state_dict['aux_optimizer'])
                self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
                current_epoch = state_dict['last_epoch'] + 1
            else:
                current_epoch = 1
            print("parameters loaded")
        else:
            current_epoch = 1
        return current_epoch

    def save_parameters(self, epoch, save_loc):
        state_dict = dict()
        state_dict['compression_net'] = self.compression_net.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['aux_optimizer'] = self.aux_optimizer.state_dict()
        state_dict['lr_scheduler'] = self.lr_scheduler.state_dict()
        state_dict['last_epoch'] = epoch
        torch.save(state_dict, save_loc)
        print("parameters saved")

    def fit(self, img):
        log = dict()
        self.optimizer.zero_grad()
        self.aux_optimizer.zero_grad()

        # to device
        img = img.to(self.dev)

        # forward
        predict = self.compression_net(img)

        loss_dict = self.distortion_rate_loss(predict, img)
        loss = loss_dict['loss']
        loss.backward()

        if self.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.compression_net.parameters(), self.clip_max_norm)
        self.optimizer.step()

        log['bpp_loss'] = loss_dict['bpp_loss'].item()
        log['dist_loss'] = loss_dict['dist_loss'].item()

        aux_loss = self.compression_net.aux_loss()
        aux_loss.backward()
        self.aux_optimizer.step()

        log['aux_loss'] = aux_loss.item()

        return log

    @torch.no_grad()
    def val(self, img):
        log = dict()

        # to device
        img = img.to(self.dev)

        # forward
        predict = self.compression_net(img)
        loss_dict = self.distortion_rate_loss(predict, img)

        log['loss'] = loss_dict['loss']
        log['bpp_loss'] = loss_dict['bpp_loss'].item()
        log['dist_loss'] = loss_dict['dist_loss'].item()
        return log

    @torch.no_grad()
    def test(self, img):
        bitstream_dict = self.compression_net.compress(img)
        decomp_result = self.compression_net.decompress(bitstream_dict['strings'], bitstream_dict['shape'])

        return {
            "bitstream_dict": bitstream_dict,
            "x_hat": decomp_result['x_hat']
        }

    def test_mode(self):
        self.update()
        self.compression_net.eval()

    def update(self):
        self.compression_net.update()

    def lr_scheduler_step(self, loss):
        self.lr_scheduler.step(loss)

    def get_lr_value(self):
        lr = self.optimizer.param_groups[0]['lr']
        return lr

    def train(self):
        self.compression_net.train()
    
    def eval(self):
        self.compression_net.eval()


