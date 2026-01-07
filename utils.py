import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random
import logging
import logging.handlers
from matplotlib import pyplot as plt
# =========================
# FIVES normalization stats (computed from TRAIN set)
# =========================
FIVES_MEAN = np.array([85.86317539, 38.87655518, 16.15345802], dtype=np.float32)
FIVES_STD  = np.array([62.03288873, 31.96086046, 14.49887466], dtype=np.float32)

# dùng để chỉ định nội suy cho ảnh vs mask
from torchvision.transforms.functional import InterpolationMode

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    from scipy import ndimage as ndi
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def remove_small_objects(binary_mask, min_area=50):
    """
    binary_mask: numpy array (0/1) hoặc (0/255)
    min_area: bỏ các component nhỏ hơn min_area pixels
    """
    m = (binary_mask > 0).astype(np.uint8)

    if _HAS_CV2:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        out = np.zeros_like(m)
        for i in range(1, num_labels):  # 0 là background
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                out[labels == i] = 1
        return out

    if _HAS_SCIPY:
        labels, num = ndi.label(m)
        if num == 0:
            return m
        sizes = ndi.sum(m, labels, index=np.arange(1, num + 1))
        out = np.zeros_like(m)
        for i, area in enumerate(sizes, start=1):
            if area >= min_area:
                out[labels == i] = 1
        return out

    # không có cv2/scipy thì trả nguyên bản (không crash)
    return m


try:
    from skimage.morphology import remove_small_objects as sk_remove_small_objects
    from skimage.morphology import remove_small_holes as sk_remove_small_holes
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


def postprocess_mask(msk_bin, min_area=20, min_hole=0):
    """
    msk_bin: numpy mask nhị phân 0/1 (hoặc 0/255 cũng ok)
    min_area: bỏ các blob nhỏ (noise) dưới ngưỡng này
    min_hole: lấp các lỗ nhỏ bên trong mạch (0 = không lấp)
    """
    m = (msk_bin > 0)

    # Ưu tiên skimage vì chuẩn & ổn định (bạn đã cài scikit-image rồi)
    if _HAS_SKIMAGE:
        m = sk_remove_small_objects(m, min_size=int(min_area))
        if min_hole and int(min_hole) > 0:
            m = sk_remove_small_holes(m, area_threshold=int(min_hole))
        return m.astype(np.uint8)

    # fallback: dùng hàm remove_small_objects bạn tự viết (cv2/scipy)
    m = remove_small_objects(m.astype(np.uint8), min_area=int(min_area)).astype(bool)

    # min_hole: nếu không có skimage thì tạm bỏ qua (để không phức tạp)
    return m.astype(np.uint8)


def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # tránh addHandler nhiều lần
    if logger.handlers:
        return logger

    info_name = os.path.join(log_dir, f"{name}.info.log")
    info_handler = logging.handlers.TimedRotatingFileHandler(
        info_name, when='D', encoding='utf-8'
    )
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)
    return logger


def log_config_info(config, logger):
    config_dict = config.__dict__
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        else:
            log_info = f'{k}: {v},'
            logger.info(log_info)


def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr=config.lr,
            rho=config.rho,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr=config.lr,
            lr_decay=config.lr_decay,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr=config.lr,
            lambd=config.lambd,
            alpha=config.alpha,
            t0=config.t0,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            alpha=config.alpha,
            eps=config.eps,
            centered=config.centered,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr=config.lr,
            etas=config.etas,
            step_sizes=config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            dampening=config.dampening,
            nesterov=config.nesterov
        )
    else:  # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.05,
        )


def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                          'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.milestones,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.T_max,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.mode,
            factor=config.factor,
            patience=config.patience,
            threshold=config.threshold,
            threshold_mode=config.threshold_mode,
            cooldown=config.cooldown,
            min_lr=config.min_lr,
            eps=config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.T_0,
            T_mult=config.T_mult,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma**len(
            [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
            math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler


def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    """
    img: tensor (1,C,H,W) đã normalize
    msk: numpy/tensor mask
    msk_pred: numpy prediction (prob)
    """
    # --- img: de-normalize để hiển thị đúng ---
    img_np = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)  # HWC

    # de-normalize theo dataset (GIỐNG myNormalize)
    if datasets == 'fives':
        mean = FIVES_MEAN
        std  = FIVES_STD
        img_vis = img_np * std + mean   # broadcast theo kênh RGB
        img_vis = np.clip(img_vis, 0, 255) / 255.0
    else:
        # fallback: nếu không rõ normalize thì cố đưa về 0-1
        img_vis = img_np
        img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-8)

    # --- mask / pred ---
    if datasets == 'retinal':
        msk = np.squeeze(msk, axis=0)
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)
        #msk_pred = postprocess_mask(msk_pred, min_area=3, min_hole=0)

    plt.figure(figsize=(7, 15))

    plt.subplot(3, 1, 1)
    plt.imshow(img_vis)
    plt.axis('off')

    plt.subplot(3, 1, 2)
    plt.imshow(msk, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 1, 3)
    plt.imshow(msk_pred, cmap='gray')
    plt.axis('off')

    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'
    plt.savefig(save_path + str(i) + '.png', bbox_inches='tight')
    plt.close()


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size
        return dice_loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)
        loss = self.wd * diceloss + self.wb * bceloss
        return loss


class GT_BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(GT_BceDiceLoss, self).__init__()
        self.bcedice = BceDiceLoss(wb, wd)

    def forward(self, gt_pre, out, target):
        bcediceloss = self.bcedice(out, target)
        gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
        gt_loss = (
            self.bcedice(gt_pre5, target) * 0.1 +
            self.bcedice(gt_pre4, target) * 0.2 +
            self.bcedice(gt_pre3, target) * 0.3 +
            self.bcedice(gt_pre2, target) * 0.4 +
            self.bcedice(gt_pre1, target) * 0.5
        )
        return bcediceloss + gt_loss


class myToTensor:
    def __call__(self, data):
        image, mask = data
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        return image, mask


class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w

    def __call__(self, data):
        image, mask = data
        # ảnh dùng BILINEAR, mask dùng NEAREST để giữ 0/1
        image = TF.resize(image, [self.size_h, self.size_w], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.size_h, self.size_w], interpolation=InterpolationMode.NEAREST)
        return image, mask


class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.hflip(image), TF.hflip(mask)
        else:
            return image, mask


class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.vflip(image), TF.vflip(mask)
        else:
            return image, mask


class myRandomRotation:
    def __init__(self, p=0.5, degree=[-15, 15]):  #đang dùng [-15,15] trong config thì để vậy cho thống nhất
        self.p = p
        self.degree = degree

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            angle = random.uniform(self.degree[0], self.degree[1])
            # ảnh BILINEAR, mask NEAREST
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
            return image, mask
        return image, mask


class myRandomCrop:
    def __init__(self, crop_h=512, crop_w=512, p=1.0):
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() > self.p:
            return image, mask

        # image/mask đang là numpy HWC (trước ToTensor)
        H, W = image.shape[:2]
        ch, cw = self.crop_h, self.crop_w

        # nếu ảnh nhỏ hơn crop thì không crop
        if H <= ch or W <= cw:
            return image, mask

        top = random.randint(0, H - ch)
        left = random.randint(0, W - cw)
        image = image[top:top + ch, left:left + cw, :]
        mask = mask[top:top + ch, left:left + cw, :]
        return image, mask
class myRandomCropVesselAware:
    def __init__(self, crop_h=512, crop_w=512, p=1.0, vessel_prob=0.8):
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.p = p
        self.vessel_prob = vessel_prob

    def __call__(self, data):
        image, mask = data
        if random.random() > self.p:
            return image, mask

        H, W = image.shape[:2]
        ch, cw = self.crop_h, self.crop_w
        if H <= ch or W <= cw:
            return image, mask

        # mask: HWC (H,W,1) hoặc (H,W)
        m = mask[..., 0] if mask.ndim == 3 else mask
        vessel_idx = np.argwhere(m > 0.5)

        # ưu tiên crop có vessel
        if vessel_idx.size > 0 and random.random() < self.vessel_prob:
            y, x = vessel_idx[np.random.randint(len(vessel_idx))]
            top = int(np.clip(y - ch // 2, 0, H - ch))
            left = int(np.clip(x - cw // 2, 0, W - cw))
        else:
            top = random.randint(0, H - ch)
            left = random.randint(0, W - cw)

        image = image[top:top+ch, left:left+cw, :]
        mask  = mask[top:top+ch, left:left+cw, :]
        return image, mask


class myBinarizeMask:
    def __call__(self, data):
        image, mask = data
        # ADDED: chốt mask về 0/1 sau các biến đổi hình học
        mask = (mask > 0.5).float()
        return image, mask


class myNormalize:
    def __init__(self, data_name, train=True):
        if data_name == 'isic18':
            if train:
                self.mean = 157.561
                self.std = 26.706
            else:
                self.mean = 149.034
                self.std = 32.022
        elif data_name == 'isic17':
            if train:
                self.mean = 159.922
                self.std = 28.871
            else:
                self.mean = 148.429
                self.std = 25.748
        elif data_name == 'isic18_82':
            if train:
                self.mean = 156.2899
                self.std = 26.5457
            else:
                self.mean = 149.8485
                self.std = 35.3346
        elif data_name == 'fives':
            #dùng chung mean/std từ TRAIN set cho mọi split
            self.mean = FIVES_MEAN
            self.std  = FIVES_STD
        else:
            raise ValueError(f"Unknown dataset name: {data_name}")

    def __call__(self, data):
        img, msk = data
        img = img.astype(np.float32)
        msk = msk.astype(np.float32)
        img = (img - self.mean) / (self.std + 1e-8)
        return img, msk


class TverskyLoss(nn.Module):
    """
    pred, target: [B,1,H,W], pred in [0,1] (model bạn đã sigmoid rồi)
    Tversky = TP / (TP + alpha*FP + beta*FN)
    loss = 1 - Tversky
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.clamp(1e-6, 1 - 1e-6)
        target = target.float()
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)

        TP = (pred * target).sum(dim=1)
        FP = (pred * (1 - target)).sum(dim=1)
        FN = ((1 - pred) * target).sum(dim=1)

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        loss = 1 - tversky
        return loss.mean()


class FocalTverskyLoss(nn.Module):
    """
    FocalTversky = (1 - Tversky)^gamma
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        self.gamma = gamma

    def forward(self, pred, target):
        # tversky_loss = 1 - tversky
        tversky_loss = self.tversky(pred, target)
        return torch.pow(tversky_loss, self.gamma)


class GT_FocalTverskyLoss(nn.Module):
    """
    Wrapper cho GT deep supervision giống GT_BceDiceLoss của bạn.
    Dùng weights y chang: 0.1 0.2 0.3 0.4 0.5
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75):
        super().__init__()
        self.base = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)

    def forward(self, gt_pre, out, target):
        main_loss = self.base(out, target)
        gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
        gt_loss = (
            self.base(gt_pre5, target) * 0.1 +
            self.base(gt_pre4, target) * 0.2 +
            self.base(gt_pre3, target) * 0.3 +
            self.base(gt_pre2, target) * 0.4 +
            self.base(gt_pre1, target) * 0.5
        )
        return main_loss + gt_loss
