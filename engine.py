import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast
from utils import save_imgs
import torch.nn.functional as F

def update_confusion(pred, true, TP, FP, FN, TN):
    # pred/true: uint8 tensor [B,1,H,W]
    TP += (pred & true).sum().item()
    FP += (pred & (1 - true)).sum().item()
    FN += ((1 - pred) & true).sum().item()
    TN += ((1 - pred) & (1 - true)).sum().item()
    return TP, FP, FN, TN

def metrics_from_confusion(TP, FP, FN, TN):
    den = TP + TN + FP + FN
    accuracy = (TP + TN) / den if den else 0.0
    sensitivity = TP / (TP + FN) if (TP + FN) else 0.0
    specificity = TN / (TN + FP) if (TN + FP) else 0.0
    f1 = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) else 0.0
    miou = TP / (TP + FP + FN) if (TP + FP + FN) else 0.0
    return accuracy, sensitivity, specificity, f1, miou
def infer_sliding_window(model, img, patch=512, stride=256):
    """
    img: torch tensor [1,3,H,W] (đã normalize)
    return: prob map [1,1,H,W] (0..1)
    """
    model.eval()
    B, C, H, W = img.shape
    assert B == 1, "Sliding-window inference currently supports batch=1."

    pad_h = (patch - H % patch) % patch
    pad_w = (patch - W % patch) % patch
    img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, Hp, Wp = img_pad.shape

    prob_sum = torch.zeros((1, 1, Hp, Wp), device=img.device)
    cnt_sum  = torch.zeros((1, 1, Hp, Wp), device=img.device)

    tops = list(range(0, Hp - patch + 1, stride))
    lefts = list(range(0, Wp - patch + 1, stride))
    if len(tops) == 0: tops = [0]
    if len(lefts) == 0: lefts = [0]
    if tops[-1] != Hp - patch: tops.append(Hp - patch)
    if lefts[-1] != Wp - patch: lefts.append(Wp - patch)

    for top in tops:
        for left in lefts:
            patch_img = img_pad[:, :, top:top+patch, left:left+patch]
            gt_pre, out = model(patch_img)

            if isinstance(out, tuple):
                out = out[0]

            # đảm bảo out là prob
            if out.min() < 0 or out.max() > 1:
                out = torch.sigmoid(out)

            prob_sum[:, :, top:top+patch, left:left+patch] += out
            cnt_sum[:, :, top:top+patch, left:left+patch]  += 1.0

    prob = prob_sum / (cnt_sum + 1e-8)
    prob = prob[:, :, :H, :W]
    return prob


def train_one_epoch(train_loader, model, criterion, optimizer, scheduler,
                    epoch, step, logger, config, writer, scaler=None):
    model.train()
    loss_list = []
    TP = FP = FN = TN = 0

    optimizer.zero_grad(set_to_none=True)

    if config.amp and scaler is None:
        raise ValueError("config.amp=True nhưng scaler=None. Hãy truyền scaler vào train_one_epoch().")

    for it, data in enumerate(train_loader):
        step += 1
        images, targets = data
        images = images.cuda(non_blocking=True).float()
        targets = targets.cuda(non_blocking=True).float()

        with autocast(enabled=config.amp):
            gt_pre, out = model(images)

            # handle tuple output
            if isinstance(out, tuple):
                out = out[0]

            # đảm bảo out/gt_pre là probability (0..1) trước khi tính Tversky
            if out.min() < 0 or out.max() > 1:
                out = torch.sigmoid(out)

            if isinstance(gt_pre, (list, tuple)):
                gt_pre = [torch.sigmoid(x) if (x.min() < 0 or x.max() > 1) else x for x in gt_pre]

            loss = criterion(gt_pre, out, targets)

        if config.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        loss_list.append(loss.item())

        # train metrics (threshold giống VAL/TEST)
        pred = (out >= config.threshold).to(torch.uint8)
        true = (targets >= 0.5).to(torch.uint8)
        TP, FP, FN, TN = update_confusion(pred, true, TP, FP, FN, TN)

        now_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('loss', loss.item(), global_step=step)

        if it % config.print_interval == 0:
            log_info = f"train: epoch {epoch}, iter:{it}, loss: {np.mean(loss_list):.6e}, lr: {now_lr}"
            print(log_info)
            logger.info(log_info)

    scheduler.step()
     # train metrics summary
    avg_train_loss = float(np.mean(loss_list)) if len(loss_list) > 0 else 0.0
    acc, sens, spec, f1, miou = metrics_from_confusion(TP, FP, FN, TN)

    log_info = (
        f"train epoch: {epoch}, "
        f"loss: {avg_train_loss:.6e}, "
        f"miou: {miou:.4f}, f1: {f1:.4f}, "
        f"sens: {sens:.4f}, spec: {spec:.4f}"
    )
    print(log_info)
    logger.info(log_info)

    return step, avg_train_loss, miou, f1, sens, spec


def val_one_epoch(test_loader, model, criterion, epoch, logger, config):
    model.eval()
    loss_list = []

    # accumulate confusion (tránh lưu full preds/gts -> đỡ crash RAM)
    TP = FP = FN = TN = 0

    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img = img.cuda(non_blocking=True).float()
            msk = msk.cuda(non_blocking=True).float()

            out = infer_sliding_window(model, img, patch=config.patch_size, stride=config.stride)

            # loss log: base only (không deep supervision vì sliding-window)
            if hasattr(criterion, "base"):
                loss = criterion.base(out, msk).item()
            else:
                loss = 0.0
            loss_list.append(loss)

            pred = (out >= config.threshold).to(torch.uint8)
            true = (msk >= 0.5).to(torch.uint8)

            TP += (pred & true).sum().item()
            FP += (pred & (1 - true)).sum().item()
            FN += ((1 - pred) & true).sum().item()
            TN += ((1 - pred) & (1 - true)).sum().item()

    val_loss = float(np.mean(loss_list)) if len(loss_list) > 0 else 0.0

    den = TP + TN + FP + FN
    accuracy = (TP + TN) / den if den else 0.0
    sensitivity = TP / (TP + FN) if (TP + FN) else 0.0
    specificity = TN / (TN + FP) if (TN + FP) else 0.0
    f1_or_dsc = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) else 0.0
    miou = TP / (TP + FP + FN) if (TP + FP + FN) else 0.0

    confusion = np.array([[TN, FP], [FN, TP]], dtype=np.int64)

    log_info = (
        f"val epoch: {epoch}, loss: {val_loss:.6e}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, "
        f"accuracy: {accuracy}, specificity: {specificity}, sensitivity: {sensitivity}, "
        f"confusion_matrix: {confusion}"
    )
    print(log_info)
    logger.info(log_info)

    return val_loss, miou, f1_or_dsc, sensitivity, specificity


def test_one_epoch(test_loader, model, criterion, logger, config, test_data_name=None):
    model.eval()
    loss_list = []

    TP = FP = FN = TN = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img = img.cuda(non_blocking=True).float()
            msk = msk.cuda(non_blocking=True).float()

            out = infer_sliding_window(model, img, patch=config.patch_size, stride=config.stride)

            # loss log
            if hasattr(criterion, "base"):
                loss = criterion.base(out, msk).item()
            else:
                loss = 0.0
            loss_list.append(loss)

            pred = (out >= config.threshold).to(torch.uint8)
            true = (msk >= 0.5).to(torch.uint8)

            TP += (pred & true).sum().item()
            FP += (pred & (1 - true)).sum().item()
            FN += ((1 - pred) & true).sum().item()
            TN += ((1 - pred) & (1 - true)).sum().item()

            # save ảnh (nếu cần)
            if i % config.save_interval == 0:
                msk_np = msk.squeeze(1).detach().cpu().numpy()
                out_np = out.squeeze(1).detach().cpu().numpy()
                save_imgs(img, msk_np, out_np, i,
                          config.work_dir + 'outputs/',
                          config.datasets, config.threshold,
                          test_data_name=test_data_name)

    test_loss = float(np.mean(loss_list)) if len(loss_list) > 0 else 0.0

    den = TP + TN + FP + FN
    accuracy = (TP + TN) / den if den else 0.0
    sensitivity = TP / (TP + FN) if (TP + FN) else 0.0
    specificity = TN / (TN + FP) if (TN + FP) else 0.0
    f1_or_dsc = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) else 0.0
    miou = TP / (TP + FP + FN) if (TP + FP + FN) else 0.0

    confusion = np.array([[TN, FP], [FN, TP]], dtype=np.int64)

    log_info = (
        f"test of best model, loss: {test_loss:.6e}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, "
        f"accuracy: {accuracy}, specificity: {specificity}, sensitivity: {sensitivity}, "
        f"confusion_matrix: {confusion}"
    )
    print(log_info)
    logger.info(log_info)

    return test_loss
