import torch
from torch.utils.data import DataLoader
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.egeunet import EGEUNet
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from engine import *
import os
import sys
from utils import *
from configs.config_setting import setting_config
import warnings
warnings.filterwarnings("ignore")



def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = getattr(config, "resume_path", "")
    if getattr(config, "force_new", False):
        resume_model = ""
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)



    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()



    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=config.pin_memory,
                                num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=config.pin_memory, 
                                num_workers=config.num_workers,
                                drop_last=False)


    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'egeunet':
        model = EGEUNet(num_classes=model_cfg['num_classes'], 
                        input_channels=model_cfg['input_channels'], 
                        c_list=model_cfg['c_list'], 
                        bridge=model_cfg['bridge'],
                        gt_ds=model_cfg['gt_ds'],
                        )
    else: raise Exception('network in not right!')
    model = model.cuda()


    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

    # ---------- QUICK CHECK: only test ----------
    if getattr(config, "only_test", False):
        print('#----------Only Testing (quick check)----------#')

        if not os.path.exists(resume_model):
            print("resume_model not found:", resume_model)
            return

        checkpoint = torch.load(resume_model, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.cuda()

        test_one_epoch(val_loader, model, criterion, logger, config)
        return

    print('#----------Set other params----------#')
    start_epoch = 1
    best_miou = -1.0
    best_epoch = 0
    step = 0

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        saved_epoch = checkpoint.get('epoch', 0)
        start_epoch = saved_epoch + 1

        best_miou = checkpoint.get('best_miou', -1.0)
        best_epoch = checkpoint.get('best_epoch', 0)
        last_loss = checkpoint.get('loss', None)

        step = checkpoint.get('step', 0)

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, best_miou: {best_miou:.4f}, best_epoch: {best_epoch}, last_loss: {last_loss}'
        logger.info(log_info)


    history = {
        "train_loss": [],
        "val_loss": [],
        "train_miou": [],
        "val_miou": [],
        "train_f1": [],
        "val_f1": [],
        "train_sens": [],
        "val_sens": [],
        "train_spec": [],
        "val_spec": [],
    }

    
    print('#----------Training----------#')


    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step, train_loss, train_miou, train_f1, train_sens, train_spec = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer,
            scaler=scaler
        )
        history["train_loss"].append(train_loss)
        history["train_miou"].append(train_miou)
        history["train_f1"].append(train_f1)
        history["train_sens"].append(train_sens)
        history["train_spec"].append(train_spec)

        val_loss, miou, f1, sens, spec = val_one_epoch(
            val_loader,
            model,
            criterion,
            epoch,
            logger,
            config
        )
        history["val_loss"].append(val_loss)
        history["val_miou"].append(miou)
        history["val_f1"].append(f1)
        history["val_sens"].append(sens)
        history["val_spec"].append(spec)

        if miou > best_miou:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            best_miou = miou
            best_epoch = epoch

        torch.save(
            {
                'epoch': epoch,
                'step': step,
                'loss': val_loss,
                'miou': miou,
                'f1': f1,
                'best_miou': best_miou,
                'best_epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth'))
    save_training_plots(history, config.work_dir)
 

    best_path = os.path.join(checkpoint_dir, 'best.pth')
    if os.path.exists(best_path):
        print('#----------Testing----------#')
        best_state = torch.load(best_path, map_location='cpu')
        model.load_state_dict(best_state)

        loss = test_one_epoch(val_loader, model, criterion, logger, config)

        os.replace(best_path, os.path.join(checkpoint_dir, f'best-epoch{best_epoch}-miou{best_miou:.4f}.pth'))      

def save_training_plots(history, work_dir):
    plot_dir = os.path.join(work_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Loss
    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.title("Loss Curve")
    plt.savefig(os.path.join(plot_dir, "loss_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Dice/F1 (Train vs Val)
    plt.figure()
    plt.plot(history["train_f1"], label="Train Dice/F1")
    plt.plot(history["val_f1"], label="Val Dice/F1")
    plt.xlabel("Epoch"); plt.ylabel("Dice/F1")
    plt.legend(); plt.title("Dice/F1 Curve")
    plt.savefig(os.path.join(plot_dir, "dice_f1_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # mIoU (Train vs Val)
    plt.figure()
    plt.plot(history["train_miou"], label="Train mIoU")
    plt.plot(history["val_miou"], label="Val mIoU")
    plt.xlabel("Epoch"); plt.ylabel("mIoU")
    plt.legend(); plt.title("mIoU Curve")
    plt.savefig(os.path.join(plot_dir, "miou_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Sensitivity (Train vs Val) - riêng
    plt.figure()
    plt.plot(history["train_sens"], label="Train Sensitivity")
    plt.plot(history["val_sens"], label="Val Sensitivity")
    plt.xlabel("Epoch"); plt.ylabel("Score")
    plt.legend(); plt.title("Sensitivity Curve")
    plt.savefig(os.path.join(plot_dir, "sensitivity_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Specificity (Train vs Val) - riêng
    plt.figure()
    plt.plot(history["train_spec"], label="Train Specificity")
    plt.plot(history["val_spec"], label="Val Specificity")
    plt.xlabel("Epoch"); plt.ylabel("Score")
    plt.legend(); plt.title("Specificity Curve")
    plt.savefig(os.path.join(plot_dir, "specificity_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved plots to:", plot_dir)
if __name__ == '__main__':
    config = setting_config
    main(config)