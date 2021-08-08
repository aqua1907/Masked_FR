import os

from tqdm import tqdm
from casia_dataset import MyLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from pathlib import Path
from torch.cuda.amp import GradScaler
from nets import iresnet
from loss.arc_face import ArcFaceLoss
import utils
import torch
import torch.nn as nn
import numpy as np
import logging
import yaml


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - np.cos(x * np.pi / steps)) / 2) * (y2 - y1) + y1


def train(device, hyp):
    train_dir, log_dir, batch_size, epochs, resume, binary = \
        hyp["train_dir"], hyp["log_dir"], hyp["batch_size"], hyp["epochs"], hyp['resume'], hyp['binary']

    # Summary writer
    if not resume:
        run_dir = Path(log_dir) / utils.create_run_folder(log_dir)
        ckpt_dir = run_dir / "weights"
    else:
        ckpt_dir = os.path.split(resume)[0]
        run_dir = ckpt_dir.replace('weights', '')
        ckpt_dir = Path(ckpt_dir)
    best = ckpt_dir / f"best.pt"
    print(best)
    writer = SummaryWriter(log_dir=run_dir)

    myloader = MyLoader(train_dir, batch_size=batch_size, test_size=0.25, seed=123)
    train_loader, val_loader = myloader.create_loaders()

    half = True if device == 'cuda' else False
    model = iresnet.iresnet34(binary=binary)
    model.to(device)

    last_module = list(model.children())[-1]    # get last module

    model_params = sum(p.numel() for p in model.parameters())
    model_params = utils.human_format(model_params)
    logging.info(f"Number of parameters = {model_params}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params = utils.human_format(trainable_params)
    logging.info(f"Number of trainable parameters = {trainable_params}")

    # Optimizer
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            if v.bias.requires_grad:
                pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            if v.weight.requires_grad:
                pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            if v.weight.requires_grad:
                pg1.append(v.weight)  # apply decay

    if hyp["adam"]:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.9))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)

    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    if hyp["linear_lr"]:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scaler = GradScaler()

    print(f"num classes = {myloader.num_classes}")
    arc_face = ArcFaceLoss(512, myloader.num_classes, device)
    arc_face.to(device)

    face_ce = torch.nn.CrossEntropyLoss()
    if isinstance(last_module, torch.nn.Sigmoid):
        mask_ce = torch.nn.BCELoss()
    else:
        mask_ce = torch.nn.BCEWithLogitsLoss()

    nb = len(train_loader)  # number of batches
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)

    start_epoch = 0
    best_val_loss = np.inf

    if resume:
        ckpt = torch.load(resume)
        # Epochs
        print(ckpt['epoch'])
        assert start_epoch > 0, f'model training to {epochs} epochs is finished, nothing to resume.'

        model.load_state_dict(ckpt['model'])

        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_val_loss = ckpt['best_val']

        if epochs < start_epoch:
            logger.info(f"model has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt

    scheduler.last_epoch = start_epoch - 1
    # Train
    for epoch in range(start_epoch, epochs):
        # Train
        model.train()
        losses = []
        face_losses = []
        mask_losses = []
        face_accs = []
        mask_accs = []

        avg_train_face_acc = 0.0
        avg_train_mask_acc = 0.0
        avg_train_loss = 0.0
        avg_train_face_loss = 0.0
        avg_train_mask_loss = 0.0

        loop = tqdm(enumerate(train_loader), total=nb)

        for i, (images, face_targets, mask_targets) in loop:
            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            images, face_targets, mask_targets = images.to(device), face_targets.to(device), mask_targets.to(device)

            ni = i + nb * epoch  # number integrated batches (since train start)
            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            if binary:
                mask_targets = mask_targets.unsqueeze(1).float()
            # forward
            with torch.cuda.amp.autocast():
                features, mask_preds = model(images)
                logits = arc_face(features, face_targets)
                face_loss = face_ce(logits, face_targets)
                mask_loss = mask_ce(mask_preds, mask_targets)
                loss = face_loss + (mask_loss + 1.0)

            optimizer.zero_grad(set_to_none=True)
            if not half:
                loss.backward()
            else:
                scaler.scale(loss).backward()

            losses.append(loss.item())
            face_losses.append(face_loss.item())
            mask_losses.append(mask_loss.item())
            face_acc = utils.calculate_acc(logits, face_targets, False)
            mask_acc = utils.calculate_acc(mask_preds, mask_targets, binary)
            face_accs.append(face_acc)
            mask_accs.append(mask_acc)

            avg_train_loss = sum(losses) / len(losses)
            avg_train_face_loss = sum(face_losses) / len(face_losses)
            avg_train_mask_loss = sum(mask_losses) / len(mask_losses)
            avg_train_face_acc = sum(face_accs) / len(face_accs)
            avg_train_mask_acc = sum(mask_accs) / len(mask_accs)

            # Optimize
            if not half:
                optimizer.step()
            else:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()

            info = dict(train_loss=avg_train_loss,
                        train_face_loss=avg_train_face_loss,
                        train_mask_loss=avg_train_mask_loss,
                        train_face_acc=avg_train_face_acc,
                        train_mask_acc=avg_train_mask_acc)
            loop.set_postfix(info)

        # Test
        losses = []
        face_losses = []
        mask_losses = []
        face_accs = []
        mask_accs = []

        avg_val_face_acc = 0.0
        avg_val_mask_acc = 0.0
        avg_val_loss = 0.0
        avg_val_face_loss = 0.0
        avg_val_mask_loss = 0.0

        model.eval()
        loop = tqdm(enumerate(val_loader), total=len(val_loader))

        for i, (images, face_targets, mask_targets) in loop:
            loop.set_description(f"Validation")
            images, face_targets, mask_targets = images.to(device), face_targets.to(device), mask_targets.to(device)

            if binary:
                mask_targets = mask_targets.unsqueeze(1).float()
            # forward
            with torch.no_grad():
                features, mask_preds = model(images)
            logits = arc_face(features, face_targets)
            face_loss = face_ce(logits, face_targets)
            mask_loss = mask_ce(mask_preds, mask_targets)
            loss = face_loss + (mask_loss + 1.0)

            losses.append(loss.item())
            face_losses.append(face_loss.item())
            mask_losses.append(mask_loss.item())

            face_acc = utils.calculate_acc(logits, face_targets, False)
            mask_acc = utils.calculate_acc(mask_preds, face_targets, binary)
            face_accs.append(face_acc)
            mask_accs.append(mask_acc)

            avg_val_loss = sum(losses) / len(losses)
            avg_val_face_loss = sum(face_losses) / len(face_losses)
            avg_val_mask_loss = sum(mask_losses) / len(mask_losses)
            avg_val_face_acc = sum(face_accs) / len(face_accs)
            avg_val_mask_acc = sum(mask_accs) / len(mask_accs)

            info = dict(val_loss=avg_val_loss,
                        val_face_loss=avg_val_face_loss,
                        val_mask_loss=avg_val_mask_loss,
                        val_face_acc=avg_val_face_acc,
                        val_mask_acc=avg_val_mask_acc)
            loop.set_postfix(info)

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        results = [avg_train_loss, avg_train_face_loss, avg_train_mask_loss,
                   avg_val_loss, avg_val_face_loss, avg_val_mask_loss,
                   avg_train_face_acc, avg_train_mask_acc, avg_val_face_acc, avg_val_mask_acc]
        tags = ['Train/loss/train_loss', 'Train//loss/face_loss', 'Train/loss/mask_loss',
                'Validation/loss/val_loss', 'Validation/loss/face_loss', 'Validation/loss/mask_loss',
                'Train/accuracy/face_acc', 'Train/accuracy/mask_acc',
                'Validation/accuracy/face_acc', 'Validation/accuracy/mask_acc',
                'x/lr0', 'x/lr1', 'x/lr2']

        for result, tag in zip(results + lr, tags):
            writer.add_scalar(tag, result, epoch)

        final_epoch = epoch + 1 == epochs
        # Save best network
        if best_val_loss > avg_val_loss:
            if final_epoch:
                torch.save(model.state_dict(),
                           best)
            else:
                ckpt = {'epoch': epoch + 1,
                        'best_val': best_val_loss,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        }
                torch.save(ckpt, best)

            best_val_loss = avg_val_loss

            del ckpt


if __name__ == "__main__":
    # Hyperparameters
    with open(r"hyperp.yaml") as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyperparameters and additional parameters
    f.close()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    train(device, hyp)
