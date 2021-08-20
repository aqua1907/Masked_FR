import torch
from nets.iresnet import iresnet34
from loss.arc_face import ArcFaceLoss
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import classification_report
from datasets.infer_mask_dataset import InferMask
from datasets.cfp_dataset import CFPDataset
import utils
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'
half = True if device == 'cuda' else False

dataset = CFPDataset(r'data/CASIA-WebFace')

# data_ids = np.random.permutation(len(dataset))
# data_ids = data_ids[:50000]
data_ids = list(range(50000))
subset = Subset(dataset, data_ids)


loader = DataLoader(subset, 1, pin_memory=True, shuffle=False)


model = iresnet34()
ckpt = torch.load(r'weights/iresnet34_arc_0791.pt')
model.load_state_dict(ckpt)
model.eval().to(device)

arc_face = ArcFaceLoss(512, dataset.num_classes, device)
print(dataset.num_classes)
arc_face.to(device)
# if half:
#     model.half()

# dataset = InferMask(r'data\unmasked', r'data\masked')

target_names = [str(i) for i in range(10575)]
preds = []
targets = []
accs = []
loop = tqdm(loader, total=len(loader))
for batch in loop:
    imgs, labels = batch
    imgs, labels = imgs.to(device), labels.to(device)
    # if half:
    #     imgs, labels = imgs.half(), labels.half()

    with torch.no_grad():
        face_preds, _ = model(imgs)
        print(face_preds[0][:10])
        face_preds = arc_face(face_preds, labels)

    face_acc = utils.calculate_acc(face_preds, labels)
    accs.append(face_acc)

    _, out = torch.max(face_preds.data, 1)
    out = out.view(-1)
    out = out.detach().cpu().numpy()

    labels = labels.cpu().numpy()

    preds.extend(out.tolist())
    targets.extend(labels.tolist())

    info = dict(accuracy=sum(accs) / len(accs))
    loop.set_postfix(info)


result = classification_report(targets, preds, target_names=target_names)
print(result)

