import warnings

warnings.filterwarnings("ignore")
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
import time


class IndoorObjectDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        start_time = time.time()
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.classes = self.annotations['class'].unique()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        print(f"Dataset initialization took {time.time() - start_time:.2f} seconds")
        print(f"Number of images: {len(self.annotations['filename'].unique())}")

    def __len__(self):
        return len(self.annotations['filename'].unique())

    def __getitem__(self, idx):
        start_time = time.time()

        img_name = self.annotations['filename'].unique()[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image_load_time = time.time()
        img = Image.open(img_path).convert("RGB")
        print(f"Image loading took {time.time() - image_load_time:.4f} seconds")

        annotation_time = time.time()
        img_annotations = self.annotations[self.annotations['filename'] == img_name]

        boxes = []
        labels = []
        for _, row in img_annotations.iterrows():
            boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            labels.append(self.class_to_idx[row['class']])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        print(f"Annotation processing took {time.time() - annotation_time:.4f} seconds")

        transform_time = time.time()
        img = self.transform(img)
        print(f"Image transformation took {time.time() - transform_time:.4f} seconds")

        print(f"Total time for item {idx}: {time.time() - start_time:.4f} seconds")
        return img, target


def get_model(num_classes):
    start_time = time.time()
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    print(f"Model initialization took {time.time() - start_time:.2f} seconds")
    return model


def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    start_time = time.time()
    for i, (images, targets) in enumerate(data_loader):
        batch_start_time = time.time()
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        forward_time = time.time()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
        print(f"Forward pass took {time.time() - forward_time:.4f} seconds")

        backward_time = time.time()
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        print(f"Backward pass took {time.time() - backward_time:.4f} seconds")

        print(f"Batch {i}, Loss: {losses.item():.4f}, Time: {time.time() - batch_start_time:.4f} seconds")

    return total_loss / len(data_loader)


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    start_time = time.time()
    train_dataset = IndoorObjectDataset(csv_file='train/_annotations.csv', img_dir='train')
    valid_dataset = IndoorObjectDataset(csv_file='valid/_annotations.csv', img_dir='valid')
    print(f"Dataset loading took {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0,
                              collate_fn=lambda x: tuple(zip(*x)))
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0,
                              collate_fn=lambda x: tuple(zip(*x)))
    print(f"DataLoader initialization took {time.time() - start_time:.2f} seconds")

    num_classes = len(train_dataset.classes)
    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = 10
    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        avg_loss = train_one_epoch(model, optimizer, train_loader, device)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f} seconds")

    torch.save(model.state_dict(), 'faster_rcnn_indoor_object_detection.pth')
    print("Model saved successfully")


if __name__ == "__main__":
    main()
