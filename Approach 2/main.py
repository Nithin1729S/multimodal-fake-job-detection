import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import cv2
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')


class ImageFeatureExtractor:
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=100)
        self.orb = cv2.ORB_create(nfeatures=100)

    def extract_edge_features(self, img_array):
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        edge_angle = np.arctan2(sobely, sobelx)
        features = [
            edge_density,
            np.mean(edge_magnitude),
            np.std(edge_magnitude),
            np.mean(edge_angle),
            np.std(edge_angle),
            np.sum(np.abs(sobelx) > 50) / sobelx.size,
            np.sum(np.abs(sobely) > 50) / sobely.size
        ]
        return features

    def extract_corner_features(self, img_array):
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        gray32 = np.float32(gray)
        harris = cv2.cornerHarris(gray32, 2, 3, 0.04)
        harris_corners = np.sum(harris > 0.01 * harris.max())
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        shi_count = len(corners) if corners is not None else 0
        if corners is not None:
            coords = corners.reshape(-1, 2)
            std_x = np.std(coords[:, 0])
            std_y = np.std(coords[:, 1])
        else:
            std_x = 0
            std_y = 0
        features = [
            harris_corners / (gray.shape[0] * gray.shape[1]),
            shi_count / 100,
            std_x / gray.shape[1],
            std_y / gray.shape[0]
        ]
        return features

    def extract_blob_features(self, img_array):
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 50
        params.maxArea = 5000
        params.filterByInertia = True
        params.minInertiaRatio = 0.01
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)
        count = len(keypoints)
        if count > 0:
            sizes = [kp.size for kp in keypoints]
            avg_size = np.mean(sizes)
            std_size = np.std(sizes)
        else:
            avg_size = 0
            std_size = 0
        features = [
            count / 100,
            avg_size / 100,
            std_size / 100
        ]
        return features

    def extract_sift_features(self, img_array):
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        try:
            k, d = self.sift.detectAndCompute(gray, None)
            if d is not None:
                count = len(k)
                desc_mean = np.mean(d)
                desc_std = np.std(d)
                desc_max = np.max(d)
                resp = [x.response for x in k]
                avg_resp = np.mean(resp) if resp else 0
                sizes = [x.size for x in k]
                avg_size = np.mean(sizes) if sizes else 0
                std_size = np.std(sizes) if sizes else 0
            else:
                count = 0
                desc_mean = 0
                desc_std = 0
                desc_max = 0
                avg_resp = 0
                avg_size = 0
                std_size = 0
        except:
            count = 0
            desc_mean = 0
            desc_std = 0
            desc_max = 0
            avg_resp = 0
            avg_size = 0
            std_size = 0
        features = [
            count / 100,
            desc_mean / 255,
            desc_std / 255,
            desc_max / 255,
            avg_resp,
            avg_size / 100,
            std_size / 100
        ]
        return features

    def extract_orb_features(self, img_array):
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        try:
            k, d = self.orb.detectAndCompute(gray, None)
            if d is not None:
                count = len(k)
                desc_mean = np.mean(d)
                desc_std = np.std(d)
            else:
                count = 0
                desc_mean = 0
                desc_std = 0
        except:
            count = 0
            desc_mean = 0
            desc_std = 0
        features = [
            count / 100,
            desc_mean / 255,
            desc_std / 255
        ]
        return features

    def extract_texture_features(self, img_array):
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        k = 5
        local_std = cv2.blur(gray**2, (k, k)) - cv2.blur(gray, (k, k))**2
        local_std = np.sqrt(np.maximum(local_std, 0))
        mean = np.mean(local_std)
        std = np.std(local_std)
        sk = skew(local_std.flatten())
        kurt = kurtosis(local_std.flatten())
        features = [
            mean / 255,
            std / 255,
            sk,
            kurt / 10
        ]
        return features

    def extract_color_layout_features(self, img_array):
        color_mean = np.mean(img_array, axis=(0, 1)) / 255
        color_std = np.std(img_array, axis=(0, 1)) / 255
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
        hist = hist / hist.sum()
        reduced = (img_array // 32) * 32
        uniq = len(np.unique(reduced.reshape(-1, 3), axis=0))
        div = uniq / 1000
        features = list(color_mean) + list(color_std) + list(hist) + [div]
        return features

    def extract_all_features(self, path):
        try:
            img = cv2.imread(path)
            if img is None:
                img = np.ones((224, 224, 3), dtype=np.uint8) * 128
            else:
                img = cv2.resize(img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        f = []
        f.extend(self.extract_edge_features(img))
        f.extend(self.extract_corner_features(img))
        f.extend(self.extract_blob_features(img))
        f.extend(self.extract_sift_features(img))
        f.extend(self.extract_orb_features(img))
        f.extend(self.extract_texture_features(img))
        f.extend(self.extract_color_layout_features(img))
        return np.array(f, dtype=np.float32)


class EarlyStopping:
    def __init__(self, patience=7, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_f1 = 0

    def __call__(self, val_loss, f1):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_f1 = f1
        elif val_loss > self.best_loss - self.delta and f1 <= self.best_f1:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_f1 = f1
            self.counter = 0
        return self.early_stop


df = pd.read_csv("/kaggle/input/multimodal-real-fake-job-posting-prediction/fake_job_postings.csv")
df = df.iloc[: int(0.1 * len(df))]
base = "/kaggle/input/multimodal-real-fake-job-posting-prediction/images"
df["webpage_screenshot"] = df.apply(lambda x: f"{base}/{x['fraudulent']}/{x['job_id']}.png", axis=1)

txt_cols = ['title', 'location', 'department', 'company_profile', 'description', 'requirements', 'benefits']
df["text"] = df[txt_cols].fillna("").agg(" ".join, axis=1)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
class_weights = compute_class_weight('balanced', classes=np.unique(df["fraudulent"]), y=df["fraudulent"])
class_weight_tensor = torch.FloatTensor(class_weights)

tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
bow = CountVectorizer(max_features=500, binary=True)

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["fraudulent"], random_state=42)

tfidf_train = tfidf.fit_transform(train_df["text"]).toarray()
tfidf_test = tfidf.transform(test_df["text"]).toarray()

bow_train = bow.fit_transform(train_df["text"]).toarray()
bow_test = bow.transform(test_df["text"]).toarray()

feature_extractor = ImageFeatureExtractor()

print("Extracting computer vision features from images")

train_paths = train_df["webpage_screenshot"].tolist()
test_paths = test_df["webpage_screenshot"].tolist()

cv_train = []
for i, path in enumerate(train_paths):
    if i % 100 == 0:
        print(f"Processing train image {i}/{len(train_paths)}")
    features = feature_extractor.extract_all_features(path)
    cv_train.append(features)
cv_train = np.array(cv_train, dtype=np.float32)

cv_test = []
for i, path in enumerate(test_paths):
    if i % 100 == 0:
        print(f"Processing test image {i}/{len(test_paths)}")
    features = feature_extractor.extract_all_features(path)
    cv_test.append(features)
cv_test = np.array(cv_test, dtype=np.float32)

cv_train = np.nan_to_num(cv_train, nan=0.0, posinf=0.0, neginf=0.0)
cv_test = np.nan_to_num(cv_test, nan=0.0, posinf=0.0, neginf=0.0)

cv_train = np.clip(cv_train, -10, 10)
cv_test = np.clip(cv_test, -10, 10)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
cv_train = scaler.fit_transform(cv_train)
cv_test = scaler.transform(cv_test)

print(cv_train.shape)
print(cv_test.shape)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


class JobDataset(Dataset):
    def __init__(self, df, tokenizer, tfidf, bow, cv, max_len=256, augment=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tfidf = tfidf
        self.bow = bow
        self.cv = cv
        self.augment = augment

        if augment:
            self.tfm = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.tfm = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        path = row["webpage_screenshot"]
        try:
            img = Image.open(path).convert("RGB")
        except:
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        img = self.tfm(img)

        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            img,
            torch.FloatTensor(self.cv[idx]),
            torch.FloatTensor(self.tfidf[idx]),
            torch.FloatTensor(self.bow[idx]),
            torch.tensor(row["fraudulent"], dtype=torch.long)
        )


train_dataset = JobDataset(train_df, tokenizer, tfidf_train, bow_train, cv_train, augment=True)
test_dataset = JobDataset(test_df, tokenizer, tfidf_test, bow_test, cv_test, augment=False)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


class MultiKernelBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid = out_ch // 3
        self.branch3 = nn.Conv2d(in_ch, mid, kernel_size=3, padding=1, bias=False)
        self.branch5 = nn.Conv2d(in_ch, mid, kernel_size=5, padding=2, bias=False)
        self.branch7 = nn.Conv2d(in_ch, out_ch - 2 * mid, kernel_size=7, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        out = torch.cat([b3, b5, b7], dim=1)
        out = self.bn(out)
        sc = self.shortcut(x)
        out = self.relu(out + sc)
        return out


class MultiKernelResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.layer1 = nn.Sequential(
            MultiKernelBlock(64, 128),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            MultiKernelBlock(128, 256),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            MultiKernelBlock(256, 512),
            nn.MaxPool2d(2)
        )
        self.layer4 = nn.Sequential(
            MultiKernelBlock(512, 512)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 512

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x


bert = BertModel.from_pretrained("bert-base-uncased")
for p in bert.parameters():
    p.requires_grad = False

resnet = MultiKernelResNet()


class MultiModalClassifier(nn.Module):
    def __init__(self, bert, resnet, cv_dim, tfidf_dim=1000, bow_dim=500, img_feat_dim=512, drop=0.3):
        super().__init__()
        self.bert = bert
        self.resnet = resnet

        self.cv_fc = nn.Sequential(
            nn.Linear(cv_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(drop)
        )
        self.tfidf_fc = nn.Sequential(
            nn.Linear(tfidf_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(drop)
        )
        self.bow_fc = nn.Sequential(
            nn.Linear(bow_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(drop)
        )
        self.bert_fc = nn.Sequential(
            nn.Linear(768, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(drop)
        )
        self.img_fc = nn.Sequential(
            nn.Linear(img_feat_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(drop)
        )

        self.fusion_fc = nn.Sequential(
            nn.Linear(640, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(128, 2)
        )

    def forward(self, ids, mask, imgs, cv, tfidf, bow):
        with torch.no_grad():
            bert_out = self.bert(input_ids=ids, attention_mask=mask).pooler_output

        img_feat = self.resnet(imgs)

        cv_p = self.cv_fc(cv)
        tfidf_p = self.tfidf_fc(tfidf)
        bow_p = self.bow_fc(bow)
        bert_p = self.bert_fc(bert_out)
        img_p = self.img_fc(img_feat)

        combined = torch.cat([cv_p, img_p, tfidf_p, bow_p, bert_p], dim=1)
        fusion_l = self.fusion_fc(combined)

        return fusion_l


device = "cuda" if torch.cuda.is_available() else "cpu"

model = MultiModalClassifier(
    bert,
    resnet,
    cv_dim=cv_train.shape[1],
    tfidf_dim=tfidf_train.shape[1],
    bow_dim=bow_train.shape[1],
    img_feat_dim=resnet.out_dim
).to(device)

opt = torch.optim.AdamW([
    {'params': model.cv_fc.parameters(), 'lr': 1e-4},
    {'params': model.tfidf_fc.parameters(), 'lr': 1e-4},
    {'params': model.bow_fc.parameters(), 'lr': 1e-4},
    {'params': model.bert_fc.parameters(), 'lr': 5e-5},
    {'params': model.img_fc.parameters(), 'lr': 5e-5},
    {'params': model.resnet.parameters(), 'lr': 1e-5},
    {'params': model.fusion_fc.parameters(), 'lr': 1e-4}
], weight_decay=0.01)

crit = nn.CrossEntropyLoss(weight=class_weight_tensor.to(device))
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
early = EarlyStopping(patience=7)

best_f1 = 0
max_epochs = 50

for epoch in range(max_epochs):
    model.train()
    train_loss = 0

    for ids, mask, imgs, cvf, tfi, bowf, labels in train_loader:
        ids = ids.to(device)
        mask = mask.to(device)
        imgs = imgs.to(device)
        cvf = cvf.to(device)
        tfi = tfi.to(device)
        bowf = bowf.to(device)
        labels = labels.to(device)

        opt.zero_grad()
        outputs = model(ids, mask, imgs, cvf, tfi, bowf)
        loss = crit(outputs, labels)

        if torch.isnan(loss):
            print("NaN detected! Skipping batch...")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        train_loss += loss.item()

    model.eval()
    y_true, y_pred, y_prob = [], [], []
    val_loss = 0

    with torch.no_grad():
        for ids, mask, imgs, cvf, tfi, bowf, labels in test_loader:
            ids = ids.to(device)
            mask = mask.to(device)
            imgs = imgs.to(device)
            cvf = cvf.to(device)
            tfi = tfi.to(device)
            bowf = bowf.to(device)
            labels = labels.to(device)

            outputs = model(ids, mask, imgs, cvf, tfi, bowf)
            loss = crit(outputs, labels)
            val_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs[:, 1].cpu().numpy())

    avg_t = train_loss / len(train_loader)
    avg_v = val_loss / len(test_loader)

    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    sched.step(avg_v)

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  *** New best F1: {best_f1:.4f} ***")

    print(f"Epoch {epoch+1}: Train Loss {avg_t:.4f}, Val Loss {avg_v:.4f}, Acc {acc:.4f}, F1 {f1:.4f}")

    if early(avg_v, f1):
        print(f"Early stopping after {epoch+1}")
        break

import os
if os.path.exists("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    print("Loaded best model from training")
else:
    print("No saved model found, using final model state")

model.eval()

y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for ids, mask, imgs, cvf, tfi, bowf, labels in test_loader:
        ids = ids.to(device)
        mask = mask.to(device)
        imgs = imgs.to(device)
        cvf = cvf.to(device)
        tfi = tfi.to(device)
        bowf = bowf.to(device)
        labels = labels.to(device)

        outputs = model(ids, mask, imgs, cvf, tfi, bowf)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs[:, 1].cpu().numpy())

print("\nFinal Results")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_true, y_pred, zero_division=0):.4f}")
print(f"F1-Score: {f1_score(y_true, y_pred, zero_division=0):.4f}")
if len(np.unique(y_true)) > 1:
    print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}")
print("\n" + classification_report(y_true, y_pred, zero_division=0))
print(confusion_matrix(y_true, y_pred))
 