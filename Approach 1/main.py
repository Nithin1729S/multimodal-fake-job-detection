import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

class EarlyStopping:
    def __init__(self, patience=7, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_f1 = 0

    def __call__(self, val_loss, f1_score):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_f1 = f1_score
        elif val_loss > self.best_loss - self.delta and f1_score <= self.best_f1:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_f1 = f1_score
            self.counter = 0
        return self.early_stop

df = pd.read_csv("/kaggle/input/multimodal-real-fake-job-posting-prediction/fake_job_postings.csv")
base_path = "/kaggle/input/multimodal-real-fake-job-posting-prediction/images"
df['webpage_screenshot'] = df.apply(lambda x: f"{base_path}/{x['fraudulent']}/{x['job_id']}.png", axis=1)

text_features = ['title', 'location', 'department', 'company_profile', 'description', 'requirements', 'benefits']
df['text'] = df[text_features].fillna("").agg(" ".join, axis=1)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class_weights = compute_class_weight('balanced', classes=np.unique(df['fraudulent']), y=df['fraudulent'])
class_weight_tensor = torch.FloatTensor(class_weights)

tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
bow_vectorizer = CountVectorizer(max_features=500, binary=True)

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['fraudulent'], random_state=42)

tfidf_features_train = tfidf_vectorizer.fit_transform(train_df['text']).toarray()
tfidf_features_test = tfidf_vectorizer.transform(test_df['text']).toarray()

bow_features_train = bow_vectorizer.fit_transform(train_df['text']).toarray()
bow_features_test = bow_vectorizer.transform(test_df['text']).toarray()

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

class JobDataset(Dataset):
    def __init__(self, df, tokenizer, tfidf_features, bow_features, max_len=256, augment=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tfidf_features = tfidf_features
        self.bow_features = bow_features
        self.augment = augment
        
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(row['text'], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        img_path = row['webpage_screenshot']
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        img = self.transform(img)
        
        tfidf_feat = torch.FloatTensor(self.tfidf_features[idx])
        bow_feat = torch.FloatTensor(self.bow_features[idx])
        
        return (
            enc["input_ids"].squeeze(0), 
            enc["attention_mask"].squeeze(0), 
            img, 
            tfidf_feat,
            bow_feat,
            torch.tensor(row['fraudulent'], dtype=torch.long)
        )

train_dataset = JobDataset(train_df, tokenizer, tfidf_features_train, bow_features_train, augment=True)
test_dataset = JobDataset(test_df, tokenizer, tfidf_features_test, bow_features_test, augment=False)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

bert = BertModel.from_pretrained("bert-base-uncased")
resnet = models.resnet50(weights="IMAGENET1K_V1")
resnet.fc = nn.Identity()

for param in bert.parameters():
    param.requires_grad = False
for name, param in resnet.named_parameters():
    if 'layer4' in name or 'layer3' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

class MultiModalClassifier(nn.Module):
    def __init__(self, bert, resnet, tfidf_dim=1000, bow_dim=500, dropout_rate=0.4):
        super().__init__()
        self.bert = bert
        self.resnet = resnet
        
        self.tfidf_fc = nn.Sequential(
            nn.Linear(tfidf_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate)
        )
        
        self.bow_fc = nn.Sequential(
            nn.Linear(bow_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate)
        )
        
        self.bert_fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate)
        )
        
        self.img_fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate)
        )
        
        self.fusion_fc = nn.Sequential(
            nn.Linear(896, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2)
        )
        
        self.tfidf_classifier = nn.Linear(256, 2)
        self.bow_classifier = nn.Linear(128, 2)
        self.bert_classifier = nn.Linear(256, 2)
        self.img_classifier = nn.Linear(256, 2)
        
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.delta = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, mask, images, tfidf_feats, bow_feats):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=mask).pooler_output
        
        img_feat = self.resnet(images)
        
        tfidf_processed = self.tfidf_fc(tfidf_feats)
        bow_processed = self.bow_fc(bow_feats)
        bert_processed = self.bert_fc(bert_output)
        img_processed = self.img_fc(img_feat)
        
        tfidf_logits = self.tfidf_classifier(tfidf_processed)
        bow_logits = self.bow_classifier(bow_processed)
        bert_logits = self.bert_classifier(bert_processed)
        img_logits = self.img_classifier(img_processed)
        
        combined_feat = torch.cat([tfidf_processed, bow_processed, bert_processed, img_processed], dim=1)
        fusion_logits = self.fusion_fc(combined_feat)
        
        return fusion_logits, tfidf_logits, bow_logits, bert_logits, img_logits

class AdaptiveMultiTaskLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
    def forward(self, model, fusion_logits, tfidf_logits, bow_logits, bert_logits, img_logits, labels):
        fusion_loss = self.criterion(fusion_logits, labels)
        tfidf_loss = self.criterion(tfidf_logits, labels)
        bow_loss = self.criterion(bow_logits, labels)
        bert_loss = self.criterion(bert_logits, labels)
        img_loss = self.criterion(img_logits, labels)
        
        alpha_norm = torch.exp(model.alpha) / (torch.exp(model.alpha) + torch.exp(model.beta) + torch.exp(model.gamma) + torch.exp(model.delta))
        beta_norm = torch.exp(model.beta) / (torch.exp(model.alpha) + torch.exp(model.beta) + torch.exp(model.gamma) + torch.exp(model.delta))
        gamma_norm = torch.exp(model.gamma) / (torch.exp(model.alpha) + torch.exp(model.beta) + torch.exp(model.gamma) + torch.exp(model.delta))
        delta_norm = torch.exp(model.delta) / (torch.exp(model.alpha) + torch.exp(model.beta) + torch.exp(model.gamma) + torch.exp(model.delta))
        
        total_loss = (fusion_loss + 
                     alpha_norm * tfidf_loss + 
                     beta_norm * bow_loss + 
                     gamma_norm * bert_loss + 
                     delta_norm * img_loss)
        
        return total_loss, fusion_loss, tfidf_loss, bow_loss, bert_loss, img_loss, alpha_norm, beta_norm, gamma_norm, delta_norm

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MultiModalClassifier(bert, resnet).to(device)

optimizer = torch.optim.AdamW([
    {'params': model.tfidf_fc.parameters(), 'lr': 3e-4},
    {'params': model.bow_fc.parameters(), 'lr': 3e-4},
    {'params': model.bert_fc.parameters(), 'lr': 1e-4},
    {'params': model.img_fc.parameters(), 'lr': 1e-4},
    {'params': model.resnet.parameters(), 'lr': 1e-5},
    {'params': model.fusion_fc.parameters(), 'lr': 2e-4},
    {'params': model.tfidf_classifier.parameters(), 'lr': 3e-4},
    {'params': model.bow_classifier.parameters(), 'lr': 3e-4},
    {'params': model.bert_classifier.parameters(), 'lr': 2e-4},
    {'params': model.img_classifier.parameters(), 'lr': 2e-4},
    {'params': [model.alpha, model.beta, model.gamma, model.delta], 'lr': 1e-3}
], weight_decay=0.01)

criterion = AdaptiveMultiTaskLoss(class_weights=class_weight_tensor.to(device))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
early_stopping = EarlyStopping(patience=7)

best_f1 = 0
max_epochs = 50

for epoch in range(max_epochs):
    model.train()
    train_loss = 0
    
    for batch_idx, (ids, mask, imgs, tfidf_feats, bow_feats, labels) in enumerate(train_loader):
        ids = ids.to(device)
        mask = mask.to(device)
        imgs = imgs.to(device)
        tfidf_feats = tfidf_feats.to(device)
        bow_feats = bow_feats.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        fusion_logits, tfidf_logits, bow_logits, bert_logits, img_logits = model(ids, mask, imgs, tfidf_feats, bow_feats)
        loss, f_loss, t_loss, b_loss, bert_loss, i_loss, a, b, g, d = criterion(model, fusion_logits, tfidf_logits, bow_logits, bert_logits, img_logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    y_true, y_pred, y_prob = [], [], []
    val_loss = 0
    
    with torch.no_grad():
        for ids, mask, imgs, tfidf_feats, bow_feats, labels in test_loader:
            ids = ids.to(device)
            mask = mask.to(device)
            imgs = imgs.to(device)
            tfidf_feats = tfidf_feats.to(device)
            bow_feats = bow_feats.to(device)
            
            fusion_logits, tfidf_logits, bow_logits, bert_logits, img_logits = model(ids, mask, imgs, tfidf_feats, bow_feats)
            loss, _, _, _, _, _, a, b, g, d = criterion(model, fusion_logits, tfidf_logits, bow_logits, bert_logits, img_logits, labels.to(device))
            val_loss += loss.item()
            
            probs = torch.softmax(fusion_logits, dim=1)
            preds = torch.argmax(fusion_logits, 1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_prob.extend(probs[:, 1].cpu().numpy())

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(test_loader)
    
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    scheduler.step(avg_val_loss)
    
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), 'best_model.pth')

    print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")
    print(f"  Weights - α(TFIDF): {a.item():.3f}, β(BoW): {b.item():.3f}, γ(BERT): {g.item():.3f}, δ(Image): {d.item():.3f}")

    if early_stopping(avg_val_loss, f1):
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

model.load_state_dict(torch.load('best_model.pth'))
model.eval()

y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for ids, mask, imgs, tfidf_feats, bow_feats, labels in test_loader:
        ids = ids.to(device)
        mask = mask.to(device)
        imgs = imgs.to(device)
        tfidf_feats = tfidf_feats.to(device)
        bow_feats = bow_feats.to(device)
        
        fusion_logits, tfidf_logits, bow_logits, bert_logits, img_logits = model(ids, mask, imgs, tfidf_feats, bow_feats)
        probs = torch.softmax(fusion_logits, dim=1)
        preds = torch.argmax(fusion_logits, 1).cpu().numpy()
        y_true.extend(labels.numpy())
        y_pred.extend(preds)
        y_prob.extend(probs[:, 1].cpu().numpy())

print("\nFinal Results:")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall: {recall_score(y_true, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)
 
 