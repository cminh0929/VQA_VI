import os
import json
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import albumentations as A
from albumentations.pytorch import ToTensorV2
from underthesea import word_tokenize

class SpecializedVQADataset(Dataset):
    def __init__(self, config, json_path, transform=None, tokenizer=None, max_length=128):
        self.config = config
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def preprocess_text(self, text):
        # Vietnamese word segmentation
        tokens = word_tokenize(text, format="text")
        return tokens

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load Image
        img_path = os.path.join(self.config.IMAGES_DIR, item['image_id'])
        if not os.path.exists(img_path):
            for sub in ['training-images', 'dev-images', 'test-images']:
                p = os.path.join(self.config.IMAGES_DIR, sub, item['image_id'])
                if os.path.exists(p):
                    img_path = p
                    break

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image=image)['image']
            
        # Preprocess Question
        question = self.preprocess_text(item['question'])
        
        # Tokenize if tokenizer is provided (for E2E models)
        if self.tokenizer:
            inputs = self.tokenizer(
                question,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            question_ids = inputs['input_ids'].squeeze(0)
            attention_mask = inputs['attention_mask'].squeeze(0)
        else:
            question_ids = question # Return plain text or handle later in model
            attention_mask = None
            
        # Preprocess Answer (for training)
        answer = item.get('answer', "")
        if self.tokenizer and answer:
            labels = self.tokenizer(
                answer,
                padding='max_length',
                truncation=True,
                max_length=self.config.MAX_ANSWER_LENGTH,
                return_tensors="pt"
            )['input_ids'].squeeze(0)
        else:
            labels = answer

        res = {
            'image': image,
            'question': question_ids,
            'answer': labels,
            'original_item': item
        }
        if attention_mask is not None:
            res['attention_mask'] = attention_mask
        return res

def get_transforms(config, is_train=True, normalize=True):
    transforms_list = [A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE)]
    
    if is_train:
        transforms_list.extend([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, p=0.2),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ])
    
    if normalize:
        transforms_list.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    
    transforms_list.append(ToTensorV2())
    return A.Compose(transforms_list)

def get_dataloader(config, json_path, tokenizer=None, is_train=True, batch_size=32, normalize=True):
    transform = get_transforms(config, is_train, normalize=normalize)
    dataset = SpecializedVQADataset(config, json_path, transform=transform, tokenizer=tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=0)
