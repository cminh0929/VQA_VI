import os
import json
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoTokenizer, BlipProcessor
import albumentations as A
from albumentations.pytorch import ToTensorV2
from underthesea import word_tokenize
import numpy as np

class VQADataset(Dataset):
    def __init__(self, config, json_path, direction='A', tokenizer=None, processor=None, is_train=True, limit=None):
        self.config = config
        self.direction = direction
        self.is_train = is_train
        
        # Load Data
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        if limit:
            self.data = self.data[:limit]

        # Pre-process word segmentation once to save CPU during training
        print(f"Pre-processing word segmentation for {len(self.data)} items...")
        for item in self.data:
            item['question'] = self._preprocess_text(item['question'])
            if 'answers' in item and isinstance(item['answers'], list):
                item['answers'] = [self._preprocess_text(a) for a in item['answers']]
            if 'answer' in item:
                item['answer'] = self._preprocess_text(item['answer'])
            
        # Direction A uses PhoBERT Tokenizer, Direction B uses BlipProcessor
        self.tokenizer = tokenizer
        self.processor = processor
        
        # Image Transform
        size = config.MODULAR_IMAGE_SIZE if direction == 'A' else config.MULTIMODAL_IMAGE_SIZE
        self.transform = self._get_transforms(size)

    def _get_transforms(self, size):
        if self.is_train:
            return A.Compose([
                A.Resize(size, size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(size, size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def _preprocess_text(self, text):
        if not text: return ""
        # Vietnamese word segmentation
        return word_tokenize(text.lower(), format="text")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load Image
        img_path = os.path.join(self.config.IMAGES_DIR, item['image_id'])
        # Fallback for subdirs if data is split
        if not os.path.exists(img_path):
            for sub in ['training-images', 'dev-images', 'test-images']:
                p = os.path.join(self.config.IMAGES_DIR, sub, item['image_id'])
                if os.path.exists(p):
                    img_path = p
                    break
        
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Text already preprocessed in __init__
        question = item['question']
        
        if 'answers' in item and isinstance(item['answers'], list) and len(item['answers']) > 0:
            raw_answer = item['answers'][0]
        else:
            raw_answer = item.get('answer', "")
            
        answer = raw_answer
        
        # Category Handling
        cat_to_id = {
            "object": 0, "yes_no": 1, "color": 2, 
            "location": 3, "action": 4, "appearance": 5
        }
        category = item.get('category', 'object').lower()
        category_id = cat_to_id.get(category, 0)

        if self.direction == 'A':
            # Direction A: Manual processing
            image = self.transform(image=image)['image']
            
            # Tokenize Question
            q_enc = self.tokenizer(
                question,
                padding='max_length',
                truncation=True,
                max_length=self.config.MAX_QUESTION_LENGTH,
                return_tensors="pt"
            )
            
            # Tokenize Answer (for training labels)
            a_enc = self.tokenizer(
                answer,
                padding='max_length',
                truncation=True,
                max_length=self.config.MAX_ANSWER_LENGTH,
                return_tensors="pt"
            )
            
            return {
                'image': image,
                'input_ids': q_enc['input_ids'].squeeze(0),
                'attention_mask': q_enc['attention_mask'].squeeze(0),
                'labels': a_enc['input_ids'].squeeze(0),
                'category_id': torch.tensor(category_id, dtype=torch.long),
                'question_raw': question,
                'answer_raw': answer,
                'answers_raw': item.get('answers', [raw_answer])
            }
        else:
            # Direction B: BLIP (will be processed in collate_fn to use processor)
            return {
                'image': Image.fromarray(image),
                'question': question,
                'answer': answer,
                'category': category,
                'answers_raw': item.get('answers', [raw_answer])
            }

def blip_collate_fn(batch, processor):
    images = [x['image'] for x in batch]
    # Integrate category into prompt for better performance
    questions = [f"[{x['category']}] {x['question']}" if 'category' in x else x['question'] for x in batch]
    answers = [x['answer'] for x in batch]
    
    inputs = processor(images=images, text=questions, return_tensors="pt", padding=True)
    labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True).input_ids
    inputs['labels'] = labels
    
    # Keep raw data for evaluation
    inputs['questions_raw'] = questions
    inputs['answers_raw'] = [x.get('answers_raw', [x['answer']]) for x in batch]
    
    return inputs

def modular_collate_fn(batch):
    # Standard collate except for answers_raw
    answers_raw = [x.pop('answers_raw') for x in batch]
    collated = torch.utils.data.dataloader.default_collate(batch)
    collated['answers_raw'] = answers_raw
    return collated

def get_dataloader(config, json_path, direction='A', is_train=True, batch_size=None, limit=None):
    if direction == 'A':
        tokenizer = AutoTokenizer.from_pretrained("weight")
        dataset = VQADataset(config, json_path, direction='A', tokenizer=tokenizer, is_train=is_train, limit=limit)
        return DataLoader(
            dataset, 
            batch_size=batch_size or config.BATCH_SIZE_A, 
            shuffle=is_train,
            num_workers=2,
            collate_fn=modular_collate_fn
        )
    else:
        processor = BlipProcessor.from_pretrained(config.BLIP_MODEL_ID)
        dataset = VQADataset(config, json_path, direction='B', is_train=is_train, limit=limit)
        return DataLoader(
            dataset, 
            batch_size=batch_size or config.BATCH_SIZE_B, 
            shuffle=is_train,
            num_workers=2,
            collate_fn=lambda b: blip_collate_fn(b, processor)
        )
