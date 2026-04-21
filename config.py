import os
from dataclasses import dataclass

@dataclass
class Config:
    # Execution Mode
    MODE = 'TRAIN'  # 'TRAIN', 'EVAL', 'DEMO'
    
    # Dataset Paths (Provided by User)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    IMAGES_DIR = os.path.join(DATA_DIR, "images")
    TRAIN_JSON = os.path.join(DATA_DIR, "train.json")
    VAL_JSON = os.path.join(DATA_DIR, "val.json")
    TEST_JSON = os.path.join(DATA_DIR, "test.json")
    
    # Results & Checkpoints
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
    
    # Image Parameters
    IMAGE_SIZE = 224
    
    # Direction A (Modular) Parameters
    HIDDEN_SIZE = 512
    EMBED_SIZE = 768  # PhoBERT embedding size
    NUM_LAYERS = 2
    DROPOUT = 0.1
    LEARNING_RATE_A = 1e-4
    BATCH_SIZE_A = 32
    EPOCHS_A = 20
    
    # Direction B (Pretrained) Parameters - PaliGemma
    MODEL_ID_B = "google/paligemma-3b-pt-224"
    LEARNING_RATE_B = 2e-5
    BATCH_SIZE_B = 8
    EPOCHS_B = 5
    LORA_R = 16
    LORA_ALPHA = 32
    
    # RL / DPO Parameters
    DPO_LEARNING_RATE = 1e-6
    DPO_BATCH_SIZE = 4
    DPO_EPOCHS = 3
    PREFERENCE_DATA_JSON = os.path.join(DATA_DIR, "preference_data.json")
    
    # Metrics
    MAX_ANSWER_LENGTH = 10
    BEAM_SIZE = 3

# Create directories if they don't exist
for d in [Config.RESULTS_DIR, Config.CHECKPOINT_DIR]:
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
