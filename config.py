import os

class Config:
    # --- Kaggle Detection ---
    IS_KAGGLE = os.path.exists('/kaggle/input')
    KAGGLE_DATA_PATH = "/kaggle/input/datasets/minhngcng3/dataset/data"
    
    # --- Paths ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    if IS_KAGGLE:
        DATA_DIR = KAGGLE_DATA_PATH
        CHECKPOINT_DIR = "/kaggle/working/results/checkpoints"
        LOG_DIR = "/kaggle/working/results/logs"
    else:
        DATA_DIR = os.path.join(BASE_DIR, "data")
        CHECKPOINT_DIR = os.path.join(BASE_DIR, "results", "checkpoints")
        LOG_DIR = os.path.join(BASE_DIR, "results", "logs")
    
    IMAGES_DIR = os.path.join(DATA_DIR, "images")
    TRAIN_JSON = os.path.join(DATA_DIR, "train_final.json")
    VAL_JSON = os.path.join(DATA_DIR, "val_final.json")
    TEST_JSON = os.path.join(DATA_DIR, "test_final.json")
    
    # --- Model Configs (Common) ---
    DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" else "cpu"
    MAX_QUESTION_LENGTH = 128
    MAX_ANSWER_LENGTH = 30  # Increased to match BLIP default decoder length
    
    # --- Direction A (Modular) ---
    IMAGE_ENCODER = "resnet50" # resnet50, vit_base_patch16_224
    TEXT_ENCODER = "vinai/phobert-base"
    MODULAR_IMAGE_SIZE = 224
    EMBED_SIZE = 768
    HIDDEN_SIZE = 512
    NUM_CATEGORIES = 6
    
    # --- Direction B (Multimodal) ---
    BLIP_MODEL_ID = "Salesforce/blip-vqa-base"
    MULTIMODAL_IMAGE_SIZE = 384
    
    # --- Training Configs ---
    BATCH_SIZE_A = 32
    BATCH_SIZE_B = 8
    EPOCHS = 10
    LEARNING_RATE_A = 1e-4
    LEARNING_RATE_B = 2e-5
    WEIGHT_DECAY = 0.01
    
    # --- LoRA Configs (for BLIP) ---
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["query", "key", "value", "dense"]

    @classmethod
    def ensure_dirs(cls):
        for d in [cls.CHECKPOINT_DIR, cls.LOG_DIR, cls.IMAGES_DIR]:
            os.makedirs(d, exist_ok=True)

if __name__ == "__main__":
    Config.ensure_dirs()
    print("Project directories initialized.")
