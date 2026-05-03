import os

class Config:
    # Check if running on Kaggle
    IS_KAGGLE = os.path.exists('/kaggle/working')
    
    if IS_KAGGLE:
        # Kaggle paths
        BASE_DIR = "/kaggle/working"
        # Assuming you uploaded your data as a Kaggle dataset named 'openvivqa-vietnamese-vqa'
        DATA_DIR = "/kaggle/input/openvivqa-vietnamese-vqa"
        CHECKPOINT_DIR = "/kaggle/working/checkpoints"
    else:
        # Local paths
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, "data")
        CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

    IMAGES_DIR = os.path.join(DATA_DIR, "images")
    TRAIN_JSON = os.path.join(DATA_DIR, "train.json")
    VAL_JSON = os.path.join(DATA_DIR, "val.json")
    TEST_JSON = os.path.join(DATA_DIR, "test.json")
    
    # Model configs
    MODEL_ID_B = "google/paligemma-3b-pt-224"
    IMAGE_SIZE = 224
    MAX_ANSWER_LENGTH = 20
    EMBED_SIZE = 768
    
    # Training configs
    BATCH_SIZE_A = 32
    BATCH_SIZE_B = 4
    EPOCHS_A = 10
    EPOCHS_B = 3
    LEARNING_RATE_A = 1e-4
    LEARNING_RATE_B = 2e-5
    
    # LoRA configs
    LORA_R = 8
    LORA_ALPHA = 32
