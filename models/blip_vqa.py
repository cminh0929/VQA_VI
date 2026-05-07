from transformers import BlipForConditionalGeneration, BlipProcessor
from peft import LoraConfig, get_peft_model
import torch

class BlipVQAModel:
    @staticmethod
    def get_model(config, is_train=True):
        device = config.DEVICE
        model = BlipForConditionalGeneration.from_pretrained(
            config.BLIP_MODEL_ID,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        if is_train:
            lora_config = LoraConfig(
                r=config.LORA_R,
                lora_alpha=config.LORA_ALPHA,
                target_modules=config.LORA_TARGET_MODULES,
                lora_dropout=config.LORA_DROPOUT,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
        return model.to(device)

    @staticmethod
    def get_processor(config):
        return BlipProcessor.from_pretrained(config.BLIP_MODEL_ID)

    @staticmethod
    def prepare_prompt(question, category=None):
        """
        Prepare a prompt for BLIP.
        Format: [Category] Question
        """
        if category:
            return f"[{category}] {question}"
        return question
