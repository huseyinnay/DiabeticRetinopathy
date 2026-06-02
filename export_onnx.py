import os
import torch
import timm
from pathlib import Path
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    dr_weights: Path = Path("runs/binary_efficientnet_b0/best.pt")
    dr_model_name: str = "efficientnet_b0"
    dr_image_size: int = 512
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pt_weights = settings.dr_weights.with_suffix(".pt")
    
    logger.info(f"Yükleniyor: {pt_weights}")
    if not pt_weights.exists():
        logger.error(f"Ağırlık dosyası bulunamadı: {pt_weights}")
        return
    
    print(f"Loading weights from {pt_weights}...")
    model = timm.create_model(settings.dr_model_name, pretrained=False, num_classes=1)
    
    state = torch.load(pt_weights, map_location=device, weights_only=True)
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]
    
    # Strip potential 'model.' prefix
    state = {k.replace("model.", ""): v for k, v in state.items()}
    
    model.load_state_dict(state, strict=True)
    model.eval()

    # Dummy tensor
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, settings.dr_image_size, settings.dr_image_size, requires_grad=True)

    # Export
    out_path = settings.dr_weights.with_suffix(".onnx")
    logger.info(f"ONNX modeline dönüştürülüyor... Çıktı: {out_path}")

    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    logger.info("ONNX export tamamlandı!")

if __name__ == "__main__":
    main()
