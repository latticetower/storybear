
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline

device = "mps"
dtype = torch.bfloat16
flux_pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-base-4B", torch_dtype=dtype)
flux_pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU


def flux_i2i_func(prompt, file_path):
    #pipe = Flux2KleinPipeline.from_pretrained(
    #    "black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16
    #).to("cuda")
    #pipe.load_lora_weights(
    #    "stephenbtl/ugly-kontext-klein-4b-lora",
    #    weight_name="ugly_kontext_klein_4b_v1.safetensors",
    #)

    reference = Image.open(file_path).convert("RGB") #.resize((1024, 1024))
    img = flux_pipe(
        prompt=prompt, 
        image=reference, 
        num_inference_steps=4, 
        guidance_scale=4.0
    ).images[0]

    # print(img)
    img.save(file_path)
    return img
    # img.save("flux_processed.png")