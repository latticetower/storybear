
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from PIL import Image
from diffusers import Flux2KleinPipeline

from pathlib import Path
from typing import Union
from typing import List, Tuple
# from llama_cpp import Llama

from storybear.data_structures import PlotRecord

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
dtype = torch.bfloat16

print("Local inference, using device:", device)

flux_pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-base-4B", torch_dtype=dtype).to(device)
if device != "cpu":
    flux_pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU


it2t_model_path = "openbmb/MiniCPM-V-4.6"  # or "openbmb/MiniCPM-V-4.6-Thinking"

it2t_processor = AutoProcessor.from_pretrained(it2t_model_path)
it2t_model = AutoModelForImageTextToText.from_pretrained(
    it2t_model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
).eval().to(device)


def flux_i2i_func(prompt: str, file_path: Path):
    """
    Gets the prompt with the image path as an input, returns image (or image path) processed by VLM.
    
    :param prompt: String with text prompt describing style which should be applied to the image.
    :type prompt: str
    :param file_path: Path to the image file to modify
    :type file_path: Path
    """
    #pipe = Flux2KleinPipeline.from_pretrained(
    #    "black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16
    #).to("cuda")
    #pipe.load_lora_weights(
    #    "stephenbtl/ugly-kontext-klein-4b-lora",
    #    weight_name="ugly_kontext_klein_4b_v1.safetensors",
    #)
    # from diffusers import Flux2KleinPipeline

    reference = Image.open(file_path).convert("RGB") #.resize((1024, 1024))
    print(reference.size)
    img = flux_pipe(
        prompt=prompt, 
        image=reference, 
        num_inference_steps=4, 
        guidance_scale=4.0
    ).images[0]
    save_file_path = file_path.parent / (file_path.stem + "_mod.png")

    # print(img)
    img.save(save_file_path)
    return save_file_path
    # img.save("flux_processed.png")




# # image1 = Image.open("flux-klein.png").convert("RGB")
# # image2 = Image.open("flux_processed.png").convert("RGB")

# # messages = [
# #    {"role": 'system', "content": "Summarize findings available in images and annotations. Use no more than 3 sentences"},
# #    {
# #     "role": "user",
# #     "content": [
# #         {"type": "image", "image": image1},
# #         #{"type": "image", "image": image2},
# #         {"type": "text",  "text":  "Cats can live with dogs"}, #Compare the two images, tell me about the differences between them."},
# #         ],
# #    },
# #    {"role": "user", "content": [
# #         {"type": "image", "image": image2},
# #         {"type": "text", "text": "It is known that the cats prefer to be alone"}]}
# # ]
# def get_messages(system_prompt, image_data: List[Tuple[str, Path]]):
#     messages = [{"role": "system", "content": system_prompt}]
#     for (text, image_path) in image_data:
#         img = Image.open(image_path)
#         img.thumbnail((50, 50))
#         messages.append({
#             "role": "user", "content": [
#                 {"type": "text", "text": text}, 
#                 {"type": "image", "image": img}
#             ]
#         })
#     return messages

def build_user_messages(record_list: List[Union[PlotRecord, str]]):
    user_messages = [
        {
            "role": "user", 
            "content": record.caption if isinstance(record, PlotRecord) else record
        }
        for record in record_list
    ]
    return user_messages


def it2t_summary_func(system_prompt, record_list: List[Union[PlotRecord, str]]):
    user_messages = build_user_messages(record_list)
    messages = [{"role": "system", "content": system_prompt}] + user_messages
    # messages = get_messages(prompt, image_data)
    # print(messages)
    inputs = it2t_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(it2t_model.device)

    out_ids = it2t_model.generate(**inputs, max_new_tokens=512)
    answer = it2t_processor.decode(
        out_ids[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
    # print(answer)
    return answer


# T2T_MODEL_ID = "openbmb/MiniCPM5-1B-GGUF"
# T2T_MODEL_NAME = "MiniCPM5-1B-Q4_K_M.gguf"

# T2T_MODEL_ID = "openbmb/MiniCPM-V-4.6-gguf"
# T2T_MODEL_NAME = "MiniCPM-V-4_6-F16.gguf"

# llm = Llama.from_pretrained(
#   repo_id=T2T_MODEL_ID,
#   filename=T2T_MODEL_NAME,
#   # chat_format="chatml"
# )

# def t2t_summary_func(system_prompt, user_messages):
#     messages = [{"role": "system", "content": system_prompt}] + user_messages
#     res = llm.create_chat_completion(
#         messages=messages,
#         # response_format=response_format,
#         temperature=0.7,
#     )
#     # print(repr(res))
#     return res
