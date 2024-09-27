import os
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
import requests
from transformers import AutoProcessor, VipLlavaForConditionalGeneration


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


model = VipLlavaForConditionalGeneration.from_pretrained(
    "llava-hf/vip-llava-7b-hf", torch_dtype=torch.float16
).to(DEVICE)
processor = AutoProcessor.from_pretrained("llava-hf/vip-llava-7b-hf")

prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives number as output like 0,1,2,... ###Human: <image>\n{}###Assistant:"

filtered = []
for pic in tqdm(list(os.walk("../Datasets/COCO/train2014/train2014/"))[0][2]):
    question = "How many people are in the image? answer with number"
    prompt = prompt.format(question)
    image = Image.open("../Datasets/COCO/train2014/train2014/" + pic)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(
        "cuda", torch.float16
    )
    generate_ids = model.generate(**inputs, max_new_tokens=20)
    answer = processor.decode(
        generate_ids[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
    )
    if answer.strip() not in ["0", "zero"]:
        filtered.append(pic)

pd.DataFrame({"path": filtered}).to_csv("filtered.csv", index=False)
