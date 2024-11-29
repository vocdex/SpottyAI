"""Based on pre-defined text prompts, this script classifies images into different categories."""


import torch
import os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
# Load CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Dynamic text prompts
text_prompts = [
    "This is a kitchen.",
    "This is an office.",
    "This is a hallway.",
    "This is a bedroom.",
]
def convert_rgb_to_grayscale(image):
    return image.convert("L")


def classify_images(images):
    # Process text descriptions
    text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True)
    text_features = model.get_text_features(**text_inputs)

    labels = []
    for image_path in images:
        image = convert_rgb_to_grayscale(Image.open(image_path))
        # show image
        # image.show()
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt", padding=True)
        image_features = model.get_image_features(**inputs)

        # Compute similarity
        similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
        print(similarity)
        labels.append(text_prompts[similarity.argmax().item()])
    return labels


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str)
    args = parser.parse_args()
    imagedir = args.image_dir
    images = os.listdir(imagedir)
    images = [os.path.join(imagedir, image) for image in images]
    labels = classify_images(images)
    for image, label in zip(images, labels):
        print(f"{image}: {label}")