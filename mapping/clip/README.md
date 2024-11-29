## CLIP-similarity
This is an example of how to use the CLIP model to calculate the similarity between an image and a text prompt.
In our case, we have a pre-defined labels for rooms and we want to calculate the similarity between an image and a room label. This would allows us to predict the room label of an image.

## Requirements
Install the required packages using the following command:
```bash
pip install transformers torch
```
## Usage
To calculate the similarity between an image and a text prompt, you can use the following code:
```bash
clip_similarity.py --image_dir /path/to/image/directory                    
```         
This would calculate the similarity between the image and the room labels and return the most similar room label.
                        
