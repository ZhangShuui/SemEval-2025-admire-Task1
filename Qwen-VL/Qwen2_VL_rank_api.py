import io
import os
import re
import gc
import ast
import json
import torch
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers.generation import GenerationConfig
import matplotlib.image as mpimg
from openai import OpenAI
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import functools
import itertools
import multiprocessing as mp
from argparse import ArgumentParser
from multiprocessing import Pool


def plot_images(image_paths):
    num_images = len(image_paths)

    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))

    for i, image_path in enumerate(image_paths):
        img = mpimg.imread(image_path)
        if num_images == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.imshow(img)
        ax.set_title(f"Image {i+1}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


from openai import OpenAI
import base64

# 初始化客户端
client = OpenAI(
    api_key="sk-aBuN1XQipxl7OGBLas1mP4bxO7uVCT33CNj4UdUswxj9SimY",
    base_url="https://api.aiproxy.io/v1",
)


# 将本地图像转为Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        img_str = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"


# base64_image = encode_image("your_image.jpg")


def extract_list(text):
    # Use regex to find the list in the text
    match = re.search(r"\[(.*?)\]", text)
    if match:
        # Extract the content inside the brackets
        content = match.group(1)
        # Split by comma and strip whitespace
        items = [item.strip() for item in content.split(",")]
        return items
    return []


import matplotlib.patches as patches


### model path and model base
# model_path = "./share_models/Qwen2-VL-2B-Instruct"
# ori_processor_path = "./share_models/Qwen2-VL-2B-Instruct"

model_path = "./share_models/Qwen2-VL-7B-Instruct"  # after RL
ori_processor_path = "./share_models/Qwen2-VL-7B-Instruct"
image_path_base = "./classification/train/"  # image path base


def get_order_list(
    expected_order, image_path1, image_path2, image_path3, image_path4, image_path5
):
    # convert the expected order list to list of integers
    expected_order = ast.literal_eval(expected_order)
    order_list = []
    for image in expected_order:
        if image == image_path1:
            order_list.append(1)
        elif image == image_path2:
            order_list.append(2)
        elif image == image_path3:
            order_list.append(3)
        elif image == image_path4:
            order_list.append(4)
        elif image == image_path5:
            order_list.append(5)
    return order_list


def load_train_data():
    train_tsv = pd.read_csv(
        "/home/szhangfa/Visual-RFT/classification/train/subtask_a_train.tsv",
        sep="\t",
    )
    train_split = []
    # convert to list
    for i in range(len(train_tsv)):
        # compound	subset	sentence_type	sentence	expected_order	image1_name	image1_caption	image2_name	image2_caption	image3_name	image3_caption	image4_name	image4_caption	image5_name	image5_caption
        # elbow grease	Train	idiomatic	It took a lot of elbow grease to get the old engine running again.	['35234427395.png', '53378381715.png', '39938261459.png', '74852536462.png', '54879908369.png']	35234427395.png	The image depicts a hand holding a sponge and cleaning a glass cooktop stove. The cooktop is black with three circular burners, and there are some food residue stains on it. The sponge appears to be scrubbing off the stains, indicating that the person is cleaning the cooktop. The background is plain white, focusing attention on the cleaning activity.	39938261459.png	The image depicts a hand wearing a yellow work glove holding a rusty metal pipe. The pipe is being dipped into a can of orange-colored paint. The paint is dripping from the pipe, indicating that it is being applied to the pipe's surface. The background is white, which makes the colors and objects stand out clearly.	53378381715.png	The image depicts a hand holding a duster with a long, black and orange handle. The duster is being used to clean a surface, likely a ceiling or high wall, as indicated by the angle of the duster and the presence of dust particles being dislodged. The dust particles are shown in motion, suggesting that the duster is actively being used to remove dirt or debris from the surface. The background is plain white, which helps to emphasize the action taking place.	54879908369.png	The image depicts a person wearing knee pads and hiking shoes. The knee pads are black with orange accents and have adjustable straps for securing them in place. The person is also wearing short pants and socks, and the hiking shoes are brown with orange laces. The overall appearance suggests that the individual is prepared for outdoor activities, possibly involving hiking or other physical exertion.	74852536462.png	The image depicts a person wearing a black outfit, holding a bowl of honey in one hand and pouring some of the honey onto their other hand. The honey is dripping from the hand into the bowl. The person appears to be in the process of using the honey for a skincare or beauty treatment.
        # night owl	Train	idiomatic	It's a constant battle for us, as he is a morning person and I am a night owl, so I find that going to sleep at 9.30 really cuts out the best hours.	['61697797701.png', '93189810779.png', '89375227504.png', '00982495584.png', '93541983868.png']	00982495584.png	The image depicts a nighttime scene with a large, bright full moon in the background. The moon is prominently positioned in the upper left quadrant of the image, casting a warm, golden light that illuminates the surrounding area. The sky is dark, suggesting it is nighttime, and there are a few small stars scattered across it.\n\nIn the foreground, there is a black owl perched on a branch. The owl has large, round eyes that are glowing orange, giving it a striking and somewhat eerie appearance. Its feathers are detailed, with shades of black and brown, and its talons are visible as it grips the branch firmly. The owl's wings are folded neatly against its body, and it appears to be looking directly at the viewer.\n\nThe branch on which the owl is perched is dark and bare, with no leaves or foliage. It extends diagonally from the bottom left to the top right of the image, creating a sense of depth and perspective. The overall atmosphere of the image is mysterious and somewhat unsettling, enhanced by the contrast between the bright moonlight and the dark surroundings.	61697797701.png	The image depicts a cartoon-style illustration of a person sitting at a desk, working on a computer. The setting appears to be a cozy, well-lit room with a starry night sky visible through a large circular window behind the desk. The desk is cluttered with various items, including a laptop, a cup of pens and pencils, a cactus plant in a pot, and some books. The person is wearing glasses and is focused on their work. The overall atmosphere is calm and productive.	89375227504.png	The image depicts a cartoon-style owl perched on a branch of a tree. The owl has large, round eyes and an orange beak. The tree branches are adorned with autumn leaves, which are mostly orange and yellow, indicating the season. In the background, there is a serene landscape featuring a body of water, possibly a lake or pond, with reflections of the trees and sky. The trees in the background are also in shades of orange and yellow, consistent with the autumn theme. The overall scene is peaceful and picturesque, capturing the essence of a calm autumn day.	93189810779.png	The image depicts a cartoon-style illustration of a young child sitting on a small couch. The child has dark hair and is wearing striped pajamas with orange and white colors. The couch is brown with visible buttons, and the child appears to be in a relaxed or sleepy state, with their eyes closed and a slight smile on their face. The background is plain and white, which helps to focus attention on the child and the couch.	93541983868.png	The image depicts a dumbbell, which is a common piece of exercise equipment used for strength training. The dumbbell has two large, round weights attached to opposite ends of a central handle. The weights appear to be made of metal and are encased in black rubber or plastic sleeves, which provide a non-slip grip and protect the weights from scratches and damage. The central handle is metallic and has a textured surface for better grip. The overall design suggests that it is intended for use in weightlifting exercises to build muscle strength and endurance.
        # heart of gold	Train	idiomatic	Even the somewhat seedy failed private eye has a heart of gold (and a bad hairstyle).	['86137977215.png', '78062290185.png', '54240592941.png', '92088849364.png', '90660547751.png']	54240592941.png	The image depicts a large, metallic safe with its door open, revealing a gold bar inside. The safe has a robust design with thick walls and reinforced corners, featuring multiple bolts and a circular lock mechanism on the door. The gold bar is prominently displayed, showcasing its shiny, golden surface with embossed markings. The overall appearance suggests a secure and valuable storage solution, likely used for storing precious metals or other high-value items.	78062290185.png	The image depicts a joyful scene featuring a young boy and his dog. The boy is sitting inside an open cardboard box, which is decorated with snowflakes and stars. He has a big smile on his face, indicating he is excited or happy. The dog, wearing an orange collar with a bell, is jumping towards the boy with its mouth open, as if it is eager to interact with him. The background is plain white, which helps to emphasize the subjects in the foreground. The overall atmosphere of the image is cheerful and playful, suggesting a moment of joy and companionship between the boy and his dog.	86137977215.png	The image depicts a cartoon scene where a young boy is feeding a group of puppies. The boy, who has brown hair and is wearing an orange shirt and brown shorts, is holding a bowl filled with food. There are four puppies in the scene: one brown puppy standing close to the boy, two black and white puppies standing next to each other, and another brown puppy standing slightly behind them. Each puppy has a bowl of food in front of it on the ground. The background shows a simple, plain wall with a door on the left side and a stone pillar on the right side. The overall setting appears to be indoors, possibly in a garage or a similar area.	90660547751.png	The image depicts a futuristic, stylized spacecraft with a sleek, aerodynamic design. The main body of the spacecraft is white with orange accents and various panels and buttons. It has a large, rounded front section with a prominent window, suggesting it is designed for observation or communication. The side panels feature multiple control panels and lights, indicating advanced technology and functionality. The rear of the spacecraft has several thrusters, likely for propulsion and maneuverability. The overall design suggests a blend of science fiction and modern engineering aesthetics.	92088849364.png	The image depicts a stylized, artistic representation of a human heart. The heart is rendered in a glossy, golden-orange color with a smooth, reflective surface. It features intricate details such as the coronary arteries and veins, which are depicted in a darker shade of orange and black, respectively. The heart also has small, sparkling gemstones embedded in various locations, adding a decorative and luxurious touch. The overall design combines elements of realism with a fantastical, embellished aesthetic.
        # agony aunt	Sample	idiomatic	ESA's Space Weather Office is like Europe's stellar agony aunt, offering forecasts, advice and information for any organisation, individual or institution vulnerable to space phenomena.	['83600499282.png', '57658144685.png', '02512838127.png', '32964421720.png', '92533456778.png']	02512838127.png	The image depicts a serene outdoor scene featuring a woman and two children in a garden setting. The woman, who has brown hair tied up in a bun, is standing to the right side of the image. She is wearing a long-sleeved dark-colored top and black pants, along with high-heeled shoes. Her expression appears to be one of gentle guidance or supervision as she watches over the children.\n\nThe two children, who also have brown hair, are standing close to each other in the center of the image. They are both dressed in orange shirts and black pants. One child is slightly ahead of the other, and they seem to be engaged in some form of play or interaction, possibly holding hands or playing together.\n\nIn the background, there is a small tree with green leaves and orange flowers. The tree is situated on the left side of the image. The ground around the tree and the children is covered with grass and small rocks, adding to the natural feel of the scene. There are also several orange flowers scattered around the area, enhancing the vibrant and lively atmosphere of the garden.\n\nAdditionally, there is a butterfly flying near the top right corner of the image, adding a touch of whimsy and movement to the scene. The overall mood of the image is warm and nurturing, suggesting a moment of connection and care between the woman and the children.	32964421720.png	The image depicts a cartoon-style illustration of a woman lying in bed, appearing distressed or in pain. She has a bloodstain on her face and is wearing a white shirt. The bed has a brown headboard and is covered with a brown blanket. The woman's expression and body language suggest she is experiencing discomfort or distress.	57658144685.png	The image depicts a scene of a person sitting at a desk, engaged in writing or typing on an old-fashioned typewriter. The individual is wearing glasses and has their hair tied back in a bun. They are dressed in a light-colored sweater over a brown top, and they appear to be focused on the task at hand.\n\nThe desk is cluttered with numerous books, both stacked and scattered around. The books vary in size and color, suggesting a diverse collection. Some books are tied together with ribbons or strings, indicating they might be part of a series or collected set. The books are predominantly placed on the lower half of the desk, while the upper half is occupied by the typewriter and some additional books.\n\nTo the right of the typewriter, there is a small desk lamp with a classic design, featuring a metal base and a bell-shaped shade. The lamp is turned off, but its presence adds to the vintage ambiance of the setting.\n\nBehind the person, there is another stack of books, neatly arranged in a pile. This stack is slightly taller than the one on the desk, and it appears to be organized in a somewhat orderly fashion.\n\nThe overall atmosphere of the image suggests a dedicated workspace, possibly belonging to someone who enjoys reading and writing. The presence of the typewriter indicates a preference for traditional methods of writing, which could imply a connection to literature, history, or academia.\n\nIn summary, the image portrays a person at a desk surrounded by a large number of books, using a typewriter, with a vintage desk lamp nearby. The scene conveys a sense of intellectual engagement and a love for literature.	83600499282.png	The image depicts a person sitting at a desk, engaged in writing or typing. The individual is holding a pencil and appears to be focused on the task at hand. The desk is cluttered with various items, including a typewriter, several envelopes, a stack of papers, and a cup filled with writing utensils such as pencils and pens. The person is wearing a white blouse and has their hair tied up in a bun. The setting suggests a workspace or an office environment, possibly from a bygone era given the presence of the typewriter.	92533456778.png	The image depicts a cartoon character of a woman dressed in professional attire, holding a microphone. She is wearing a brown suit with a white shirt and an orange tie. The character has brown hair styled in a bob cut and is standing with one hand on her hip and the other holding the microphone. The background is plain white, emphasizing the character.
        base_dir = image_path_base + train_tsv["compound"][i] + "/"
        train_split.append(
            {
                "compound": train_tsv["compound"][i],
                "subset": train_tsv["subset"][i],
                "sentence_type": train_tsv["sentence_type"][i],
                "sentence": train_tsv["sentence"][i],
                "expected_order": get_order_list(
                    train_tsv["expected_order"][i],
                    train_tsv["image1_name"][i],
                    train_tsv["image2_name"][i],
                    train_tsv["image3_name"][i],
                    train_tsv["image4_name"][i],
                    train_tsv["image5_name"][i],
                ),
                "base_dir": base_dir,
                "image1_name": train_tsv["image1_name"][i],
                "image2_name": train_tsv["image2_name"][i],
                "image3_name": train_tsv["image3_name"][i],
                "image4_name": train_tsv["image4_name"][i],
                "image5_name": train_tsv["image5_name"][i],
            }
        )
    return train_split


def run(rank, world_size):

    train_split = load_train_data()
    pred_results = []
    right_count = 0
    wrong_count = 0
    error_count = 0
    for image in tqdm(train_split):
        ### 获取图片信息
        compound = image["compound"]
        subset = image["subset"]
        sentence_type = image["sentence_type"]
        sentence = image["sentence"]
        expected_order = image["expected_order"]
        base_dir = image["base_dir"]
        image_path_1 = image["image1_name"]
        image_path_2 = image["image2_name"]
        image_path_3 = image["image3_name"]
        image_path_4 = image["image4_name"]
        image_path_5 = image["image5_name"]
        image_path = [
            image_path_1,
            image_path_2,
            image_path_3,
            image_path_4,
            image_path_5,
        ]

        ### Traverse all class in image.

        # question = (
        #     f"Detect all objects belonging to the category '{category}' in the image, and provide the bounding boxes (between 0 and 1000, integer) and confidence (between 0 and 1, with two decimal places).\n"
        #     f"If no object belonging to the category '{category}' in the image, return 'No Objects'.\n"
        #     "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
        #     "The output answer format should be as follows:\n"
        #     "<think> ... </think> <answer>[{'Position': [x1, y1, x2, y2], 'Confidence': number}, ...]</answer>\n"
        #     "Please strictly follow the format."
        # )
        question = (
            f"Given images: 1. <image>\n 2. <image>\n 3. <image>\n 4. <image>\n 5. <image>\n\n"
            f"Below is a compound word, with a example usage sentence\n"
            f"Compound: {compound}\n"
            f"Usage: {sentence}\n"
            f"Given the five images, please identify the expected order of the images that best match the compound word.\n"
            f"Your output should be in the format of a list, with the image names in the expected order, which means the first image in the list should be the one that best matches the compound word,\n"
            f"and the last image in the list should be the one that least matches the compound word.\n"
            f"for example, if the expected order is imageA, imageB, imageC, imageD, imageE, your output should be: [A, B, C, D, E]\n"
            f"where A, B, C, D, E are the image given order numbers.\n"
            f"Now, please provide the expected order of the given images:\n"
        )
        query = question
        # logger.info(RED + query + RESET)

        try:
            messages = [
                {
                    "role": "user",
                    # "content": [{"type": "image", "image": image_path}]
                    # + [{"type": "text", "text": query}],
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": encode_image(
                                    os.path.join(base_dir, image_path[0])
                                )
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": encode_image(
                                    os.path.join(base_dir, image_path[1])
                                )
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": encode_image(
                                    os.path.join(base_dir, image_path[2])
                                )
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": encode_image(
                                    os.path.join(base_dir, image_path[3])
                                )
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": encode_image(
                                    os.path.join(base_dir, image_path[4])
                                )
                            },
                        },
                        {"type": "text", "text": query},
                    ],
                }
            ]
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=1024,
                temperature=0.1,
                top_p=0.5,
            )
            response = response.choices[0].message.content

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory error")
            error_count += 1
            torch.cuda.empty_cache()
            gc.collect()
            continue
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            error_count += 1
            continue
        # Fix possible formatting errors in the response.
        response = response.replace("[[", "[")
        response = response.replace("]]", "]")
        response = response.replace("\n", "")
        response = response.replace(", ...", "")
        # logger.info("\033[92m" + response + "\033[0m")

        # extract list

        try:
            pred_result = extract_list(response)
            pred_result = [int(i) for i in pred_result]
            logger.info(GREEN + str(pred_result) + RESET)
            pred_results.append(
                {
                    "compound": compound,
                    "subset": subset,
                    "sentence_type": sentence_type,
                    "sentence": sentence,
                    "expected_order": expected_order,
                    "pred_result": pred_result,
                }
            )
        except Exception as e:
            logger.error(f"Error extracting list: {str(e)}")
            error_count += 1
            continue

        # check the top1 accuracy
        pred_result_top1 = pred_result[0]
        if pred_result_top1 == expected_order[0]:
            logger.info(GREEN + "Top1 Right" + RESET)
            right_count += 1
        else:
            logger.info(RED + "Top1 Wrong" + RESET)
            wrong_count += 1

    return [error_count, right_count, wrong_count, pred_results]


# def main():
#     multiprocess = torch.cuda.device_count() >= 2
#     mp.set_start_method("spawn")
#     if multiprocess:
#         logger.info("started generation")
#         n_gpus = torch.cuda.device_count()
#         world_size = n_gpus
#         with Pool(world_size) as pool:
#             func = functools.partial(run, world_size=world_size)
#             result_lists = pool.map(func, range(world_size))

#         global_count_error = 0
#         global_results = []
#         for i in range(world_size):
#             global_count_error += int(result_lists[i][0])
#             global_results = global_results + result_lists[i][1]


#         logger.info("Error number: " + str(global_count_error))
#         ### save path
#         with open("prediction_results.json", "w") as json_file:
#             json.dump(global_results, json_file)
#         logger.info("Done")
#         logger.info("finished running")
#     else:
#         logger.info("Not enough GPUs")
def main():
    # Check if at least one GPU is available
    if torch.cuda.is_available():
        logger.info("Started generation on single GPU")
        world_size = 1
        # Run the 'run' function directly for a single process
        result = run(rank=0, world_size=world_size)
        global_count_error = result[0]
        global_count_right = result[1]
        global_count_wrong = result[2]
        global_results = result[3]

        logger.info("Error number: " + str(global_count_error))
        logger.info("Total Right Number: " + str(global_count_right))
        logger.info("Total Wrong Number: " + str(global_count_wrong))
        ### save path
        with open("prediction_results_qwen2_vl.json", "w") as json_file:
            json.dump(global_results, json_file)
        logger.info("Done")
        logger.info("Finished running")
    else:
        logger.info("No GPUs available for computation")


if __name__ == "__main__":
    main()
