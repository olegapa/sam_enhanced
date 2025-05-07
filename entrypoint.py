import argparse
import base64
import json
import os
import time
import cv2
import logging
import numpy as np

from container_status import ContainerStatus as CS

color_mapping = {
    0: (0, 0, 0),  # Background
    1: (128, 0, 0),  # Jacket/Coat
    2: (0, 128, 0),  # Shirt/Blouse
    3: (128, 128, 0),  # Sweater/Sweatshirt/Hoodie
    4: (0, 0, 128),  # Dress/Romper
    5: (128, 0, 128),  # Pants/Jeans/Leggings
    6: (0, 128, 128),  # Shorts
    7: (128, 128, 128),  # Skirt
    8: (64, 0, 0),  # Shoes
    9: (192, 0, 0),  # Vest
    10: (64, 128, 0),  # Boots
    11: (192, 128, 0),  # Bodysuit/T-shirt/Top
    12: (64, 0, 128),  # Bag/Purse
    13: (192, 0, 128),  # Hat
    14: (64, 128, 128),  # Scarf/Tie
    15: (192, 128, 128),  # Gloves
    16: (0, 64, 0),  # Blazer/Suit
    17: (128, 64, 0),  # Underwear/Swim
}

run_time = time.time()
# Настройка логгера
logging.basicConfig(level=logging.INFO, filename='/output/sam.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Process some images.")

parser.add_argument('--input_data', type=str, help='Path to the input image')
parser.add_argument("--clipes_mode", action="store_true", help="Flag for training mode")
parser.add_argument("--host_web", type=str, help="url host with web")
parser.add_argument("--demo_mode", action="store_true", help="Flag for demo mode")
parser.add_argument("--min_width", type=str, help='Path to the input image')
parser.add_argument("--min_height", type=str, help='Path to the input image')

# Парсим аргументы
args = parser.parse_args()

# Получаем значения аргументов

OUTPUT_PATH = "/output"
CLIPES_MODE = args.clipes_mode

DEMO_MODE = args.demo_mode

TEMP_ROOT = OUTPUT_PATH if DEMO_MODE else "temp_image_root"

IMAGE_TEMP_PATH = f"{TEMP_ROOT}/bboxes"
MASK_TEMP_PATH = f"{TEMP_ROOT}/masks"
SAM_TEMP_PATH = f"{TEMP_ROOT}/sam"
COLORED_MASKS = f"{TEMP_ROOT}/colored"
INPUT_DATA = "/markups"
INPUT_DATA_ARG = args.input_data
SAM_CHECKPOINT = "segment_anything/sam_checkpoint/sam_vit_h_4b8939.pth"
HOST_WEB = args.host_web
cs = CS(HOST_WEB)
cs.post_start()
# Необязательные параметры на ограничения изображений
if args.min_width:
    min_width = int(args.min_width)
else:
    min_width = 5
if args.min_height:
    min_height = int(args.min_height)
else:
    min_height = 5

if INPUT_DATA_ARG:
    json_input_arg = json.loads(INPUT_DATA_ARG.replace("'", "\""))
    PROCESS_FREQ = int(json_input_arg.get("frame_frequency", 1))
else:
    PROCESS_FREQ = 1

CLIPES_LABELS = "CLIP-ES/custom_dataset/custom_labels.txt"


def check_video_extension(video_path):
    valid_extensions = {"avi", "mp4", "m4v", "mov", "mpg", "mpeg", "wmv"}
    ext = video_path.split('.')[-1].lower()
    return ext in valid_extensions


def get_image_name(video, frame, person):
    return f"{video.split('.')[0].split('/')[-1]}_frame_{frame}_person_{person}.png"



def prepare_image_dir(file_path, prepared_data, out_path, mask_out_path=None):
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(mask_out_path, exist_ok=True)
    cap = cv2.VideoCapture(file_path)
    frame_count = 0
    success = True

    while success:
        success, frame = cap.read()

        if not success:
            break

        # Проверяем, есть ли данный кадр в prepared_data
        if frame_count in prepared_data:
            frame_data = prepared_data[frame_count]

            for person_id, bbox in frame_data.items():
                x, y, width, height = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                person_image = frame[y:y + height, x:x + width]

                # Сохранение изображения человека
                image_name = get_image_name(file_path, frame_count, person_id)
                person_image_path = os.path.join(out_path, image_name)

                cv2.imwrite(person_image_path, person_image)
                if not CLIPES_MODE:
                    mask = bbox['mask']
                    image_data = base64.b64decode(mask)
                    with open(os.path.join(mask_out_path, image_name), 'wb') as output_file:
                        output_file.write(image_data)
                else:
                    with open(CLIPES_LABELS, "a") as file:
                        l = bbox["labels"].split()[1:]
                        if len(l) == 0:
                            continue
                        shifted_l = [str(int(i)-1) for i in l]
                        file.write(f'{image_name.split(".")[0]} {" ".join(shifted_l)}\n')
        frame_count += 1
    cap.release()


processed_frames = {"small": 0, 'ok': 0}


def frame_process_condition(num, markup_path):
    mask = markup_path.get('mask', None)
    if ((PROCESS_FREQ < 2 or int(num) % PROCESS_FREQ == 1) and markup_path['x'] > 0 and markup_path['y'] > 0
            and markup_path['width'] > min_width and markup_path['height'] > min_height and mask):
        if CLIPES_MODE:
            if len(markup_path['labels'].split()) < 2:
                return False
        if markup_path['width'] < 50 or markup_path['height'] < 100:
            processed_frames["small"] += 1
        else:
            processed_frames['ok'] += 1
        return True


def get_img_str(file_name, mask_dir):
    with open(f'{mask_dir}_processed/{file_name}', 'rb') as image_file:
        # Читаем содержимое изображения
        image_data = image_file.read()
        # Кодируем содержимое в Base64
        base64_encoded_data = base64.b64encode(image_data)
        # Преобразуем закодированные данные в строку
        base64_image_str = base64_encoded_data.decode('utf-8')
    return base64_image_str


def apply_color_mapping(mask, color_mapping):
    # Создание пустого изображения того же размера, что и маска, с 3 каналами для RGB
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Применение цвета для каждого класса согласно маппингу
    for class_id, color in color_mapping.items():
        colored_mask[mask == class_id] = color

    return colored_mask


def mask_to_polygons(mask, num_classes=17):
    polygons_by_class = {}
    for class_id in range(1, num_classes + 1):  # Пропускаем 0, так как это background
        # Создание бинарной маски для каждого класса
        class_mask = (mask == class_id).astype(np.uint8) * 255

        # Нахождение контуров
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Аппроксимация контуров полигонами
        polygons = [cv2.approxPolyDP(contour, epsilon=3, closed=True).tolist() for contour in contours]

        if polygons:
            polygons_by_class[class_id] = polygons
    return polygons_by_class


def process_frame(file_name, mask_dir, num_classes, color_mapping, colored_masks=False, output_dir=None):
    # Чтение маски
    mask = cv2.imread(f'{mask_dir}_processed/{file_name}', 0)  # Чтение маски в градациях серого

    # Преобразование маски в полигоны
    polygons = mask_to_polygons(mask, num_classes)

    # Если включена опция colored_masks, то сохраняем покрашенную маску
    if colored_masks:
        if output_dir is None:
            raise ValueError("Не указана директория для сохранения покрашенных масок")

        # Применение цветового маппинга к маске
        colored_mask = apply_color_mapping(mask, color_mapping)

        colored_mask_path = os.path.join(output_dir, file_name)

        # Создание директории, если она не существует
        os.makedirs(output_dir, exist_ok=True)

        # Сохранение цветной маски
        cv2.imwrite(colored_mask_path, colored_mask)

    return polygons


def prepare_output(input_data, mask_dir):
    for item in input_data['files']:
        video_path = item['file_name']
        file_chains = item['file_chains']

        for chain in file_chains:
            chain_id = chain['chain_name']
            for frame in chain['chain_markups']:
                # думаю, надо красить каждый 10 фрейм
                frame_num = str(frame['markup_frame'])
                if frame_process_condition(frame_num, frame["markup_path"]):
                    file_name = get_image_name(video_path, frame_num, chain_id)
                    frame['markup_path']['mask'] = get_img_str(file_name, mask_dir)
                    frame['markup_path']['polygons'] = process_frame(file_name, mask_dir, 17, color_mapping, DEMO_MODE, COLORED_MASKS)

    return input_data

def verify_file_name(postfix_name, common_name):
    _, postfix_tail = os.path.split(postfix_name)
    tail1 = '.'.join(postfix_tail.split('.')[0:-1])
    _, tail2 = os.path.split(common_name)
    logger.info(f"comparing {tail1} and {tail2}")
    return tail1 == f'IN_{tail2}'


files_in_directory = [
    os.path.join(INPUT_DATA, f)
    for f in os.listdir(INPUT_DATA)
    if (
        os.path.isfile(os.path.join(INPUT_DATA, f))
    )
]
files_in_directory = [
    file for file in files_in_directory if check_video_extension('.'.join(file.split('.')[0:-1]))
]

directories = list()
total_images = 0
start_time = time.time()
count = 0

for file in files_in_directory:
    cs.post_progress({"stage": "1 из 2", "progress": round(100*(count/len(files_in_directory)), 2)})
    with open(file, 'r', encoding='utf-8') as input_json:
        json_data = json.load(input_json)

    video_path = None
    _, dir_postfix = os.path.split(file)
    dir_postfix = dir_postfix.split('.')[0]
    im_dir = f'{IMAGE_TEMP_PATH}_{dir_postfix}'
    mask_dir = f'{MASK_TEMP_PATH}_{dir_postfix}'
    sam_dir = f'{SAM_TEMP_PATH}_{dir_postfix}'
    for item in json_data['files']:
        prepared_data = dict()
        video_path = item.get('file_name', "no_video")
        _, video_path = os.path.split(video_path)
        if not os.path.isfile(f'/input/{video_path}'):
            logger.warning(f"File name {video_path} doesn't exist - it is skipped")
            continue
        if not verify_file_name(file, video_path):
            logger.warning(f"File name {file} doesn't correspond to file_name key in json {video_path} - it is skipped")
            continue
        # if not check_video_extension(video_path):
        #     continue
        file_chains = item['file_chains']
        for chain in file_chains:
            chain_name = chain['chain_name']
            for frame in chain['chain_markups']:
                frame_num = frame['markup_frame']
                if frame_process_condition(frame_num, frame["markup_path"]):
                    # {"1frame":{"1pers": "bounder box", ...}, "11frame": {...}...}
                    if frame_num not in prepared_data.keys():
                        prepared_data[frame_num] = dict()
                    prepared_data[frame_num][chain_name] = frame["markup_path"]
        if not prepared_data:
            continue
        prepare_image_dir(f'/input/{video_path}', prepared_data, im_dir, mask_dir)
    logger.info(f'Data preparation took {time.time() - start_time} seconds')
    logger.info(f'Amount of small images (with height < 100 or width < 50): {processed_frames["small"]} '
                f'Amount of other images: {processed_frames["ok"]}. Total: {processed_frames["small"] + processed_frames["ok"]}')
    if not video_path:
        logger.warning(f"For {file} correspondent videos")
        continue
    if processed_frames["small"] + processed_frames["ok"] == 0:
        logger.warning(f"For {file} no bounder boxes were found")
        continue
    directories.append((json_data, im_dir, mask_dir, sam_dir, file, processed_frames["small"] + processed_frames["ok"]))
    total_images += (processed_frames["small"] + processed_frames["ok"])
    processed_frames["small"] = 0
    processed_frames["ok"] = 0

cs.post_progress({"stage": "1 из 3", "progress": 100})
cs.post_progress({"stage": "2 из 3", "progress": 0})
if CLIPES_MODE:
    start_time = time.time()
    output_weights = f'{OUTPUT_PATH}/deeplab_weights.pt'
    command = f'CUDA_VISIBLE_DEVICES=0 python CLIP-ES/genetate_cams_custom.py --img_root {IMAGE_TEMP_PATH} --split_file {CLIPES_LABELS} --model ./CLIP-ES/pretrained_models/clip/ViT-B-16.pt --num_workers 3 --cam_out_dir ./output/clipes/cams'
    os.system(command)
    command = f'python CLIP-ES/eval_cam_with_crf.py --cam_out_dir ./output/clipes/cams --image_root {IMAGE_TEMP_PATH} --split_file {CLIPES_LABELS} --pseudo_mask_save_path {MASK_TEMP_PATH}'
    os.system(command)
    logger.info(f'Clip-es inference took {time.time() - start_time} seconds')

start_time = time.time()
processed = 0
for json_data, image_directory, mask_directory, sam_dir, f, frame_amount in directories:
    command = (f'python segment_anything/scripts/amg.py --checkpoint {SAM_CHECKPOINT}'
               f' --model-type default --input {image_directory} --output {sam_dir} '
               f'--host_web {HOST_WEB} --total {total_images} --processed {processed}')
    os.system(command)
    processed += frame_amount
    logger.info(f'SAM inference took {time.time() - start_time} seconds')
tart_time = time.time()
count = 0
cs.post_progress({"stage": "3 из 3", "progress": 0})
for json_data, image_directory, mask_directory, sam_dir, f, frame_amount in directories:
    command = f'python SAM_WSSS/main.py --pseudo_path {mask_directory} --sam_path {sam_dir} --number_class 18'
    os.system(command)

    result = prepare_output(json_data, mask_directory)
    with open(f"{OUTPUT_PATH}/{os.path.basename(f)}", "w") as outfile:
        json.dump(result, outfile, ensure_ascii=False)
    count += 1
    cs.post_progress({"stage": "3 из 3", "progress": round(100 * (count / len(directories)), 2),
                      "statistics": {"out_file": os.path.basename(f)}})

logger.info(f'Result file generation took {time.time() - start_time} seconds')
logger.info(f'The whole process took {time.time() - run_time} seconds, clip-es mode = {CLIPES_MODE}')
