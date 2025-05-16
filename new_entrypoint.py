import argparse
import base64
import json
import os
import sys
import time
import cv2
import logging
import numpy as np
from PIL import Image
import pickle

sys.path.insert(0, './segment_anything/scripts')

# import segment_anything.scripts.amg as amg
import amg

from container_status import ContainerStatus as CS
from progress_counter import ProgressCounter
import visualization

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

SCORE_THRESHOLD = {
    0: 0.0,  # Background
    1: 0.0,  # Jacket/Coat
    2: 0.0,  # Shirt/Blouse
    3: 0.0,  # Sweater/Sweatshirt/Hoodie
    4: 0.0,  # Dress/Romper
    5: 0.0,  # Pants/Jeans/Leggings
    6: 0.0,  # Shorts
    7: 0.0,  # Skirt
    8: 0.0,  # Shoes
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

N_CLS = 8
# Получаем значения аргументов

OUTPUT_PATH = "/output"
CLIPES_MODE = args.clipes_mode

DEMO_MODE = args.demo_mode

TEMP_ROOT = OUTPUT_PATH if DEMO_MODE else "temp_image_root"

IMAGE_TEMP_PATH = f"{TEMP_ROOT}/bboxes"
MASK_TEMP_PATH = f"{TEMP_ROOT}/masks"
SAM_TEMP_PATH = f"{TEMP_ROOT}/sam"
COLORED_MASKS = f"{TEMP_ROOT}/colored"
INPUT_DATA = "/input_data"
INPUT = "/input_videos"
INPUT_DATA_ARG = args.input_data
SAM_CHECKPOINT = "segment_anything/sam_checkpoint/sam_vit_h_4b8939.pth"
SAM2_CHECKPOINT = "/segmentation/sam2/checkpoints/sam2.1_hiera_large.pt"
HOST_WEB = args.host_web
STAGE1, STAGE2, STAGE3 = "Подготовка данных", "Обработка SAM", "Улучшение исходных масок SEPL"
cs = CS(HOST_WEB, logger=logger)
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
    try:
        json_input_arg = json.loads(INPUT_DATA_ARG.replace("'", "\""))
    except Exception:
        cs.post_error({"msg": "Wrong input_data argument format", "details": f"input_data is following {INPUT_DATA_ARG}"})
        logger.info(f"Wrong input_data argument format {INPUT_DATA_ARG}")
        exit(-1)
    PROCESS_FREQ = int(json_input_arg.get("frame_frequency", 1))
    SAM2_MODE = eval(json_input_arg.get("sam2", "False"))
    MAKE_VISUALIZATION = eval(json_input_arg.get("visualize", "False"))
    score_thresholds = json_input_arg.get("score_thresholds", None)
    approx_eps = json_input_arg.get("approx_eps", 0.02)
    approx_eps = float(approx_eps) if approx_eps else None
    if score_thresholds:
        score_thresholds = eval(score_thresholds)
        for cls, score in score_thresholds.items():
            SCORE_THRESHOLD[cls] = score
else:
    PROCESS_FREQ = 1
    SAM2_MODE = False
    MAKE_VISUALIZATION = False
    approx_eps = 0.02

CLIPES_LABELS = "CLIP-ES/custom_dataset/custom_labels.txt"

def check_video_extension(video_path):
    valid_extensions = {"avi", "mp4", "m4v", "mov", "mpg", "mpeg", "wmv"}
    ext = video_path.split('.')[-1].lower()
    return ext in valid_extensions


def get_image_name(video, frame, person):
    return f"{video.split('.')[0].split('/')[-1]}_frame_{frame}_person_{person}.png"


def check_class_score_criteria(class_id: int, score: float):
    if score is None:
        return True
    return score > SCORE_THRESHOLD[class_id]


def prepare_image_dir(file_path, prepared_data, out_path, mask_out_path=None):
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(mask_out_path, exist_ok=True)
    cap = cv2.VideoCapture(file_path)
    max_width, max_height = cap.get(3), cap.get(4)
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
                if x + width > max_width or y + height > max_height:
                    continue
                person_image = frame[y:y + height, x:x + width]

                # Сохранение изображения человека
                image_name = get_image_name(file_path, frame_count, person_id)

                if not CLIPES_MODE:
                    polygons = bbox['polygons']
                    mask = np.zeros((height, width), dtype=np.uint8)
                    # logger.info(f"shape = {mask.shape} width height = {(width, height)}")
                    skip_bbox = False
                    for class_id, class_dict in polygons.items():
                        if not check_class_score_criteria(int(class_id), class_dict['score']):
                            skip_bbox = True
                            break
                        # logger.info(f'for {image_name} class_id = {class_id}, poly_list = {poly_list}')
                        for poly in class_dict['polygons']:
                            # Преобразуем список координат в массив точек формы (-1, 1, 2)
                            poly_shifted = [(px - x, py - y) for px, py in zip(poly[::2], poly[1::2])]
                            points = np.array(poly_shifted, dtype=np.int32).reshape(-1, 2)
                            # points = points[:, [1, 0]]
                            # Заполняем область полигона на маске
                            cv2.fillPoly(mask, [points], int(class_id))
                    if skip_bbox:
                        continue
                    # Сохраняем маску, восстановленную по полигонам
                    polygonized_pred_img = Image.fromarray(mask)
                    polygonized_pred_img.save(os.path.join(mask_out_path, image_name))
                else:
                    with open(CLIPES_LABELS, "a") as file:
                        l = bbox["labels"].split()[1:]
                        if len(l) == 0:
                            continue
                        shifted_l = [str(int(i) - 1) for i in l]
                        file.write(f'{image_name.split(".")[0]} {" ".join(shifted_l)}\n')
                person_image_path = os.path.join(out_path, image_name)

                cv2.imwrite(person_image_path, person_image)

        frame_count += 1
    cap.release()


processed_frames = {"small": 0, 'ok': 0}


def frame_process_condition(num, bbox_data):
    if ((PROCESS_FREQ < 2 or int(num) % PROCESS_FREQ == 1) and bbox_data['x'] > 0 and bbox_data['y'] > 0
            and bbox_data['width'] > min_width and bbox_data['height'] > min_height):
        if CLIPES_MODE:
            if len(bbox_data['labels'].split()) < 2:
                return False
        if bbox_data['width'] < 50 or bbox_data['height'] < 100:
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


def mask_to_polygons(mask, num_classes=8):
    polygons_by_class = {}
    for class_id in range(1, num_classes + 1):  # Пропускаем 0, так как это background
        # Создание бинарной маски для каждого класса
        class_mask = (mask == class_id).astype(np.uint8) * 255

        # Нахождение контуров
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if approx_eps != 0:
            approximated_contours = list()
            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, approx_eps * peri, True)
                approximated_contours.append(approx)
            contours = approximated_contours
        polygons_by_class[int(class_id)] = [contour.squeeze().flatten().tolist() for contour in contours]
        # if DEMO_MODE:
        #     cv2.drawContours(polygonized_pred, contours, -1, color, thickness=cv2.FILLED)

    return polygons_by_class


def process_frame(file_name, mask_dir, cls, num_classes, color_mapping, colored_masks=False, output_dir=None):
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
    # logger.info(f'ploygons = {polygons}, cls = {cls}')
    return polygons[int(cls)]


def prepare_output(input_data, mask_dir):
    chain_count, markup_count = 0, 0
    for item in input_data['files']:
        video_path = item['file_name']
        file_chains = item['file_chains']
        for chain in file_chains:
            chain_count += 1
            chain_id = chain['chain_name']
            chain['chain_vector'] = None
            i = 0
            while i < len(chain['chain_markups']):
                frame_num = str(chain['chain_markups'][i]['markup_frame'])
                file_name = get_image_name(video_path, frame_num, chain_id)
                if not os.path.isfile(f"{mask_dir}_processed/{file_name}"):
                    chain['chain_markups'].pop(i)
                else:
                    i += 1
            for frame in chain['chain_markups']:
                markup_count += 1
                frame_num = str(frame['markup_frame'])
                bbox_data = {"x": round(frame["markup_path"]["x"]), "y": round(frame["markup_path"]["y"]),
                             "width": round(frame["markup_path"]["width"]), "height": round(frame["markup_path"]["height"])}
                if frame_process_condition(frame_num, bbox_data) and int(frame['markup_path']['class']) != 0:
                    file_name = get_image_name(video_path, frame_num, chain_id)
                    poly = process_frame(file_name, mask_dir, frame['markup_path']['class'],
                                                                    N_CLS, color_mapping, DEMO_MODE,
                                                                    COLORED_MASKS)
                    resized_polygons = list()
                    for polygon in poly:
                        resized_polygon = []
                        for i in range(0, len(polygon), 2):
                            x = int(bbox_data['x']) + int(polygon[i])
                            y = int(bbox_data['y']) + int(polygon[i + 1])
                            resized_polygon.extend([x, y])
                        resized_polygons.append(resized_polygon)
                    frame['markup_path']['polygons'] = resized_polygons
                    frame['markup_vector'] = None
    return input_data, chain_count, markup_count


def save_empty_file(data, f):
    files = data.get("files", None)
    if files:
        file_id = files[0].get("file_id", None)
        file_name = files[0].get("file_name", None)
    else:
        file_id, file_name = None, None
    result = {"files": [{"file_id": file_id, "file_name": file_name, "file_chains": []}]}
    spl = os.path.basename(f).split('_')
    output_file_name = '_'.join(spl[1:len(spl)]) if os.path.basename(f).startswith('IN_') else os.path.basename(
        f)
    spl = output_file_name.split('.')
    outp_without_ext = '.'.join(spl[0:len(spl) - 1])
    output_file_name = "OUT_" + output_file_name
    result['files'][0]['file_name'] = outp_without_ext

    with open(f"{OUTPUT_PATH}/{output_file_name}", "w") as outfile:
        json.dump(result, outfile, ensure_ascii=False)

    empty_vectors, empty_vector_chains = list(), list()

    with open(f"{OUTPUT_PATH}/{outp_without_ext}_chains_vectors.pkl", "wb") as pickle_file:
        pickle.dump(empty_vector_chains, pickle_file)
    with open(f"{OUTPUT_PATH}/{outp_without_ext}_markups_vectors.pkl", "wb") as pickle_file:
        pickle.dump(empty_vectors, pickle_file)
    cs.post_progress(
        {"stage": STAGE3, "progress": 100,
         "statistics": {"out_file": output_file_name, "chains_count": 0, "markups_count": 0}})


def verify_file_name(postfix_name, common_name):
    _, postfix_tail = os.path.split(postfix_name)
    tail1 = '.'.join(postfix_tail.split('.')[0:-1])
    _, tail2 = os.path.split(common_name)
    logger.info(f"comparing {tail1} and {tail2}")
    return tail1 == f'IN_{tail2}'


def verify_additional_file_name(postfix_name, common_name):
    _, postfix_tail = os.path.split(postfix_name)
    tail1 = '.'.join(postfix_tail.split('.')[0:-1])
    _, tail2 = os.path.split(common_name)
    logger.info(f"comparing {tail1} and {tail2}")
    return tail1 == f'{tail2}'


def get_bbox_data(frame_data, id_bbox):
    if not id_bbox:
        raise Exception("No bbox_data is found")
    return id_bbox[frame_data['markup_parent_id']]


def prepare_bbox_info(json_file):
    bbox_res = dict()
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for i in data['files']:
        chains = i['file_chains']
        for ch in chains:
            for fr in ch['chain_markups']:
                bbox_res[fr['markup_id']] = fr['markup_path']
    return bbox_res


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

directories, empty_files = list(), list()
total_images = 0
start_time = time.time()
count = 0
for file in files_in_directory:
    cs.post_progress({"stage": STAGE1, "progress": round(100 * (count / len(files_in_directory)), 2)})
    with open(file, 'r', encoding='utf-8') as input_json:
        try:
            json_data = json.load(input_json)
        except Exception:
            cs.post_error({"msg": "Error when loading json file", "details": f"File: {file}"})
            logger.info(f"Error when loading json file {file}")
            continue

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
        if not (os.path.isfile(f'{INPUT}/{video_path}') or os.path.islink(f'{INPUT}/{video_path}')):
            cs.post_error({"msg": "Video file hasn't been found", "details": f"File name: {INPUT}/{video_path}"})
            logger.warning(f"File name {INPUT}/{video_path} doesn't exist - it is skipped")
            break
        if not verify_file_name(file, video_path):
            cs.post_error({"msg": "Invalid file name",
                           "details": f"File name {file} doesn't correspond to file_name key in json {INPUT}/{video_path}"})
            logger.warning(f"File name {file} doesn't correspond to file_name key in json {video_path}")
            break
        # if not check_video_extension(video_path):
        #     continue

        # To convert time to frame_num
        cap = cv2.VideoCapture(f'{INPUT}/{video_path}')
        fps = cap.get(cv2.CAP_PROP_FPS)

        file_chains = item.get('file_chains', None)
        if not file_chains:
            # cs.post_error({"msg": "file_chains key has not been found",
            #                "details": f"File name {file}"})
            continue
        for chain in file_chains:
            chain_name = chain.get('chain_name', None)
            if chain_name is None:
                cs.post_error({"msg": "Chain name is not specified",
                               "details": f"File name {file}"})
                logger.info(f"Chain name is not specified {file}")
                continue
            for frame in chain['chain_markups']:
                frame_num = frame.get('markup_frame', None)
                if frame_num is None:
                    m_time = frame.get('markup_time', None)
                    if m_time is None:
                        cs.post_error({"msg": "Nor markup_frame, nor markup_time are set",
                                       "details": f"File name {file}, chain_name {chain_name}"})
                        continue
                    frame['markup_frame'] = round(float(frame['markup_time']) * float(fps))
                    frame_num = frame['markup_frame']
                bbox_data = {"x": frame["markup_path"]["x"], "y": frame["markup_path"]["y"], "width": frame["markup_path"]["width"], "height": frame["markup_path"]["height"]}
                if frame_process_condition(frame_num, bbox_data):
                    if frame_num not in prepared_data.keys():
                        prepared_data[frame_num] = dict()
                    if chain_name not in prepared_data[frame_num].keys():
                        prepared_data[frame_num][chain_name] = bbox_data
                        prepared_data[frame_num][chain_name]['polygons'] = dict()
                    prepared_data[frame_num][chain_name]['polygons'][frame['markup_path']['class']] = dict()
                    prepared_data[frame_num][chain_name]['polygons'][frame['markup_path']['class']]['polygons'] = frame["markup_path"]['polygons']
                    markup_confidence = frame.get('markup_confidence', None)
                    if markup_confidence:
                        markup_confidence = float(markup_confidence)
                    prepared_data[frame_num][chain_name]['polygons'][frame['markup_path']['class']]['score'] = markup_confidence

        if not prepared_data:
            empty_files.append((json_data, file))
            continue
        cap.release()
        prepare_image_dir(f'/{INPUT}/{video_path}', prepared_data, im_dir, mask_dir)
    logger.info(f'Data preparation took {time.time() - start_time} seconds')
    logger.info(f'Amount of small images (with height < 100 or width < 50): {processed_frames["small"]} '
                f'Amount of other images: {processed_frames["ok"]}. Total: {processed_frames["small"] + processed_frames["ok"]}')
    if not video_path:
        logger.warning(f"For {file} correspondent videos")
        empty_files.append((json_data, file))
        continue
    if processed_frames["small"] + processed_frames["ok"] == 0:
        logger.warning(f"For {file} no bounder boxes were found")
        empty_files.append((json_data, file))
        continue
    directories.append(
        (json_data, im_dir, mask_dir, video_path, sam_dir, file, processed_frames["small"] + processed_frames["ok"]))
    total_images += (processed_frames["small"] + processed_frames["ok"])
    processed_frames["small"] = 0
    processed_frames["ok"] = 0

cs.post_progress({"stage": STAGE1, "progress": 100})
cs.post_progress({"stage": STAGE2, "progress": 0})
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
for json_data, image_directory, mask_directory, vp, sam_dir, f, frame_amount in directories:
    if SAM2_MODE:
        logger.info("Sam v2 is used")
        command = (f'python sam2/sam2_predict.py --checkpoint {SAM2_CHECKPOINT}'
                   f' --input {image_directory} --output {sam_dir} '
                   f'--host_web {HOST_WEB} --total {total_images} --processed {processed}')
        os.system(command)
    else:
        logger.info("Sam v1 is used")
        counter = ProgressCounter(total=int(total_images), processed=int(processed), cs=cs, logger=logger)
        # command = (f'python segment_anything/scripts/amg.py --checkpoint {SAM_CHECKPOINT}'
        #            f' --model-type default --input {image_directory} --output {sam_dir} '
        #            f'--host_web {HOST_WEB} --total {total_images} --processed {processed}')
        amg.main(args=args, checkpoint=SAM_CHECKPOINT, model_type="default", input=image_directory,
                 output=sam_dir, counter=counter)
    processed += frame_amount
    logger.info(f'SAM inference took {time.time() - start_time} seconds')


tart_time = time.time()
count = 0
cs.post_progress({"stage": STAGE3, "progress": 0})
for json_data, image_directory, mask_directory, vp, sam_dir, f, frame_amount in directories:
    command = f'python SAM_WSSS/main.py --pseudo_path {mask_directory} --sam_path {sam_dir} --number_class 9'
    os.system(command)

    result, chain_c, markup_c = prepare_output(json_data, mask_directory)
    spl = os.path.basename(f).split('_')
    output_file_name = '_'.join(spl[1:len(spl)]) if os.path.basename(f).startswith('IN_') else os.path.basename(f)

    spl = output_file_name.split('.')
    outp_without_ext = '.'.join(spl[0:len(spl) - 1])
    with open(f"{OUTPUT_PATH}/{outp_without_ext}_chains_vectors.pkl", "wb") as pickle_file:
        pickle.dump([], pickle_file)
    with open(f"{OUTPUT_PATH}/{outp_without_ext}_markups_vectors.pkl", "wb") as pickle_file:
        pickle.dump([], pickle_file)

    output_file_name = "OUT_" + output_file_name
    result['files'][0]['file_name'] = outp_without_ext

    with open(f"{OUTPUT_PATH}/{output_file_name}", "w") as outfile:
        json.dump(result, outfile, ensure_ascii=False)
    count += 1
    cs.post_progress({"stage": STAGE3, "progress": round(100 * (count / len(directories)), 2),
                      "statistics": {"out_file": output_file_name, "chains_count": chain_c, "markups_count": markup_c}})
    if MAKE_VISUALIZATION:
        visualization.visualize_masks(f'{INPUT}/{vp}', result, OUTPUT_PATH)
for params in empty_files:
    save_empty_file(*params)
cs.post_progress({"stage": STAGE3, "progress": 100})
cs.post_end()
logger.info(f'Result file generation took {time.time() - start_time} seconds')
logger.info(f'The whole process took {time.time() - run_time} seconds, clip-es mode = {CLIPES_MODE}')
