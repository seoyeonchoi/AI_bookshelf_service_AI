from google.colab import drive
import ultralytics
import time
import os
import io
import cv2
import sys
import numpy as np
# import matplotlib.pyplot as plt
# import platform
# import types
from ultralytics import YOLO
from PIL import Image#, ImageFont, ImageDraw
from google.cloud import vision
from flask import Flask, request, jsonify
import nest_asyncio
import base64
import shutil
from pyngrok import ngrok
from flask_cors import CORS
# from multiprocessing import Pool

drive.mount('/content/drive')
sys.path.append("..")
ultralytics.checks()

# google cloud vision API access
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/content/drive/MyDrive/big project/yolov8/alpine-practice-390604-5960e4078cde.json'
client_options = {'api_endpoint': 'eu-vision.googleapis.com'}
client = vision.ImageAnnotatorClient(client_options=client_options)

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}}, supports_credentials=True)
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 최대 이미지 크기 설정 (10MB)
model = YOLO('/content/drive/MyDrive/big project/yolov8/runs/segment/augment2/weights/aug_160.pt')
second_model = YOLO('/content/drive/MyDrive/big project/yolov8/runs/title/train/mistake_crop_300/weights/coin_best.pt')
def clear_directories():
    if os.path.exists('/content/request'):
        shutil.rmtree('/content/request')
    if os.path.exists('/content/runs'):
        shutil.rmtree('/content/runs')
    if os.path.exists('/content/cropped_images'):
        shutil.rmtree('/content/cropped_images')

def parse_labels(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()

    labels = []
    for line in lines:
        # class와 좌표들을 분리합니다
        label_parts = line.split()
        class_label = label_parts[0]
        coordinates = list(map(float, label_parts[1:]))
        labels.append((class_label, coordinates))

    return labels

def crop_polygon(img, coordinates):
    # label을 클래스와 좌표로 분리합니다
    coordinates = list(map(float, coordinates))

    # 좌표를 실제 픽셀 좌표로 변환합니다
    h, w = img.shape[:2]
    coordinates = np.array([(int(x * w), int(y * h)) for x, y in zip(coordinates[::2], coordinates[1::2])])

    # 폴리곤에 해당하는 마스크를 만듭니다
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [coordinates], (255,)*img.shape[2])

    # 마스크를 이용하여 이미지를 잘라냅니다
    cropped_img = np.where(mask==255, img, 0)

    # Bounding rectangle를 찾습니다
    x, y, w, h = cv2.boundingRect(coordinates.reshape(-1, 1, 2).astype(np.int32))
    cropped_img = cropped_img[y:y+h, x:x+w]

    return cropped_img

def save_image_file(image_file, save_dir):
    filename = image_file.filename
    file_path = os.path.join(save_dir, filename)

    # 이미지 크기 체크
    image_data = image_file.read()
    image_size = len(image_data)
    # print('이미지 사이즈:', image_size)
    # print('제한 사이즈:', MAX_IMAGE_SIZE)
    if image_size > MAX_IMAGE_SIZE:
        raise Exception(f"Image file {filename} is too large.")

    with open(file_path, "wb") as f:
        f.write(image_data)
    if not os.path.exists(file_path):
        raise Exception(f"Failed to save the image file {filename}.")

    return file_path

@app.route('/')
def hello_world():
    return "Hello Flask and Ngrok!"

@app.route("/img2title/", methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.getlist("image"):
        start_time = time.time()
        save_dir = "/content/request"
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs('/content/cropped_images', exist_ok=True)
        file_paths = []
        for image_file in request.files.getlist("image"):
            print(image_file)
            file_path = save_image_file(image_file, save_dir)
            file_paths.append(file_path)
    results = model.predict(source=save_dir,
                            conf=0.7,
                            iou=0.4,
                            save=True,
                            save_txt=True,
                            name='model_result',
                            device='cpu',
                            retina_masks=True,
                            show_labels=False,
                            show_conf=False,
                            boxes=False,
                            )

    responses = {"data": {}, "segment_images": {}, "success": 1}
    id_counter = 1
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        filename_without_ext = os.path.splitext(filename)[0]
        # detected image show
        # Possible extensions
        extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        # Check for all possible extensions
        segment_image = None
        for ext in extensions:
            potential_path = f'/content/runs/segment/model_result/{filename_without_ext}{ext}'
            if os.path.exists(potential_path):
                segment_image = potential_path
                # img2txt by base64 encoding
                with open(segment_image, "rb") as f:
                    segment_image_base64 = base64.b64encode(f.read()).decode()
                responses["segment_images"][filename_without_ext] = segment_image_base64
                break

        # If no image was found
        if segment_image is None:
            print(f"No segmented image found for {filename_without_ext}. Skipping this image.")
            continue

        img = cv2.imread(file_path)
        label_path = f'/content/runs/segment/model_result/labels/{filename_without_ext}.txt'
        labels = parse_labels(label_path)
        cropped_images = []

        for i, label in enumerate(labels):
            class_label, coordinates = label
            cropped_img = crop_polygon(img, coordinates)
            cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            cropped_images.append(cropped_img_rgb)
            cv2.imwrite(f'/content/cropped_images/{filename_without_ext}_{i}.jpg', cv2.cvtColor(cropped_img_rgb, cv2.COLOR_RGB2BGR))
        # second yolo model
    second_results = second_model.predict(source='/content/cropped_images/',
                                        conf=0.7,
                                        iou=0.4,
                                        save=True,
                                        save_txt=True,
                                        name='second_model_result',
                                        device='cpu',
                                        retina_masks=True,
                                        show_labels=False,
                                        show_conf=False,
                                        boxes=False,
                                        )

    image_dir = "/content/runs/segment/second_model_result"
    label_dir = "/content/runs/segment/second_model_result/labels"

    # 잘린 이미지를 저장할 리스트 초기화
    cropped_titles = []

    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg"):
            img_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt"))

            # 이미지 로드
            img = cv2.imread(img_path)

            # 라벨 파일이 존재하는 경우
            if os.path.exists(label_path):
                with open(label_path, "r") as file:
                    for line in file:
                        # 라벨 파싱
                        class_name, *coordinates = line.strip().split()
                        coordinates = list(map(float, coordinates))

                        # 이미지 자르기
                        cropped_img = crop_polygon(img, coordinates)

                        # Grayscale 변환
                        cropped_img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

                        # 잘린 이미지를 리스트에 추가
                        cropped_titles.append(cropped_img_gray)
            # 라벨 파일이 존재하지 않는 경우
            else:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                cropped_titles.append(img)

    extracted_titles = []
    for i, data in enumerate(cropped_titles):
        image = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        image = vision.Image(content=cv2.imencode('.jpg', image)[1].tobytes())
        response = client.text_detection(image=image)
        texts = response.text_annotations
        if not texts:  # texts가 비어있는 경우
            print(f"No text detected in cropped image {i}. Skipping this image.")
            continue
        # text box별 정보 추출
        txt_blocks,title = [],''
        for text in texts[1:]:
            if text :
                ocr_text = text.description
                txt_blocks.append(ocr_text) # 검출한 text box의 폭,너비,글자 넣기
            else :
                title='no_title_in_this_segmentation'
                continue
        # title 추출(책등에서 제목이 가장 큰 글씨라고 가정)
        for txt in txt_blocks:
            title += txt
        print(txt_blocks)
        extracted_titles.append(title)

    for title in extracted_titles:
        responses["data"][id_counter] = title
        id_counter += 1

    clear_directories()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: {} seconds".format(elapsed_time))
    print(f"Received images: {len(file_paths)}")
    print(f"Detected books: {len(extracted_titles)}")
    print('data: ', responses['data'])
    # print(texts)
    # print(txt_blocks)
    return jsonify(responses)

ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
app.run(host="0.0.0.0", port=8000)