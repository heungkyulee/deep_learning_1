# C:\Users\Lee\Desktop\2025\deeplearning1\ProtoViT\prepare_stanford_dogs.py
import os
import xml.etree.ElementTree as ET
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm

# --- 설정값 (사용자 환경에 맞게 수정) ---
IMAGE_ROOT_DIR = r"C:\Users\Lee\Downloads\archive\images\Images"
ANNOTATION_ROOT_DIR = r"C:\Users\Lee\Downloads\Annotation"
OUTPUT_DATASET_DIR = r"./datasets/stanford_dogs_cropped"  # 프로젝트 내 datasets 폴더 하위
TRAIN_DIR = os.path.join(OUTPUT_DATASET_DIR, "train_cropped")
TEST_DIR = os.path.join(OUTPUT_DATASET_DIR, "test_cropped")
IMAGE_SIZE = (224, 224)  # ProtoViT settings.py의 img_size와 일치
TEST_SPLIT_RATIO = 0.2 # 테스트셋 비율
RANDOM_SEED = 42
# --- 설정값 끝 ---

def parse_annotation(annotation_path):
    """XML Annotation 파일을 파싱하여 bounding box와 클래스 이름을 반환"""
    try:
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        filename = root.find('filename').text
        # 클래스 이름은 object/name 또는 folder에서 가져올 수 있습니다.
        # 여기서는 object/name을 사용합니다. folder는 n02108915-French_bulldog 형태일 수 있습니다.
        class_name_tag = root.find('object/name')
        if class_name_tag is None or not class_name_tag.text: # object/name이 없는 경우 folder 사용
            folder_name_full = root.find('folder').text
            # "n02108915-French_bulldog" -> "French_bulldog"
            class_name = folder_name_full.split('-', 1)[1] if '-' in folder_name_full else folder_name_full
        else:
            class_name = class_name_tag.text.replace(' ', '_') # 공백을 언더스코어로 변경

        bndbox = root.find('object/bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        return filename, class_name, (xmin, ymin, xmax, ymax)
    except Exception as e:
        print(f"Error parsing {annotation_path}: {e}")
        return None, None, None

def crop_and_save_image(image_path, bbox, output_path, class_name_dir):
    """이미지를 crop하고 지정된 경로에 저장"""
    try:
        img = Image.open(image_path)
        cropped_img = img.crop(bbox)
        cropped_img = cropped_img.resize(IMAGE_SIZE) # ProtoViT 입력 크기에 맞게 리사이즈

        # 클래스별 디렉토리 생성
        class_output_dir = os.path.join(output_path, class_name_dir)
        if not os.path.exists(class_output_dir):
            os.makedirs(class_output_dir)

        # 파일명은 원본 파일명(확장자 포함)을 그대로 사용하거나, 이미지 ID만 사용할 수 있음
        # 여기서는 원본 파일명(또는 이미지 ID)을 그대로 사용한다고 가정
        file_name_only = os.path.basename(image_path) # 확장자 포함
        # 만약 annotation의 filename이 확장자 없이 ID만 있다면 아래처럼
        # file_name_only = os.path.splitext(os.path.basename(image_path))[0] + ".jpg" # 또는 원본 확장자

        save_to = os.path.join(class_output_dir, file_name_only)
        cropped_img.save(save_to)
        return True
    except FileNotFoundError:
        print(f"Image not found: {image_path}")
        return False
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def main():
    print("Starting Stanford Dogs dataset preparation...")

    # 출력 디렉토리 생성
    if not os.path.exists(OUTPUT_DATASET_DIR):
        os.makedirs(OUTPUT_DATASET_DIR)
    if not os.path.exists(TRAIN_DIR):
        os.makedirs(TRAIN_DIR)
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)

    all_image_files = [] # (image_path, annotation_path, class_name_raw)
    class_names_set = set()

    print("Scanning annotations and images...")
    # Annotation 폴더를 기준으로 이미지와 매칭
    for class_folder in tqdm(os.listdir(ANNOTATION_ROOT_DIR), desc="Processing class folders"):
        annotation_class_path = os.path.join(ANNOTATION_ROOT_DIR, class_folder)
        image_class_path = os.path.join(IMAGE_ROOT_DIR, class_folder) # 이미지 폴더도 동일한 구조라고 가정

        if os.path.isdir(annotation_class_path) and os.path.isdir(image_class_path):
            for annotation_file in os.listdir(annotation_class_path):
                if annotation_file.endswith(".xml") or True: # ProCore Class 파일도 XML 형식일 가능성이 높음
                    annotation_full_path = os.path.join(annotation_class_path, annotation_file)
                    
                    # Annotation 파일명(확장자 제외)과 동일한 이미지 파일명(JPG)을 찾음
                    image_id = os.path.splitext(annotation_file)[0]
                    image_full_path = os.path.join(image_class_path, image_id + ".jpg") # JPG 확장자로 가정

                    if os.path.exists(image_full_path):
                        # class_folder가 "n02108915-French_bulldog"와 같은 형식이므로, 여기서 클래스 이름 추출
                        # 또는 annotation 내부의 <name> 태그 사용 (parse_annotation에서 처리)
                        class_name_from_folder = class_folder.split('-', 1)[1] if '-' in class_folder else class_folder
                        class_names_set.add(class_name_from_folder.replace(' ', '_'))
                        all_image_files.append((image_full_path, annotation_full_path, class_name_from_folder.replace(' ', '_')))
                    else:
                        print(f"Warning: Image for annotation {annotation_full_path} not found at {image_full_path}")

    print(f"Found {len(all_image_files)} image-annotation pairs.")
    print(f"Found {len(class_names_set)} unique classes.")

    # Train/Test 분할 (클래스 비율 유지 - stratify)
    # sklearn.model_selection.train_test_split은 파일 경로 리스트와 레이블 리스트가 필요함
    image_paths = [item[0] for item in all_image_files]
    annotation_paths = [item[1] for item in all_image_files]
    class_labels_for_split = [item[2] for item in all_image_files] # 폴더명 기반 클래스

    if not all_image_files:
        print("No image files found to process. Exiting.")
        return

    train_files_info, test_files_info = train_test_split(
        list(zip(image_paths, annotation_paths, class_labels_for_split)),
        test_size=TEST_SPLIT_RATIO,
        random_state=RANDOM_SEED,
        stratify=class_labels_for_split # 클래스 분포 유지
    )

    print(f"Training set size: {len(train_files_info)}")
    print(f"Test set size: {len(test_files_info)}")

    # 학습 데이터 처리 및 저장
    print("\nProcessing and saving training images...")
    for img_path, ann_path, class_name_dir in tqdm(train_files_info, desc="Train images"):
        filename, class_name_ann, bbox = parse_annotation(ann_path)
        if filename and bbox:
            # 클래스 이름은 폴더 기반(class_name_dir)을 사용할지, annotation 내부(class_name_ann)를 사용할지 결정
            # 여기서는 일관성을 위해 폴더 기반의 class_name_dir 사용
            crop_and_save_image(img_path, bbox, TRAIN_DIR, class_name_dir)

    # 테스트 데이터 처리 및 저장
    print("\nProcessing and saving test images...")
    for img_path, ann_path, class_name_dir in tqdm(test_files_info, desc="Test images"):
        filename, class_name_ann, bbox = parse_annotation(ann_path)
        if filename and bbox:
            crop_and_save_image(img_path, bbox, TEST_DIR, class_name_dir)

    print("\nDataset preparation finished!")
    print(f"Cropped training images saved to: {TRAIN_DIR}")
    print(f"Cropped test images saved to: {TEST_DIR}")
    print(f"Total classes: {len(os.listdir(TRAIN_DIR))}") # 실제 생성된 클래스 폴더 수 확인

if __name__ == "__main__":
    main()