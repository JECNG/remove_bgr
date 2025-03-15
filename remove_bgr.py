import streamlit as st
from rembg import remove
import cv2
import numpy as np
import os
import io
from PIL import Image

def remove_background(image):
    img_data = io.BytesIO()
    image.save(img_data, format="PNG")
    output = remove(img_data.getvalue())
    nparr = np.frombuffer(output, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    img = refine_alpha_channel(img)
    img = remove_object_outline(img)
    return img

def refine_alpha_channel(img):
    if img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        _, alpha_mask = cv2.threshold(a, 220, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        alpha_mask = cv2.morphologyEx(alpha_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        alpha_mask = cv2.morphologyEx(alpha_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        a[alpha_mask == 0] = 0
        a[a < 220] = 0
        a = cv2.erode(a, kernel, iterations=2)
        a = cv2.GaussianBlur(a, (7, 7), 0)
        img = cv2.merge([b, g, r, a])
    return img

def remove_object_outline(img):
    if img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        gray = cv2.cvtColor(cv2.merge([b, g, r]), cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(a)
        cv2.drawContours(mask, contours, -1, (255), thickness=3)
        a[mask > 0] = 0
        img = cv2.merge([b, g, r, a])
    return img

def convert_to_white_background(img):
    if img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        white_bg = np.full((img.shape[0], img.shape[1], 3), 255, dtype=np.uint8)
        mask = a > 50
        img_rgb = cv2.merge([b, g, r])
        img_rgb = cv2.addWeighted(img_rgb, 0.95, white_bg, 0.05, 0)
        white_bg[mask] = img_rgb[mask]
        return white_bg
    return img

# Streamlit UI 설정
st.title("배경 제거 프로그램")

uploaded_files = st.file_uploader("이미지 파일 업로드 (여러 개 가능)", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

save_as_png = st.checkbox("PNG로 저장 (투명 배경 유지)", value=True)
save_as_jpg = st.checkbox("JPG로 저장 (흰색 배경 적용)", value=False)

if st.button("배경 제거 실행"):
    if not uploaded_files:
        st.error("파일을 업로드하세요!")
    else:
        output_folder = "배경 제거"
        os.makedirs(output_folder, exist_ok=True)
        
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGBA")
            processed_img = remove_background(image)
            file_name, ext = os.path.splitext(uploaded_file.name)
            
            if save_as_png:
                output_path = os.path.join(output_folder, file_name + ".png")
                cv2.imwrite(output_path, processed_img)
                st.success(f"저장 완료: {output_path}")
            
            if save_as_jpg:
                jpg_img = convert_to_white_background(processed_img)
                output_path = os.path.join(output_folder, file_name + ".jpg")
                cv2.imwrite(output_path, jpg_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                st.success(f"저장 완료: {output_path}")
        
        st.success(f"배경 제거 완료! {len(uploaded_files)}개 파일 처리 완료.")
        st.balloons()

st.info("배경이 제거된 이미지는 '배경 제거' 폴더에 자동 저장됩니다.")
