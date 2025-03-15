import streamlit as st
import numpy as np
import cv2
import zipfile
import os
from rembg import remove
from io import BytesIO
from PIL import Image

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
        a = cv2.GaussianBlur(a, (11, 11), 0)
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

# Streamlit UI
st.title("🔥 배경 제거 프로그램 (Streamlit)")

# 파일 업로드
uploaded_files = st.file_uploader("이미지 파일 업로드 (여러 개 가능)", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

# 변환 옵션
save_as_png = st.checkbox("PNG로 저장 (투명 배경 유지)", value=True)
save_as_jpg = st.checkbox("JPG로 저장 (흰색 배경 적용)", value=False)

processed_files = []

# 배경 제거 실행
if st.button("배경 제거 실행"):
    if uploaded_files:
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for uploaded_file in uploaded_files:
                try:
                    image = Image.open(uploaded_file).convert("RGBA")
                    img_data = uploaded_file.getvalue()
                    
                    output = remove(img_data)
                    nparr = np.frombuffer(output, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

                    img = refine_alpha_channel(img)
                    img = remove_object_outline(img)

                    file_name = uploaded_file.name.rsplit('.', 1)[0]

                    if save_as_png:
                        png_bytes = cv2.imencode(".png", img)[1].tobytes()
                        zip_file.writestr(f"{file_name}_no_bg.png", png_bytes)
                        processed_files.append((f"{file_name}_no_bg.png", png_bytes))

                    if save_as_jpg:
                        img_jpg = convert_to_white_background(img)
                        jpg_bytes = cv2.imencode(".jpg", img_jpg, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
                        zip_file.writestr(f"{file_name}_no_bg.jpg", jpg_bytes)
                        processed_files.append((f"{file_name}_no_bg.jpg", jpg_bytes))

                    st.success(f"✅ {file_name} 배경 제거 완료!")

                except Exception as e:
                    st.error(f"❌ 오류 발생: {e}")

        zip_buffer.seek(0)
        st.download_button(
            label="📦 모든 파일 ZIP으로 다운로드",
            data=zip_buffer,
            file_name="processed_images.zip",
            mime="application/zip"
        )

    else:
        st.warning("⚠️ 파일을 업로드하세요!")
