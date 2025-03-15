import streamlit as st
import numpy as np
import cv2
from rembg import remove
from io import BytesIO
from PIL import Image

def refine_alpha_channel(img):
    if img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        _, alpha_mask = cv2.threshold(a, 220, 255, cv2.THRESH_BINARY)  # 임계값 상향 조정
        kernel = np.ones((3, 3), np.uint8)
        alpha_mask = cv2.morphologyEx(alpha_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        alpha_mask = cv2.morphologyEx(alpha_mask, cv2.MORPH_CLOSE, kernel, iterations=3)  # 내부 배경 제거
        a[alpha_mask == 0] = 0
        a[a < 220] = 0  # 잔여 배경 강제 제거
        a = cv2.erode(a, kernel, iterations=2)  # 경계 부드럽게
        a = cv2.GaussianBlur(a, (11, 11), 0)  # 블러 강도를 높임
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
    """ 투명 배경을 흰색 배경으로 변경 """
    if img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        white_bg = np.full((img.shape[0], img.shape[1], 3), 255, dtype=np.uint8)
        mask = a > 50  # 투명도가 낮은 부분까지 포함
        img_rgb = cv2.merge([b, g, r])
        img_rgb = cv2.addWeighted(img_rgb, 0.95, white_bg, 0.05, 0)  # 경계 블렌딩
        white_bg[mask] = img_rgb[mask]  # 객체만 유지하고 배경을 흰색으로 채움
        return white_bg
    return img

# Streamlit UI 구성
st.title("🔥 배경 제거 프로그램 (Streamlit)")

# 파일 업로드
uploaded_files = st.file_uploader("이미지 파일 업로드 (여러 개 가능)", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

# 변환 옵션
save_as_png = st.checkbox("PNG로 저장 (투명 배경 유지)", value=True)
save_as_jpg = st.checkbox("JPG로 저장 (흰색 배경 적용)", value=False)

# 변환 실행
if st.button("배경 제거 실행"):
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                # 이미지 읽기
                image = Image.open(uploaded_file).convert("RGBA")
                img_data = uploaded_file.getvalue()
                
                # 배경 제거
                output = remove(img_data)
                nparr = np.frombuffer(output, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

                # 후처리 (알파 채널 개선 및 아웃라인 제거)
                img = refine_alpha_channel(img)
                img = remove_object_outline(img)

                # 파일 이름 처리
                file_name = uploaded_file.name.rsplit('.', 1)[0]

                # PNG 저장 (투명 배경 유지)
                if save_as_png:
                    png_bytes = cv2.imencode(".png", img)[1].tobytes()
                    st.download_button(
                        label=f"{file_name}_no_bg.png 다운로드",
                        data=BytesIO(png_bytes),
                        file_name=f"{file_name}_no_bg.png",
                        mime="image/png"
                    )

                # JPG 저장 (흰색 배경 적용)
                if save_as_jpg:
                    img_jpg = convert_to_white_background(img)
                    jpg_bytes = cv2.imencode(".jpg", img_jpg, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
                    st.download_button(
                        label=f"{file_name}_no_bg.jpg 다운로드",
                        data=BytesIO(jpg_bytes),
                        file_name=f"{file_name}_no_bg.jpg",
                        mime="image/jpeg"
                    )

                st.success(f"✅ {file_name} 배경 제거 완료!")

            except Exception as e:
                st.error(f"❌ 오류 발생: {e}")

    else:
        st.warning("⚠️ 파일을 업로드하세요!")
