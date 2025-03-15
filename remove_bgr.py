import tkinter as tk
from tkinter import filedialog
from rembg import remove
import cv2
import numpy as np
import os

def select_files():
    file_paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.webp")])
    if file_paths:
        entry_files.delete(0, tk.END)
        entry_files.insert(0, ", ".join(file_paths))

def select_output_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        entry_output.delete(0, tk.END)
        entry_output.insert(0, folder_path)

def remove_background():
    input_paths = entry_files.get().split(", ")
    output_folder = entry_output.get()
    save_as_png = var_png.get()
    save_as_jpg = var_jpg.get()

    if not input_paths or not output_folder or (not save_as_png and not save_as_jpg):
        lbl_status.config(text="파일, 저장 경로, 변환 형식을 선택하세요!", fg="red")
        return

    try:
        for input_path in input_paths:
            with open(input_path, 'rb') as inp_file:
                img_data = inp_file.read()
            
            output = remove(img_data)
            nparr = np.frombuffer(output, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            
            img = refine_alpha_channel(img)
            img = remove_object_outline(img)

            file_name = os.path.basename(input_path).split('.')[0]
            
            if save_as_png:
                output_path_png = os.path.join(output_folder, f"{file_name}_no_bg.png")
                cv2.imwrite(output_path_png, img)
            
            if save_as_jpg:
                img_jpg = convert_to_white_background(img)
                output_path_jpg = os.path.join(output_folder, f"{file_name}_no_bg.jpg")
                cv2.imwrite(output_path_jpg, img_jpg, [cv2.IMWRITE_JPEG_QUALITY, 95])

        lbl_status.config(text=f"배경 제거 완료! {len(input_paths)}개 파일 처리 완료.", fg="green")
    except Exception as e:
        lbl_status.config(text=f"오류 발생: {e}", fg="red")

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

# GUI 설정
root = tk.Tk()
root.title("고기 배경 제거 프로그램")
root.geometry("500x400")

tk.Label(root, text="이미지 파일 선택 (여러 개 가능):").pack()
entry_files = tk.Entry(root, width=60)
entry_files.pack()
tk.Button(root, text="파일 찾기", command=select_files).pack()

tk.Label(root, text="저장할 폴더 선택:").pack()
entry_output = tk.Entry(root, width=60)
entry_output.pack()
tk.Button(root, text="폴더 선택", command=select_output_folder).pack()

# 출력 형식 선택
var_png = tk.BooleanVar(value=True)
var_jpg = tk.BooleanVar(value=False)
tk.Checkbutton(root, text="PNG로 저장 (투명 배경 유지)", variable=var_png).pack()
tk.Checkbutton(root, text="JPG로 저장 (흰색 배경 적용)", variable=var_jpg).pack()

tk.Button(root, text="배경 제거 실행", command=remove_background, bg="blue", fg="white").pack(pady=10)

lbl_status = tk.Label(root, text="", fg="black")
lbl_status.pack()

root.mainloop()
