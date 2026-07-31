import streamlit as st
from PIL import Image, ImageChops, ImageEnhance, ExifTags
import torch
from transformers import pipeline
import numpy as np
import io
import datetime

# 頁面配置
st.set_page_config(
    page_title="「狗」眼看真偽 - 影像真偽鑑定平台",
    page_icon="🐶",
    layout="wide"
)

# 主題標頭
st.title("🐶「狗」眼看真偽：基於 AI 檢測與 ELA 熱圖之影像真偽鑑定平台")
st.markdown("##### 🐾 資安警犬隊出動！結合 AI 嗅覺辨識、ELA 熱圖顯影與相機數位履歷搜查")
st.success("✅ 資安警犬鑑識引擎已成功啟動，隨時準備進行影像嗅檢！")

# 1. 載入 AI 模型
@st.cache_resource
def load_ai_model():
    return pipeline("image-classification", model="umm-maybe/AI-image-detector")

try:
    classifier = load_ai_model()
except Exception as e:
    st.error(f"AI 模型載入中: {e}")
    classifier = None

# ELA 熱圖產生器
def generate_ela(image, quality=90):
    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    ela_im = ImageChops.difference(image, compressed_image)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema]) if extrema else 1
    scale = 255.0 / (max_diff if max_diff > 0 else 1)
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale * 1.8)
    return ela_im

# 反偵查檢測 (LSB 與平滑化)
def analyze_anti_forensics(image):
    img_gray = image.convert("L")
    img_array = np.array(img_gray)
    lsb_array = (img_array & 1) * 255
    lsb_image = Image.fromarray(lsb_array.astype(np.uint8))
    laplacian_var = np.var(np.gradient(img_array))
    is_smoothed = laplacian_var < 100.0
    return lsb_image, laplacian_var, is_smoothed

# EXIF 中文映射表
EXIF_ZH_MAP = {
    "Make": "拍攝裝置品牌",
    "Model": "拍攝裝置型號",
    "DateTime": "檔案修改時間",
    "DateTimeOriginal": "照片原始拍攝時間",
    "DateTimeDigitized": "照片數位化時間",
    "Software": "處理軟體 / 來源系統",
    "ColorSpace": "色彩空間格式",
    "ExifImageWidth": "照片寬度 (像素)",
    "ExifImageHeight": "照片高度 (像素)"
}
IGNORE_TAGS = ["ExifOffset", "MakerNote", "UserComment", "GPSInfo"]

# 檔案上傳 UI
uploaded_file = st.file_uploader("📂 請上傳待鑑定之照片檔案", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 待鑑定原始照片", use_container_width=True)
    
    # ------------------ 第一防线 ------------------
    st.markdown("---")
    st.subheader("🐕 第一防線｜AI 鷹眼辨識（AI 深度偽造嗅檢）")
    
    ai_score = 0.0
    ai_result_text = "分析中..."
    
    if classifier:
        with st.spinner("警犬正在進行 AI 特徵嗅檢中..."):
            predictions = classifier(image)
            fake_score = 0.0
            for pred in predictions:
                if pred['label'].lower() in ['fake', 'artificial', 'ai-generated']:
                    fake_score = pred['score'] * 100
                elif pred['label'].lower() in ['real', 'human']:
                    fake_score = (1.0 - pred['score']) * 100
            
            if fake_score < 5.0:
                fake_score = 0.0
            ai_score = fake_score
            
            if ai_score > 60:
                ai_result_text = "⚠️ 警報！高度懷疑為 AI 生成或 Deepfake 偽造圖片"
                st.error(f"**🐕 警犬嗅檢風險指數：{ai_score:.2f}%** — {ai_result_text}")
            else:
                ai_result_text = "✅ 正常！未發現明顯 AI 生成痕跡（判定為真實拍攝）"
                st.success(f"**🐕 警犬嗅檢風險指數：{ai_score:.2f}%** — {ai_result_text}")

    # ------------------ 第二防线 ------------------
    st.markdown("---")
    st.subheader("🔍 第二防線｜ELA 靈犬顯影（修圖壓縮差異熱圖）")
    
    with st.spinner("警犬正在進行 ELA 壓縮顯影..."):
        ela_img = generate_ela(image)
        st.image(ela_img, caption="ELA 熱圖顯影（發亮高亮區域代表局部改圖、合成或修圖痕跡）", use_container_width=True)

    # ------------------ 第三防线 ------------------
    st.markdown("---")
    st.subheader("🕵️ 第三防線｜反偵查隱寫追蹤（痕跡抹除檢測）")
    
    with st.spinner("警犬正在追蹤是否有故意抹除痕跡..."):
        lsb_img, lap_var, is_smoothed = analyze_anti_forensics(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(lsb_img, caption="LSB 最低有效位點陣圖 (若出現規律圖樣代表藏有隱寫訊息)", use_container_width=True)
        with col2:
            st.markdown(f"**高頻雜訊變異度:** `{lap_var:.2f}`")
            if is_smoothed:
                st.warning("⚠️ 偵測到異常平滑區域，疑有過度降噪或磨皮來掩蓋 P 圖邊緣。")
            else:
                st.info("✅ 雜訊分佈自然，未發現故意抹除痕跡。")

    # ------------------ 第四防线 ------------------
    st.markdown("---")
    st.subheader("📜 第四防線｜相機履歷搜查（中文 EXIF 數位元資料）")
    
    exif_data_dict = {}
    info = image._getexif()
    if info:
        for tag_id, value in info.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            if tag_name in IGNORE_TAGS or str(tag_id) in IGNORE_TAGS:
                continue
            zh_name = EXIF_ZH_MAP.get(tag_name, tag_name)
            
            val_str = str(value)
            if tag_name == "ColorSpace":
                val_str = "sRGB 標準色彩空間" if str(value) == "1" else "標準色彩格式"
            elif "Screenshot" in val_str:
                val_str = "螢幕截圖照片 (Screenshot)"
            elif tag_name in ["ExifImageWidth", "PixelXDimension"]:
                val_str = f"{value} 像素 (px)"
            elif tag_name in ["ExifImageHeight", "PixelYDimension"]:
                val_str = f"{value} 像素 (px)"
                
            exif_data_dict[zh_name] = val_str
            
        st.json(exif_data_dict)
    else:
        st.warning("⚠️ 未偵測到原始相機 EXIF 元資料（可能為轉傳照片、網路下載或螢幕截圖）。")
        exif_data_dict = {"狀態說明": "未偵測到原始相機拍攝紀錄"}

    # ------------------ 第五防线 ------------------
    st.markdown("---")
    st.subheader("📄 第五防線｜資安採證報告匯出")
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_content = f"""==================================================
      「狗」眼看真偽 - 影像真偽鑑定報告 (Official)
==================================================
【照片基本資料】
- 檢測時間: {current_time}
- 檔案名稱: {uploaded_file.name}
- 照片尺寸: {image.size[0]} x {image.size[1]} px
- 檔案格式: {image.format}

【1. AI 鷹眼辨識結果】
- 偽造風險指數: {ai_score:.2f}%
- 警犬鑑定結論: {ai_result_text}

【2. ELA 靈犬顯影熱圖】
- 檢測狀態: 已成功繪製 ELA 差異熱圖

【3. 反偵查痕跡追蹤】
- 雜訊變異數值: {lap_var:.2f}
- 抹除痕跡判定: {"⚠️ 疑有降噪抹除邊緣痕跡" if is_smoothed else "✅ 雜訊分佈自然"}

【4. 相機履歷搜查 (EXIF)】
{exif_data_dict}

==================================================
此報告由「『狗』眼看真偽 影像真偽鑑定平台」自動生成
==================================================
"""

    st.download_button(
        label="📥 一鍵下載「狗」眼看真偽採證報告 (.txt)",
        data=report_content,
        file_name=f"狗眼看真偽_採證報告_{uploaded_file.name}.txt",
        mime="text/plain"
    )
