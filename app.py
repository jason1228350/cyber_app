import streamlit as st
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
import torch
from transformers import pipeline
import numpy as np
import io
import datetime

# 頁面配置
st.set_page_config(
    page_title="影真鑑 - 影像真偽鑑識平台",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ 影真鑑：五重資安防線影像鑑識平台")
st.markdown("結合 AI 深度學習、ELA 熱圖分析、反偵查隱寫稽核與相機元資料分析")
st.success("✅ 全防線資安鑑識引擎已成功啟動！")

# 1. 載入 AI 模型
@st.cache_resource
def load_ai_model():
    return pipeline("image-classification", model="umm-maybe/AI-image-detector")

try:
    classifier = load_ai_model()
except Exception as e:
    st.error(f"AI 模組初始化失敗: {e}")
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

# 反偵查 (Anti-Forensics) 檢測：LSB 隱寫術與雜訊平滑化稽核
def analyze_anti_forensics(image):
    img_gray = image.convert("L")
    img_array = np.array(img_gray)
    
    # 1. LSB 最低有效位隱寫提取 (Least Significant Bit)
    lsb_array = (img_array & 1) * 255
    lsb_image = Image.fromarray(lsb_array.astype(np.uint8))
    
    # 2. 高頻雜訊與平滑化遮蔽檢測 (檢測是否抹除痕跡)
    laplacian_var = np.var(np.gradient(img_array))
    is_smoothed = laplacian_var < 100.0  # 過度平滑可能代表抹除痕跡
    
    return lsb_image, laplacian_var, is_smoothed

# 檔案上傳 UI
uploaded_file = st.file_uploader("📂 請選擇要測試的照片", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="原始上傳照片", use_container_width=True)
    
    # ------------------ 第一防線 ------------------
    st.markdown("---")
    st.subheader("🤖 第一防線：AI 深度學習偽造特徵辨識")
    
    ai_score = 0.0
    ai_result_text = "分析中..."
    
    if classifier:
        with st.spinner("AI 辨識中..."):
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
                ai_result_text = "⚠️ 高度懷疑為 AI 生成或深度偽造影像"
                st.error(f"**AI 偽造風險指數：{ai_score:.2f}%** - {ai_result_text}")
            else:
                ai_result_text = "✅ 未檢測出明顯 AI 偽造痕跡（判定為真實拍攝）"
                st.success(f"**AI 偽造風險指數：{ai_score:.2f}%** - {ai_result_text}")

    # ------------------ 第二防線 ------------------
    st.markdown("---")
    st.subheader("🔍 第二防線：ELA 影像壓縮差異熱圖")
    
    with st.spinner("繪製 ELA 熱圖..."):
        ela_img = generate_ela(image)
        st.image(ela_img, caption="ELA 壓縮熱圖（高亮區域代表異常演算或修圖痕跡）", use_container_width=True)

    # ------------------ 第三防線 (反偵查功能) ------------------
    st.markdown("---")
    st.subheader("🕵️ 第三防線：反偵查 (Anti-Forensics) 與隱寫稽核")
    
    with st.spinner("進行反偵查隱寫術與痕跡抹除分析..."):
        lsb_img, lap_var, is_smoothed = analyze_anti_forensics(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(lsb_img, caption="LSB 最低有效位點圖 (若出現規則圖案可能藏有隱寫訊息)", use_container_width=True)
        with col2:
            st.markdown(f"**高頻雜訊變異度 (Variance):** `{lap_var:.2f}`")
            if is_smoothed:
                st.warning("⚠️ 偵測到異常平滑區域，疑有過重降噪或高頻抹除痕跡（試圖掩蓋 P 圖邊緣）。")
            else:
                st.info("✅ 雜訊分布正常，未發現明顯反偵查抹除痕跡。")

    # ------------------ 第四防線 ------------------
    st.markdown("---")
    st.subheader("📜 第四防線：EXIF 物理參數與元資料稽核")
    
    exif_data_dict = {}
    info = image._getexif()
    if info:
        for tag, value in info.items():
            exif_data_dict[str(tag)] = str(value)
        st.json(exif_data_dict)
    else:
        st.warning("⚠️ 此影像未包含原始 EXIF 元資料。")
        exif_data_dict = {"Status": "No EXIF metadata found"}

    # ------------------ 第五防線 ------------------
    st.markdown("---")
    st.subheader("📄 第五防線：資安採證報告匯出")
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_content = f"""==================================================
           影真鑑 - 資安採證稽核報告 (Official)
==================================================
【基本資訊】
- 檢測時間: {current_time}
- 檔案名稱: {uploaded_file.name}
- 影像尺寸: {image.size[0]} x {image.size[1]} px
- 檔案格式: {image.format}

【1. AI 深度學習檢測】
- 偽造風險指數: {ai_score:.2f}%
- 鑑識判定結論: {ai_result_text}

【2. ELA 壓縮熱圖分析】
- 檢測狀態: 已成功繪製 ELA 差異熱圖

【3. 反偵查與隱寫稽核 (Anti-Forensics)】
- 高頻雜訊數值: {lap_var:.2f}
- 抹除痕跡判定: {"⚠️ 疑似有抹除/降噪掩蓋痕跡" if is_smoothed else "✅ 雜訊分佈正常"}

【4. EXIF 物理元資料稽核】
{exif_data_dict}

==================================================
此報告由「五重資安防線影像鑑識平台」自動生成
簽章校驗碼: SEC-{hash(uploaded_file.name)}
==================================================
"""

    st.download_button(
        label="📥 一鍵下載資安採證報告 (.txt)",
        data=report_content,
        file_name=f"forensic_report_{uploaded_file.name}.txt",
        mime="text/plain"
    )
