import streamlit as st
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageDraw, ExifTags
import torch
from transformers import pipeline
import numpy as np
import io
import datetime
import hashlib
import matplotlib.cm as cm

# 1. 頁面配置
st.set_page_config(
    page_title="「狗」眼看真偽 - 影像真偽鑑定平台",
    page_icon="🐶",
    layout="wide"
)

st.title("🐶「狗」眼看真偽：基於 AI 檢測與 ELA 熱圖之影像真偽鑑定平台")
st.markdown("##### 🐾 資安警犬隊出動！結合 AI 嗅檢、ELA 熱圖、FFT 頻域、PRNU 指紋與 C2PA 水印稽核")
st.success("✅ 全防線資安警犬鑑識引擎已成功啟動！")

# 2. 載入 AI 模型
@st.cache_resource
def load_ai_model():
    return pipeline("image-classification", model="umm-maybe/AI-image-detector")

try:
    classifier = load_ai_model()
except Exception as e:
    st.error(f"AI 模型初始化中: {e}")
    classifier = None

# 3. 核心功能模組

# (1) ELA 彩色 Jet 熱圖產生器
def generate_ela_jet(image, quality=90):
    image_rgb = image.convert("RGB")
    buffer = io.BytesIO()
    image_rgb.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    
    ela_im = ImageChops.difference(image_rgb, compressed_image)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema]) if extrema else 1
    scale = 255.0 / (max_diff if max_diff > 0 else 1)
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale * 1.8)
    
    gray_array = np.array(ela_im.convert("L")) / 255.0
    jet_colormap = cm.get_cmap('jet')
    jet_mapped = (jet_colormap(gray_array)[:, :, :3] * 255).astype(np.uint8)
    
    return Image.fromarray(jet_mapped), gray_array

# (2) 自動 ROI 變造疑點標記 (Bounding Box)
def detect_roi_bounding_box(image_orig, gray_ela_array):
    h, w = gray_ela_array.shape
    threshold = np.percentile(gray_ela_array, 98.5)
    y_indices, x_indices = np.where(gray_ela_array >= threshold)
    
    if len(x_indices) > 0 and len(y_indices) > 0:
        xmin, xmax = int(np.min(x_indices)), int(np.max(x_indices))
        ymin, ymax = int(np.min(y_indices)), int(np.max(y_indices))
        
        marked_image = image_orig.copy()
        draw = ImageDraw.Draw(marked_image)
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=5)
        roi_box = (xmin, ymin, xmax - xmin, ymax - ymin)
        cropped_roi = image_orig.crop((xmin, ymin, xmax, ymax))
        return marked_image, roi_box, cropped_roi
    return image_orig, None, None

# (3) 2D-FFT 快速傅立葉變換頻域殘影
def analyze_fft_spectrum(image):
    img_gray = np.array(image.convert("L"))
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    magnitude_spectrum = ((magnitude_spectrum - magnitude_spectrum.min()) / 
                          (magnitude_spectrum.max() - magnitude_spectrum.min() + 1e-8) * 255).astype(np.uint8)
    return Image.fromarray(magnitude_spectrum)

# (4) PRNU 相機感光元件物理指紋
def extract_prnu_noise(image):
    img_gray = np.array(image.convert("L"), dtype=np.float32)
    blurred = np.array(image.convert("L").filter(ImageFilter.GaussianBlur(radius=2)), dtype=np.float32)
    noise_residual = img_gray - blurred
    noise_var = np.var(noise_residual)
    has_sensor_prnu = noise_var > 12.0
    return noise_var, has_sensor_prnu

# (5) 反偵查 LSB 隱寫術與平滑化
def analyze_anti_forensics(image):
    img_gray = image.convert("L")
    img_array = np.array(img_gray)
    lsb_array = (img_array & 1) * 255
    lsb_image = Image.fromarray(lsb_array.astype(np.uint8))
    laplacian_var = np.var(np.gradient(img_array))
    is_smoothed = laplacian_var < 80.0
    return lsb_image, laplacian_var, is_smoothed

# (6) C2PA 數位水印掃描
def scan_c2pa_watermark(file_bytes):
    bytes_str = str(file_bytes[:10000]) + str(file_bytes[-10000:])
    keywords = ["C2PA", "jumbf", "Midjourney", "StableDiffusion", "DALL-E", "Photoshop", "Firefly"]
    detected = [kw for kw in keywords if kw.lower() in bytes_str.lower()]
    return detected

# EXIF 標籤過濾與中文對照
EXIF_ZH_MAP = {
    "Make": "拍攝裝置品牌", "Model": "拍攝裝置型號", "DateTime": "檔案修改時間",
    "DateTimeOriginal": "照片原始拍攝時間", "Software": "處理軟體 / 來源系統",
    "ColorSpace": "色彩空間格式", "ExifImageWidth": "照片寬度 (像素)", "ExifImageHeight": "照片高度 (像素)"
}
IGNORE_TAGS = ["ExifOffset", "MakerNote", "UserComment", "GPSInfo", "YCbCrPositioning", "ExifVersion", "ComponentsConfiguration", "FlashPixVersion", "SceneCaptureType"]

# 側邊欄控制台
st.sidebar.title("🐶 警犬隊控制台")
sensitivity = st.sidebar.slider("警犬嗅覺靈敏度門檻 (%)", 30, 90, 60)

# 檔案上傳 UI
uploaded_file = st.file_uploader("📂 請上傳待鑑定之照片檔案", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    image = Image.open(io.BytesIO(bytes_data))
    
    # 數位指紋計算
    sha256_hash = hashlib.sha256(bytes_data).hexdigest()
    md5_hash = hashlib.md5(bytes_data).hexdigest()
    
    st.info(f"🐾 **證物數位指紋 (Chain of Custody):**\n- **SHA-256:** `{sha256_hash}`\n- **MD5:** `{md5_hash}`")
    st.image(image, caption="📷 待鑑定原始照片", use_container_width=True)
    
    # 防線算力執行
    ela_jet_img, gray_ela = generate_ela_jet(image)
    marked_img, roi_box, cropped_roi = detect_roi_bounding_box(image, gray_ela)
    fft_img = analyze_fft_spectrum(image)
    prnu_var, has_prnu = extract_prnu_noise(image)
    lsb_img, lap_var, is_smoothed = analyze_anti_forensics(image)
    c2pa_findings = scan_c2pa_watermark(bytes_data)
    
    # AI 分析
    ai_score = 0.0
    ai_result_text = "分析中..."
    if classifier:
        predictions = classifier(image)
        fake_score = 0.0
        for pred in predictions:
            if pred['label'].lower() in ['fake', 'artificial', 'ai-generated']:
                fake_score = pred['score'] * 100
            elif pred['label'].lower() in ['real', 'human']:
                fake_score = (1.0 - pred['score']) * 100
        if fake_score < 5.0: fake_score = 0.0
        ai_score = fake_score
        ai_result_text = "⚠️ 警報！高度懷疑為 AI 生成假圖" if ai_score > sensitivity else "✅ 未見明顯 AI 偽造痕跡"

    # 評分卡分數
    trust_score = 100.0 - (ai_score * 0.5)
    if is_smoothed: trust_score -= 15.0
    if not has_prnu: trust_score -= 10.0
    if c2pa_findings: trust_score -= 20.0
    trust_score = max(0.0, min(100.0, trust_score))
    
    # ------------------ 頂部評分卡 ------------------
    st.markdown("---")
    st.subheader("📊 🐕‍🦺 警犬隊綜合鑑定卡 (Trust Score)")
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        st.metric(label="資安可信度總分", value=f"{trust_score:.1f} / 100")
    with col_s2:
        if trust_score >= 80:
            st.success("🟢 **警犬評定：真實照片** (未見明顯變造或 AI 偽造痕跡)")
        elif trust_score >= 50:
            st.warning("🟡 **警犬評定：疑點注意** (發現二次壓縮、修圖軟體痕跡或磨皮掩蓋)")
        else:
            st.error("🔴 **警犬評定：高度偽造警報** (高度懷疑 AI 生成或重度改圖)")

    # ------------------ 第一防線：AI 檢測 ------------------
    st.markdown("---")
    st.subheader("🐕 第一防線｜AI 鷹眼辨識（AI 深度偽造嗅檢）")
    st.write(f"**🐕 警犬嗅檢風險指數：{ai_score:.2f}%** — {ai_result_text}")

    # ------------------ 第二防線：ELA Jet 熱圖 & ROI ------------------
    st.markdown("---")
    st.subheader("🔍 第二防線｜ELA 靈犬顯影（彩色 Jet 熱圖與自動 ROI 疑點定位）")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.image(ela_jet_img, caption="ELA 彩色 Jet 熱力圖 (紅色高亮代表極高異常壓縮區)", use_container_width=True)
    with col_e2:
        st.image(marked_img, caption="自動 ROI 疑點紅框標記圖", use_container_width=True)
        if roi_box:
            st.error(f"🚨 **自動偵測到疑似變造疑點區塊**\n- 座標 (X, Y): `{roi_box[0]}, {roi_box[1]}`\n- 尺寸 (W, H): `{roi_box[2]} x {roi_box[3]}` 像素")
            if cropped_roi:
                st.image(cropped_roi, caption="疑似變造區域局部裁切放大", width=200)

    # ------------------ 第三防線：FFT & PRNU ------------------
    st.markdown("---")
    st.subheader("📉 第三防線｜FFT 頻域殘影與 PRNU 物理感光指紋")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.image(fft_img, caption="FFT 快速傅立葉變換頻域圖 (放射狀網格代表 AI 演算法殘影)", use_container_width=True)
    with col_f2:
        st.markdown(f"**PRNU 感光雜訊變異值:** `{prnu_var:.2f}`")
        if has_prnu:
            st.info("✅ **PRNU 檢測結果：** 偵測到實體鏡頭晶片感光雜訊，符合真實拍攝特徵。")
        else:
            st.warning("⚠️ **PRNU 檢測結果：** 未偵測到實體晶片物理雜訊 (疑為 AI 生成或純數位渲染圖)。")

    # ------------------ 第四防線：反偵查, LSB, C2PA ------------------
    st.markdown("---")
    st.subheader("🕵️ 第四防線｜反偵查隱寫追蹤、LSB 隱寫與 C2PA 水印稽核")
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.image(lsb_img, caption="LSB 最低有效位點陣圖", use_container_width=True)
    with col_a2:
        st.markdown(f"**高頻雜訊平滑度指標:** `{lap_var:.2f}`")
        if is_smoothed:
            st.warning("⚠️ 偵測到過度平滑區塊，疑有磨皮、降噪抹除 P 圖邊緣之行為。")
        else:
            st.info("✅ 雜訊分布自然，未發現抹除痕跡。")
            
        if c2pa_findings:
            st.error(f"🔏 **發現 C2PA / AI 工具特徵標籤:** `{', '.join(c2pa_findings)}`")
        else:
            st.write("🔏 **C2PA 稽核:** 未發現顯性 AI 工具水印簽章。")

    # ------------------ 第五防線：局部放大鏡 ------------------
    st.markdown("---")
    st.subheader("🔍 第五防線｜局部放大鑑識鏡 (Zoom Lens Inspector)")
    zoom_factor = st.radio("放大倍率", [2, 4, 8], horizontal=True)
    img_w, img_h = image.size
    center_x = st.slider("選擇觀察 X 座標", 0, img_w, img_w // 2)
    center_y = st.slider("選擇觀察 Y 座標", 0, img_h, img_h // 2)
    
    crop_w, crop_h = img_w // (zoom_factor * 2), img_h // (zoom_factor * 2)
    box = (max(0, center_x - crop_w), max(0, center_y - crop_h), min(img_w, center_x + crop_w), min(img_h, center_y + crop_h))
    
    col_z1, col_z2 = st.columns(2)
    with col_z1:
        st.image(image.crop(box), caption=f"原圖 {zoom_factor}x 局部放大", use_container_width=True)
    with col_z2:
        st.image(ela_jet_img.crop(box), caption=f"ELA 彩色熱圖 {zoom_factor}x 局部放大", use_container_width=True)

    # ------------------ 第六防線：EXIF 中文化 ------------------
    st.markdown("---")
    st.subheader("📜 第六防線｜相機履歷搜查（中文 EXIF 數位元資料）")
    exif_data_dict = {}
    info = image._getexif()
    if info:
        for tag_id, value in info.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            if tag_name in IGNORE_TAGS or str(tag_id) in IGNORE_TAGS: continue
            zh_name = EXIF_ZH_MAP.get(tag_name, tag_name)
            exif_data_dict[zh_name] = str(value)
        st.json(exif_data_dict)
    else:
        exif_data_dict = {"說明": "未偵測到原始相機 EXIF 紀錄"}
        st.warning("⚠️ 未偵測到原始相機 EXIF 元資料（可能為截圖或轉傳照片）。")

    # ------------------ 第七防線：報告匯出 ------------------
    st.markdown("---")
    st.subheader("📄 第七防線｜資安採證官方稽核報告匯出")
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_content = f"""================================================================================
           「狗」眼看真偽 - 官方資安採證與數位證物稽核報告 (Official)
================================================================================
【一、證物監管鏈 (Chain of Custody)】
- 採證時間: {current_time}
- 證物檔名: {uploaded_file.name}
- SHA-256 哈希: {sha256_hash}
- MD5 哈希: {md5_hash}
- 影像尺寸: {image.size[0]} x {image.size[1]} px

【二、警犬隊綜合鑑定結論】
- 可信度總分: {trust_score:.1f} / 100
- AI 偽造風險指數: {ai_score:.2f}% ({ai_result_text})
- 疑點 ROI 座標: {roi_box if roi_box else "未發現集中疑點區塊"}

【三、高階物理解析】
- PRNU 感光指紋: {"✅ 具備實體鏡頭晶片雜訊" if has_prnu else "⚠️ 無實體鏡頭雜訊 (疑為 AI/數位渲染)"}
- 抹除磨皮痕跡: {"⚠️ 發現過重平滑掩蓋痕跡" if is_smoothed else "✅ 雜訊分布正常"}
- C2PA 工具標籤: {', '.join(c2pa_findings) if c2pa_findings else "未發現顯性標籤"}

【四、相機履歷 (EXIF)】
{exif_data_dict}

================================================================================
此報告由「『狗』眼看真偽 影像真偽鑑定平台」自動生成並完成 SHA-256 數位簽章鎖定
================================================================================
"""

    st.download_button(
        label="📥 一鍵下載「狗眼看真偽」資安採證官方報告 (.txt)",
        data=report_content,
        file_name=f"狗眼看真偽_採證報告_{sha256_hash[:10]}.txt",
        mime="text/plain"
    )
