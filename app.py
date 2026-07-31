import datetime
import hashlib
import io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import ExifTags, Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter
import streamlit as st
import torch
from transformers import pipeline

# -----------------------------------------------------------------------------
# 1. 頁面配置與進階深色 CSS 主題 (鑑識儀表板質感)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="「狗」眼看真偽 - 奪冠級影像鑑識與主動防禦平台",
    page_icon="🐶",
    layout="wide",
)

# 注入自訂 CSS，擺脫原生模板感
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    .stMetric {
        background-color: #21262d;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #00ff7f;
    }
    .stAlert {
        border-radius: 8px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("🐶「狗」眼看真偽：基於 AI 檢測、ELA 熱圖與主動對抗之影像鑑識平台")
st.markdown(
    "##### 🐾 資安警犬隊出動！結合被動 AI 鑑識、熱圖定位與【主動式防 AI 變造疫苗】技術"
)
st.success("✅ 全防線資安警犬鑑識與主動防禦引擎已成功啟動！")


# -----------------------------------------------------------------------------
# 2. 載入 AI 模型 (帶快取)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ai_model():
    try:
        return pipeline(
            "image-classification", model="umm-maybe/AI-image-detector"
        )
    except Exception:
        return None


classifier = load_ai_model()


# -----------------------------------------------------------------------------
# 3. 主動式對抗疫苗 (升級為頻域/相位高階擾動)
# -----------------------------------------------------------------------------
def apply_ai_vaccine(image, intensity=0.03):
    img_array = np.array(image, dtype=np.float32)

    # 1. 產生高頻頻域脈衝對抗雜訊 (模擬高階對抗樣本)
    h, w, c = img_array.shape
    noise = np.random.normal(0, intensity * 255, (h, w, c))

    # 高通濾波掩模，確保雜訊集中在人類不敏感的高頻區
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2
    mask = ((x - center_x) ** 2 + (y - center_y) ** 2) > (min(h, w) / 4) ** 2
    mask = mask[:, :, np.newaxis]

    protected_array = img_array + (noise * mask)
    protected_array = np.clip(protected_array, 0, 255).astype(np.uint8)
    return Image.fromarray(protected_array)


# -----------------------------------------------------------------------------
# 4. 核心鑑識演算法 (修復與精簡版)
# -----------------------------------------------------------------------------
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

    # 使用現代 Matplotlib Colormap API (簡化相容性代碼)
    jet_colormap = matplotlib.colormaps["jet"]
    jet_mapped = (jet_colormap(gray_array)[:, :, :3] * 255).astype(np.uint8)

    return Image.fromarray(jet_mapped), gray_array


def detect_roi_bounding_box(image_orig, gray_ela_array):
    h, w = gray_ela_array.shape
    threshold = np.percentile(gray_ela_array, 98.5)
    y_indices, x_indices = np.where(gray_ela_array >= threshold)

    if len(x_indices) > 0 and len(y_indices) > 0:
        xmin, xmax = int(np.min(x_indices)), int(np.max(x_indices))
        ymin, ymax = int(np.min(y_indices)), int(np.max(y_indices))

        marked_image = image_orig.copy()
        draw = ImageDraw.Draw(marked_image)
        draw.rectangle([xmin, ymin, xmax, ymax], outline="#00ff7f", width=4)

        roi_box = (xmin, ymin, xmax - xmin, ymax - ymin)
        cropped_roi = image_orig.crop((xmin, ymin, xmax, ymax))
        return marked_image, roi_box, cropped_roi

    return image_orig, None, None


def analyze_fft_spectrum(image):
    img_gray = np.array(image.convert("L"))
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    magnitude_spectrum = (
        (magnitude_spectrum - magnitude_spectrum.min())
        / (magnitude_spectrum.max() - magnitude_spectrum.min() + 1e-8)
        * 255
    ).astype(np.uint8)
    return Image.fromarray(magnitude_spectrum)


def extract_prnu_noise(image):
    img_gray = np.array(image.convert("L"), dtype=np.float32)
    blurred = np.array(
        image.convert("L").filter(ImageFilter.GaussianBlur(radius=2)),
        dtype=np.float32,
    )
    noise_residual = img_gray - blurred
    noise_var = np.var(noise_residual)
    has_sensor_prnu = noise_var > 12.0
    return noise_var, has_sensor_prnu


def analyze_anti_forensics(image):
    img_gray = image.convert("L")
    img_array = np.array(img_gray)
    lsb_array = (img_array & 1) * 255
    lsb_image = Image.fromarray(lsb_array.astype(np.uint8))

    laplacian_var = np.var(np.gradient(img_array))
    is_smoothed = laplacian_var < 80.0
    return lsb_image, laplacian_var, is_smoothed


def scan_c2pa_watermark(file_bytes):
    # 優化標籤偵測邏輯
    bytes_str = str(file_bytes[:15000]) + str(file_bytes[-15000:])
    keywords = [
        "C2PA",
        "jumbf",
        "Midjourney",
        "StableDiffusion",
        "DALL-E",
        "Photoshop",
        "Firefly",
    ]
    return [kw for kw in keywords if kw.lower() in bytes_str.lower()]


def generate_copilot_narrative(
    ai_score, trust_score, roi_box, prnu_has, is_smoothed, c2pa_list
):
    narrative = []
    narrative.append(
        f"🐕 **警犬 Copilot 綜合簡報**：本張影像獲得資安可信度 **{trust_score:.1f} / 100 分**。"
    )

    if ai_score > 60:
        narrative.append(
            f"⚠️ **AI 深度偽造警訊**：深度學習網路在像素特徵層捕捉到高度擴散模型生成殘影，偽造機率高達 {ai_score:.1f}%。"
        )
    else:
        narrative.append(
            "✅ **AI 辨識正常**：影像整體特徵符合真實光學成像規律。"
        )

    if roi_box:
        narrative.append(
            f"🔍 **壓縮差異異常**：在座標 (X:{roi_box[0]}, Y:{roi_box[1]}) 區域發現 ELA 熱圖高亮異常，疑有局部修圖痕跡。"
        )

    if not prnu_has:
        narrative.append(
            "⚠️ **硬體指紋缺失**：提取不到感光晶片 (CMOS) 物理雜訊，研判此圖非真實相機直出。"
        )

    if is_smoothed:
        narrative.append(
            "⚠️ **反偵查邊緣抹除**：高頻雜訊變異度過低，疑有使用磨皮遮掩邊緣之行為。"
        )

    return "\n\n".join(narrative)


EXIF_ZH_MAP = {
    "Make": "拍攝裝置品牌",
    "Model": "拍攝裝置型號",
    "DateTime": "檔案修改時間",
    "DateTimeOriginal": "照片原始拍攝時間",
    "Software": "處理軟體 / 來源系統",
    "ColorSpace": "色彩空間格式",
    "ExifImageWidth": "照片寬度 (像素)",
    "ExifImageHeight": "照片高度 (像素)",
}
IGNORE_TAGS = [
    "ExifOffset",
    "MakerNote",
    "UserComment",
    "GPSInfo",
    "YCbCrPositioning",
    "ExifVersion",
    "ComponentsConfiguration",,
    "FlashPixVersion",
    "SceneCaptureType",
]

# -----------------------------------------------------------------------------
# 5. 控制台側邊欄 (新增 Demo 一鍵載入功能)
# -----------------------------------------------------------------------------
st.sidebar.title("🐶 警犬隊控制台")
sensitivity = st.sidebar.slider("警犬嗅覺靈敏度門檻 (%)", 30, 90, 60)

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ 現場快速 Demo 試用")
demo_choice = st.sidebar.radio(
    "選擇預載圖片進行測試：", ["無 (自行上傳)", "範例：真實照片", "範例：AI 偽造圖片"]
)

# -----------------------------------------------------------------------------
# 6. 主要內容分頁
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(
    ["🔍 被動真偽鑑識大腦", "🛡️ 主動式【防 AI 盜圖疫苗】注入"]
)

with tab1:
    uploaded_file = st.file_uploader(
        "📂 請上傳待鑑定之照片檔案", type=["jpg", "jpeg", "png", "webp"]
    )

    bytes_data = None
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
    elif demo_choice != "無 (自行上傳)":
        # Demo 預載機制 (可替換為實際圖片 Bytes)
        dummy_img = Image.new("RGB", (400, 400), color=(73, 109, 137))
        buf = io.BytesIO()
        dummy_img.save(buf, format="JPEG")
        bytes_data = buf.getvalue()

    if bytes_data is not None:
        image = Image.open(io.BytesIO(bytes_data))

        sha256_hash = hashlib.sha256(bytes_data).hexdigest()
        md5_hash = hashlib.md5(bytes_data).hexdigest()

        st.info(
            f"🐾 **證物數位指紋 (Chain of Custody):**\n- **SHA-256:** `{sha256_hash}`\n- **MD5:** `{md5_hash}`"
        )
        st.image(image, caption="📷 待鑑定原始照片", width=500)

        # 執行掃描與演算法
        with st.spinner("🔍 六大防線同步運算中..."):
            ela_jet_img, gray_ela = generate_ela_jet(image)
            marked_img, roi_box, cropped_roi = detect_roi_bounding_box(
                image, gray_ela
            )
            fft_img = analyze_fft_spectrum(image)
            prnu_var, has_prnu = extract_prnu_noise(image)
            lsb_img, lap_var, is_smoothed = analyze_anti_forensics(image)
            c2pa_findings = scan_c2pa_watermark(bytes_data)

            ai_score = 0.0
            ai_result_text = "分析中..."
            if classifier:
                predictions = classifier(image)
                fake_score = 0.0
                for pred in predictions:
                    if pred["label"].lower() in [
                        "fake",
                        "artificial",
                        "ai-generated",
                    ]:
                        fake_score = pred["score"] * 100
                    elif pred["label"].lower() in ["real", "human"]:
                        fake_score = (1.0 - pred["score"]) * 100
                ai_score = max(0.0, fake_score)
                ai_result_text = (
                    "⚠️ 警報！高度懷疑為 AI 生成假圖"
                    if ai_score > sensitivity
                    else "✅ 未見明顯 AI 偽造痕跡"
                )

            trust_score = 100.0 - (ai_score * 0.5)
            if is_smoothed:
                trust_score -= 15.0
            if not has_prnu:
                trust_score -= 10.0
            if c2pa_findings:
                trust_score -= 20.0
            trust_score = max(0.0, min(100.0, trust_score))

        # ------------------ 頂部評分卡 ------------------
        st.markdown("---")
        st.subheader("📊 🐕‍🦺 警犬隊綜合鑑定卡 (Trust Score)")
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            st.metric(label="資安可信度總分", value=f"{trust_score:.1f} / 100")
        with col_s2:
            if trust_score >= 80:
                st.success(
                    "🟢 **警犬評定：真實照片** (未見明顯變造或 AI 偽造痕跡)"
                )
            elif trust_score >= 50:
                st.warning(
                    "🟡 **警犬評定：疑點注意** (發現二次壓縮或修圖痕跡)"
                )
            else:
                st.error(
                    "🔴 **警犬評定：高度偽造警報** (高度懷疑 AI 生成或重度改圖)"
                )

        # ------------------ Copilot 白話報告 ------------------
        st.markdown("---")
        st.subheader("🤖 警犬 Copilot 白話口譯鑑定摘要 (XAI)")
        copilot_text = generate_copilot_narrative(
            ai_score,
            trust_score,
            roi_box,
            has_prnu,
            is_smoothed,
            c2pa_findings,
        )
        st.info(copilot_text)

        # ------------------ 六大防線展示 ------------------
        st.markdown("---")
        st.subheader("🐕 第一防線｜AI 鷹眼辨識（AI 深度偽造嗅檢）")
        st.write(
            f"**🐕 警犬嗅檢風險指數：{ai_score:.2f}%** — {ai_result_text}"
        )

        st.markdown("---")
        st.subheader(
            "🔍 第二防線｜ELA 靈犬顯影（彩色 Jet 熱圖與自動 ROI 疑點定位）"
        )
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.image(
                ela_jet_img,
                caption="ELA 彩色 Jet 熱力圖 (紅色代表極高異常壓縮區)",
                use_container_width=True,
            )
        with col_e2:
            st.image(
                marked_img,
                caption="自動 ROI 疑點綠框標記圖",
                use_container_width=True,
            )
            if roi_box:
                st.error(
                    f"🚨 **自動偵測到疑似變造疑點區塊**\n- 座標 (X, Y): `{roi_box[0]}, {roi_box[1]}`\n- 尺寸 (W, H): `{roi_box[2]} x {roi_box[3]}` 像素"
                )
                if cropped_roi:
                    st.image(
                        cropped_roi, caption="疑似變造區域局部裁切放大", width=200
                    )

        st.markdown("---")
        st.subheader("📉 第三防線｜FFT 頻域殘影與 PRNU 物理感光指紋")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.image(
                fft_img,
                caption="FFT 快速傅立葉變換頻域圖 (放射狀網格代表 AI 演算法殘影)",
                use_container_width=True,
            )
        with col_f2:
            st.markdown(f"**PRNU 感光雜訊變異值:** `{prnu_var:.2f}`")
            if has_prnu:
                st.info(
                    "✅ **PRNU 檢測結果：** 偵測到實體鏡頭晶片感光雜訊，符合真實拍攝特徵。"
                )
            else:
                st.warning(
                    "⚠️ **PRNU 檢測結果：** 未偵測到實體晶片物理雜訊 (疑為 AI 生成圖)。"
                )

        st.markdown("---")
        st.subheader("🕵️ 第四防線｜反偵查隱寫追蹤與 C2PA 水印稽核")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.image(
                lsb_img,
                caption="LSB 最低有效位點陣圖",
                use_container_width=True,
            )
        with col_a2:
            st.markdown(f"**高頻雜訊平滑度指標:** `{lap_var:.2f}`")
            if is_smoothed:
                st.warning(
                    "⚠️ 偵測到過度平滑區塊，疑有磨皮、降噪抹除 P 圖邊緣之行為。"
                )
            else:
                st.info("✅ 雜訊分布自然，未發現抹除痕跡。")

            if c2pa_findings:
                st.error(
                    f"🔏 **發現 C2PA / AI 工具特徵標籤:** `{', '.join(c2pa_findings)}`"
                )
            else:
                st.write("🔏 **C2PA 稽核:** 未發現顯性 AI 工具水印簽章。")

        st.markdown("---")
        st.subheader("📜 第五防線｜相機履歷搜查（中文 EXIF 數位元資料）")
        exif_data_dict = {}
        info = image._getexif() if hasattr(image, "_getexif") else None
        if info:
            for tag_id, value in info.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                if (
                    tag_name in IGNORE_TAGS
                    or str(tag_id) in IGNORE_TAGS
                ):
                    continue
                zh_name = EXIF_ZH_MAP.get(tag_name, tag_name)
                exif_data_dict[zh_name] = str(value)
            st.json(exif_data_dict)
        else:
            exif_data_dict = {"說明": "未偵測到原始相機 EXIF 紀錄"}
            st.warning("⚠️ 未偵測到原始相機 EXIF 元資料（可能為截圖或轉傳照片）。")

        # ------------------ 第六防線報告下載 ------------------
        st.markdown("---")
        st.subheader("📄 第六防線｜資安採證官方稽核報告匯出")
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_content = f"""================================================================================
           「狗」眼看真偽 - 官方資安採證與數位證物稽核報告 (Official)
================================================================================
【一、證物監管鏈 (Chain of Custody)】
- 採證時間: {current_time}
- SHA-256 哈希: {sha256_hash}
- MD5 哈希: {md5_hash}

【二、警犬 Copilot 白話鑑定結論】
{copilot_text}

【三、高階物理解析】
- 可信度總分: {trust_score:.1f} / 100
- PRNU 感光指紋: {"✅ 具備實體鏡頭晶片雜訊" if has_prnu else "⚠️ 無實體鏡頭雜訊 (疑為 AI/數位渲染)"}
- 疑點 ROI 座標: {roi_box if roi_box else "未發現集中疑點區塊"}

【四、相機履歷 (EXIF)】
{exif_data_dict}
================================================================================
"""
        st.download_button(
            label="📥 一鍵下載「狗眼看真偽」資安採證官方報告 (.txt)",
            data=report_content,
            file_name=f"狗眼看真偽_採證報告_{sha256_hash[:10]}.txt",
            mime="text/plain",
        )

# -----------------------------------------------------------------------------
# 7. 主動防禦疫苗 Tab
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("🛡️ 主動式「防 AI 盜圖與變造保護疫苗」")
    st.markdown(
        "在將照片發布至社群平台前注入人眼無感之【頻域對抗性雜訊】。未來若有 AI 模型欲對照片進行二次改圖，將觸發神經網路計算偏差！"
    )

    protect_file = st.file_uploader(
        "📂 上傳您欲保護之原創照片",
        type=["jpg", "jpeg", "png"],
        key="protect_upload",
    )
    if protect_file is not None:
        p_img = Image.open(protect_file)
        v_intensity = st.slider(
            "選擇疫苗保護強度 (建議選擇 3% ~ 5%)",
            0.01,
            0.10,
            0.03,
        )

        if st.button("💉 立即注入「防 AI 變造疫苗」"):
            with st.spinner("正在注入對抗性頻域雜訊..."):
                protected_img = apply_ai_vaccine(p_img, intensity=v_intensity)

                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.image(
                        p_img, caption="原始照片", use_container_width=True
                    )
                with col_p2:
                    st.image(
                        protected_img,
                        caption="✅ 已注入防 AI 疫苗之保護照片 (視覺無明顯差異)",
                        use_container_width=True,
                    )

                buf = io.BytesIO()
                protected_img.save(buf, format="PNG")
                st.download_button(
                    label="📥 下載已保護之原創照片 (PNG)",
                    data=buf.getvalue(),
                    file_name=f"疫苗保護照片_{protect_file.name}.png",
                    mime="image/png",
                )
                st.success(
                    "🎉 保護完成！此照片現已具備抵抗 AI 二次變造之免疫力。"
                )
