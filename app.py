import streamlit as st
import torch
import cv2
import os
from PIL import Image
from core import FSRCNN, rgb_to_ycbcr, ycbcr_to_rgb

st.set_page_config(page_title="AI Super Resolution", layout="wide")

# إعداد المجلد المؤقت
TEMP_DIR = "temp"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

@st.cache_resource
def load_trained_model(scale):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FSRCNN(scale_factor=scale).to(device)
    path = f"weights/best_x{scale}.pth"
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model, device
    except:
        return None, device

st.title("✨ محسن الصور الذكي")

uploaded_file = st.file_uploader("ارفع صورة (JPG/PNG/JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 1. تحميل وعرض الصورة الأصلية فوراً
    img = Image.open(uploaded_file).convert('RGB')
    file_name = os.path.splitext(uploaded_file.name)[0]
    
    # إنشاء أعمدة لعرض المقارنة
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("before")
        st.image(img, use_container_width=True)
    
    # 2. خيارات التحكم (ستظهر تحت الصورة الأصلية)
    with st.sidebar:
        st.header("إعدادات التحسين")
        scale = st.select_slider("اختر درجة التكبير", options=[2, 3, 4])
        run_button = st.button(" ابدأ التحسين الآن")

    # 3. منطق المعالجة عند الضغط على الزر
    if run_button:
        model, device = load_trained_model(scale)
        
        if model is None:
            st.error(f"ملف الأوزان x{scale} غير موجود.")
        else:
            with col2:
                st.subheader("after")
                with st.spinner('جاري التحسين...'):
                    # المعالجة
                    ycbcr = rgb_to_ycbcr(img)
                    y = ycbcr[..., 0]
                    input_y = torch.from_numpy(y/255.).view(1,1,y.shape[0],y.shape[1]).to(device).float()
                    
                    with torch.no_grad():
                        output_y = model(input_y).clamp(0,1).cpu().numpy().squeeze() * 255.
                    
                    h, w = output_y.shape
                    cb = cv2.resize(ycbcr[..., 1], (w, h), interpolation=cv2.INTER_CUBIC)
                    cr = cv2.resize(ycbcr[..., 2], (w, h), interpolation=cv2.INTER_CUBIC)
                    result = ycbcr_to_rgb(output_y, cb, cr)
                    
                    # عرض النتيجة في العمود الثاني
                    st.image(result, use_container_width=True)
                    
                    # حفظ وتجهيز التحميل
                    final_pil = Image.fromarray(result)
                    save_path = os.path.join(TEMP_DIR, f"{file_name}_x{scale}.png")
                    final_pil.save(save_path)
                    
                    with open(save_path, "rb") as f:
                        st.download_button(
                            label=" تحميل الصورة المحسنة",
                            data=f,
                            file_name=f"{file_name}_x{scale}.png",
                            mime="image/png"
                        )
                    st.success(f"تم الحفظ في {TEMP_DIR}")