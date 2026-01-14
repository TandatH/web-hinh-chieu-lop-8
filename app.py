import streamlit as st
import matplotlib.pyplot as plt
import google.generativeai as genai
import plotly.graph_objects as go
import numpy as np
from PIL import Image

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Phân Tích Khối 3D", page_icon="📐", layout="wide")

# --- 1. SIDEBAR: CẤU HÌNH & NHẬP LIỆU ---
with st.sidebar:
    st.header("⚙️ Cấu hình")
    api_key_input = st.text_input("Nhập Google AI API Key", type="password", help="Nhập key bắt đầu bằng AIza...")
    if api_key_input:
        try:
            genai.configure(api_key=api_key_input)
            st.success("Đã kết nối AI! ✅")
        except:
            st.error("Key không hợp lệ.")
            api_key_input = None
    else:
        st.warning("Chưa nhập API Key (AI sẽ tắt)")

    st.divider()
    st.header("🖼️ Chế độ hoạt động")
    
    # TÙY CHỌN MỚI: TẢI ẢNH KHỐI BẤT KỲ
    uploaded_new_block = st.file_uploader("Tải ảnh khối 3D mới (Thay thế khối L):", type=["png", "jpg", "jpeg"])
    
    if not uploaded_new_block:
        # Chỉ hiện tùy chỉnh khối L nếu KHÔNG tải ảnh mới
        st.subheader("🎛️ Tùy chỉnh Khối L mặc định")
        h1 = st.slider("Chiều cao (Đứng)", 2, 6, 3)
        w1 = st.slider("Chiều rộng (Đứng)", 1, 3, 1)
        l2 = st.slider("Chiều dài (Ngang)", 1, 5, 2)
        st.info("Kéo thanh trượt để thay đổi khối L bên cạnh.")
    else:
        st.success("Đang sử dụng ảnh khối mới tải lên!")
        # Đặt giá trị mặc định để tránh lỗi code phía dưới
        h1, w1, l2 = 3, 1, 2 

# --- 2. CÁC HÀM VẼ (GIỮ NGUYÊN CHO KHỐI L) ---
def create_dynamic_L_block(h1, w1, l2):
    # ... (Code vẽ 3D giữ nguyên như cũ) ...
    def get_cube_trace(x_start, y_start, z_start, dx, dy, dz, color, name):
        x = np.array([0, 1, 1, 0, 0, 1, 1, 0]) * dx + x_start
        y = np.array([0, 0, 1, 1, 0, 0, 1, 1]) * dy + y_start
        z = np.array([0, 0, 0, 0, 1, 1, 1, 1]) * dz + z_start
        return go.Mesh3d(
            x=x, y=y, z=z,
            i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            opacity=0.9, color=color, flatshading=True, name=name
        )
    box_v = get_cube_trace(0, 0, 0, w1, w1, h1, '#FF7043', 'Đứng')
    h_base = 1 
    box_h = get_cube_trace(w1, 0, 0, l2, w1, h_base, '#26A69A', 'Ngang')
    fig = go.Figure(data=[box_v, box_h])
    fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0), height=350)
    return fig

def plot_dynamic_projections(h1, w1, l2):
    # ... (Code vẽ 2D giữ nguyên như cũ) ...
    h_base = 1
    total_width = w1 + l2
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax in axes: ax.set_aspect('equal'); ax.set_xlim(-0.5, total_width + 0.5); ax.set_ylim(-0.5, h1 + 0.5); ax.axis('off')
    axes[0].set_title("1. Chiếu Đứng", color='blue'); x_pts = [0, total_width, total_width, w1, w1, 0, 0]; y_pts = [0, 0, h_base, h_base, h1, h1, 0]; axes[0].plot(x_pts, y_pts, 'k-', lw=2); axes[0].fill(x_pts, y_pts, 'salmon', alpha=0.3)
    axes[1].set_title("2. Chiếu Bằng", color='blue'); axes[1].set_ylim(-0.5, total_width + 0.5); axes[1].plot([0, total_width, total_width, 0, 0], [0, 0, w1, w1, 0], 'k-', lw=2); axes[1].plot([w1, w1], [0, w1], 'k-', lw=2) 
    axes[2].set_title("3. Chiếu Cạnh", color='blue'); axes[2].plot([0, w1, w1, 0, 0], [0, 0, h1, h1, 0], 'k-', lw=2); axes[2].plot([0, w1], [h_base, h_base], 'k-', lw=2)
    plt.tight_layout(); return fig

# --- 3. HÀM AI PHÂN TÍCH (NÂNG CẤP) ---
def ask_ai_analyze_block(image_file=None, h1=None, w1=None, l2=None):
    if not api_key_input: return "⚠️ Vui lòng nhập API Key."
    
    # Dùng model Flash cho nhanh
    model = genai.GenerativeModel('gemini-1.5-flash')

    if image_file:
        # Trường hợp 1: Phân tích ảnh khối mới tải lên
        img = Image.open(image_file)
        prompt = """
        Bạn là giáo viên Vẽ Kỹ Thuật. Hãy quan sát khối vật thể 3D trong bức ảnh này và:
        1. Mô tả ngắn gọn hình dáng của vật thể này (Nó được tạo thành từ các khối cơ bản nào?).
        2. Dự đoán hình chiếu đứng (nhìn từ mặt trước) của nó sẽ có hình dạng gì?
        3. Dự đoán hình chiếu bằng (nhìn từ trên xuống) của nó sẽ có hình dạng gì?
        """
        response = model.generate_content([prompt, img])
    else:
        # Trường hợp 2: Phân tích khối L mặc định
        prompt = f"""
        Bạn là giáo viên Vẽ Kỹ Thuật. Vật thể là khối chữ L có kích thước: Phần đứng cao {h1}, rộng {w1}. Phần ngang dài thêm {l2}.
        Hãy giải thích tại sao hình chiếu cạnh của nó lại có một nét gạch ngang ở giữa?
        """
        response = model.generate_content(prompt)
        
    return response.text

# --- 4. GIAO DIỆN CHÍNH (LOGIC HIỂN THỊ MỚI) ---
st.title("🛠️ Phân Tích Vật Thể 3D & Hình Chiếu")

col1, col2 = st.columns([1, 1.5])

# --- CỘT 1: MÔ HÌNH 3D ---
with col1:
    if uploaded_new_block:
        # NẾU CÓ ẢNH MỚI: Hiển thị ảnh đó
        st.subheader("📸 Ảnh vật thể mới")
        st.image(uploaded_new_block, caption="Vật thể bạn tải lên", use_column_width=True)
        st.info("AI sẽ phân tích ảnh này thay vì khối L.")
    else:
        # NẾU KHÔNG CÓ ẢNH: Hiển thị khối L tương tác mặc định
        st.subheader("🧊 Mô hình 3D Tương tác (Khối L)")
        fig_3d = create_dynamic_L_block(h1, w1, l2)
        st.plotly_chart(fig_3d, use_container_width=True)

# --- CỘT 2: BẢN VẼ 2D ---
with col2:
    st.subheader("📐 Bản vẽ Hình chiếu tương ứng")
    if uploaded_new_block:
        # Nếu là ảnh mới, không vẽ được 2D chính xác ngay, hiện thông báo chờ AI
        st.warning("Đang hiển thị ảnh vật thể mới. Vui lòng nhấn nút bên dưới để AI phân tích hình chiếu của vật thể này.")
        # Có thể hiển thị một hình ảnh placeholder hoặc để trống
    else:
        # Nếu là khối L, vẽ 2D như bình thường
        fig_2d = plot_dynamic_projections(h1, w1, l2)
        st.pyplot(fig_2d)

st.divider()

# --- KHU VỰC AI ---
st.subheader("🤖 Giáo viên AI phân tích")

if st.button("Nhờ AI phân tích vật thể đang hiển thị"):
    with st.spinner("AI đang quan sát và suy nghĩ..."):
        # Truyền đúng tham số tùy vào việc có ảnh mới hay không
        if uploaded_new_block:
            analysis_result = ask_ai_analyze_block(image_file=uploaded_new_block)
        else:
            analysis_result = ask_ai_analyze_block(h1=h1, w1=w1, l2=l2)
        st.markdown(analysis_result)
