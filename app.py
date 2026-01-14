import streamlit as st
import matplotlib.pyplot as plt
import google.generativeai as genai
import plotly.graph_objects as go
import numpy as np
from PIL import Image

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Vẽ Kỹ Thuật 3D Động", page_icon="📐", layout="wide")

# --- 1. SIDEBAR: CẤU HÌNH & NHẬP LIỆU ---
with st.sidebar:
    st.header("⚙️ Cấu hình")
    
    # Nhập API Key an toàn
    api_key_input = st.text_input("Nhập Google AI API Key", type="password", help="Nhập key bắt đầu bằng AIza...")
    
    if api_key_input:
        try:
            genai.configure(api_key=api_key_input)
            st.success("Đã kết nối AI! ✅")
        except:
            st.error("Key không hợp lệ.")
    else:
        st.warning("Chưa nhập API Key (AI sẽ tắt)")

    st.divider()
    st.header("🎛️ Tùy chỉnh Khối L")
    # Thông số khối đứng
    h1 = st.slider("Chiều cao (Đứng)", 2, 6, 3)
    w1 = st.slider("Chiều rộng (Đứng)", 1, 3, 1)
    
    # Thông số khối ngang
    l2 = st.slider("Chiều dài (Ngang)", 1, 5, 2)
    
    st.info("Thay đổi thanh trượt để cập nhật hình chiếu!")

# --- 2. HÀM TẠO KHỐI 3D ĐỘNG (PARAMETRIC) ---
def create_dynamic_L_block(h1, w1, l2):
    # Khối 1: Trụ đứng (Gốc 0,0,0)
    # Kích thước: Rộng=w1, Sâu=w1 (giả sử vuông), Cao=h1
    
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

    # Phần đứng (Màu cam)
    box_v = get_cube_trace(0, 0, 0, w1, w1, h1, '#FF7043', 'Đứng')
    
    # Phần ngang (Màu xanh) - Gắn vào bên phải phần đứng
    # Bắt đầu từ x=w1, độ cao mặc định là 1 đơn vị (để tạo hình L)
    h_base = 1 
    box_h = get_cube_trace(w1, 0, 0, l2, w1, h_base, '#26A69A', 'Ngang')

    fig = go.Figure(data=[box_v, box_h])
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0), height=350
    )
    return fig

# --- 3. HÀM VẼ 2D ĐỘNG (MATPLOTLIB) ---
def plot_dynamic_projections(h1, w1, l2):
    h_base = 1 # Độ cao phần đế ngang
    total_width = w1 + l2
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, total_width + 0.5)
        ax.set_ylim(-0.5, h1 + 0.5)
        ax.axis('off')

    # 1. HÌNH CHIẾU ĐỨNG (Nhìn từ mặt trước - Trục XZ)
    # Thấy hình chữ L
    axes[0].set_title("1. Chiếu Đứng", color='blue', fontsize=12)
    # Vẽ biên dạng chữ L
    x_pts = [0, total_width, total_width, w1, w1, 0, 0]
    y_pts = [0, 0, h_base, h_base, h1, h1, 0]
    axes[0].plot(x_pts, y_pts, 'k-', lw=2)
    axes[0].fill(x_pts, y_pts, 'salmon', alpha=0.3)

    # 2. HÌNH CHIẾU BẰNG (Nhìn từ trên xuống - Trục XY)
    # Thấy hình chữ nhật dài chia làm 2 phần
    axes[1].set_title("2. Chiếu Bằng", color='blue', fontsize=12)
    axes[1].set_ylim(-0.5, total_width + 0.5) # Resize lại cho cân
    # Khung bao ngoài (w1 x total_width) -> Ở đây vẽ đơn giản hóa chiều sâu = w1
    axes[1].plot([0, total_width, total_width, 0, 0], [0, 0, w1, w1, 0], 'k-', lw=2)
    # Nét liền phân chia 2 khối
    axes[1].plot([w1, w1], [0, w1], 'k-', lw=2) 

    # 3. HÌNH CHIẾU CẠNH (Nhìn từ trái sang - Trục YZ)
    # Thấy hình chữ nhật đứng (w1 x h1)
    axes[2].set_title("3. Chiếu Cạnh", color='blue', fontsize=12)
    axes[2].plot([0, w1, w1, 0, 0], [0, 0, h1, h1, 0], 'k-', lw=2) # Bao ngoài
    # Nét liền thể hiện bậc ngang (nếu nhìn từ trái thì thấy bậc)
    axes[2].plot([0, w1], [h_base, h_base], 'k-', lw=2)

    plt.tight_layout()
    return fig

def ask_ai(h1, w1, l2, uploaded_file=None):
    if not api_key_input:
        return "⚠️ Vui lòng nhập API Key trước."

    try:
        model = genai.GenerativeModel(
            model_name="models/gemini-1.5-flash"
        )

        # Prompt cho giáo viên Công nghệ 8
        prompt = f"""
        Tôi đang dạy vẽ kỹ thuật lớp 8.
        Vật thể là khối chữ L có kích thước:
        - Phần đứng cao {h1} đơn vị, rộng {w1} đơn vị.
        - Phần ngang dài thêm {l2} đơn vị, cao 1 đơn vị.

        Hãy giải thích NGẮN GỌN, DỄ HIỂU cho học sinh:
        1. Kích thước hình chiếu đứng.
        2. Vì sao hình chiếu cạnh có một đường ngang ở cao độ 1.
        """

        # Nếu có ảnh học sinh vẽ
        if uploaded_file:
            img = Image.open(uploaded_file)
            response = model.generate_content([
                "Đây là bản vẽ hình chiếu của học sinh lớp 8. Hãy nhận xét đúng – sai và góp ý ngắn gọn.",
                img
            ])
        else:
            response = model.generate_content(prompt)

        return response.text

    except Exception as e:
        return f"❌ Lỗi AI: {e}"
text

# --- 5. GIAO DIỆN CHÍNH ---
st.title("🛠️ Tạo & Phân Tích Khối Chữ L (Dynamic)")
st.caption("Chỉnh thông số bên trái -> Hình thay đổi ngay lập tức.")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("Mô hình 3D")
    fig_3d = create_dynamic_L_block(h1, w1, l2)
    st.plotly_chart(fig_3d, use_container_width=True)

with col2:
    st.subheader("Bản vẽ 2D Tương ứng")
    fig_2d = plot_dynamic_projections(h1, w1, l2)
    st.pyplot(fig_2d)

st.divider()

# --- KHU VỰC AI ---
st.subheader("🤖 Trợ lý AI (Giáo viên ảo)")
tab1, tab2 = st.tabs(["Giải thích thông số hiện tại", "Chấm bài (Tải ảnh lên)"])

with tab1:
    if st.button("Giải thích hình này"):
        with st.spinner("AI đang suy nghĩ..."):
            st.write(ask_ai(h1, w1, l2))

with tab2:
    uploaded_file = st.file_uploader("Tải ảnh bài vẽ tay của bạn lên để AI chấm:", type=["png", "jpg", "jpeg"])
    if uploaded_file and st.button("Chấm bài"):
        with st.spinner("AI đang soi bản vẽ..."):
            st.image(uploaded_file, width=200)
            st.write(ask_ai(h1, w1, l2, uploaded_file))

