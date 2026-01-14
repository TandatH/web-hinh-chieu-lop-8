import streamlit as st
import matplotlib.pyplot as plt
from google import genai
import plotly.graph_objects as go
import numpy as np

# --- 1. CẤU HÌNH API KEY ---
try:
    api_key = st.secrets["AIzaSyA-TYnWFvS4YByH0NW_e98vqcTQR6lnw44"]
    genai.configure(api_key=api_key)
    api_status = "Đã kết nối AI thành công! ✅"

except Exception as e:
    api_status = "⚠️ Chưa tìm thấy API Key (Chế độ xem offline)"
    api_key = None

# --- 2. HÀM TẠO KHỐI 3D TƯƠNG TÁC (PLOTLY) ---
def create_3d_block():
    """
    Tạo khối chữ L 3D tương tác bằng cách ghép 2 hình hộp chữ nhật.
    - Hộp 1: Phần đứng (Cao 3, Rộng 1, Sâu 1)
    - Hộp 2: Phần ngang (Cao 1, Rộng 1, Sâu 1) - Ghép thêm vào bên cạnh
    """
    def make_cube(x_offset, y_offset, z_offset, dx, dy, dz, color):
        # Định nghĩa 8 đỉnh của hình hộp
        x = np.array([0, 1, 1, 0, 0, 1, 1, 0]) * dx + x_offset
        y = np.array([0, 0, 1, 1, 0, 0, 1, 1]) * dy + y_offset
        z = np.array([0, 0, 0, 0, 1, 1, 1, 1]) * dz + z_offset
        
        # Định nghĩa các mặt (dựa trên index của đỉnh)
        return go.Mesh3d(
            x=x, y=y, z=z,
            # i, j, k là chỉ số các đỉnh để tạo thành mặt tam giác
            i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            opacity=0.8,
            color=color,
            flatshading=True,
            name='Khối'
        )

    # Tạo phần đứng (Màu cam)
    box_vertical = make_cube(0, 0, 0, 1, 1, 3, '#FFAB91') # x=0->1, y=0->1, z=0->3
    
    # Tạo phần ngang (Màu xanh) - Ghép vào bên cạnh
    box_horizontal = make_cube(1, 0, 0, 1, 1, 1, '#80CBC4') # x=1->2, y=0->1, z=0->1

    # Tạo khung cảnh 3D
    fig = go.Figure(data=[box_vertical, box_horizontal])

    # Cấu hình giao diện (Bỏ lưới cho đẹp)
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data' # Giữ đúng tỷ lệ
        ),
        margin=dict(l=0, r=0, b=0, t=0), # Canh lề sát
        height=300,
    )
    return fig

# --- 3. HÀM VẼ 2D (MATPLOTLIB) ---
def plot_projections():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax in axes:
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.5, 3.5)

    # 1. Chiếu Đứng (Nhìn từ trục Y -> Thấy mặt XZ) -> Chữ L
    axes[0].set_title("1. Chiếu Đứng", color='blue')
    axes[0].plot([0, 2, 2, 1, 1, 0, 0], [0, 0, 1, 1, 3, 3, 0], 'k-', lw=3)
    axes[0].fill([0, 2, 2, 1, 1, 0], [0, 0, 1, 1, 3, 3], 'skyblue', alpha=0.3)

    # 2. Chiếu Bằng (Nhìn từ Z xuống -> Thấy mặt XY)
    axes[1].set_title("2. Chiếu Bằng", color='blue')
    axes[1].plot([0, 2, 2, 0, 0], [0, 0, 1, 1, 0], 'k-', lw=3) # Bao ngoài
    axes[1].plot([1, 1], [0, 1], 'k-', lw=2) # Đường phân chia (nếu nhìn khối ghép)

    # 3. Chiếu Cạnh (Nhìn từ trục X -> Thấy mặt YZ)
    axes[2].set_title("3. Chiếu Cạnh", color='blue')
    axes[2].plot([0, 1, 1, 0, 0], [0, 0, 3, 3, 0], 'k-', lw=3) # Bao ngoài
    axes[2].plot([0, 1], [1, 1], 'k-', lw=2) # Nét liền (bậc ngang)

    plt.tight_layout()
    return fig

# --- 4. HÀM AI ---
def get_ai_review():
    if not api_key: return "⚠️ Vui lòng nhập API Key để dùng AI."
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = """
    Bạn là giáo viên Công Nghệ 8. Hãy giải thích ngắn gọn về "Khối Chữ L":
    1. Tại sao Hình chiếu đứng lại có dạng chữ L?
    2. Tại sao Hình chiếu cạnh lại là hình chữ nhật có gạch ngang?
    """
    return model.generate_content(prompt).text

# --- 5. GIAO DIỆN CHÍNH ---
st.set_page_config(page_title="Vẽ Kỹ Thuật 3D", page_icon="📐", layout="wide")

st.title("📐 Công Nghệ 8: Thực Hành Hình Chiếu")
st.caption("Dùng chuột xoay khối 3D để hiểu rõ các mặt của vật thể.")

# Chia cột: Bên trái là 3D, Bên phải là 2D
col1, col2 = st.columns([1, 1.5])

with col1:
    st.info("🖱️ Xoay chuột vào hình dưới để xem:")
    fig_3d = create_3d_block()
    # HIỂN THỊ 3D TƯƠNG TÁC
    st.plotly_chart(fig_3d, use_container_width=True)

with col2:
    st.success("📝 Bản vẽ 3 hình chiếu vuông góc:")
    fig_2d = plot_projections()
    st.pyplot(fig_2d)

st.divider()

if st.button("🤖 Nhờ AI giải thích bài học"):
    if api_key:
        with st.spinner("Đang phân tích vật thể..."):
            st.markdown(get_ai_review())
    else:
        st.error("Chưa kết nối API Key.")

st.markdown("---")

st.caption("Dự án hỗ trợ học tập - Tương tác 3D với Streamlit & Plotly")




