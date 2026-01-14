import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from PIL import Image
from google import genai

# ================== CẤU HÌNH TRANG ==================
st.set_page_config(
    page_title="Phân tích vật thể 3D & hình chiếu",
    page_icon="📐",
    layout="wide"
)

# ================== SIDEBAR ==================
with st.sidebar:
    st.header("⚙️ Cấu hình AI")

    api_key_input = st.text_input(
        "Nhập Google AI API Key",
        type="password",
        help="Key bắt đầu bằng AIza..."
    )

    client = None
    if api_key_input:
        try:
            client = genai.Client(api_key=api_key_input)
            st.success("Đã kết nối AI ✅")
        except Exception:
            st.error("API Key không hợp lệ")
            client = None
    else:
        st.warning("Chưa nhập API Key")

    st.divider()
    st.header("🖼️ Chế độ vật thể")

    uploaded_new_block = st.file_uploader(
        "Tải ảnh vật thể 3D",
        type=["png", "jpg", "jpeg"]
    )

    if not uploaded_new_block:
        st.subheader("🎛️ Khối L mặc định")
        h1 = st.slider("Chiều cao", 2, 6, 3)
        w1 = st.slider("Chiều rộng", 1, 3, 1)
        l2 = st.slider("Chiều dài", 1, 5, 2)
    else:
        st.success("Đang dùng ảnh tải lên")
        h1, w1, l2 = 3, 1, 2   # giá trị giả để tránh lỗi

# ================== HÀM VẼ 3D KHỐI L ==================
def create_dynamic_L_block(h1, w1, l2):
    def cube(x0, y0, z0, dx, dy, dz, color, name):
        x = np.array([0,1,1,0,0,1,1,0])*dx + x0
        y = np.array([0,0,1,1,0,0,1,1])*dy + y0
        z = np.array([0,0,0,0,1,1,1,1])*dz + z0
        return go.Mesh3d(
            x=x, y=y, z=z,
            i=[7,0,0,0,4,4,6,6,4,0,3,2],
            j=[3,4,1,2,5,6,5,2,0,1,6,3],
            k=[0,7,2,3,6,7,1,1,5,5,7,6],
            opacity=0.9,
            color=color,
            name=name
        )

    v = cube(0,0,0,w1,w1,h1,"#FF7043","Đứng")
    h = cube(w1,0,0,l2,w1,1,"#26A69A","Ngang")

    fig = go.Figure(data=[v,h])
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data"
        ),
        margin=dict(l=0,r=0,t=0,b=0),
        height=350
    )
    return fig

# ================== HÀM VẼ HÌNH CHIẾU KHỐI L ==================
def plot_dynamic_projections(h1, w1, l2):
    fig, axes = plt.subplots(1,3,figsize=(10,4))
    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")

    # Chiếu đứng
    axes[0].set_title("Chiếu đứng")
    axes[0].plot(
        [0,w1+l2,w1+l2,w1,w1,0,0],
        [0,0,1,1,h1,h1,0], lw=2
    )

    # Chiếu bằng
    axes[1].set_title("Chiếu bằng")
    axes[1].plot(
        [0,w1+l2,w1+l2,0,0],
        [0,0,w1,w1,0], lw=2
    )
    axes[1].plot([w1,w1],[0,w1], lw=2)

    # Chiếu cạnh
    axes[2].set_title("Chiếu cạnh")
    axes[2].plot(
        [0,w1,w1,0,0],
        [0,0,h1,h1,0], lw=2
    )
    axes[2].plot([0,w1],[1,1], lw=2)

    plt.tight_layout()
    return fig

# ================== VẼ HÌNH CHIẾU TỪ ẢNH (MINH HỌA) ==================
def draw_projection_from_image():
    fig, axes = plt.subplots(1,3,figsize=(10,4))
    titles = ["Chiếu đứng", "Chiếu bằng", "Chiếu cạnh"]

    for ax, title in zip(axes, titles):
        ax.set_title(title, color="blue")
        ax.set_aspect("equal")
        ax.axis("off")

    # Chiếu đứng – dạng khối bậc
    axes[0].plot(
        [0,4,4,2,2,0,0],
        [0,0,2,2,4,4,0],
        lw=2
    )

    # Chiếu bằng
    axes[1].plot(
        [0,4,4,0,0],
        [0,3,3,3,0],
        lw=2
    )
    axes[1].plot([2,2],[0,3], lw=2)

    # Chiếu cạnh
    axes[2].plot(
        [0,3,3,0,0],
        [0,4,4,4,0],
        lw=2
    )
    axes[2].plot([0,3],[2,2], lw=2)

    plt.tight_layout()
    return fig

# ================== AI PHÂN TÍCH ==================
def ask_ai_analyze_block(image_file=None, h1=None, w1=None, l2=None):
    if not client:
        return "⚠️ Chưa kết nối AI"

    try:
        if image_file:
            img = Image.open(image_file)
            prompt = """
            Bạn là giáo viên Vẽ Kỹ Thuật THCS.
            Hãy:
            1. Mô tả dạng hình học của vật thể.
            2. Nhận xét hình chiếu đứng, bằng, cạnh.
            Trình bày ngắn gọn, dễ hiểu cho học sinh lớp 8.
            """
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[prompt, img]
            )
        else:
            prompt = f"""
            Vật thể là khối chữ L:
            - Cao {h1}
            - Rộng {w1}
            - Dài {l2}

            Giải thích vì sao hình chiếu cạnh có một đường gạch ngang.
            """
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt
            )

        return response.text

    except Exception as e:
        return f"❌ Lỗi AI: {e}"

# ================== GIAO DIỆN CHÍNH ==================
st.title("🛠️ Phân tích vật thể 3D & hình chiếu")

col1, col2 = st.columns([1,1.5])

with col1:
    if uploaded_new_block:
        st.image(uploaded_new_block, caption="Ảnh vật thể 3D")
    else:
        st.plotly_chart(
            create_dynamic_L_block(h1,w1,l2),
            use_container_width=True
        )

with col2:
    if uploaded_new_block:
        st.subheader("📐 Hình chiếu minh họa (AI suy luận)")
        st.pyplot(draw_projection_from_image())
        st.caption("Hình chiếu dùng cho học tập – không yêu cầu đúng kích thước")
    else:
        st.pyplot(plot_dynamic_projections(h1,w1,l2))

st.divider()

if st.button("🤖 Nhờ AI phân tích"):
    with st.spinner("AI đang phân tích..."):
        if uploaded_new_block:
            result = ask_ai_analyze_block(image_file=uploaded_new_block)
        else:
            result = ask_ai_analyze_block(h1=h1,w1=w1,l2=l2)
        st.markdown(result)
