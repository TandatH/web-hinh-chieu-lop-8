import streamlit as st
import matplotlib.pyplot as plt
from google import genai
import plotly.graph_objects as go
import numpy as np

# ======================================================
# 1. CẤU HÌNH TRANG
# ======================================================
st.set_page_config(
    page_title="Công Nghệ 8 – Hình Chiếu",
    page_icon="📐",
    layout="wide"
)

# ======================================================
# 2. CẤU HÌNH AI (ẨN NẾU CHƯA CÓ KEY)
# ======================================================
api_key = st.secrets.get("GEMINI_API_KEY", None)
ai_ready = False

if api_key:
    genai.configure(api_key=api_key)
    ai_ready = True

# ======================================================
# 3. HÀM TẠO KHỐI 3D
# ======================================================
def make_cube(x0, y0, z0, dx, dy, dz, color):
    x = np.array([0,1,1,0,0,1,1,0])*dx + x0
    y = np.array([0,0,1,1,0,0,1,1])*dy + y0
    z = np.array([0,0,0,0,1,1,1,1])*dz + z0

    return go.Mesh3d(
        x=x, y=y, z=z,
        i=[7,0,0,0,4,4,6,6,4,0,3,2],
        j=[3,4,1,2,5,6,5,2,0,1,6,3],
        k=[0,7,2,3,6,7,1,1,5,5,7,6],
        opacity=0.8,
        color=color
    )

def create_khoi_L():
    return go.Figure(data=[
        make_cube(0,0,0,1,1,3,"#FFAB91"),
        make_cube(1,0,0,1,1,1,"#80CBC4")
    ])

def create_khoi_hop():
    return go.Figure(data=[
        make_cube(0,0,0,2,1,1,"#90CAF9")
    ])

def format_3d(fig):
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data"
        ),
        margin=dict(l=0,r=0,t=0,b=0),
        height=320
    )
    return fig

# ======================================================
# 4. HÀM VẼ HÌNH CHIẾU 2D
# ======================================================
def plot_chieu_L():
    fig, ax = plt.subplots(1,3,figsize=(12,4))
    for a in ax:
        a.axis("off")
        a.set_aspect("equal")

    ax[0].set_title("Chiếu đứng")
    ax[0].plot([0,2,2,1,1,0,0],[0,0,1,1,3,3,0],"k",lw=3)

    ax[1].set_title("Chiếu bằng")
    ax[1].plot([0,2,2,0,0],[0,0,1,1,0],"k",lw=3)

    ax[2].set_title("Chiếu cạnh")
    ax[2].plot([0,1,1,0,0],[0,0,3,3,0],"k",lw=3)

    return fig

def plot_chieu_hop():
    fig, ax = plt.subplots(1,3,figsize=(12,4))
    for a in ax:
        a.axis("off")
        a.set_aspect("equal")

    for i in range(3):
        ax[i].plot([0,2,2,0,0],[0,0,1,1,0],"k",lw=3)

    ax[0].set_title("Chiếu đứng")
    ax[1].set_title("Chiếu bằng")
    ax[2].set_title("Chiếu cạnh")

    return fig

# ======================================================
# 5. GIAO DIỆN CHÍNH
# ======================================================
st.title("📐 Công Nghệ 8 – Thực Hành Hình Chiếu")
st.caption("Quan sát vật thể → Xoay khối 3D → Phân tích hình chiếu")

vat_the = st.selectbox(
    "🧱 Chọn vật thể",
    ["Khối chữ L", "Khối hộp"]
)

col1, col2 = st.columns([1,1.5])

# ======================================================
# 6. CỘT TRÁI: ẢNH + 3D
# ======================================================
with col1:
    if vat_the == "Khối chữ L":
        st.image("images/khoi_L.png", caption="Ảnh vật thể", width="stretch")
        fig3d = format_3d(create_khoi_L())

    else:
        st.image("images/khoi_hop.png", caption="Ảnh vật thể", width="stretch")
        fig3d = format_3d(create_khoi_hop())

    st.plotly_chart(fig3d, width="stretch")

# ======================================================
# 7. CỘT PHẢI: HÌNH CHIẾU
# ======================================================
with col2:
    st.success("📝 Ba hình chiếu vuông góc")

    if vat_the == "Khối chữ L":
        st.pyplot(plot_chieu_L())
    else:
        st.pyplot(plot_chieu_hop())

# ======================================================
# 8. AI – CHỈ HIỆN KHI CÓ KEY
# ======================================================
if ai_ready:
    st.divider()
    if st.button("🤖 Nhờ AI giải thích"):
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Giải thích ngắn gọn hình chiếu của {vat_the} cho học sinh lớp 8."
        st.markdown(model.generate_content(prompt).text)

st.caption("Ứng dụng học tập Công Nghệ 8 – Streamlit 2026")
