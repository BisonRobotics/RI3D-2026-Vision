import time
import ntcore
import streamlit as st

st.set_page_config(layout="wide")
st.title("Vision Feed (NetworkTables JPEG)")

# --- NT setup ---
inst = ntcore.NetworkTableInstance.getDefault()
inst.setServer("vision")              # docker-compose service name
inst.startClient4("vision-ui")

table = inst.getTable("Vision")
img_sub = table.getRawTopic("image").subscribe("image/jpeg", b"")
id_sub = table.getIntegerArrayTopic("ids").subscribe([])

# --- UI elements ---
status = st.empty()
frame_slot = st.empty()

last_len = 0
last_t = time.time()
frames = 0
fps = 0.0

# Streamlit runs top-to-bottom; keep it alive
while True:
    data = img_sub.get()
    ids = id_sub.get()
    connected = inst.isConnected()

    if data:
        # Streamlit can render JPEG bytes directly
        frame_slot.image(data, channels="BGR")  # channels param is ignored for JPEG bytes, but harmless
        frames += 1
        last_len = len(data)

    now = time.time()
    if now - last_t >= 1.0:
        fps = frames / (now - last_t)
        frames = 0
        last_t = now

    status.write(
        f"Connected: **{connected}** | Last frame bytes: **{last_len}** | IDs: {ids}"
    )

    time.sleep(0.02)  # small sleep to avoid pegging CPU
