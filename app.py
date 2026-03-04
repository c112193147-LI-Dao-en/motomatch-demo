import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import json
from streamlit_javascript import st_javascript # 需安裝：pip install streamlit-javascript

# --- 1. 網頁設定與數據載入 ---
st.set_page_config(page_title="MotoMatch AI - 智慧購車顧問", page_icon="🛵", layout="wide")

@st.cache_data 
def load_data():
    try:
        df = pd.read_csv("labeled_data.csv")
        # 品牌識別與進階規格標籤 (ABS/TCS)
        brand_map = {"山葉": ["YAMAHA", "R15", "MT", "勁戰"], "三陽": ["SYM", "DRG", "JET", "曼巴"], "光陽": ["KYMCO", "KRV", "雷霆"]}
        def augment_data(row):
            m = str(row['Model']).upper()
            row['ABS'] = True if any(x in m for x in ["ABS", "GR", "MMBCU", "DRG", "AUGUR"]) else False
            row['TCS'] = True if any(x in m for x in ["TCS", "MMBCU", "DRG"]) else False
            for b, k in brand_map.items():
                if any(x.upper() in m for x in k): row['Brand'] = b; break
            return row
        df = df.apply(augment_data, axis=1)
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
        df['id'] = df.index
        return df
    except:
        return pd.DataFrame(columns=['id', 'Store', 'Brand', 'Model', 'Price', 'Image_URL', 'Shop_Link', 'ABS', 'TCS'])

df = load_data()

# --- 2. Cookie (LocalStorage) 持久化邏輯 ---
# 這裡使用 JavaScript 來讀取與寫入瀏覽器本地儲存，達成「隔天打開還在」的需求
def sync_cookie():
    # 讀取 LocalStorage
    js_read = "parent.localStorage.getItem('moto_history');"
    stored_data = st_javascript(js_read)
    if stored_data and 'view_history' not in st.session_state:
        st.session_state.view_history = json.loads(stored_data)

def save_to_cookie(data_list):
    # 寫入 LocalStorage
    js_write = f"parent.localStorage.setItem('moto_history', '{json.dumps(data_list)}');"
    st_javascript(js_write)

# --- 3. 推薦演算法 (死忠於歷史總和) ---
def get_loyal_recommendations(history, data):
    if not history: return pd.DataFrame()
    # 計算歷史特徵總和
    hist_df = pd.DataFrame(history)
    avg_price = hist_df['Price'].mean()
    fav_brand = hist_df['Brand'].mode()[0] if not hist_df['Brand'].empty else None
    prefers_abs = (hist_df['ABS'] == True).sum() > (len(hist_df) / 2)
    
    # 過濾與歷史特徵相近的車
    mask = (data['Price'] >= avg_price * 0.8) & (data['Price'] <= avg_price * 1.2)
    if fav_brand: mask &= (data['Brand'] == fav_brand)
    if prefers_abs: mask &= (data['ABS'] == True)
    
    return data[mask].head(6)

# --- 4. 初始化 Session State ---
sync_cookie() # 啟動時讀取 Cookie
if 'view_history' not in st.session_state: st.session_state.view_history = []
if 'liked_cars' not in st.session_state: st.session_state.liked_cars = []
if 'chat_stage' not in st.session_state: st.session_state.chat_stage = 0
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "你好！我是 MotoBot。我已準備好為您紀錄購車偏好，您同意開啟 Cookie 記憶功能嗎？"}]

# --- 5. 主介面標籤頁配置 ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 AI 顧問", "🏠 現場庫存", "🔮 猜你喜歡", "⚖️ 規格比較", "🕒 最近瀏覽"])

# ==========================================
# Tab 2: 🏠 現場庫存 (大卡片)
# ==========================================
with tab2:
    cols = st.columns(3)
    for i, (_, row) in enumerate(df.head(12).iterrows()):
        with cols[i % 3]:
            with st.container(border=True):
                st.image(row['Image_URL'], use_container_width=True)
                st.subheader(f"💰 NT$ {int(row['Price']):,}")
                st.write(f"**{row['Model']}**")
                if st.button("❤️ 關注並紀錄", key=f"main_{row['id']}"):
                    car_dict = row.to_dict()
                    # 更新關注清單
                    if car_dict['id'] not in [c['id'] for c in st.session_state.liked_cars]:
                        st.session_state.liked_cars.append(car_dict)
                    # 更新瀏覽紀錄 (Cookie)
                    if car_dict['id'] not in [c['id'] for c in st.session_state.view_history]:
                        st.session_state.view_history.append(car_dict)
                        save_to_cookie(st.session_state.view_history)
                    st.rerun()

# ==========================================
# Tab 5: 🕒 最近瀏覽 (中/小卡片 + ABS/TCS 標籤)
# ==========================================
with tab5:
    st.header("🕒 您所有看過的愛車 (已儲存至 Cookie)")
    if st.button("🗑️ 清除所有記憶"):
        st.session_state.view_history = []
        save_to_cookie([])
        st.rerun()
    
    if not st.session_state.view_history:
        st.info("尚無歷史紀錄，您的購車 DNA 正在孵化中...")
    else:
        # 分類顯示：今天、更早以前
        st.write("### 📅 歷史總覽")
        v_cols = st.columns(5) # 中/小卡片：一列 5 欄
        for i, car in enumerate(reversed(st.session_state.view_history)):
            with v_cols[i % 5]:
                with st.container(border=True):
                    st.image(car['Image_URL'], use_container_width=True)
                    st.caption(f"**{car['Model']}**")
                    st.write(f"NT$ {int(car['Price']):,}")
                    # A 方案標籤：顯示 ABS / TCS
                    tag_html = ""
                    if car.get('ABS'): tag_html += "🛡️ ABS "
                    if car.get('TCS'): tag_html += "🛣️ TCS"
                    st.markdown(f"<small>{tag_html}</small>", unsafe_allow_html=True)

# ==========================================
# Tab 3: 🔮 猜你喜歡 (死忠推薦)
# ==========================================
with tab3:
    st.header("🔮 根據您的購車 DNA 推薦")
    if not st.session_state.view_history:
        st.warning("⚠️ 提示：請先在商場點擊❤️，系統將根據您的長期歷史總和進行死忠推薦。")
    else:
        # 顯示 DNA 分析訊息
        hist_df = pd.DataFrame(st.session_state.view_history)
        fav_brand = hist_df['Brand'].mode()[0]
        st.success(f"MotoBot 觀察：您歷史紀錄中有 {len(hist_df)} 筆資料，其中最常看的是 **{fav_brand}**。")
        
        recs = get_loyal_recommendations(st.session_state.view_history, df)
        r_cols = st.columns(3)
        for i, (_, row) in enumerate(recs.iterrows()):
            with r_cols[i % 3]:
                with st.container(border=True):
                    st.image(row['Image_URL'], use_container_width=True)
                    st.write(f"**{row['Model']}**")
                    st.write(f"💰 ${int(row['Price']):,}")

# ==========================================
# Tab 4: ⚖️ 規格比較 (✅/❌ 直觀比較)
# ==========================================
with tab4:
    st.header("⚖️ 規格直觀比對")
    if len(st.session_state.liked_cars) < 2:
        st.info("請至少關注 2 台車以進行比對。")
    else:
        comp_df = pd.DataFrame(st.session_state.liked_cars)[["Model", "Price", "Brand", "ABS", "TCS"]]
        comp_df['ABS'] = comp_df['ABS'].map({True: "✅ 有", False: "❌ 無"})
        comp_df['TCS'] = comp_df['TCS'].map({True: "✅ 有", False: "❌ 無"})
        st.table(comp_df.set_index("Model").T)