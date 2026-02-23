import streamlit as st
import pandas as pd  # 修正原本 pd 未定義的問題
import numpy as np
import datetime      # 用於行為紀錄時間
import os            # 用於檔案路徑檢查
import re

# --- 1. 網頁設定 ---
st.set_page_config(
    page_title="MotoMatch AI - 智慧購車顧問", 
    page_icon="🛵", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. 數據紀錄函數 (產學數據收集核心) ---
def log_action(action_type, details):
    """
    僅在使用者同意 Cookie 後，將行為匿名紀錄至 CSV，供期末分析報告使用。
    """
    if st.session_state.get('cookie_consent', False):
        log_file = "user_behavior_logs.csv"
        log_data = {
            "timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "location": [st.session_state.get('chat_data', {}).get('location', 'Unknown')],
            "action": [action_type],
            "details": [details],
            "budget": [st.session_state.get('chat_data', {}).get('budget', 0)]
        }
        log_df = pd.DataFrame(log_data)
        # 檔案不存在就建標題，存在就續寫 (append)
        if not os.path.isfile(log_file):
            log_df.to_csv(log_file, index=False, encoding='utf-8-sig')
        else:
            log_df.to_csv(log_file, mode='a', index=False, header=False, encoding='utf-8-sig')

# --- 3. 讀取資料 ---
@st.cache_data 
def load_data():
    try:
        # 讀取你的主要數據檔案
        df = pd.read_csv("labeled_data.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=['id', 'Store', 'Brand', 'Style', 'Model', 'Price', 'Image_URL', 'Shop_Link'])
    
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
    df['id'] = df.index
    return df

df = load_data()

# --- 4. 核心演算法 (餘弦相似度) ---
@st.cache_resource
def build_similarity_model(data):
    if len(data) < 2: return np.zeros((len(data), len(data)))
    max_price = data['Price'].max() if data['Price'].max() > 0 else 1
    price_norm = data[['Price']] / max_price
    brands_ohe = pd.get_dummies(data['Brand']) * 0.5
    styles_ohe = pd.get_dummies(data['Style']) * 1.0
    features = np.hstack([price_norm.values, brands_ohe.values, styles_ohe.values])
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1, norm)
    features_normalized = features / norm
    cosine_sim = np.dot(features_normalized, features_normalized.T)
    return cosine_sim

# --- 5. 初始化 Session State ---
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "你好！我是 MotoBot。(1/5) 請問您**居住在哪個縣市**？(例如：高雄)"}]
if 'chat_stage' not in st.session_state: st.session_state.chat_stage = 0
if 'chat_data' not in st.session_state: st.session_state.chat_data = {}
if 'liked_cars' not in st.session_state: st.session_state.liked_cars = []
if 'last_clicked_car' not in st.session_state: st.session_state.last_clicked_car = None
if 'cookie_consent' not in st.session_state: st.session_state.cookie_consent = False

# --- 6. CSS 樣式 ---
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; }
    .hero-box {
        background: linear-gradient(135deg, #0f172a 0%, #334155 100%);
        padding: 25px; border-radius: 12px; color: white; text-align: center; margin-bottom: 20px;
    }
    .price-tag { color: #dc2626; font-weight: 800; font-size: 1.2rem; }
    footer { display: none !important; }
</style>
""", unsafe_allow_html=True)

# --- 7. Cookie 同意聲明 (置頂顯示，獲取授權後隱藏) ---
if not st.session_state.cookie_consent:
    with st.container():
        st.warning("🍪 **數據分析授權聲明**")
        st.markdown("為了優化推薦體驗，本系統會匿名記錄行為數據。點擊代表您同意專案分析使用。")
        if st.button("我同意並繼續使用"):
            st.session_state.cookie_consent = True
            st.rerun()

# --- 8. 側邊欄：僅保留關注清單 ---
with st.sidebar:
    st.title("📍 系統設定")
    selected_region = st.selectbox("所在分店", ["全台分店"] + sorted(list(df['Store'].unique() if not df.empty else [])))
    st.divider()
    liked_count = len(st.session_state.liked_cars)
    with st.expander(f"❤️ 我的關注清單 ({liked_count})", expanded=True):
        if liked_count == 0: st.caption("尚未收藏車輛")
        else:
            for i, car in enumerate(st.session_state.liked_cars):
                st.markdown(f"**{car['Model']}**")
                if st.button("❌ 移除", key=f"del_{car['id']}"):
                    st.session_state.liked_cars.pop(i); st.rerun()

# --- 9. 主介面佈局 ---
st.markdown('<div class="hero-box"><h1>🛵 MotoMatch AI</h1><p>HTTPS 加密 · 智慧導購與數據分析系統</p></div>', unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["💬 AI 購車顧問", "🏠 現場庫存", "🔮 猜你喜歡"])

# ==========================================
# Tab 1: AI 購車顧問 (整合按鈕與原地顯示)
# ==========================================
with tab1:
    col_btn1, col_btn2 = st.columns([5, 1])
    with col_btn2:
        if st.button("🔄 重製對話", use_container_width=True):
            st.session_state.chat_stage = 0
            st.session_state.chat_data = {}
            st.session_state.messages = [{"role": "assistant", "content": "你好！我是 MotoBot。(1/5) 請問您**居住在哪個縣市**？"}]
            st.rerun()
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.write(msg["content"])

    stage = st.session_state.chat_stage
    if prompt := st.chat_input("請輸入您的回答...", key=f"chat_input_s{stage}"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = ""
        
        if stage == 0:
            if re.search(r'[a-zA-Z]', prompt): response = "🚫 請輸入中文縣市名稱。"
            else:
                st.session_state.chat_data['location'] = prompt
                response = f"收到，您在 {prompt}。(2/5) 預算上限是多少？(2萬~12萬)"
                st.session_state.chat_stage = 1
        elif stage == 1:
            try:
                clean = prompt.replace('萬', '0000').replace(',', '')
                budget = int(re.findall(r'\d+', clean)[0])
                if budget <= 120: budget *= 10000 
                if 20000 <= budget <= 150000:
                    st.session_state.chat_data['budget'] = budget
                    response = f"好的，預算 **${budget:,.0f}** 內。(3/5) 主要用途是？"
                    st.session_stage = 2
                    st.session_state.chat_stage = 2
                else: response = "💰 預算請在 2萬~12萬 之間。"
            except: response = "🔢 請輸入有效數字。"
        elif stage == 2:
            st.session_state.chat_data['usage'] = prompt
            response = "(4/5) 需要 ABS 嗎？(提示：可問「什麼是 ABS」)"
            st.session_state.chat_stage = 3
        elif stage == 3:
            if any(k in prompt for k in ["什麼", "科普"]): response = "🛡️ **小科普：什麼是 ABS？**\n能在緊急煞車時防止輪胎鎖死。您需要嗎？"
            else:
                st.session_state.chat_data['abs'] = any(k in prompt for k in ["是", "要", "需"])
                response = "(5/5) 最後一題：願意付 $1500 運費調車嗎？"
                st.session_state.chat_stage = 4
        elif stage == 4:
            st.session_state.chat_stage = 5
            response = "🎉 分析完成！推薦車款如下："
            log_action("AI_SEARCH", f"Budget:{st.session_state.chat_data.get('budget')}")

        if stage != 5:
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

    if st.session_state.chat_stage == 5:
        st.divider()
        budget = st.session_state.chat_data.get('budget', 120000)
        final_df = df[df['Price'] <= budget].copy()
        if st.session_state.chat_data.get('abs'):
            final_df = final_df[final_df['Model'].str.contains("ABS", case=False, na=False)]
        
        if not final_df.empty:
            res_cols = st.columns(3)
            for i, (_, row) in enumerate(final_df.head(6).iterrows()):
                with res_cols[i % 3]:
                    with st.container(border=True):
                        st.image(row['Image_URL'], use_container_width=True)
                        st.markdown(f"**{row['Model']}**\n\n<div class='price-tag'>${row['Price']:,.0f}</div>", unsafe_allow_html=True)
                        if st.link_button("查看詳情", row['Shop_Link'], use_container_width=True):
                            log_action("VIEW", row['Model'])
        else:
            st.warning("😢 找不到完全吻合車款，請點擊上方重置調整需求。")

# ==========================================
# Tab 2 & 3: 現場庫存與猜你喜歡 (紀錄關注行為)
# ==========================================
with tab2:
    current_df = df[df['Store'] == selected_region] if selected_region != "全台分店" else df
    cols = st.columns(3)
    for i, (_, row) in enumerate(current_df.head(12).iterrows()):
        with cols[i % 3]:
            with st.container(border=True):
                st.image(row['Image_URL'], use_container_width=True)
                st.markdown(f"**{row['Model']}**\n\n<div class='price-tag'>${row['Price']:,.0f}</div>", unsafe_allow_html=True)
                if st.button("❤️ 關注", key=f"lk_{row['id']}"):
                    if row['id'] not in [c['id'] for c in st.session_state.liked_cars]:
                        st.session_state.liked_cars.append(row.to_dict())
                        st.session_state.last_clicked_car = row.to_dict()
                        log_action("LIKE", row['Model'])
                        st.rerun()

with tab3:
    if not st.session_state.liked_cars: st.info("💡 請先關注感興趣的車輛。")
    else:
        target = st.session_state.last_clicked_car or st.session_state.liked_cars[-1]
        sim_model = build_similarity_model(df)
        idx = df[df['id'] == target['id']].index[0]
        scores = sorted(list(enumerate(sim_model[idx])), key=lambda x: x[1], reverse=True)[1:7]
        cols = st.columns(3)
        for i, (s_idx, score) in enumerate(scores):
            r = df.iloc[s_idx]
            with cols[i % 3]:
                with st.container(border=True):
                    st.image(r['Image_URL'], use_container_width=True)
                    st.caption(f"🧬 相似度 {int(score*100)}%")
                    st.markdown(f"**{r['Model']}**")
                    if st.link_button("查看車輛", r['Shop_Link'], use_container_width=True):
                        log_action("VIEW", r['Model'])

# ==========================================
# Footer: 責任歸屬與免責聲明
# ==========================================
st.divider()
with st.expander("⚖️ 責任歸屬界定與免責聲明 [產學合作技術展示]"):
    st.markdown(f"""
    <div style="font-size: 0.85rem; color: #64748b; line-height: 1.8;">
    1. <b>數據合規：</b> 本系統在獲得授權後匿名記錄行為數據。<br>
    2. <b>資訊準確：</b> 庫存資料以 <b>貳輪嶼門市現場</b> 為準。<br>
    3. <b>責任界定：</b> 本平台為媒合工具，不參與交易，亦不負擔任何交易糾紛責任。
    </div>
    """, unsafe_allow_html=True)

st.markdown("""<div style='text-align:center; color:#94a3b8; font-size: 0.75rem; margin-top: 20px;'>
MotoMatch AI System © 2026 | MIS Team 專案研發<br>數據源：貳輪嶼二手機車連鎖</div>""", unsafe_allow_html=True)