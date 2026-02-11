import streamlit as st
import pandas as pd
import numpy as np
import math
import re

# --- 1. 網頁設定 ---
st.set_page_config(
    page_title="MotoMatch", 
    page_icon="🛵", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. 讀取資料 ---
@st.cache_data 
def load_data():
    try:
        df = pd.read_csv("labeled_data.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=['Store', 'Brand', 'Style', 'Model', 'Price', 'Image_URL', 'Shop_Link'])
    
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
    df['Image_URL'] = df['Image_URL'].fillna('https://cdn-icons-png.flaticon.com/512/3097/3097180.png')
    for col in ['Store', 'Brand', 'Style']:
        if col not in df.columns: df[col] = '未知'
    df['id'] = df.index
    return df

df = load_data()

# --- 3. 核心演算法 (權重優化版) ---
@st.cache_resource
def build_similarity_model(data):
    if len(data) < 2: return np.zeros((len(data), len(data)))
    
    # 1. 價格標準化 (Price Weight: 40%)
    max_price = data['Price'].max() if data['Price'].max() > 0 else 1
    price_norm = data[['Price']] / max_price
    
    # 2. 品牌 One-Hot (Brand Weight: 20%)
    brands_ohe = pd.get_dummies(data['Brand']) * 0.5
    
    # 3. 風格 One-Hot (Style Weight: 40%)
    styles_ohe = pd.get_dummies(data['Style']) * 1.0
    
    # 合併特徵
    features = np.hstack([price_norm.values * 1.0, brands_ohe.values, styles_ohe.values])
    
    # 餘弦相似度計算
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1, norm)
    features_normalized = features / norm
    cosine_sim = np.dot(features_normalized, features_normalized.T)
    return cosine_sim

# --- 4. CSS 美化 ---
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; }
    .hero-box {
        background: linear-gradient(135deg, #0f172a 0%, #334155 100%);
        padding: 25px 20px; border-radius: 12px; color: white; text-align: center; margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0, 0.2);
    }
    .hero-title { font-size: 2.2rem; font-weight: 800; margin:0; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: white; border-radius: 10px; border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02); overflow: hidden;
        border-top: 4px solid #3b82f6; 
    }
    .card-content { padding: 12px; }
    .moto-title {
        font-weight: 700; font-size: 16px; color: #1e293b; margin: 5px 0; height: 45px;
        display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
    }
    .price-tag { color: #dc2626; font-weight: 800; font-size: 1.3rem; margin-top:5px; }
    .tag-box { display: flex; gap: 5px; margin-bottom: 8px; flex-wrap: wrap; }
    .pill { padding: 3px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; }
    .pill-loc { background-color: #eff6ff; color: #1d4ed8; }
    .pill-abs { background-color: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; } 
    .pill-ship { background-color: #f0fdf4; color: #15803d; border: 1px solid #bbf7d0; } 
    .stChatMessage { background-color: white; border-radius: 10px; border: 1px solid #e2e8f0; }
    div.stButton > button { width: 100%; border-radius: 8px; font-weight: bold; }
    [data-testid="stBottomBlock"] {
        padding-bottom: 0px !important;
        padding-top: 10px !important;
        background-color: #f8fafc;
    }
    footer { display: none !important; }
    .stChatInput { padding-bottom: 1rem !important; }
</style>
""", unsafe_allow_html=True)

# --- 5. 初始化 Session State ---
# 這裡改成 list 來存多台車
if 'liked_cars' not in st.session_state:
    st.session_state.liked_cars = [] 

if 'last_clicked_car' not in st.session_state:
    st.session_state.last_clicked_car = None

if 'chat_stage' not in st.session_state:
    st.session_state.chat_stage = 0
    st.session_state.chat_data = {}
    st.session_state.messages = [{"role": "assistant", "content": "你好！我是 MotoBot。(1/5) 請問您**居住在哪個縣市**？(例如：高雄)"}]

# --- 6. 側邊欄 (新增：關注清單功能) ---
with st.sidebar:
    st.markdown("### 📍 全域設定")
    all_stores = ["全台分店"] + sorted(list(df['Store'].unique()))
    selected_region = st.selectbox("您的所在位置", all_stores)
    
    st.divider()
    
    # ★★★ 新增：我的關注清單 ★★★
    liked_count = len(st.session_state.liked_cars)
    with st.expander(f"❤️ 我的關注清單 ({liked_count})", expanded=True):
        if liked_count == 0:
            st.caption("尚未關注任何車輛")
        else:
            for i, car in enumerate(st.session_state.liked_cars):
                st.markdown(f"**{i+1}. {car['Model']}**")
                st.caption(f"💲 {car['Price']:,.0f} | 📍 {car['Store']}")
                if st.button("❌ 移除", key=f"del_{i}"):
                    st.session_state.liked_cars.pop(i)
                    st.rerun()
            
            if st.button("🗑️ 清空全部", type="primary"):
                st.session_state.liked_cars = []
                st.session_state.last_clicked_car = None
                st.rerun()

    st.divider()
    st.warning("遇到問題嗎？")
    if st.button("🔄 重置 AI 對話", type="primary"):
        st.session_state.chat_stage = 0
        st.session_state.messages = [{"role": "assistant", "content": "你好！我是 MotoBot。(1/5) 請問您**居住在哪個縣市**？(例如：高雄)"}]
        st.session_state.chat_data = {}
        st.rerun()

# --- 7. 資料預處理 ---
current_df = df.copy()
if selected_region != "全台分店":
    current_df = current_df[current_df['Store'] == selected_region]

# --- 8. 主介面 ---
st.markdown(f"""
<div class="hero-box">
    <div class="hero-title">🛵 MotoMatch {selected_region if selected_region != '全台分店' else '全台'}</div>
    <div style="opacity:0.8; margin-top:5px;">AI 智慧媒合 · 懂車更懂你</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["💬 AI 購車顧問", "🏠 現場庫存", "🔮 猜你喜歡"])

# ==========================================
# Tab 1: AI 購車顧問
# ==========================================
with tab1:
    st.markdown("### 🤖 MotoBot 智慧助理")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.write(msg["content"])

    stage = st.session_state.chat_stage
    placeholders = {
        0: "請輸入居住縣市 (例如: 高雄)...",
        1: "請輸入預算 (限制 2萬 ~ 12萬)...",
        2: "例如: 跑山、買菜、通勤...",
        3: "請回答: 需要 / 不需要...",
        4: "請回答: 願意 / 不願意..."
    }
    
    if prompt := st.chat_input(placeholders.get(stage, "..."), key=f"chat_s{stage}"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
        
        response = ""
        should_rerun = True

        if stage == 0:
            st.session_state.chat_data['location'] = prompt
            response = f"收到，您在 **{prompt}**。(2/5) 請問您的購車**預算上限**是多少？(請輸入 **2萬 ~ 12萬** 之間的金額)"
            st.session_state.chat_stage = 1

        elif stage == 1:
            try:
                clean = prompt.replace(',', '').replace('萬', '0000')
                if len(clean) > 8: response = "😱 數字太大了！請輸入 12萬 以內的金額。"
                else:
                    nums = re.findall(r'\d+', clean)
                    if nums:
                        budget = int(nums[0])
                        if budget <= 100: budget *= 10000 
                        if 20000 <= budget <= 120000:
                            st.session_state.chat_data['budget'] = budget
                            response = f"好的，預算 **{budget/10000:.1f}萬** 以內。(3/5) 請問您的**主要用途**是？"
                            st.session_state.chat_stage = 2
                        else:
                            if budget > 120000: response = "💰 預算太高了！我們只推薦 12萬 以內的車款。"
                            else: response = "💸 預算太低囉！2萬 以下很難買到好車。"
                    else: response = "不好意思，我沒讀到數字。"
            except: response = "請輸入有效數字。"

        elif stage == 2:
            st.session_state.chat_data['usage'] = prompt
            tag = "標準"
            if any(k in prompt for k in ["跑山", "運動"]): tag = "⛰️ 跑山"
            elif any(k in prompt for k in ["買菜", "代步"]): tag = "🛒 代步"
            elif any(k in prompt for k in ["長途", "環島"]): tag = "🛣️ 長途"
            elif any(k in prompt for k in ["檔車"]): tag = "🏍️ 檔車"
            st.session_state.chat_data['tag'] = tag
            response = f"了解 ({tag})。(4/5) 需要 ABS 嗎？"
            st.session_state.chat_stage = 3

        elif stage == 3:
            st.session_state.chat_data['abs'] = any(k in prompt for k in ["是", "要", "yes"])
            response = f"收到。(5/5) 若無車，願意付 $1500 運費嗎？"
            st.session_state.chat_stage = 4

        elif stage == 4:
            shipping = any(k in prompt for k in ["願意", "好", "ok"])
            final_df = df.copy()
            final_df = final_df[final_df['Price'] <= st.session_state.chat_data.get('budget', 120000)]
            if st.session_state.chat_data.get('abs'):
                final_df = final_df[final_df['Model'].str.contains("ABS", case=False, na=False)]
            
            count = len(final_df)
            response = f"🎉 分析完成！找到 {count} 台車。(請切換到庫存分頁查看)"
            st.session_state.chat_stage = 5
        
        elif stage == 5:
            st.session_state.chat_stage = 0
            st.session_state.messages = [{"role": "assistant", "content": "🔄 已重置。請問您居住在哪個縣市？"}]

        if stage != 4: st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)

# ==========================================
# Tab 2: 🏠 現場庫存 (關注按鈕升級)
# ==========================================
with tab2:
    col1, col2 = st.columns([3, 1])
    with col1: keyword = st.text_input("搜尋車名", placeholder="例如: 勁戰")
    with col2: max_budget = st.number_input("預算上限", value=150000, step=5000)

    filtered_df = current_df.copy()
    if keyword: filtered_df = filtered_df[filtered_df['Model'].str.contains(keyword, case=False)]
    filtered_df = filtered_df[filtered_df['Price'] <= max_budget]

    if filtered_df.empty:
        st.warning("無符合車輛。")
    else:
        ITEMS_PER_PAGE = 12
        if 'page_number' not in st.session_state: st.session_state.page_number = 1
        total_pages = math.ceil(len(filtered_df) / ITEMS_PER_PAGE)
        
        st.caption(f"共找到 {len(filtered_df)} 台車 | 目前第 {st.session_state.page_number} / {total_pages} 頁")

        start_idx = (st.session_state.page_number - 1) * ITEMS_PER_PAGE
        end_idx = start_idx + ITEMS_PER_PAGE
        display_df = filtered_df.iloc[start_idx:end_idx]

        for i in range(0, len(display_df), 3):
            cols = st.columns(3)
            batch = display_df.iloc[i:i+3]
            for col, (_, row) in zip(cols, batch.iterrows()):
                with col:
                    with st.container(border=True):
                        st.image(row['Image_URL'], use_container_width=True)
                        st.markdown(f"""<div class="card-content">
                            <div class="tag-box"><span class="pill pill-loc">{row["Store"]}</span></div>
                            <div class="moto-title">{row["Model"]}</div>
                            <div class="price-tag">${row["Price"]:,.0f}</div>
                        </div>""", unsafe_allow_html=True)
                        
                        # 關注按鈕邏輯
                        c_btn1, c_btn2 = st.columns([1, 1])
                        with c_btn1:
                            # 檢查是否已在清單中
                            is_liked = any(c['id'] == row['id'] for c in st.session_state.liked_cars)
                            btn_label = "❤️ 已關注" if is_liked else "🤍 關注"
                            
                            if st.button(btn_label, key=f"like_{row['id']}", disabled=is_liked):
                                # 1. 加入清單
                                st.session_state.liked_cars.append(row.to_dict())
                                # 2. 設定為「最後點擊」，觸發推薦
                                st.session_state.last_clicked_car = row.to_dict()
                                st.rerun()
                                
                        with c_btn2:
                            st.link_button("查看", row['Shop_Link'], use_container_width=True)

        st.markdown("<br><br>", unsafe_allow_html=True)
        cols = st.columns(5)
        if cols[1].button("◀", key="prev"): st.session_state.page_number = max(1, st.session_state.page_number-1); st.rerun()
        with cols[2]: st.write(f"第 {st.session_state.page_number} 頁")
        if cols[3].button("▶", key="next"): st.session_state.page_number = min(total_pages, st.session_state.page_number+1); st.rerun()

# ==========================================
# Tab 3: 🔮 猜你喜歡 (6台推薦版)
# ==========================================
with tab3:
    if not st.session_state.liked_cars:
        st.info("👋 您還沒有關注任何車輛！")
        st.markdown("請回到 **「🏠 現場庫存」** 分頁，點擊 **「🤍 關注」** 按鈕，我們會根據您的收藏進行推薦。")
    else:
        # 使用最後一次加入關注的車作為推薦基準
        target_car = st.session_state.last_clicked_car
        # 如果是剛打開網頁且有歷史紀錄，預設取清單最後一台
        if target_car is None and st.session_state.liked_cars:
            target_car = st.session_state.liked_cars[-1]
            
        st.success(f"正在根據您最新關注的 **【{target_car['Model']}】** 進行推薦...")
        
        local_sim = build_similarity_model(current_df)
        
        try:
            target_idx_list = current_df.index[
                (current_df['Model'] == target_car['Model']) & 
                (current_df['Price'] == target_car['Price'])
            ].tolist()
            
            if not target_idx_list:
                st.warning("資料庫更新中，請重新關注其他車輛。")
            else:
                target_idx = target_idx_list[0]
                # ★★★ 擴充推薦數量：取前 6 名 (索引 1~7) ★★★
                scores = sorted(list(enumerate(local_sim[target_idx])), key=lambda x: x[1], reverse=True)[1:7]
                
                st.divider()
                st.markdown("### 🔥 AI 精選 6 款推薦")
                
                # 自動排版：每行 3 台，顯示 2 行
                cols = st.columns(3)
                for i, (idx, score) in enumerate(scores):
                    if idx < len(current_df):
                        r = current_df.iloc[idx]
                        with cols[i % 3]: # 餘數 0,1,2 自動換行
                            with st.container(border=True):
                                st.image(r['Image_URL'], use_container_width=True)
                                st.caption(f"🧬 相似度 {int(score*100)}%")
                                st.markdown(f"**{r['Model']}**")
                                st.markdown(f'<div class="price-tag">${r["Price"]:,.0f}</div>', unsafe_allow_html=True)
                                st.link_button("查看", r['Shop_Link'], use_container_width=True)
                                
        except Exception as e:
            st.error(f"運算發生錯誤：{e}")

# ==========================================
# Footer
# ==========================================
st.divider()
st.markdown("<div style='text-align:center; color:#94a3b8; font-size: 0.8rem; margin-bottom: 80px;'>MotoMatch AI System © 2026</div>", unsafe_allow_html=True)