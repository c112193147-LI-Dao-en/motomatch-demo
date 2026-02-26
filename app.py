import streamlit as st
import pandas as pd
import numpy as np
import os
import re

# --- 1. 網頁設定與數據載入 ---
st.set_page_config(page_title="MotoMatch AI - 智慧購車顧問", page_icon="🛵", layout="wide")

# 台灣縣市白名單 (用於嚴格地址攔截與地理推薦)
taiwan_cities = [
    "台北", "新北", "基隆", "桃園", "新竹", "苗栗", "台中", "彰化", 
    "南投", "雲林", "嘉義", "台南", "高雄", "屏東", "宜蘭", "花蓮", 
    "台東", "澎湖", "金門", "連江"
]

@st.cache_data 
def load_data():
    try:
        # 讀取標記資料
        df = pd.read_csv("labeled_data.csv")
        # 品牌自動識別 (確保規格表不顯示「其他」)
        brand_map = {
            "山葉": ["YAMAHA", "山葉", "R15", "MT", "勁戰", "FORCE", "BWS", "AUGUR"],
            "三陽": ["SYM", "三陽", "DRG", "JET", "曼巴", "MMBCU", "FIDDLE", "CLBCU"],
            "光陽": ["KYMCO", "光陽", "KRV", "雷霆", "MANY", "VJR", "ROMA"],
            "偉士牌": ["VESPA", "偉士牌", "春天", "衝刺", "PRIMAVERA", "SPRINT"],
            "睿能": ["GOGORO", "睿能", "VIVA", "MIX", "DELIGHT"]
        }
        def fix_brand(row):
            m = str(row['Model']).upper()
            for b_name, keywords in brand_map.items():
                if any(k.upper() in m for k in keywords): return b_name
            return row.get('Brand', '其他')
        
        df['Brand'] = df.apply(fix_brand, axis=1)
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
        df['id'] = df.index
        if 'Style' not in df.columns: df['Style'] = "一般通勤"
        return df
    except:
        return pd.DataFrame(columns=['id', 'Store', 'Brand', 'Style', 'Model', 'Price', 'Image_URL', 'Shop_Link'])

df = load_data()

# --- 2. 相似度演算法 (推薦核心) ---
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
    return np.dot(features_normalized, features_normalized.T)

# --- 3. 初始化 Session State ---
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "你好！我是 MotoBot。(1/5) 請問您居住在哪個縣市？"}]
if 'chat_stage' not in st.session_state: st.session_state.chat_stage = 0
if 'chat_data' not in st.session_state: st.session_state.chat_data = {}
if 'liked_cars' not in st.session_state: st.session_state.liked_cars = []
if 'view_history' not in st.session_state: st.session_state.view_history = []
if 'current_page' not in st.session_state: st.session_state.current_page = 1

# --- 4. 側邊欄 ---
with st.sidebar:
    st.title("📍 系統設定")
    selected_region = st.selectbox("所在分店", ["全台分店"] + sorted(list(df['Store'].unique() if not df.empty else [])))
    st.divider()
    liked_count = len(st.session_state.liked_cars)
    with st.expander(f"❤️ 關注清單 ({liked_count})", expanded=True):
        if not st.session_state.liked_cars: st.caption("尚未收藏車輛")
        else:
            for i, car in enumerate(st.session_state.liked_cars):
                st.write(f"**{car['Model']}**")
                if st.button("❌ 移除", key=f"side_del_{car['id']}"):
                    st.session_state.liked_cars.pop(i); st.rerun()

# --- 5. 主介面標籤頁 ---
st.title("🛵 MotoMatch AI 智慧導購")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 AI 顧問", "🏠 現場庫存", "🔮 猜你喜歡", "⚖️ 規格比較", "🕒 最近瀏覽"])

# ==========================================
# Tab 1: 💬 AI 顧問 (預算、ABS與地理過濾細節)
# ==========================================
with tab1:
    if st.button("🔄 重製對話"):
        st.session_state.chat_stage = 0
        st.session_state.chat_data = {}
        st.session_state.messages = [{"role": "assistant", "content": "你好！我是 MotoBot。(1/5) 請問您居住在哪個縣市？"}]
        st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.write(msg["content"])

    stage = st.session_state.chat_stage
    if stage < 5 and (prompt := st.chat_input("請回答...")):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # --- Stage 0: 嚴格地址攔截 ---
        if stage == 0:
            user_loc = next((city for city in taiwan_cities if city in prompt), None)
            if user_loc:
                st.session_state.chat_data['location'] = user_loc
                st.session_state.chat_stage = 1
                response = f"收到，您在 **{user_loc}**。(2/5) 預算上限是多少？(例如：8萬 或 85000)"
            else:
                response = "📍 抱歉，目前的服務僅限台灣地區。請輸入正確的台灣縣市名稱。"
        
        # --- Stage 1: 預算數字解析 ---
        elif stage == 1:
            try:
                clean = prompt.replace('萬', '0000').replace(',', '').replace(' ', '')
                nums = re.findall(r'\d+', clean)
                val = int(nums[0])
                final_budget = val * 10000 if val < 200 else val
                st.session_state.chat_data['budget'] = final_budget
                st.session_state.chat_stage = 2
                response = f"預算已設定為 **${final_budget:,.0f}**。接下來 (3/5) 主要用途是？"
            except:
                response = "🔢 請輸入有效的數字金額。"
        
        # --- Stage 2: 用途 ---
        elif stage == 2:
            st.session_state.chat_data['usage'] = prompt
            st.session_state.chat_stage = 3
            response = "(4/5) 需要 **ABS 安全系統** 嗎？(如果不清楚可以問：什麼是 ABS？)"
        
        # --- Stage 3: ABS 科普細節 ---
        elif stage == 3:
            if any(k in prompt for k in ["什麼", "不懂", "?", "科普", "是啥"]):
                response = "🛡️ **MotoBot 小百科**：ABS 能防止急煞時輪胎鎖死導致打滑，是雨天或緊急狀況的保命關鍵！您覺得需要配備嗎？"
            else:
                st.session_state.chat_data['abs'] = any(k in prompt for k in ["要", "是", "需", "有"])
                st.session_state.chat_stage = 4
                response = "(5/5) 若心儀車款在其他縣市，願意付 $1500 跨店調車費嗎？"
        
        # --- Stage 4: 地理調度意願 ---
        elif stage == 4:
            is_negative = any(n in prompt for n in ["不", "否", "沒", "拒絕"])
            st.session_state.chat_data['shipping'] = not is_negative
            st.session_state.chat_stage = 5
            response = "🎉 分析完成！推薦結果如下："
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    if stage == 5:
        st.divider()
        u_loc = st.session_state.chat_data.get('location', '')
        u_shipping = st.session_state.chat_data.get('shipping', True)
        u_budget = st.session_state.chat_data.get('budget', 150000)
        
        # 嚴格篩選邏輯
        rec_df = df[df['Price'] <= u_budget]
        if not u_shipping:
            rec_df = rec_df[rec_df['Store'].str.contains(u_loc, na=False)]
        
        if rec_df.empty:
            st.warning(f"目前在 **{u_loc}** 門市暫無符合預算的車款。建議調整條件或選擇願意調度。")
        else:
            cols = st.columns(3)
            for i, (_, row) in enumerate(rec_df.head(6).iterrows()):
                with cols[i % 3]:
                    with st.container(border=True):
                        st.image(row['Image_URL'], use_container_width=True)
                        st.subheader(f"💰 ${int(row['Price']):,}")
                        st.write(f"**{row['Model']}**")
                        st.caption(f"📍 {row['Store']}")
                        st.link_button("👉 查看詳情", row['Shop_Link'], use_container_width=True)

# ==========================================
# Tab 2: 🏠 現場庫存 (1-60 分頁)
# ==========================================
with tab2:
    current_df = df[df['Store'] == selected_region] if selected_region != "全台分店" else df
    items_per_page = 12
    total_pages = max(1, (len(current_df) - 1) // items_per_page + 1)
    if st.session_state.current_page > total_pages: st.session_state.current_page = 1
    
    start_idx = (st.session_state.current_page - 1) * items_per_page
    page_df = current_df.iloc[start_idx : start_idx + items_per_page]
    
    cols = st.columns(3)
    for i, (_, row) in enumerate(page_df.iterrows()):
        with cols[i % 3]:
            with st.container(border=True):
                st.image(row['Image_URL'], use_container_width=True)
                st.subheader(f"💰 NT$ {int(row['Price']):,}")
                st.markdown(f"**{row['Model']}**")
                
                c1, c2 = st.columns(2)
                if c1.button("❤️ 關注", key=f"fav_{row['id']}"):
                    car_dict = row.to_dict()
                    if car_dict['id'] not in [c['id'] for c in st.session_state.liked_cars]:
                        st.session_state.liked_cars.append(car_dict)
                    if car_dict['id'] not in [c['id'] for c in st.session_state.view_history]:
                        st.session_state.view_history.append(car_dict)
                    st.rerun()
                c2.link_button("🌐 網站", row['Shop_Link'], use_container_width=True)

    st.divider()
    p_cols = st.columns(min(total_pages, 12) + 2)
    for i, p in enumerate(range(max(1, st.session_state.current_page-5), min(total_pages, st.session_state.current_page+5)+1)):
        label = f"★{p}" if p == st.session_state.current_page else str(p)
        if p_cols[i+1].button(label, key=f"pg_{p}"):
            st.session_state.current_page = p; st.rerun()

# ==========================================
# Tab 3: 🔮 猜你喜歡 (推薦演算法)
# ==========================================
with tab3:
    if not st.session_state.liked_cars: st.info("請先關注車子")
    else:
        target = st.session_state.liked_cars[-1]
        sim_model = build_similarity_model(df)
        idx = df[df['id'] == target['id']].index[0]
        scores = sorted(list(enumerate(sim_model[idx])), key=lambda x: x[1], reverse=True)[1:7]
        cols = st.columns(3)
        for i, (s_idx, _) in enumerate(scores):
            r = df.iloc[s_idx]
            with cols[i % 3]:
                with st.container(border=True):
                    st.image(r['Image_URL'], use_container_width=True)
                    st.subheader(f"💰 ${int(r['Price']):,}")
                    st.write(r['Model'])

# ==========================================
# Tab 4: ⚖️ 規格比較 & Tab 5: 最近瀏覽
# ==========================================
with tab4:
    st.header("⚖️ 車款規格對照")
    if len(st.session_state.liked_cars) < 2: st.info("請關注至少 2 台車。")
    else:
        comp_df = pd.DataFrame(st.session_state.liked_cars)[["Model", "Price", "Brand", "Store"]]
        comp_df.columns = ["型號", "售價", "品牌", "所在地"]
        comp_df["售價"] = comp_df["售價"].apply(lambda x: f"${int(x):,}")
        st.table(comp_df.set_index("型號").T)

with tab5:
    st.header("🕒 最近查看紀錄")
    if not st.session_state.view_history: st.info("尚無紀錄。")
    else:
        v_cols = st.columns(3)
        for i, car in enumerate(reversed(st.session_state.view_history[-9:])):
            with v_cols[i % 3]:
                with st.container(border=True):
                    st.image(car['Image_URL'], use_container_width=True)
                    st.write(f"**{car['Model']}**")
                    st.subheader(f"💰 ${int(car['Price']):,}")