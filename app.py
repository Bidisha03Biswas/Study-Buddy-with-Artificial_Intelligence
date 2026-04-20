import streamlit as st
import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from groq import Groq
from ml_models_tab import render_ml_models_tab

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StudyBuddy AI — ML Edition",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif !important; }
.stApp { background: #0a0a12; }
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 16px; padding: 24px 32px; margin-bottom: 20px;
}
.main-header h1 { color: #a78bfa; font-size: 26px; font-weight: 700; margin: 0; }
.main-header p  { color: #8888aa; font-size: 13px; margin: 6px 0 0; }
.api-banner {
    background: linear-gradient(135deg, rgba(52,211,153,0.1), rgba(56,189,248,0.08));
    border: 1px solid rgba(52,211,153,0.3); border-radius: 12px;
    padding: 12px 18px; margin-bottom: 16px; font-size: 13px; color: #34d399;
}
.chat-msg-user {
    background: linear-gradient(135deg, rgba(108,99,255,0.2), rgba(56,189,248,0.12));
    border: 1px solid rgba(108,99,255,0.35);
    border-radius: 16px 16px 4px 16px;
    padding: 14px 18px; margin: 8px 0; color: #e8e8f8; font-size: 14px; line-height: 1.7;
}
.chat-msg-bot {
    background: #1e1e32; border: 1px solid rgba(120,120,255,0.15);
    border-radius: 16px 16px 16px 4px;
    padding: 14px 18px; margin: 8px 0; color: #e8e8f8; font-size: 14px; line-height: 1.7;
}
.msg-label-user { color: #f472b6; font-size: 11px; font-weight: 600; letter-spacing: 1px; margin-bottom: 4px; text-align: right; }
.msg-label-bot  { color: #6c63ff; font-size: 11px; font-weight: 600; letter-spacing: 1px; margin-bottom: 4px; }
.metric-card { background: #1e1e32; border: 1px solid rgba(120,120,255,0.15); border-radius: 12px; padding: 14px 16px; text-align: center; }
.metric-card .val { font-size: 24px; font-weight: 700; color: #a78bfa; }
.metric-card .lbl { font-size: 11px; color: #8888aa; margin-top: 2px; }
.concept-tag { display: inline-block; background: rgba(108,99,255,0.15); border: 1px solid rgba(108,99,255,0.3); border-radius: 20px; padding: 3px 10px; font-size: 11px; color: #a78bfa; margin: 2px; }
.stButton > button { background: linear-gradient(135deg, #6c63ff, #38bdf8) !important; color: white !important; border: none !important; border-radius: 10px !important; font-family: 'Sora', sans-serif !important; font-weight: 600 !important; }
.stTextArea textarea, .stTextInput input { background: #1e1e32 !important; border: 1px solid rgba(120,120,255,0.2) !important; border-radius: 12px !important; color: #e8e8f8 !important; font-family: 'Sora', sans-serif !important; }
div[data-testid="stSidebar"] { background: #0f0f1c !important; }
.stSelectbox > div > div { background: #1e1e32 !important; border-color: rgba(120,120,255,0.2) !important; color: #e8e8f8 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────
for key, default in {
    "messages": [], "q_count": 0, "streak": 0,
    "topics_covered": set(), "api_key": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Groq API helper ────────────────────────────────────────────────────────
from groq import Groq

def call_groq(api_key: str, conversation: list, system_prompt: str) -> str:
    client = Groq(api_key="add_your_groq_key_here")

    # Build messages (Groq uses OpenAI-style format)
    messages = [{"role": "system", "content": system_prompt}]

    for m in conversation:
        messages.append({
            "role": m["role"],  # "user" or "assistant"
            "content": m["content"]
        })

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",   # fast + free,   # fast + free
            messages=messages,
            temperature=0.7,
            max_tokens=1500,
        )

        return response.choices[0].message.content

    except Exception as e:
        raise Exception(f"Groq API error: {str(e)}")
# ─── Constants ────────────────────────────────────────────────────────────────
ML_CONCEPTS = [
    "Gradient Descent", "Backpropagation", "Attention Mechanism",
    "Regularization (L1/L2)", "Bias-Variance Tradeoff", "CNNs",
    "RNNs / LSTMs", "Transfer Learning",
]

MOTIVATIONAL_QUOTES = [
    "Every ML expert was once confused by backprop. Keep going! 🚀",
    "Build things that fail — then fix them. That's ML. 💡",
    "Confusion today = clarity tomorrow. You're closer than you think! 🧠",
    "Every gradient step forward counts. Don't stop now! 📈",
    "The model isn't the only thing that trains — so do you. 💪",
]

SYSTEM_PROMPT = """You are StudyBuddy AI — an expert ML tutor and motivational coach for students learning machine learning and AI.

Expertise: Classical ML, Deep Learning (CNN/RNN/Transformer), Optimization, GANs, VAEs, RL, NLP, Explainability (SHAP/LIME).

When explaining:
1. Start with intuition/analogy before math
2. Include Python code (sklearn/PyTorch) when helpful
3. Highlight common mistakes
4. Connect to real-world use cases

Formatting:
- Use **bold** for key terms
- Use bullet points for lists
- Use ```python code blocks``` for code
- Keep responses 200–400 words unless asked for more
- Always end with an encouraging note or follow-up question"""

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 StudyBuddy AI")
    st.markdown("*Advanced ML Edition*")
    st.divider()

    # API Key
    st.markdown("### 🔑 Groq API Key (Free)")
    st.markdown("""<div style='font-size:12px;color:#8888aa;margin-bottom:8px;'>
     Get free key → <a href='https://console.groq.com/keys' target='_blank' style='color:#38bdf8;'>console.groq.com</a><br>
    ✅ No credit card &nbsp;&nbsp;✅ Free tier available
    </div>""", unsafe_allow_html=True)


    api_input = st.text_input("API Key", value=st.session_state.api_key,
    type="password", placeholder="gsk_...", label_visibility="collapsed")
    if api_input:
        st.session_state.api_key = api_input

    if st.session_state.api_key:
        st.success("✅ API key ready!")
    else:
        st.warning("⚠️ Paste your free Groq key above.")

    st.divider()

    # Stats
    c1, c2 = st.columns(2)
    c1.markdown(f"<div class='metric-card'><div class='val'>{st.session_state.q_count}</div><div class='lbl'>Questions</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='val'>🔥{st.session_state.streak}</div><div class='lbl'>Streak</div></div>", unsafe_allow_html=True)

    st.divider()
    subject = st.selectbox("Focus Area", ["All ML Topics","Deep Learning","Classical ML","NLP & Transformers","Reinforcement Learning","Generative Models","ML Theory"])
    mode    = st.selectbox("Learning Mode", ["Explain & Discuss","Quiz Me","Code Walkthrough","Math Deep-Dive","Motivate Me"])

    st.divider()
    st.markdown("### 🧠 Quick Topics")
    for concept in ML_CONCEPTS:
        if st.button(concept, key=f"btn_{concept}", use_container_width=True):
            st.session_state.messages.append({"role":"user","content":f"Explain {concept} in depth — intuition, math, and Python code."})
            st.session_state.topics_covered.add(concept)
            st.rerun()

    if st.session_state.topics_covered:
        st.divider()
        st.markdown("### 📚 Covered")
        for t in st.session_state.topics_covered:
            st.markdown(f"<span class='concept-tag'>✓ {t}</span>", unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.q_count = 0
        st.rerun()
    st.markdown("---")
    st.markdown(f"*\"{random.choice(MOTIVATIONAL_QUOTES)}\"*")


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat Tutor","📊 ML Visualizer","🔬 ML Models Lab","📝 Quick Reference"])

# ══ TAB 1: Chat ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""<div class='main-header'>
        <h1>🤖 StudyBuddy AI — ML Tutor</h1>
        <p>Ask anything about ML, Deep Learning, NLP, RL and more. Powered by Groq + Llama 3 (Free API).</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='api-banner'>
        ✅ <strong>100% Free</strong> — Groq + Llama 3 8B &nbsp;|&nbsp;
        Fast inference &nbsp;|&nbsp; Free tier available &nbsp;|&nbsp; No credit card &nbsp;|&nbsp;
        Get key: <a href='https://console.groq.com/keys' target='_blank' style='color:#34d399;'>console.groq.com</a>
    </div>""", unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown("### 🚀 Try a topic to get started:")
        suggestions = [
            ("🔍 Attention Mechanism","Explain the Attention Mechanism in Transformers with intuition and code."),
            ("⚡ Backpropagation","Explain backpropagation step by step with math and Python."),
            ("⚖️ Bias-Variance","Explain the bias-variance tradeoff with examples."),
            ("🎨 GANs from scratch","How do GANs work? Explain with architecture and code."),
            ("📉 Gradient Descent","Compare SGD, Adam, RMSProp and when to use each."),
            ("🤖 Transformers","Walk me through the full Transformer architecture."),
            ("💪 I'm stuck","I'm struggling with ML and feeling discouraged. Help me."),
            ("🌲 RF vs XGBoost","Compare Random Forests and XGBoost — when to use which?"),
        ]
        cols = st.columns(4)
        for i, (label, prompt) in enumerate(suggestions):
            with cols[i % 4]:
                if st.button(label, key=f"sug_{i}", use_container_width=True):
                    st.session_state.messages.append({"role":"user","content":prompt})
                    st.rerun()

    # Render chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='msg-label-user'>YOU</div><div class='chat-msg-user'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='msg-label-bot'>🤖 STUDYBUDDY (Groq)</div>", unsafe_allow_html=True)
            st.markdown(msg["content"])

    # Auto-respond to latest user message
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        if not st.session_state.api_key:
            st.error("⚠️ Add your free Groq API key in the sidebar!")
        else:
            mode_map = {
                "Quiz Me": "Quiz the student with 3 questions. Reveal answers one at a time.",
                "Code Walkthrough": "Focus on Python (sklearn/PyTorch). Explain every key line.",
                "Math Deep-Dive": "Show full mathematical derivations and proofs.",
                "Motivate Me": "Be warm and empathetic. Acknowledge difficulty, share inspiration.",
                "Explain & Discuss": "",
            }
            api_messages = list(st.session_state.messages)
            if mode_map.get(mode):
                api_messages[-1] = {**api_messages[-1],
                    "content": api_messages[-1]["content"] + f"\n\n[Mode: {mode_map[mode]}]"}

            with st.spinner("🧠 Groq is thinking..."):
                try:
                    reply = call_groq(
                        st.session_state.api_key, api_messages,
                        SYSTEM_PROMPT + f"\n\nFocus area: {subject}"
                    )
                    st.session_state.messages.append({"role":"assistant","content":reply})
                    st.session_state.q_count += 1
                    if st.session_state.q_count % 5 == 0:
                        st.session_state.streak += 1
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Groq API error: {e}")
                    st.info("Check your Groq API key is valid and you have remaining quota.")

    st.markdown("---")
    with st.form("chat_form", clear_on_submit=True):
        col_inp, col_btn = st.columns([5,1])
        with col_inp:
            user_input = st.text_area("Ask...", placeholder="e.g. How does dropout prevent overfitting? / I'm stuck on backprop...", height=80, label_visibility="collapsed")
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Send 🚀", use_container_width=True)
        if submitted and user_input.strip():
            st.session_state.messages.append({"role":"user","content":user_input.strip()})
            st.rerun()


# ══ TAB 2: ML Visualizer ═════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Interactive ML Concept Visualizer")
    viz = st.selectbox("Concept:", ["Gradient Descent","Bias-Variance Tradeoff","Activation Functions","Learning Rate Effect","Decision Boundary (SVM)","Regularization (L1 vs L2)"])

    DARK, SURF = "#0a0a12", "#1e1e32"
    PL = dict(paper_bgcolor=DARK, plot_bgcolor=SURF, font_color="#e8e8f8",
              xaxis=dict(gridcolor="#252540"), yaxis=dict(gridcolor="#252540"),
              legend=dict(bgcolor=SURF), margin=dict(l=40,r=20,t=50,b=40))

    if viz == "Gradient Descent":
        lr = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
        steps = st.slider("Steps", 5, 50, 20)
        x = np.linspace(-3, 3, 200); y = np.linspace(-3, 3, 200)
        X, Y = np.meshgrid(x, y); Z = X**2 + 2*Y**2 + 0.5*np.sin(3*X)
        pos = np.array([2.5, 2.5]); path = [pos.copy()]
        for _ in range(steps):
            pos = pos - lr * np.array([2*pos[0]+1.5*np.cos(3*pos[0]), 4*pos[1]])
            path.append(pos.copy())
        path = np.array(path)
        fig = go.Figure([
            go.Contour(x=x, y=y, z=Z, colorscale="Viridis", opacity=0.8, showscale=False),
            go.Scatter(x=path[:,0], y=path[:,1], mode="lines+markers", line=dict(color="#a78bfa",width=3), marker=dict(size=7,color="#f472b6"), name="Path"),
            go.Scatter(x=[path[-1,0]], y=[path[-1,1]], mode="markers", marker=dict(size=14,color="#34d399",symbol="star"), name="Now"),
        ])
        fig.update_layout(**PL, height=420, title=dict(text=f"Gradient Descent (lr={lr}, {steps} steps)", font=dict(color="#a78bfa")))
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 Too-high LR overshoots the minimum. Too-low LR converges very slowly. The sweet spot depends on your loss landscape.")

    elif viz == "Bias-Variance Tradeoff":
        c = np.linspace(1,10,100)
        bias = 10/c**1.2; var = 0.3*c**1.5; total = bias+var+1.5
        fig = go.Figure([
            go.Scatter(x=c,y=bias,name="Bias²",line=dict(color="#38bdf8",width=3)),
            go.Scatter(x=c,y=var,name="Variance",line=dict(color="#f472b6",width=3)),
            go.Scatter(x=c,y=total,name="Total Error",line=dict(color="#fbbf24",width=3,dash="dash")),
        ])
        fig.add_vline(x=c[np.argmin(total)], line_color="#34d399", line_dash="dot",
                      annotation_text="Sweet Spot", annotation_font_color="#34d399")
        fig.update_layout(**PL, height=400, xaxis_title="Model Complexity", yaxis_title="Error",
                          title=dict(text="Bias-Variance Tradeoff", font=dict(color="#a78bfa")))
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 Underfitting = high bias (model too simple). Overfitting = high variance (model memorises training data).")

    elif viz == "Activation Functions":
        x = np.linspace(-5,5,300)
        acts = {"ReLU":np.maximum(0,x),"Sigmoid":1/(1+np.exp(-x)),"Tanh":np.tanh(x),
                "Leaky ReLU":np.where(x>0,x,0.1*x),
                "GELU":x*0.5*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))}
        fig = go.Figure([go.Scatter(x=x,y=v,name=k,line=dict(width=2.5,color=c))
                         for (k,v),c in zip(acts.items(),["#6c63ff","#38bdf8","#f472b6","#34d399","#fbbf24"])])
        fig.update_layout(**PL, height=420, xaxis=dict(range=[-5,5],gridcolor="#252540"),
                          yaxis=dict(range=[-1.5,5],gridcolor="#252540"),
                          title=dict(text="Activation Functions", font=dict(color="#a78bfa")))
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 ReLU is the standard default. GELU is used in GPT/BERT. Sigmoid/Tanh live inside LSTM gates.")

    elif viz == "Learning Rate Effect":
        ep = np.arange(1,51)
        cfgs = [(0.001,"#38bdf8","Very Low — Slow"),(0.01,"#34d399","Good — Optimal"),
                (0.1,"#fbbf24","High — Unstable"),(0.5,"#f472b6","Too High — Diverges")]
        fig = go.Figure()
        for lr_v,col,lbl in cfgs:
            if lr_v<=0.01: loss=2.0*np.exp(-lr_v*8*ep)+0.05+np.random.randn(50)*0.01
            elif lr_v==0.1: loss=2.0*np.exp(-lr_v*3*ep)+0.2+0.15*np.sin(ep*0.5)+np.random.randn(50)*0.05
            else: loss=np.minimum(2.0+0.3*ep*(np.random.randn(50)*0.5+1),10)
            fig.add_trace(go.Scatter(x=ep,y=np.clip(loss,0.05,10),name=lbl,line=dict(color=col,width=2.5)))
        fig.update_layout(**PL, height=400, xaxis_title="Epoch", yaxis_title="Loss",
                          title=dict(text="Learning Rate Effect on Training Loss", font=dict(color="#a78bfa")))
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 Use learning rate schedulers (cosine decay, warm-up) in production to get the best of both worlds.")

    elif viz == "Decision Boundary (SVM)":
        np.random.seed(42)
        X1=np.random.randn(50,2)+[2,2]; X2=np.random.randn(50,2)+[-2,-2]
        xr=np.linspace(-6,6,200)
        fig = go.Figure([
            go.Scatter(x=X1[:,0],y=X1[:,1],mode="markers",marker=dict(color="#38bdf8",size=8),name="Class +1"),
            go.Scatter(x=X2[:,0],y=X2[:,1],mode="markers",marker=dict(color="#f472b6",size=8,symbol="square"),name="Class -1"),
            go.Scatter(x=xr,y=-xr,mode="lines",line=dict(color="#fbbf24",width=2.5),name="Boundary"),
            go.Scatter(x=xr,y=-xr+1.5,mode="lines",line=dict(color="#fbbf24",width=1.5,dash="dash"),name="Margin +"),
            go.Scatter(x=xr,y=-xr-1.5,mode="lines",line=dict(color="#fbbf24",width=1.5,dash="dash"),name="Margin -"),
        ])
        fig.update_layout(**PL, height=420, xaxis=dict(range=[-6,6],gridcolor="#252540"),
                          yaxis=dict(range=[-6,6],gridcolor="#252540"),
                          title=dict(text="SVM Decision Boundary & Margin", font=dict(color="#a78bfa")))
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 SVM finds the hyperplane that maximises the margin. Only the points on the margin (support vectors) define it.")

    elif viz == "Regularization (L1 vs L2)":
        wo=np.random.randn(20)*2
        fig = go.Figure()
        for lam,col in zip([0.0,0.5,1.0,2.0],["#6c63ff","#38bdf8","#34d399","#fbbf24"]):
            fig.add_trace(go.Bar(name=f"λ={lam}",x=list(range(20)),y=wo/(1+lam),marker_color=col,opacity=0.8))
        fig.update_layout(**PL, barmode="group", height=400,
                          xaxis_title="Feature Index", yaxis_title="Weight",
                          title=dict(text="L2 Regularization: Weight Shrinkage vs λ", font=dict(color="#a78bfa")))
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 L2 shrinks all weights smoothly. L1 drives some weights to exactly 0 → automatic feature selection.")


# ══ TAB 3: ML Models Lab ═════════════════════════════════════════════════════
with tab3:
    render_ml_models_tab()


# ══ TAB 4: Quick Reference ═══════════════════════════════════════════════════
with tab4:
    st.markdown("### 📝 ML Quick Reference Card")
    ca, cb = st.columns(2)

    with ca:
        st.markdown("#### 🧮 Key Formulas")
        st.markdown("""
**Gradient Descent:**
```
θ = θ - α · ∇L(θ)
```
**Cross-Entropy Loss:**
```
L = -Σ y·log(ŷ) + (1-y)·log(1-ŷ)
```
**Softmax:**
```
σ(zᵢ) = exp(zᵢ) / Σ exp(zⱼ)
```
**Scaled Dot-Product Attention:**
```
Attention(Q,K,V) = softmax(QKᵀ/√dₖ)·V
```
**L2 Loss:**
```
L_total = L_data + λ·‖w‖²
```
        """)
        st.markdown("#### ⚡ Optimizer Cheat Sheet")
        st.dataframe(pd.DataFrame({
            "Optimizer":["SGD","Momentum","RMSProp","Adam","AdaGrad"],
            "Best For":["Simple","Noisy grads","RNNs","General","Sparse"],
            "Drawback":["Slow","Overshoots","LR decay","Memory","LR→0"],
        }), use_container_width=True, hide_index=True)

    with cb:
        st.markdown("#### 🏗️ Architecture Reference")
        st.dataframe(pd.DataFrame({
            "Model":["CNN","RNN/LSTM","Transformer","GAN","VAE","ResNet"],
            "Use Case":["Images","Sequences","NLP/Vision","Generation","Latent","Deep vision"],
            "Key Idea":["Conv filters","Hidden state","Self-attention","Adversarial","Reparameterise","Skip connections"],
        }), use_container_width=True, hide_index=True)

        st.markdown("#### 🐍 Python Snippets")
        st.code("""
# 5-fold Cross-Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print(f"{scores.mean():.3f} ± {scores.std():.3f}")

# PyTorch training loop
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(epochs):
    model.train()
    opt.zero_grad()
    loss = criterion(model(X_train), y_train)
    loss.backward()
    opt.step()

# Scaled dot-product attention
import torch.nn.functional as F
attn_out = F.softmax(Q@K.T / K.shape[-1]**0.5, dim=-1) @ V
        """, language="python")

    st.divider()
    st.markdown("#### 🎯 Common Interview Q&A")
    for q, a in {
        "What is the curse of dimensionality?": "As feature count grows, data becomes exponentially sparse, distances lose meaning, and models need far more samples to generalise.",
        "Explain vanishing gradients": "Gradients shrink exponentially through many layers during backprop, starving early layers. Fixed by ReLU, BatchNorm, skip connections, or LSTMs.",
        "L1 vs L2 regularisation?": "L1 (Lasso) → sparse weights, good for feature selection. L2 (Ridge) → smooth small weights. ElasticNet = both.",
        "What is dropout?": "Randomly zeroes neurons during training, forcing redundancy. Equivalent to averaging many sub-networks at inference time.",
        "Transformers vs RNNs?": "Transformers process all tokens in parallel via self-attention (no sequential bottleneck). RNNs struggle with long-range dependencies and don't parallelise.",
        "Bagging vs Boosting?": "Bagging (Random Forest): parallel models on random subsets → reduces variance. Boosting (XGBoost): sequential models each fixing previous errors → reduces bias.",
    }.items():
        with st.expander(f"❓ {q}"):
            st.markdown(f"**Answer:** {a}")
