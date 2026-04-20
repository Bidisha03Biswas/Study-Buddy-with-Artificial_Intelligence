"""
ML Models Lab — Tab 4
Live, trainable ML models with real sklearn datasets.
Students can tune hyperparameters, train, and see results instantly.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn import datasets
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, roc_curve, auc
)

# --- Classifiers ---
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE

DARK = "#0a0a12"
SURFACE = "#1e1e32"
ACCENT = "#a78bfa"
CYAN = "#38bdf8"
PINK = "#f472b6"
GREEN = "#34d399"
GOLD = "#fbbf24"
TEXT = "#e8e8f8"
MUTED = "#8888aa"

PLOT_LAYOUT = dict(
    paper_bgcolor=DARK, plot_bgcolor=SURFACE,
    font=dict(color=TEXT, family="Sora"),
    xaxis=dict(gridcolor="#252540", zerolinecolor="#252540"),
    yaxis=dict(gridcolor="#252540", zerolinecolor="#252540"),
    legend=dict(bgcolor=SURFACE, bordercolor="#252540"),
    margin=dict(l=40, r=20, t=50, b=40),
)


def load_dataset(name):
    if name == "Iris (Classification)":
        d = datasets.load_iris()
        X, y = d.data, d.target
        feature_names = d.feature_names
        target_names = d.target_names
    elif name == "Breast Cancer (Classification)":
        d = datasets.load_breast_cancer()
        X, y = d.data, d.target
        feature_names = d.feature_names
        target_names = d.target_names
    elif name == "Wine (Classification)":
        d = datasets.load_wine()
        X, y = d.data, d.target
        feature_names = d.feature_names
        target_names = d.target_names
    elif name == "Digits (Classification)":
        d = datasets.load_digits()
        X, y = d.data, d.target
        feature_names = [f"pixel_{i}" for i in range(X.shape[1])]
        target_names = [str(i) for i in range(10)]
    elif name == "Boston Housing (Regression)":
        d = datasets.fetch_california_housing()
        X, y = d.data[:500], d.target[:500]
        feature_names = d.feature_names
        target_names = None
    elif name == "Diabetes (Regression)":
        d = datasets.load_diabetes()
        X, y = d.data, d.target
        feature_names = d.feature_names
        target_names = None
    elif name == "Blobs (Clustering)":
        X, y = datasets.make_blobs(n_samples=300, centers=4, cluster_std=1.2, random_state=42)
        feature_names = ["Feature 1", "Feature 2"]
        target_names = None
    elif name == "Moons (Clustering)":
        X, y = datasets.make_moons(n_samples=300, noise=0.1, random_state=42)
        feature_names = ["Feature 1", "Feature 2"]
        target_names = None
    else:
        d = datasets.load_iris()
        X, y = d.data, d.target
        feature_names = d.feature_names
        target_names = d.target_names
    return X, y, feature_names, target_names


def plot_confusion_matrix(cm, labels):
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Purples",
        labels=dict(x="Predicted", y="Actual"),
        x=labels, y=labels,
    )
    fig.update_layout(**PLOT_LAYOUT, title=dict(text="Confusion Matrix", font=dict(color=ACCENT)))
    fig.update_coloraxes(showscale=False)
    return fig


def plot_roc_curve(y_test, y_prob, n_classes):
    fig = go.Figure()
    colors = [ACCENT, CYAN, PINK, GREEN, GOLD]
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC={roc_auc:.3f}",
                                 line=dict(color=ACCENT, width=2.5)))
    else:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_test, classes=list(range(n_classes)))
        for i in range(min(n_classes, 5)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"Class {i} (AUC={roc_auc:.2f})",
                                     line=dict(color=colors[i % len(colors)], width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color=MUTED), name="Random", showlegend=False))
    fig.update_layout(**PLOT_LAYOUT, title=dict(text="ROC Curve", font=dict(color=ACCENT)),
                      xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return fig


def plot_learning_curve(model, X, y, task="classification"):
    sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 8),
        scoring="accuracy" if task == "classification" else "r2"
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sizes, y=train_mean, name="Train Score",
        line=dict(color=ACCENT, width=2.5),
        error_y=dict(type="data", array=train_std, visible=True, color=ACCENT, thickness=1)
    ))
    fig.add_trace(go.Scatter(
        x=sizes, y=val_mean, name="Validation Score",
        line=dict(color=CYAN, width=2.5),
        error_y=dict(type="data", array=val_std, visible=True, color=CYAN, thickness=1)
    ))
    fig.update_layout(**PLOT_LAYOUT,
                      title=dict(text="Learning Curve", font=dict(color=ACCENT)),
                      xaxis_title="Training Samples", yaxis_title="Score")
    return fig


def plot_feature_importance(importances, feature_names):
    idx = np.argsort(importances)[-15:]
    fig = go.Figure(go.Bar(
        x=importances[idx], y=np.array(feature_names)[idx],
        orientation="h",
        marker=dict(
            color=importances[idx],
            colorscale=[[0, "#252540"], [1, ACCENT]],
        )
    ))
    fig.update_layout(**PLOT_LAYOUT, title=dict(text="Feature Importance", font=dict(color=ACCENT)),
                      xaxis_title="Importance", yaxis_title="Feature")
    return fig


def plot_regression_results(y_test, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test, y=y_pred, mode="markers",
        marker=dict(color=ACCENT, size=6, opacity=0.7),
        name="Predictions"
    ))
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx],
                             line=dict(color=PINK, dash="dash", width=2), name="Perfect Fit"))
    fig.update_layout(**PLOT_LAYOUT, title=dict(text="Actual vs Predicted", font=dict(color=ACCENT)),
                      xaxis_title="Actual", yaxis_title="Predicted")
    return fig


def plot_pca_2d(X, y, target_names, title="PCA Projection"):
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(StandardScaler().fit_transform(X))
    colors = [ACCENT, CYAN, PINK, GREEN, GOLD, "#fb923c", "#e879f9"]
    fig = go.Figure()
    classes = np.unique(y)
    for i, cls in enumerate(classes):
        mask = y == cls
        label = target_names[cls] if target_names is not None else str(cls)
        fig.add_trace(go.Scatter(
            x=X2[mask, 0], y=X2[mask, 1], mode="markers",
            name=label,
            marker=dict(color=colors[i % len(colors)], size=7, opacity=0.8)
        ))
    fig.update_layout(**PLOT_LAYOUT,
                      title=dict(text=f"{title} (var: {pca.explained_variance_ratio_.sum()*100:.1f}%)", font=dict(color=ACCENT)),
                      xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                      yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    return fig


def plot_clusters(X, labels, title="Clustering Result"):
    colors = [ACCENT, CYAN, PINK, GREEN, GOLD, "#fb923c", "#e879f9", MUTED]
    fig = go.Figure()
    unique = np.unique(labels)
    for i, lbl in enumerate(unique):
        mask = labels == lbl
        name = f"Cluster {lbl}" if lbl >= 0 else "Noise"
        fig.add_trace(go.Scatter(
            x=X[mask, 0], y=X[mask, 1], mode="markers",
            name=name,
            marker=dict(color=colors[i % len(colors)], size=8, opacity=0.8)
        ))
    fig.update_layout(**PLOT_LAYOUT, title=dict(text=title, font=dict(color=ACCENT)),
                      xaxis_title="Feature 1", yaxis_title="Feature 2")
    return fig


def render_ml_models_tab():
    st.markdown("### 🔬 ML Models Lab — Train & Evaluate Real Models")
    st.markdown("Pick a dataset, choose a model, tune hyperparameters, and train — all live in your browser.")

    model_category = st.radio(
        "Model Category",
        ["🏷️ Classification", "📈 Regression", "🔵 Clustering", "🧬 Dimensionality Reduction"],
        horizontal=True
    )

    st.divider()

    # ── CLASSIFICATION ─────────────────────────────────────────────────────────
    if model_category == "🏷️ Classification":
        col_ds, col_mdl = st.columns([1, 1])
        with col_ds:
            dataset_name = st.selectbox("Dataset", [
                "Iris (Classification)",
                "Breast Cancer (Classification)",
                "Wine (Classification)",
                "Digits (Classification)",
            ])
        with col_mdl:
            model_name = st.selectbox("Model", [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting",
                "SVM (RBF Kernel)",
                "K-Nearest Neighbors",
                "Naive Bayes",
                "MLP Neural Network",
            ])

        # Hyperparameters
        st.markdown("#### ⚙️ Hyperparameters")
        hp_cols = st.columns(4)

        params = {}
        if model_name == "Logistic Regression":
            params["C"] = hp_cols[0].slider("C (regularization)", 0.01, 10.0, 1.0, step=0.01)
            params["max_iter"] = hp_cols[1].slider("Max Iterations", 100, 2000, 500, step=100)
            params["solver"] = hp_cols[2].selectbox("Solver", ["lbfgs", "liblinear", "saga"])
        elif model_name == "Decision Tree":
            params["max_depth"] = hp_cols[0].slider("Max Depth", 1, 20, 5)
            params["min_samples_split"] = hp_cols[1].slider("Min Samples Split", 2, 20, 2)
            params["min_samples_leaf"] = hp_cols[2].slider("Min Samples Leaf", 1, 10, 1)
            params["criterion"] = hp_cols[3].selectbox("Criterion", ["gini", "entropy"])
        elif model_name == "Random Forest":
            params["n_estimators"] = hp_cols[0].slider("Trees", 10, 300, 100, step=10)
            params["max_depth"] = hp_cols[1].slider("Max Depth", 1, 20, 10)
            params["max_features"] = hp_cols[2].selectbox("Max Features", ["sqrt", "log2", None])
            params["min_samples_split"] = hp_cols[3].slider("Min Split", 2, 20, 2)
        elif model_name == "Gradient Boosting":
            params["n_estimators"] = hp_cols[0].slider("Estimators", 10, 300, 100, step=10)
            params["learning_rate"] = hp_cols[1].slider("Learning Rate", 0.01, 1.0, 0.1, step=0.01)
            params["max_depth"] = hp_cols[2].slider("Max Depth", 1, 10, 3)
            params["subsample"] = hp_cols[3].slider("Subsample", 0.5, 1.0, 1.0, step=0.05)
        elif model_name == "SVM (RBF Kernel)":
            params["C"] = hp_cols[0].slider("C", 0.01, 50.0, 1.0, step=0.1)
            params["gamma"] = hp_cols[1].selectbox("Gamma", ["scale", "auto"])
            params["kernel"] = hp_cols[2].selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
        elif model_name == "K-Nearest Neighbors":
            params["n_neighbors"] = hp_cols[0].slider("K (neighbors)", 1, 30, 5)
            params["weights"] = hp_cols[1].selectbox("Weights", ["uniform", "distance"])
            params["metric"] = hp_cols[2].selectbox("Distance Metric", ["euclidean", "manhattan", "minkowski"])
        elif model_name == "MLP Neural Network":
            layers = hp_cols[0].selectbox("Hidden Layers", ["(64,)", "(128,64)", "(256,128,64)", "(64,64,64)"])
            params["hidden_layer_sizes"] = eval(layers)
            params["activation"] = hp_cols[1].selectbox("Activation", ["relu", "tanh", "logistic"])
            params["learning_rate_init"] = hp_cols[2].slider("Learning Rate", 0.0001, 0.1, 0.001, step=0.0001, format="%.4f")
            params["max_iter"] = hp_cols[3].slider("Max Epochs", 100, 2000, 500, step=100)

        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, step=0.05)

        if st.button("🚀 Train Model", key="train_clf", use_container_width=True):
            with st.spinner("Training model..."):
                X, y, feature_names, target_names = load_dataset(dataset_name)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=test_size, random_state=42, stratify=y
                )

                # Build model
                model_map = {
                    "Logistic Regression": LogisticRegression(**params),
                    "Decision Tree": DecisionTreeClassifier(**params, random_state=42),
                    "Random Forest": RandomForestClassifier(**params, random_state=42, n_jobs=-1),
                    "Gradient Boosting": GradientBoostingClassifier(**params, random_state=42),
                    "SVM (RBF Kernel)": SVC(**params, probability=True),
                    "K-Nearest Neighbors": KNeighborsClassifier(**params),
                    "Naive Bayes": GaussianNB(),
                    "MLP Neural Network": MLPClassifier(**params, random_state=42),
                }
                model = model_map[model_name]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
                acc = accuracy_score(y_test, y_pred)

                # Cross-val
                cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")

            # ── Metrics ───────────────────────────────────────────────────────
            st.markdown("#### 📊 Results")
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f"""<div class='metric-card'><div class='val' style='color:{GREEN}'>{acc*100:.2f}%</div><div class='lbl'>Test Accuracy</div></div>""", unsafe_allow_html=True)
            m2.markdown(f"""<div class='metric-card'><div class='val' style='color:{CYAN}'>{cv_scores.mean()*100:.2f}%</div><div class='lbl'>CV Mean (5-fold)</div></div>""", unsafe_allow_html=True)
            m3.markdown(f"""<div class='metric-card'><div class='val' style='color:{GOLD}'>{cv_scores.std()*100:.2f}%</div><div class='lbl'>CV Std Dev</div></div>""", unsafe_allow_html=True)
            m4.markdown(f"""<div class='metric-card'><div class='val' style='color:{PINK}'>{len(X_train)}</div><div class='lbl'>Train Samples</div></div>""", unsafe_allow_html=True)

            st.markdown("#### 📋 Classification Report")
            report = classification_report(y_test, y_pred,
                target_names=target_names if target_names is not None else None,
                output_dict=True)
            report_df = pd.DataFrame(report).T.drop(columns=["support"], errors="ignore")
            st.dataframe(report_df.style.format("{:.3f}").background_gradient(
                subset=["precision", "recall", "f1-score"], cmap="Purples"), use_container_width=True)

            # ── Plots ─────────────────────────────────────────────────────────
            pc1, pc2 = st.columns(2)
            with pc1:
                n_classes = len(np.unique(y))
                cm = confusion_matrix(y_test, y_pred)
                labels = list(target_names) if target_names is not None else [str(i) for i in range(n_classes)]
                st.plotly_chart(plot_confusion_matrix(cm, labels), use_container_width=True)
            with pc2:
                if y_prob is not None:
                    st.plotly_chart(plot_roc_curve(y_test, y_prob, n_classes), use_container_width=True)

            pc3, pc4 = st.columns(2)
            with pc3:
                st.plotly_chart(plot_learning_curve(
                    model_map[model_name].__class__(**params) if model_name not in ["Naive Bayes"] else GaussianNB(),
                    X_scaled, y, task="classification"), use_container_width=True)
            with pc4:
                if hasattr(model, "feature_importances_"):
                    st.plotly_chart(plot_feature_importance(model.feature_importances_, feature_names), use_container_width=True)
                elif hasattr(model, "coef_") and len(model.coef_.shape) > 0:
                    imp = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_[0])
                    st.plotly_chart(plot_feature_importance(imp, feature_names), use_container_width=True)
                else:
                    st.plotly_chart(plot_pca_2d(X, y, target_names, title="PCA: Data Overview"), use_container_width=True)

    # ── REGRESSION ─────────────────────────────────────────────────────────────
    elif model_category == "📈 Regression":
        col_ds, col_mdl = st.columns([1, 1])
        with col_ds:
            dataset_name = st.selectbox("Dataset", [
                "Boston Housing (Regression)",
                "Diabetes (Regression)",
            ])
        with col_mdl:
            model_name = st.selectbox("Regression Model", [
                "Linear Regression",
                "Ridge Regression",
                "Lasso Regression",
                "Decision Tree Regressor",
                "Random Forest Regressor",
                "Gradient Boosting Regressor",
                "SVR",
                "MLP Regressor",
            ])

        st.markdown("#### ⚙️ Hyperparameters")
        hp_cols = st.columns(4)
        params = {}
        if model_name == "Ridge Regression":
            params["alpha"] = hp_cols[0].slider("Alpha (λ)", 0.01, 100.0, 1.0, step=0.01)
        elif model_name == "Lasso Regression":
            params["alpha"] = hp_cols[0].slider("Alpha (λ)", 0.001, 10.0, 0.1, step=0.001, format="%.3f")
            params["max_iter"] = hp_cols[1].slider("Max Iter", 500, 5000, 1000, step=500)
        elif model_name == "Decision Tree Regressor":
            params["max_depth"] = hp_cols[0].slider("Max Depth", 1, 20, 5)
            params["min_samples_split"] = hp_cols[1].slider("Min Split", 2, 20, 2)
        elif model_name in ("Random Forest Regressor",):
            params["n_estimators"] = hp_cols[0].slider("Trees", 10, 300, 100, step=10)
            params["max_depth"] = hp_cols[1].slider("Max Depth", 1, 20, 10)
        elif model_name == "Gradient Boosting Regressor":
            params["n_estimators"] = hp_cols[0].slider("Estimators", 10, 300, 100, step=10)
            params["learning_rate"] = hp_cols[1].slider("Learning Rate", 0.01, 1.0, 0.1, step=0.01)
            params["max_depth"] = hp_cols[2].slider("Max Depth", 1, 8, 3)
        elif model_name == "SVR":
            params["C"] = hp_cols[0].slider("C", 0.1, 50.0, 1.0)
            params["epsilon"] = hp_cols[1].slider("Epsilon", 0.01, 1.0, 0.1, step=0.01)
            params["kernel"] = hp_cols[2].selectbox("Kernel", ["rbf", "linear", "poly"])
        elif model_name == "MLP Regressor":
            layers = hp_cols[0].selectbox("Hidden Layers", ["(64,)", "(128,64)", "(256,128,64)"])
            params["hidden_layer_sizes"] = eval(layers)
            params["activation"] = hp_cols[1].selectbox("Activation", ["relu", "tanh"])
            params["learning_rate_init"] = hp_cols[2].slider("LR", 0.0001, 0.1, 0.001, format="%.4f")

        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, step=0.05)

        if st.button("🚀 Train Regressor", key="train_reg", use_container_width=True):
            with st.spinner("Training regressor..."):
                X, y, feature_names, _ = load_dataset(dataset_name)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

                model_map = {
                    "Linear Regression": LinearRegression(),
                    "Ridge Regression": Ridge(**params),
                    "Lasso Regression": Lasso(**params),
                    "Decision Tree Regressor": DecisionTreeRegressor(**params, random_state=42),
                    "Random Forest Regressor": RandomForestRegressor(**params, random_state=42, n_jobs=-1),
                    "Gradient Boosting Regressor": GradientBoostingRegressor(**params, random_state=42),
                    "SVR": SVR(**params),
                    "MLP Regressor": MLPRegressor(**params, random_state=42, max_iter=1000),
                }
                model = model_map[model_name]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                cv_r2 = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")

            st.markdown("#### 📊 Results")
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f"""<div class='metric-card'><div class='val' style='color:{GREEN}'>{r2:.4f}</div><div class='lbl'>R² Score</div></div>""", unsafe_allow_html=True)
            m2.markdown(f"""<div class='metric-card'><div class='val' style='color:{CYAN}'>{rmse:.4f}</div><div class='lbl'>RMSE</div></div>""", unsafe_allow_html=True)
            m3.markdown(f"""<div class='metric-card'><div class='val' style='color:{GOLD}'>{cv_r2.mean():.4f}</div><div class='lbl'>CV R² Mean</div></div>""", unsafe_allow_html=True)
            m4.markdown(f"""<div class='metric-card'><div class='val' style='color:{PINK}'>{cv_r2.std():.4f}</div><div class='lbl'>CV R² Std</div></div>""", unsafe_allow_html=True)

            pc1, pc2 = st.columns(2)
            with pc1:
                st.plotly_chart(plot_regression_results(y_test, y_pred), use_container_width=True)
            with pc2:
                residuals = y_test - y_pred
                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers",
                    marker=dict(color=ACCENT, size=6, opacity=0.7), name="Residuals"))
                fig_res.add_hline(y=0, line_color=PINK, line_dash="dash")
                fig_res.update_layout(**PLOT_LAYOUT, title=dict(text="Residual Plot", font=dict(color=ACCENT)),
                    xaxis_title="Predicted", yaxis_title="Residual")
                st.plotly_chart(fig_res, use_container_width=True)

            pc3, pc4 = st.columns(2)
            with pc3:
                st.plotly_chart(plot_learning_curve(
                    model_map[model_name].__class__(**params) if params else model_map[model_name].__class__(),
                    X_scaled, y, task="regression"), use_container_width=True)
            with pc4:
                if hasattr(model, "feature_importances_"):
                    st.plotly_chart(plot_feature_importance(model.feature_importances_, feature_names), use_container_width=True)
                elif hasattr(model, "coef_"):
                    st.plotly_chart(plot_feature_importance(np.abs(model.coef_), feature_names), use_container_width=True)

    # ── CLUSTERING ─────────────────────────────────────────────────────────────
    elif model_category == "🔵 Clustering":
        col_ds, col_mdl = st.columns([1, 1])
        with col_ds:
            dataset_name = st.selectbox("Dataset", ["Blobs (Clustering)", "Moons (Clustering)", "Iris (Classification)"])
        with col_mdl:
            model_name = st.selectbox("Clustering Algorithm", ["K-Means", "DBSCAN"])

        st.markdown("#### ⚙️ Hyperparameters")
        hp_cols = st.columns(4)
        params = {}
        if model_name == "K-Means":
            params["n_clusters"] = hp_cols[0].slider("K (clusters)", 2, 10, 4)
            params["init"] = hp_cols[1].selectbox("Init Method", ["k-means++", "random"])
            params["n_init"] = hp_cols[2].slider("N Init", 1, 20, 10)
            params["max_iter"] = hp_cols[3].slider("Max Iter", 100, 500, 300)
        elif model_name == "DBSCAN":
            params["eps"] = hp_cols[0].slider("Epsilon (ε)", 0.1, 2.0, 0.5, step=0.05)
            params["min_samples"] = hp_cols[1].slider("Min Samples", 2, 20, 5)

        if st.button("🚀 Run Clustering", key="train_cluster", use_container_width=True):
            with st.spinner("Clustering..."):
                X, y_true, feature_names, target_names = load_dataset(dataset_name)
                scaler = StandardScaler()
                X_2d = X[:, :2] if X.shape[1] > 2 else X
                X_scaled = scaler.fit_transform(X_2d)

                if model_name == "K-Means":
                    model = KMeans(**params, random_state=42)
                else:
                    model = DBSCAN(**params)

                labels = model.fit_predict(X_scaled)
                n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)

            st.markdown("#### 📊 Results")
            m1, m2, m3 = st.columns(3)
            m1.markdown(f"""<div class='metric-card'><div class='val' style='color:{GREEN}'>{n_clusters_found}</div><div class='lbl'>Clusters Found</div></div>""", unsafe_allow_html=True)
            m2.markdown(f"""<div class='metric-card'><div class='val' style='color:{CYAN}'>{(labels == -1).sum()}</div><div class='lbl'>Noise Points</div></div>""", unsafe_allow_html=True)
            m3.markdown(f"""<div class='metric-card'><div class='val' style='color:{GOLD}'>{len(X_scaled)}</div><div class='lbl'>Total Points</div></div>""", unsafe_allow_html=True)

            pc1, pc2 = st.columns(2)
            with pc1:
                st.plotly_chart(plot_clusters(X_scaled, labels, f"{model_name} Result"), use_container_width=True)
            with pc2:
                st.plotly_chart(plot_clusters(X_scaled, y_true, "True Labels (Ground Truth)"), use_container_width=True)

            if model_name == "K-Means":
                st.markdown("#### 📉 Elbow Method — Optimal K")
                inertias = []
                k_range = range(1, 12)
                for k in k_range:
                    km = KMeans(n_clusters=k, random_state=42, n_init=5)
                    km.fit(X_scaled)
                    inertias.append(km.inertia_)
                fig_elbow = go.Figure()
                fig_elbow.add_trace(go.Scatter(x=list(k_range), y=inertias,
                    mode="lines+markers", line=dict(color=ACCENT, width=2.5),
                    marker=dict(size=8, color=PINK)))
                fig_elbow.add_vline(x=params["n_clusters"], line_color=GREEN, line_dash="dot",
                    annotation_text="Your K", annotation_font_color=GREEN)
                fig_elbow.update_layout(**PLOT_LAYOUT,
                    title=dict(text="Elbow Curve (Inertia vs K)", font=dict(color=ACCENT)),
                    xaxis_title="K", yaxis_title="Inertia")
                st.plotly_chart(fig_elbow, use_container_width=True)

    # ── DIMENSIONALITY REDUCTION ───────────────────────────────────────────────
    elif model_category == "🧬 Dimensionality Reduction":
        dataset_name = st.selectbox("Dataset", [
            "Iris (Classification)",
            "Wine (Classification)",
            "Breast Cancer (Classification)",
            "Digits (Classification)",
        ])
        algo = st.selectbox("Algorithm", ["PCA", "t-SNE"])

        st.markdown("#### ⚙️ Settings")
        hp_cols = st.columns(4)
        params = {}
        if algo == "PCA":
            n_components = hp_cols[0].slider("Components to show", 2, 10, 2)
        elif algo == "t-SNE":
            params["perplexity"] = hp_cols[0].slider("Perplexity", 5, 80, 30)
            params["n_iter"] = hp_cols[1].slider("Iterations", 250, 2000, 1000, step=250)
            params["learning_rate"] = hp_cols[2].slider("Learning Rate", 10, 1000, 200)

        if st.button("🚀 Reduce Dimensions", key="train_dr", use_container_width=True):
            with st.spinner("Running dimensionality reduction..."):
                X, y, feature_names, target_names = load_dataset(dataset_name)
                X_scaled = StandardScaler().fit_transform(X)

                if algo == "PCA":
                    pca_full = PCA(n_components=min(n_components, X_scaled.shape[1]))
                    X_reduced = pca_full.fit_transform(X_scaled)

                    # Explained variance plot
                    pca_all = PCA()
                    pca_all.fit(X_scaled)
                    evr = pca_all.explained_variance_ratio_
                    cumulative = np.cumsum(evr)

                    fig_var = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_var.add_trace(go.Bar(x=list(range(1, len(evr)+1)), y=evr*100,
                        name="Explained Variance %", marker_color=ACCENT), secondary_y=False)
                    fig_var.add_trace(go.Scatter(x=list(range(1, len(evr)+1)), y=cumulative*100,
                        name="Cumulative %", line=dict(color=PINK, width=2)), secondary_y=True)
                    fig_var.update_layout(**PLOT_LAYOUT,
                        title=dict(text="PCA Explained Variance", font=dict(color=ACCENT)))
                    fig_var.update_yaxes(title_text="Individual %", secondary_y=False, gridcolor="#252540")
                    fig_var.update_yaxes(title_text="Cumulative %", secondary_y=True)
                    st.plotly_chart(fig_var, use_container_width=True)

                    # 2D scatter
                    st.plotly_chart(plot_pca_2d(X, y, target_names, "PCA 2D Projection"), use_container_width=True)

                    # Loadings
                    loadings = pd.DataFrame(
                        pca_full.components_[:2].T,
                        columns=[f"PC{i+1}" for i in range(2)],
                        index=feature_names
                    )
                    st.markdown("#### 🔢 PCA Loadings (PC1 & PC2)")
                    st.dataframe(loadings.style.format("{:.4f}").background_gradient(cmap="RdPu"), use_container_width=True)

                elif algo == "t-SNE":
                    tsne = TSNE(n_components=2, random_state=42, **params)
                    X_tsne = tsne.fit_transform(X_scaled)
                    colors = [ACCENT, CYAN, PINK, GREEN, GOLD, "#fb923c", "#e879f9", MUTED, "#a3e635", "#fb7185"]
                    fig_tsne = go.Figure()
                    for i, cls in enumerate(np.unique(y)):
                        mask = y == cls
                        label = target_names[cls] if target_names is not None else str(cls)
                        fig_tsne.add_trace(go.Scatter(
                            x=X_tsne[mask, 0], y=X_tsne[mask, 1], mode="markers",
                            name=label, marker=dict(color=colors[i % len(colors)], size=7, opacity=0.85)
                        ))
                    fig_tsne.update_layout(**PLOT_LAYOUT,
                        title=dict(text=f"t-SNE (perplexity={params['perplexity']})", font=dict(color=ACCENT)),
                        xaxis_title="t-SNE 1", yaxis_title="t-SNE 2")
                    st.plotly_chart(fig_tsne, use_container_width=True)
                    st.info("💡 t-SNE preserves local structure. Clusters that are tight = similar samples. But distances between clusters are NOT meaningful!")
