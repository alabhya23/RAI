from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance

from src.safedrive import (
    DEFAULT_LABELS,
    DEFAULT_IMAGES,
    evaluate_records,
    load_validation_records,
    match_image,
    summarize_attributes,
)


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="SafeDrive-R", page_icon="S", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#172522; --muted:#66736e; --paper:#f4f5ef; --line:#d9ded5; --lime:#d7f36a; --orange:#e16e45; }
    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; color:var(--ink); }
    .stApp { background:var(--paper); }
    [data-testid="stHeader"] { background:rgba(244,245,239,.92); }
    [data-testid="stSidebar"] { background:#e9ede4; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] > div { padding-top:1.5rem; }
    h1,h2,h3 { letter-spacing:0 !important; color:var(--ink); }
    h1 { font-size:2.1rem !important; font-weight:800 !important; }
    .eyebrow { font:500 11px 'DM Mono',monospace; text-transform:uppercase; color:#637068; letter-spacing:1px; }
    .hero { padding:1.25rem 0 1rem; border-bottom:1px solid var(--line); margin-bottom:1.5rem; }
    .hero-mark { display:inline-flex; align-items:center; justify-content:center; width:34px; height:34px; background:var(--lime); color:var(--ink); font:500 15px 'DM Mono',monospace; margin-right:10px; }
    .metric { border-top:2px solid var(--ink); padding:12px 0 8px; }
    .metric-value { font-size:1.8rem; font-weight:800; line-height:1.1; }
    .metric-label { color:var(--muted); font:400 11px 'DM Mono',monospace; text-transform:uppercase; margin-top:6px; }
    .note { border-left:3px solid var(--orange); padding:10px 14px; background:#fff8f3; color:#4b524b; font-size:13px; }
    .mono { font-family:'DM Mono',monospace; }
    div[data-testid="stTabs"] button { font-weight:700; }
    div.stButton > button[kind="primary"] { background:var(--ink); border-color:var(--ink); color:#fff; }
    </style>
    """,
    unsafe_allow_html=True,
)


def metric(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric"><div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


def get_detector(model_name: str):
    try:
        from ultralytics import YOLO

        return YOLO(model_name)
    except Exception as error:
        st.error(f"Could not load the detector: {error}")
        st.info("The first run may need internet access to download the selected pretrained weights.")
        return None


st.markdown(
    '<div class="hero"><span class="hero-mark">SR</span>'
    '<span class="eyebrow">Responsible perception / research prototype</span></div>',
    unsafe_allow_html=True,
)
st.title("SafeDrive-R")
st.caption("Pedestrian detection, environmental disparity, and robustness auditing on BDD100K.")

with st.sidebar:
    st.markdown("### Experiment setup")
    labels_path = st.text_input("Validation annotations", str(DEFAULT_LABELS))
    images_path = st.text_input("100K images folder", str(DEFAULT_IMAGES))
    model_name = st.selectbox("Pretrained detector", ["yolo11n.pt", "yolo11s.pt"])
    confidence = st.slider("Detection confidence", 0.10, 0.90, 0.25, 0.05)
    st.markdown("---")
    st.caption("Environment: local BDD100K validation split")
    st.caption("Class audited: person / pedestrian")


@st.cache_data(show_spinner="Reading BDD100K annotations...")
def cached_records(path: str):
    return load_validation_records(Path(path))


if Path(labels_path).is_file():
    try:
        records = cached_records(labels_path)
        attributes = summarize_attributes(records)
    except (OSError, ValueError, RuntimeError) as error:
        records = []
        attributes = pd.DataFrame()
        st.error(f"Unable to read validation annotations: {error}")
else:
    records = []
    attributes = pd.DataFrame()
    st.info("Validation annotations are not available here. Upload an image to run pedestrian detection.")

if records:
    people_count = sum(len(row["people"]) for row in records)
    day_count = sum(row["timeofday"] == "daytime" for row in records)
    night_count = sum(row["timeofday"] == "night" for row in records)
    top = st.columns(4)
    with top[0]:
        metric("Validation frames", f"{len(records):,}")
    with top[1]:
        metric("Person boxes", f"{people_count:,}")
    with top[2]:
        metric("Day frames", f"{day_count:,}")
    with top[3]:
        metric("Night frames", f"{night_count:,}")

tabs = st.tabs(["Overview", "Detect", "Fairness audit", "Robustness"])

with tabs[0]:
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Dataset profile")
        if not attributes.empty:
            selected_attribute = st.segmented_control(
                "Attribute", ["timeofday", "weather", "scene"], default="timeofday"
            )
            profile = attributes[selected_attribute].value_counts().sort_values(ascending=False)
            st.bar_chart(profile, color="#e16e45")
        else:
            st.info("Add the BDD100K validation JSON to show the dataset profile.")
    with right:
        st.subheader("Project status")
        st.markdown(
            '<div class="note">This workspace contains validation boxes and image attributes, '
            'but no detection-training annotations. SafeDrive-R uses a pretrained model for the '
            'baseline and evaluates it only against available validation ground truth.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(" ")
        st.markdown("**Available now**")
        st.markdown("- Single-image pedestrian inference\n- Day/night and weather comparison\n- Controlled brightness / occlusion stress tests\n- Exportable measured metrics")
        st.markdown("**Not safety-certified**")
        st.caption("This is an educational evaluation prototype, not a driving system.")

with tabs[1]:
    st.subheader("Pedestrian detection")
    source = st.radio("Image source", ["Upload image", "BDD100K validation image"], horizontal=True)
    image = None
    record = None
    if source == "Upload image":
        uploaded = st.file_uploader("Choose a road image", type=["jpg", "jpeg", "png"])
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
    elif records:
        person_records = [row for row in records if row["people"]]
        selected_name = st.selectbox("Validation image with a person annotation", [row["name"] for row in person_records])
        record = next(row for row in person_records if row["name"] == selected_name)
        image_file = match_image(Path(images_path), selected_name)
        if image_file:
            image = Image.open(image_file).convert("RGB")
        else:
            st.warning("The selected image is not present under the configured image folder.")

    if image:
        if record:
            st.caption(f"BDD100K · {record['timeofday']} · {record['weather']} · {record['scene']}")
        if record and record["people"]:
            from src.safedrive import draw_boxes

            ground_truth = draw_boxes(image.copy(), record["people"], color=(226, 110, 69), label="BDD ground truth")
        else:
            ground_truth = image
        st.image(ground_truth, caption="Orange boxes are available ground-truth annotations.", use_container_width=True)
        if st.button("Run pedestrian detector", type="primary", icon="🚦"):
            detector = get_detector(model_name)
            if detector:
                with st.spinner("Running pretrained detector..."):
                    results = detector.predict(image, conf=confidence, verbose=False)[0]
                    detections = [
                        {"box": box.xyxy[0].cpu().tolist(), "confidence": float(box.conf[0])}
                        for box in results.boxes
                        if int(box.cls[0]) == 0
                    ]
                    from src.safedrive import draw_boxes

                    annotated = draw_boxes(image.copy(), detections, color=(38, 102, 83), label="person")
                    st.image(annotated, caption=f"{len(detections)} person detections · confidence threshold {confidence:.2f}", use_container_width=True)
                    if record:
                        st.caption(f"Ground truth: {len(record['people'])} annotated persons. Inference output is not itself a correctness score.")

with tabs[2]:
    st.subheader("Day / night model audit")
    st.caption("Metrics are computed only when you run the detector on a sampled set with ground-truth boxes.")
    if records:
        col_a, col_b = st.columns(2)
        with col_a:
            sample_size = st.number_input("Validation frames to evaluate", min_value=10, max_value=200, value=40, step=10)
        with col_b:
            iou_threshold = st.slider("True-positive IoU threshold", 0.30, 0.80, 0.50, 0.05)
        if st.button("Evaluate day / night subset", type="primary"):
            detector = get_detector(model_name)
            if detector:
                with st.spinner("Evaluating a reproducible, stratified sample..."):
                    report = evaluate_records(
                        records,
                        Path(images_path),
                        detector,
                        confidence=confidence,
                        limit=int(sample_size),
                        iou_threshold=iou_threshold,
                    )
                if report.empty:
                    st.warning("No matching annotated images were found in the configured validation folder.")
                else:
                    st.dataframe(report, use_container_width=True, hide_index=True)
                    st.download_button("Download metrics CSV", report.to_csv(index=False), "safedrive_fairness_metrics.csv", "text/csv")
                    st.caption("Precision and recall are sample estimates at the selected confidence and IoU thresholds, not official benchmark mAP.")
    else:
        st.info("Load the validation annotations to run this audit.")

with tabs[3]:
    st.subheader("Controlled robustness checks")
    st.caption("These are corruption and occlusion stress tests, not claims of adversarial-attack resistance.")
    robustness_source = st.radio("Robustness image", ["Upload image", "Use current detection image"], horizontal=True, key="robustness_source")
    robustness_image = image if robustness_source == "Use current detection image" else None
    if robustness_source == "Upload image":
        robustness_upload = st.file_uploader("Choose image for stress testing", type=["jpg", "jpeg", "png"], key="robustness_upload")
        if robustness_upload:
            robustness_image = Image.open(robustness_upload).convert("RGB")
    if robustness_image:
        brightness = st.slider("Brightness factor", 0.25, 1.50, 0.65, 0.05)
        occlusion = st.slider("Center occlusion", 0, 40, 15, 5, format="%d%%")
        changed = ImageEnhance.Brightness(robustness_image).enhance(brightness)
        if occlusion:
            from src.safedrive import apply_center_occlusion

            changed = apply_center_occlusion(changed, occlusion / 100)
        previews = st.columns(2)
        previews[0].image(robustness_image, caption="Original", use_container_width=True)
        previews[1].image(changed, caption="Brightness / occlusion stress input", use_container_width=True)
        if st.button("Compare detections", type="primary"):
            detector = get_detector(model_name)
            if detector:
                first, second = st.columns(2)
                for column, candidate, title in [(first, robustness_image, "Original"), (second, changed, "Stress input")]:
                    output = detector.predict(candidate, conf=confidence, verbose=False)[0]
                    count = sum(int(box.cls[0]) == 0 for box in output.boxes)
                    column.metric(f"{title} person detections", count)
                    from src.safedrive import draw_boxes

                    detections = [
                        {"box": box.xyxy[0].cpu().tolist(), "confidence": float(box.conf[0])}
                        for box in output.boxes if int(box.cls[0]) == 0
                    ]
                    column.image(draw_boxes(candidate.copy(), detections), use_container_width=True)
                st.caption("A changed detection count indicates sensitivity for this image only; it does not establish dataset-level robustness.")

st.markdown("---")
st.markdown('<span class="eyebrow">SafeDrive-R · BDD100K · Educational prototype</span>', unsafe_allow_html=True)