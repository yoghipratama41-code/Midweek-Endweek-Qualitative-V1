import io
import re
import json
import time
import random
import difflib

import numpy as np
import streamlit as st
import PIL.Image
import PIL.ImageDraw
import google.generativeai as genai
import easyocr
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleAuthRequest

# ==========================================
# 0. CONFIGURATION
# ==========================================
st.set_page_config(page_title="GNS Censor -> Endweek", layout="wide")

CUSTOM_CSS = """
<style>
/* ORANGE & WHITE STRICT DARK THEME */
:root {
    --orange: #F5821F;
    --bg-dark: #121212;
    --bg-card: #1E1E1E;
    --white: #FFFFFF;
    --text-muted: #A0A0A0;
}

html, body, [class*="css"] {
    font-family: "Helvetica Neue", Arial, sans-serif;
    color: var(--white);
}

.stApp {
    background-color: var(--bg-dark);
}

/* Remove default Streamlit top padding */
.block-container {
    padding-top: 2rem;
}

/* ---------- Typography & Headers ---------- */
.gns-eyebrow {
    display: inline-block;
    color: var(--orange);
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
h1 {
    color: var(--white) !important;
    font-weight: 800 !important;
    padding-bottom: 0 !important;
    border-bottom: none !important;
}
h1::after {
    content: "";
    display: block;
    width: 64px;
    height: 4px;
    background-color: var(--orange);
    margin-top: 10px;
}
h2, h3, h4, h5, h6 {
    color: var(--white) !important;
    font-weight: 700 !important;
}

/* Solid square accents for subheaders instead of emojis */
h3::before {
    content: "";
    display: inline-block;
    width: 8px;
    height: 8px;
    background-color: var(--orange);
    margin-right: 8px;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background-color: var(--bg-card) !important;
    border-right: 1px solid var(--orange) !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: var(--white) !important;
}
section[data-testid="stSidebar"] h2::before {
    content: "";
    display: inline-block;
    width: 8px;
    height: 8px;
    background-color: var(--orange);
    margin-right: 8px;
}

/* ---------- Divider ---------- */
hr {
    border-top: 1px solid var(--orange) !important;
    opacity: 0.3;
}

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {
    border-radius: 2px !important; /* Sharp corners, no gradients */
    font-weight: 600 !important;
    transition: all 0.2s ease;
    background-color: var(--orange) !important;
    color: var(--bg-dark) !important; 
    border: 1px solid var(--orange) !important;
}
.stButton > button p {
    color: var(--bg-dark) !important;
}
.stButton > button:hover {
    background-color: var(--bg-dark) !important;
    color: var(--orange) !important;
}
.stButton > button:hover p {
    color: var(--orange) !important;
}

/* Secondary Button override */
.stButton > button[kind="secondary"] {
    background-color: var(--bg-card) !important;
    color: var(--orange) !important;
    border: 1px solid var(--orange) !important;
}
.stButton > button[kind="secondary"] p {
    color: var(--orange) !important;
}
.stButton > button[kind="secondary"]:hover {
    background-color: var(--orange) !important;
    color: var(--bg-dark) !important;
}
.stButton > button[kind="secondary"]:hover p {
    color: var(--bg-dark) !important;
}

/* ---------- File Uploader ---------- */
[data-testid="stFileUploaderDropzone"] {
    background-color: var(--bg-card) !important;
    border: 1px dashed var(--orange) !important;
    border-radius: 2px !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background-color: var(--bg-dark) !important;
    color: var(--white) !important;
    border: 1px solid var(--orange) !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background-color: var(--orange) !important;
    color: var(--bg-dark) !important;
}

/* ---------- Sliders (handle + filled track) ---------- */
div[data-baseweb="slider"] div[role="slider"] {
    background-color: var(--orange) !important;
    border-color: var(--orange) !important;
    box-shadow: none !important;
}
div[data-baseweb="slider"] > div > div {
    background-color: var(--orange) !important;
}
div[data-baseweb="slider"] > div:first-child {
    background-color: #333333 !important;
}

/* ---------- Radio / Checkbox ---------- */
label[data-baseweb="radio"] div:first-child,
label[data-baseweb="checkbox"] div:first-child {
    border-color: var(--orange) !important;
}

/* ---------- Alerts (Info/Success/Error/Warning) ---------- */
div[data-testid="stAlert"] {
    border-radius: 2px !important;
    border: 1px solid var(--orange) !important;
    background-color: var(--bg-card) !important;
}
div[data-testid="stAlert"] p {
    color: var(--white) !important;
}
/* Force hide alert icons (emotes) generated natively by Streamlit */
div[data-testid="stAlert"] span[role="img"] {
    display: none !important;
}

/* ---------- Progress bar ---------- */
.stProgress > div > div > div > div {
    background-color: var(--orange) !important;
}

/* ---------- Containers / Cards ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--orange) !important;
    border-radius: 2px !important;
    background-color: var(--bg-card) !important;
    box-shadow: none !important;
}

/* ---------- Text input focus ---------- */
.stTextInput > div > div {
    background-color: var(--bg-dark) !important;
    border: 1px solid #333333 !important;
    color: var(--white) !important;
}
.stTextInput > div > div:focus-within {
    border-color: var(--orange) !important;
    box-shadow: none !important;
}

/* ---------- Links ---------- */
a {
    color: var(--orange) !important;
    font-weight: 600;
}

/* ---------- Captions ---------- */
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--text-muted) !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
GOOGLE_CLIENT_ID = st.secrets["GOOGLE_CLIENT_ID"]
GOOGLE_CLIENT_SECRET = st.secrets["GOOGLE_CLIENT_SECRET"]
GOOGLE_REFRESH_TOKEN = st.secrets["GOOGLE_REFRESH_TOKEN"]
TEMPLATE_PRESENTATION_ID = st.secrets["TEMPLATE_PRESENTATION_ID"]

ENDWEEK_TEMPLATE_ID = "1PvaGfcS1dBMcW-48HQWLEXT3irKPFi9Ptm5eqloX9QA"
TARGET_SPREADSHEET_ID = "1D0CnYZwZtx75OXJGvHeJCiMe6t-pt0UhTOm7vYhbGLQ"
TARGET_SHEET_RANGE = "Extract!A:C"

SCOPES = (
    "https://www.googleapis.com/auth/drive.file "
    "https://www.googleapis.com/auth/drive.readonly "
    "https://www.googleapis.com/auth/presentations "
    "https://www.googleapis.com/auth/spreadsheets"
)
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"

MODEL_PRIORITY = [
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-3-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
]
MAX_RETRY_PER_MODEL = 3


# ==========================================
# 1. GOOGLE AUTH
# ==========================================
@st.cache_resource(ttl=1800)
def get_creds():
    creds = Credentials(
        token=None,
        refresh_token=GOOGLE_REFRESH_TOKEN,
        token_uri=TOKEN_ENDPOINT,
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        scopes=SCOPES.split(),
    )
    creds.refresh(GoogleAuthRequest())
    return creds


@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(["en"], gpu=False)


# ==========================================
# 2. GEMINI - SINGLE FALLBACK FUNCTION USED FOR ALL AI CALLS
#    (name identification for censoring, AND title/context/insight analysis)
# ==========================================
def get_model_fallback_list():
    available = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
    ordered = []
    for key in MODEL_PRIORITY:
        match = next((m for m in available if key in m), None)
        if match and match not in ordered:
            ordered.append(match)
    if not ordered:
        ordered = available[:3]
    return ordered


def panggil_gemini_fallback(model_names, prompt, gambar_list, status_box, max_retry_per_model=MAX_RETRY_PER_MODEL):
    """Send prompt+images to Gemini. Retry with backoff on rate limit, switch model if it keeps failing."""
    last_err = None
    for model_name in model_names:
        model = genai.GenerativeModel(model_name)
        delay = 10
        nama_model_pendek = model_name.split("/")[-1]

        for attempt in range(max_retry_per_model):
            try:
                time.sleep(2)
                respon = model.generate_content([prompt] + gambar_list)
                return respon.text, nama_model_pendek
            except Exception as e:
                err_msg = str(e)
                last_err = e
                if "429" in err_msg or "503" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    wait_time = delay + random.uniform(0, 5)
                    status_box.warning(
                        f"Model **{nama_model_pendek}** hit a rate limit/is busy "
                        f"(attempt {attempt + 1}/{max_retry_per_model}). Waiting {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                    delay *= 2
                else:
                    status_box.warning(f"Model **{nama_model_pendek}** error: {err_msg[:150]}")
                    break
        status_box.info(f"Switching from model **{nama_model_pendek}** to the next model...")
    raise Exception(f"All models failed. Last error: {last_err}")


# ==========================================
# 3. CENSOR STAGE (Gemini identifies names + EasyOCR finds pixel location)
# ==========================================
AIM_PROMPT = """
Analyze this social media screenshot (it could be a main post, a caption, OR a comment).
Focus on ALL USERNAMES (account names) printed in BOLD, wherever they appear in this image —
whether it's the name that posted the main content, or the account names on each comment.
Do NOT pick up names that are only mentioned/tagged within the body text of a comment or caption (not the account name itself).

Return the result in pure JSON format, with no explanatory text and no markdown.
The JSON must be an array of strings, ordered from the top to the bottom of the image. If no account names are visible, return an empty array [].
Example:
["See Toh Kwai Leng", "Pengkok Lim", "Mat Ken"]
"""


def run_ocr(image_pil, reader):
    img_np = np.array(image_pil)
    raw_results = reader.readtext(img_np, detail=1, paragraph=False)
    results = []
    for bbox, text, conf in raw_results:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        results.append({"text": text, "x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)})
    results.sort(key=lambda r: (round(r["y1"] / 10), r["x1"]))
    return results


def _normalize(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())


def find_name_bbox(ocr_results, target_name, used_indices, threshold=0.72):
    target_norm = _normalize(target_name)
    if not target_norm:
        return None
    best_score, best_box, best_indices = 0, None, []
    n = len(ocr_results)
    for i in range(n):
        if i in used_indices:
            continue
        combined_text, combined_indices = "", []
        x1 = y1 = float("inf")
        x2 = y2 = float("-inf")
        base_y = (ocr_results[i]["y1"] + ocr_results[i]["y2"]) / 2
        base_height = ocr_results[i]["y2"] - ocr_results[i]["y1"]
        for j in range(i, min(i + 6, n)):
            if j in used_indices:
                continue
            r = ocr_results[j]
            r_mid_y = (r["y1"] + r["y2"]) / 2
            if abs(r_mid_y - base_y) > max(base_height, 12) * 0.7:
                break
            combined_text += r["text"]
            combined_indices.append(j)
            x1, y1 = min(x1, r["x1"]), min(y1, r["y1"])
            x2, y2 = max(x2, r["x2"]), max(y2, r["y2"])
            score = difflib.SequenceMatcher(None, _normalize(combined_text), target_norm).ratio()
            if score > best_score:
                best_score, best_box, best_indices = score, (x1, y1, x2, y2), list(combined_indices)
    if best_score >= threshold:
        for idx in best_indices:
            used_indices.add(idx)
        return best_box
    return None


def apply_censor_pixel_boxes(image_pil, boxes, pad_ukuran, offset_y):
    censored_image = image_pil.copy()
    draw = PIL.ImageDraw.Draw(censored_image)
    for (x1, y1, x2, y2) in boxes:
        y1_adj, y2_adj = y1 + offset_y, y2 + offset_y
        draw.rectangle([x1 - pad_ukuran, y1_adj - pad_ukuran, x2 + pad_ukuran, y2_adj + pad_ukuran], fill="#FFFFFF")
    return censored_image


def sensor_satu_gambar(uploaded_file, model_fallback_list, reader, match_threshold, pad_ukuran, offset_y, status_box):
    """Run the censor pipeline for a single uploaded file, return (original_image, censored_image, name_list, unmatched_names)."""
    image = PIL.Image.open(uploaded_file).convert("RGB")
    response_text, model_dipakai = panggil_gemini_fallback(model_fallback_list, AIM_PROMPT, [image], status_box)
    json_clean = response_text.replace("```json", "").replace("```", "").strip()
    name_list = json.loads(json_clean)

    ocr_results = run_ocr(image, reader)
    used_indices = set()
    matched_boxes, unmatched_names = [], []
    for name in name_list:
        box = find_name_bbox(ocr_results, name, used_indices, threshold=match_threshold)
        if box:
            matched_boxes.append(box)
        else:
            unmatched_names.append(name)

    censored_image = apply_censor_pixel_boxes(image, matched_boxes, pad_ukuran, offset_y)
    return image, censored_image, name_list, unmatched_names


# ==========================================
# 4. ENDWEEK STAGE (AI analysis on already-censored images + Drive upload + slide generation)
# ==========================================
PROMPT_ANALISIS = """
Analyze this image (and comments if any) for a professional research slide.
Write the analysis in one single cohesive paragraph in English.

Strict Rules:
1. Refer to both users and drivers ONLY as "rider".
2. Do NOT mention any social media usernames, account names, or the image filename.
3. You MAY mention the name of the gig/delivery platform (e.g. Grab, Foodpanda, Lalamove, etc.) if it is visibly shown or referenced in the image or comments. Only usernames, handles, and filenames are off-limits — platform names are allowed and encouraged when relevant.
4. Never begin the paragraph with a generic opener such as "This image...", "This discussion...", "The illustration...", "This conversation...", or any variant that refers to "the image" or "the illustration" itself. Do not describe the fact that you are looking at an image at all. Instead, dive straight into the substance: open with the rider's situation, the specific complaint or sentiment, the operational issue, a concrete detail, or the context of the exchange. Vary the opening construction from one slide to the next (e.g. start with a cause, a location, a time reference, a rider's action, or a direct statement of the issue) so that consecutive outputs do not read as templated or repetitive.
5. The paragraph must consist of exactly 3 sentences: the first sentence serves as the Context, and the following 2 sentences serve as two distinct Insight points (do not merge them into one sentence, and do not add a 4th sentence).

Output Format:
[TITLE] Write a specific, headline-style title, roughly 9-15 words, that reads like a mini research-slide headline capturing the core theme plus a specific supporting detail (not a short generic label). For reference, match this style and length:
"Cross-Region Operational Challenges: Suggestion from Community on Working a Different Zone"
[CONTENT] Write the full paragraph here.
"""


PROMPT_SUMMARY = """
Below are the Context and Insight already written for a research slide about a rider-related social media post.

Context: {konteks}
Insight: {insight}

Task: Write ONE short Summary for this slide, in 1-2 sentences maximum, in English. Keep it strictly short: no more than 25 words total, so it fits inside a small text box on the slide. This Summary is a DIFFERENT field from the Context/Insight above and from what gets logged to the spreadsheet — it must add something on top of them, not restate them.

Tone: Stay strictly NEUTRAL and factual. Report what riders said/did and what the numbers show — do not editorialize, do not take a side, do not use judgment-loaded words (e.g. "unfortunately", "sadly", "clearly a problem", "great news"). State it as an observation, not an opinion.

Follow this priority order when deciding what the Summary should say:
1. NEW FINDING: If the Context/Insight surfaces something beyond the obvious topic (a root cause, a pattern across riders, a consequence), state that new finding.
2. ANSWER THE QUESTION: If the original post or comments posed a question, answer it directly and plainly.
3. BOTH OF THE ABOVE MUST BE FACTUAL AND NUMBER-BACKED: whichever of the two above applies, ground it in a concrete count, ratio, or percentage (e.g. how many riders, how many comments, what share of feedback) rather than a vague qualifier like "several" or "many".
4. FALLBACK: Only if there is no new finding to surface and no question to answer, describe the overall sentiment instead — but still anchor it with a number wherever the Context/Insight gives enough basis to estimate one.

Style reference (match this tone, sentence structure, and use of numbers — do NOT copy this content):
- "At least two riders explicitly mentioned that the tier system does not affect their incentives. This was further supported by some riders who explained that the tier system mainly affects the benefits they receive rather than their incentives."
- "Backed by 50% of classified feedback (4 out of 8 comments), riders conclude that while bundled batches accelerate individual incentives, their disproportionately low payouts make equitable single-order distribution far more sustainable."
- "Three riders explicitly criticized GrabExpress's strict cancellations and low fares, supported by five others."
- "2 out of 3 comments agree that the cancellation affected multiple drivers at once, thus the suspected bug on the app is the main cause of this."

Rules:
- Do not simply copy or lightly reword sentences from the Context or Insight; synthesize a fresh, standalone Summary that goes beyond them.
- Do not use generic openers such as "This shows...", "This highlights...", "Overall...", "In summary...". Go straight to the point.
- Stay within 25 words total (roughly the length of the style-reference examples above) — this is a hard limit, not a suggestion.
- Output ONLY the summary text, with no labels, quotes, or markdown.
"""


PROMPT_SUMMARY_PROMO = """
Below are the Context and Insight already written for a slide about a promotional image (e.g. a platform promo, campaign, or announcement banner) — NOT rider feedback or a social media discussion.

Context: {konteks}
Insight: {insight}

Task: Write ONE short Summary for this slide, in 1-2 sentences maximum, in English, no more than 25 words total, so it fits inside a small text box on the slide.

The Summary must simply and neutrally state what the promotion is about — e.g. what is being offered, to whom, and any key detail such as the timeframe, mechanic, or reward, if that detail is available in the Context/Insight. Do not analyze, evaluate, or speculate about impact or rider reaction; this is a promo, not feedback.

Tone: Stay strictly NEUTRAL and factual, like a plain description — no promotional language of your own (avoid words like "exciting", "amazing", "don't miss out"), and no generic openers such as "This shows...", "This highlights...", "Overall...", "In summary...".

Output ONLY the summary text, with no labels, quotes, or markdown.
"""


SUMMARY_MAX_WORDS = 25


def _potong_summary(teks, max_words=SUMMARY_MAX_WORDS):
    """Hard-enforce the word cap in case the model ignores the length instruction."""
    words = teks.strip().split()
    if len(words) <= max_words:
        return teks.strip()
    dipotong = " ".join(words[:max_words]).rstrip(",;:")
    if not dipotong.endswith((".", "!", "?")):
        dipotong += "."
    return dipotong


def buat_summary_slide(model_fallback_list, konteks, insight_teks, status_box, is_promo=False):
    """Combine Context + Insight into a short 1-2 sentence slide summary via Gemini. Falls back to Context on failure."""
    insight_bersih = insight_teks.strip() if insight_teks and insight_teks.strip() not in ("", "-") else konteks
    template_prompt = PROMPT_SUMMARY_PROMO if is_promo else PROMPT_SUMMARY
    prompt_summary = template_prompt.format(konteks=konteks, insight=insight_bersih)
    try:
        teks_summary, _ = panggil_gemini_fallback(model_fallback_list, prompt_summary, [], status_box)
        return _potong_summary(teks_summary)
    except Exception as e:
        status_box.warning(f"Failed to generate slide Summary, falling back to Context: {e}")
        return _potong_summary(konteks)


def upload_pil_ke_drive(drive_service, pil_image, filename):
    """Upload a PIL image (already censored) to Drive as PNG, return the public link."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)
    media = MediaIoBaseUpload(buf, mimetype="image/png")
    file_dr = drive_service.files().create(
        body={"name": filename}, media_body=media, fields="id, webContentLink"
    ).execute()
    drive_service.permissions().create(
        fileId=file_dr["id"], body={"type": "anyone", "role": "reader"}
    ).execute()
    return file_dr["webContentLink"]


def _get_slide_text(slide):
    texts = []
    for el in slide.get("pageElements", []):
        shape = el.get("shape")
        if shape and "text" in shape:
            for te in shape["text"].get("textElements", []):
                run = te.get("textRun")
                if run:
                    texts.append(run.get("content", ""))
    return "".join(texts)


def cari_template_slide(presentation):
    id_main = None
    id_comment = None
    id_summary = None
    for slide in presentation.get("slides", []):
        txt = _get_slide_text(slide)
        if "{{IMG}}" in txt and id_main is None:
            id_main = slide["objectId"]
        if "{{CMT}}" in txt and id_comment is None:
            id_comment = slide["objectId"]
        if "{{TITLE_SUMMARY}}" in txt and id_summary is None:
            id_summary = slide["objectId"]
    return id_main, id_comment, id_summary


def cari_template_slide_endweek(presentation):
    id_fb_main, id_fb_comment, id_promo_main = None, None, None
    for slide in presentation.get("slides", []):
        txt = _get_slide_text(slide).lower()
        if "{{img}}" in txt and "facebook group" in txt and id_fb_main is None:
            id_fb_main = slide["objectId"]
        if "{{cmt}}" in txt and "facebook group" in txt and id_fb_comment is None:
            id_fb_comment = slide["objectId"]
        if "{{img}}" in txt and "promotion" in txt and id_promo_main is None:
            id_promo_main = slide["objectId"]
    return id_fb_main, id_fb_comment, id_promo_main


def jalankan_otomatisasi_midweek_dari_sensor(creds, censored_items, week_range, progress_bar, status_box):
    drive_service = build("drive", "v3", credentials=creds)
    slides_service = build("slides", "v1", credentials=creds)
    sheets_service = build("sheets", "v4", credentials=creds)
    genai.configure(api_key=GEMINI_API_KEY)
    model_fallback_list = get_model_fallback_list()

    nama_slide_baru = f"Final Midweek - {time.strftime('%Y-%m-%d %H:%M:%S')}"
    copy = drive_service.files().copy(
        fileId=TEMPLATE_PRESENTATION_ID, body={"name": nama_slide_baru}
    ).execute()
    id_slide_baru = copy.get("id")
    link_presentasi = f"https://docs.google.com/presentation/d/{id_slide_baru}/edit"

    presentation = slides_service.presentations().get(presentationId=id_slide_baru).execute()
    id_templat_main, id_templat_comment, id_templat_summary = cari_template_slide(presentation)
    if not id_templat_main:
        raise Exception("Midweek template not found! Make sure there is a slide containing the '{{IMG}}' placeholder.")

    all_titles = []

    slide_count = len(presentation.get("slides", []))
    jumlah = len(censored_items)

    processed_data = []
    sheets_append_data = []

    for index, item in enumerate(censored_items):
        fname = item["filename"]
        try:
            status_box.info(f"[{index+1}/{jumlah}] Processing Midweek: {fname}...")

            gambar_list = [item["img_main_pil"]]
            if item.get("img_cmt_pil") is not None:
                gambar_list.append(item["img_cmt_pil"])

            prompt_ai = PROMPT_ANALISIS
            teks_raw, model_dipakai = panggil_gemini_fallback(
                model_fallback_list, prompt_ai, gambar_list, status_box
            )
            judul = teks_raw.split("[TITLE]")[1].split("[CONTENT]")[0].strip()
            full_para = teks_raw.split("[CONTENT]")[1].strip()
            sentences = re.split(r"(?<=[.!?]) +", full_para)

            if len(sentences) > 1:
                konteks = sentences[0]
                insight_list = [s for s in sentences[1:] if s.strip()]
                insight_midweek = "\n".join(insight_list)
            else:
                konteks = full_para
                insight_list = []
                insight_midweek = "-"

            sheets_append_data.append([week_range, judul, konteks])
            all_titles.append(judul)

            ringkasan_summary = buat_summary_slide(
                model_fallback_list, konteks, insight_midweek, status_box, is_promo=item.get("is_promo", False)
            )

            link_gambar_main = upload_pil_ke_drive(drive_service, item["img_main_pil"], fname)
            link_gambar_comment = None
            if item.get("img_cmt_pil") is not None:
                link_gambar_comment = upload_pil_ke_drive(drive_service, item["img_cmt_pil"], f"comment_{fname}")

            processed_data.append({
                "filename": fname,
                "title": judul,
                "context": konteks,
                "insight_list": insight_list,
                "summary": ringkasan_summary,
                "img_main": link_gambar_main,
                "img_cmt": link_gambar_comment,
            })

            res_dup = slides_service.presentations().batchUpdate(
                presentationId=id_slide_baru,
                body={"requests": [{"duplicateObject": {"objectId": id_templat_main}}]},
            ).execute()
            id_baru_main = res_dup["replies"][0]["duplicateObject"]["objectId"]

            req_main = [
                {"updateSlidesPosition": {"slideObjectIds": [id_baru_main], "insertionIndex": slide_count}},
                {"replaceAllText": {"containsText": {"text": "{{TITLE}}"}, "replaceText": judul, "pageObjectIds": [id_baru_main]}},
                {"replaceAllText": {"containsText": {"text": "{{CONTEXT}}"}, "replaceText": konteks, "pageObjectIds": [id_baru_main]}},
                {"replaceAllText": {"containsText": {"text": "{{INSIGHT}}"}, "replaceText": insight_midweek, "pageObjectIds": [id_baru_main]}},
                {"replaceAllText": {"containsText": {"text": "{{Summary}}"}, "replaceText": ringkasan_summary, "pageObjectIds": [id_baru_main]}},
                {"replaceAllShapesWithImage": {"imageUrl": link_gambar_main, "replaceMethod": "CENTER_INSIDE", "containsText": {"text": "{{IMG}}", "matchCase": True}, "pageObjectIds": [id_baru_main]}},
            ]
            slides_service.presentations().batchUpdate(presentationId=id_slide_baru, body={"requests": req_main}).execute()
            slide_count += 1

            if item.get("img_cmt_pil") is not None and id_templat_comment:
                res_dup_c = slides_service.presentations().batchUpdate(
                    presentationId=id_slide_baru,
                    body={"requests": [{"duplicateObject": {"objectId": id_templat_comment}}]},
                ).execute()
                id_baru_comment = res_dup_c["replies"][0]["duplicateObject"]["objectId"]

                req_cmt = [
                    {"updateSlidesPosition": {"slideObjectIds": [id_baru_comment], "insertionIndex": slide_count}},
                    {"replaceAllText": {"containsText": {"text": "{{TITLE}}"}, "replaceText": judul, "pageObjectIds": [id_baru_comment]}},
                    {"replaceAllShapesWithImage": {"imageUrl": link_gambar_comment, "replaceMethod": "CENTER_INSIDE", "containsText": {"text": "{{CMT}}", "matchCase": True}, "pageObjectIds": [id_baru_comment]}},
                ]
                slides_service.presentations().batchUpdate(presentationId=id_slide_baru, body={"requests": req_cmt}).execute()
                slide_count += 1

            if index < jumlah - 1:
                time.sleep(15)

        except Exception as e:
            status_box.error(f"{fname} skipped: {e}")
        finally:
            progress_bar.progress((index + 1) / jumlah)

    if id_templat_summary and all_titles:
        try:
            status_box.info("Filling in the title summary on the cover slide...")
            ringkasan_judul = "\n".join(all_titles)
            slides_service.presentations().batchUpdate(
                presentationId=id_slide_baru,
                body={"requests": [
                    {"replaceAllText": {
                        "containsText": {"text": "{{TITLE_SUMMARY}}"},
                        "replaceText": ringkasan_judul,
                        "pageObjectIds": [id_templat_summary],
                    }}
                ]},
            ).execute()
        except Exception as summary_err:
            status_box.error(f"Failed to fill in the title summary on the cover slide: {summary_err}")

    if sheets_append_data:
        try:
            status_box.info("Sending data (Title & First Sentence) to Spreadsheet...")
            sheets_service.spreadsheets().values().append(
                spreadsheetId=TARGET_SPREADSHEET_ID,
                range=TARGET_SHEET_RANGE,
                valueInputOption="USER_ENTERED",
                body={"values": sheets_append_data},
            ).execute()
            status_box.success("Data successfully added to Spreadsheet!")
        except Exception as sheet_err:
            status_box.error(f"Midweek slide succeeded, but failed to add to Spreadsheet: {sheet_err}")

    return link_presentasi, processed_data


def jalankan_otomatisasi_endweek(creds, processed_data, selections, status_box):
    drive_service = build("drive", "v3", credentials=creds)
    slides_service = build("slides", "v1", credentials=creds)

    nama_slide_baru = f"Final Endweek - {time.strftime('%Y-%m-%d %H:%M:%S')}"
    copy = drive_service.files().copy(
        fileId=ENDWEEK_TEMPLATE_ID, body={"name": nama_slide_baru}
    ).execute()
    id_slide_baru = copy.get("id")
    link_presentasi = f"https://docs.google.com/presentation/d/{id_slide_baru}/edit"

    presentation = slides_service.presentations().get(presentationId=id_slide_baru).execute()
    id_fb_main, id_fb_comment, id_promo_main = cari_template_slide_endweek(presentation)

    if not id_fb_main or not id_promo_main:
        raise Exception("Endweek template is incomplete! Make sure there is a slide with the text 'Facebook Group' & 'Promotion'.")

    slide_count = len(presentation.get("slides", []))

    for item in processed_data:
        fname = item["filename"]
        format_pilihan = selections[fname]
        status_box.info(f"Processing Endweek: {fname} as {format_pilihan}...")

        judul, konteks = item["title"], item["context"]
        img_main, img_cmt = item["img_main"], item["img_cmt"]
        insight_endweek = " ".join(item["insight_list"])
        ringkasan_summary = item.get("summary", konteks)

        if "Format 1" in format_pilihan:
            res_dup = slides_service.presentations().batchUpdate(
                presentationId=id_slide_baru, body={"requests": [{"duplicateObject": {"objectId": id_fb_main}}]}
            ).execute()
            id_baru_main = res_dup["replies"][0]["duplicateObject"]["objectId"]
            req_main = [
                {"updateSlidesPosition": {"slideObjectIds": [id_baru_main], "insertionIndex": slide_count}},
                {"replaceAllText": {"containsText": {"text": "{{TITLE}}"}, "replaceText": judul, "pageObjectIds": [id_baru_main]}},
                {"replaceAllText": {"containsText": {"text": "{{CONTEXT}}"}, "replaceText": konteks, "pageObjectIds": [id_baru_main]}},
                {"replaceAllText": {"containsText": {"text": "{{INSIGHT}}"}, "replaceText": insight_endweek, "pageObjectIds": [id_baru_main]}},
                {"replaceAllText": {"containsText": {"text": "{{Summary}}"}, "replaceText": ringkasan_summary, "pageObjectIds": [id_baru_main]}},
                {"replaceAllShapesWithImage": {"imageUrl": img_main, "replaceMethod": "CENTER_INSIDE", "containsText": {"text": "{{IMG}}", "matchCase": True}, "pageObjectIds": [id_baru_main]}},
            ]
            slides_service.presentations().batchUpdate(presentationId=id_slide_baru, body={"requests": req_main}).execute()
            slide_count += 1

            if img_cmt and id_fb_comment:
                res_dup_c = slides_service.presentations().batchUpdate(
                    presentationId=id_slide_baru, body={"requests": [{"duplicateObject": {"objectId": id_fb_comment}}]}
                ).execute()
                id_baru_comment = res_dup_c["replies"][0]["duplicateObject"]["objectId"]
                req_cmt = [
                    {"updateSlidesPosition": {"slideObjectIds": [id_baru_comment], "insertionIndex": slide_count}},
                    {"replaceAllText": {"containsText": {"text": "{{TITLE}}"}, "replaceText": judul, "pageObjectIds": [id_baru_comment]}},
                    {"replaceAllShapesWithImage": {"imageUrl": img_cmt, "replaceMethod": "CENTER_INSIDE", "containsText": {"text": "{{CMT}}", "matchCase": True}, "pageObjectIds": [id_baru_comment]}},
                ]
                slides_service.presentations().batchUpdate(presentationId=id_slide_baru, body={"requests": req_cmt}).execute()
                slide_count += 1
        else:
            res_dup = slides_service.presentations().batchUpdate(
                presentationId=id_slide_baru, body={"requests": [{"duplicateObject": {"objectId": id_promo_main}}]}
            ).execute()
            id_baru_main = res_dup["replies"][0]["duplicateObject"]["objectId"]
            req_main = [
                {"updateSlidesPosition": {"slideObjectIds": [id_baru_main], "insertionIndex": slide_count}},
                {"replaceAllText": {"containsText": {"text": "{{TITLE}}"}, "replaceText": judul, "pageObjectIds": [id_baru_main]}},
                {"replaceAllText": {"containsText": {"text": "{{CONTEXT}}"}, "replaceText": konteks, "pageObjectIds": [id_baru_main]}},
                {"replaceAllText": {"containsText": {"text": "{{INSIGHT}}"}, "replaceText": insight_endweek, "pageObjectIds": [id_baru_main]}},
                {"replaceAllText": {"containsText": {"text": "{{Summary}}"}, "replaceText": ringkasan_summary, "pageObjectIds": [id_baru_main]}},
                {"replaceAllShapesWithImage": {"imageUrl": img_main, "replaceMethod": "CENTER_INSIDE", "containsText": {"text": "{{IMG}}", "matchCase": True}, "pageObjectIds": [id_baru_main]}},
            ]
            slides_service.presentations().batchUpdate(presentationId=id_slide_baru, body={"requests": req_main}).execute()
            slide_count += 1

    return link_presentasi


# ==========================================
# 5. UI
# ==========================================
st.markdown('<div class="gns-eyebrow">Automation Tool</div>', unsafe_allow_html=True)
st.title("GNS Censor to Endweek (One-Shot)")
st.caption("Upload → Automatic censor (review & approve) → Midweek Slide (censored images) → Choose Format → Endweek Slide done.")

try:
    creds = get_creds()
except Exception as e:
    st.error(f"Failed to authenticate with Google: {e}")
    st.stop()

with st.sidebar:
    st.header("Censor Settings")
    offset_y = st.slider("Vertical Shift (Y) px:", min_value=-20, max_value=20, value=-1, step=1)
    pad_ukuran = st.slider("Extra Width (Padding px):", min_value=0, max_value=20, value=0, step=1)
    match_threshold = st.slider(
        "Name Match Threshold (fuzzy match):", min_value=0.5, max_value=1.0, value=0.72, step=0.02,
        help="If a name fails to get censored, try lowering this value a bit."
    )

for key, default in [
    ("censor_done", False), ("censored_items", []),
    ("approved", False), ("processed_data", []),
    ("midweek_link", ""), ("endweek_link", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.divider()

# ---------- STAGE 1: UPLOAD ----------
st.subheader("1. Upload Images")
week_range_input = st.text_input("Enter Date / Week Range (free text):", value="")
main_files = st.file_uploader("Upload main image", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="main_uploader")
comment_files = st.file_uploader(
    "Upload comment image (optional — filename must match its paired main image)",
    type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="comment_uploader"
)
promo_files = st.file_uploader(
    "Upload promo image (optional — NOT censored, goes straight into Midweek as-is)",
    type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="promo_uploader"
)
if promo_files:
    st.caption(f"{len(promo_files)} promo image(s) ready (not censored):")
    cols = st.columns(min(len(promo_files), 6))
    for i, pf in enumerate(promo_files):
        cols[i % len(cols)].image(pf, caption=pf.name, use_container_width=True)

news_items = []
if main_files:
    comment_by_name = {f.name: f for f in (comment_files or [])}
    for mf in main_files:
        news_items.append({"main": mf, "comment": comment_by_name.get(mf.name)})

# ---------- STAGE 2: CENSOR ----------
if news_items:
    st.divider()
    st.subheader("2. Automatic Censor")
    mulai_sensor = st.button(f"Run Censor ({len(news_items)} image pairs)", type="primary", use_container_width=True)

    if mulai_sensor:
        genai.configure(api_key=GEMINI_API_KEY)
        model_fallback_list = get_model_fallback_list()
        reader = load_ocr_reader()
        status_box = st.empty()
        progress_bar = st.progress(0)

        censored_items = []
        jumlah = len(news_items)
        for idx, item in enumerate(news_items):
            main_file, comment_file = item["main"], item["comment"]
            try:
                img_asli_main, img_sensor_main, names_main, unmatched_main = sensor_satu_gambar(
                    main_file, model_fallback_list, reader, match_threshold, pad_ukuran, offset_y, status_box
                )
                img_sensor_cmt, names_cmt, unmatched_cmt = None, [], []
                img_asli_cmt = None
                if comment_file:
                    img_asli_cmt, img_sensor_cmt, names_cmt, unmatched_cmt = sensor_satu_gambar(
                        comment_file, model_fallback_list, reader, match_threshold, pad_ukuran, offset_y, status_box
                    )

                censored_items.append({
                    "filename": main_file.name,
                    "img_main_pil": img_sensor_main,
                    "img_cmt_pil": img_sensor_cmt,
                    "preview_main_asli": img_asli_main,
                    "preview_cmt_asli": img_asli_cmt,
                    "names_main": names_main, "unmatched_main": unmatched_main,
                    "names_cmt": names_cmt, "unmatched_cmt": unmatched_cmt,
                })
            except Exception as e:
                st.error(f"Failed to process {main_file.name}: {e}")
            progress_bar.progress((idx + 1) / jumlah)

        st.session_state.censored_items = censored_items
        st.session_state.censor_done = True
        st.session_state.approved = False  # reset approval if censor is run again
        st.session_state.midweek_link = ""
        st.session_state.processed_data = []
        st.success("Censoring complete. Check the results below before continuing to Endweek.")

# ---------- REVIEW CENSOR RESULTS ----------
if st.session_state.censor_done and st.session_state.censored_items:
    st.divider()
    st.subheader("Review Censor Results")
    for item in st.session_state.censored_items:
        with st.container(border=True):
            st.markdown(f"**{item['filename']}**")
            col1, col2 = st.columns(2)
            col1.image(item["preview_main_asli"], caption="Main - Original", use_container_width=True)
            col2.image(item["img_main_pil"], caption=f"Main - Censored ({len(item['names_main'])} names)", use_container_width=True)
            if item["unmatched_main"]:
                st.warning(f"Main names not matched: {item['unmatched_main']}")

            if item["img_cmt_pil"] is not None:
                col3, col4 = st.columns(2)
                col3.image(item["preview_cmt_asli"], caption="Comment - Original", use_container_width=True)
                col4.image(item["img_cmt_pil"], caption=f"Comment - Censored ({len(item['names_cmt'])} names)", use_container_width=True)
                if item["unmatched_cmt"]:
                    st.warning(f"Comment names not matched: {item['unmatched_cmt']}")

    st.info("Not quite right? Adjust the sliders in the sidebar then click **Run Censor** again above.")

# ---------- CONTINUE TO MIDWEEK ----------
butuh_sensor_dulu = bool(news_items) and not st.session_state.censor_done
ada_yang_bisa_diproses = bool(st.session_state.censored_items) or bool(promo_files)

if ada_yang_bisa_diproses and not butuh_sensor_dulu:
    st.divider()
    if not week_range_input.strip():
        st.warning("Fill in 'Date / Week Range' above before continuing to Midweek.")
    else:
        jumlah_promo = len(promo_files or [])
        jumlah_sensor = len(st.session_state.censored_items)
        label = f"Continue to Midweek & Endweek ({jumlah_sensor} censored + {jumlah_promo} promo)"
        setuju = st.button(label, type="primary", use_container_width=True)
        if setuju:
            status_box2 = st.empty()
            progress_bar2 = st.progress(0)
            with st.spinner("Creating Midweek Slide & uploading to Drive..."):
                try:
                    promo_items = []
                    for pf in (promo_files or []):
                        img_promo = PIL.Image.open(pf).convert("RGB")
                        promo_items.append({"filename": pf.name, "img_main_pil": img_promo, "img_cmt_pil": None, "is_promo": True})

                    semua_items = st.session_state.censored_items + promo_items

                    link_midweek, processed_data = jalankan_otomatisasi_midweek_dari_sensor(
                        creds, semua_items, week_range_input, progress_bar2, status_box2
                    )
                    st.session_state.midweek_link = link_midweek
                    st.session_state.processed_data = processed_data
                    st.session_state.approved = True
                    st.success("Midweek Slide created successfully!")
                except Exception as e:
                    st.error(f"Error while creating Midweek: {e}")

if st.session_state.midweek_link:
    st.markdown(f"**[Open Midweek Presentation]({st.session_state.midweek_link})**")

# ---------- STAGE 3: CHOOSE FORMAT & GENERATE ENDWEEK ----------
if st.session_state.approved and st.session_state.processed_data:
    st.divider()
    st.subheader("3. Slide Category for Endweek")
    st.info("Choose the presentation format for each image. Images used in the slides are already the censored versions.")

    selections = {}
    for item in st.session_state.processed_data:
        selections[item["filename"]] = st.radio(
            f"Format for image: **{item['filename']}**",
            options=["Format 1 (Facebook Group)", "Format 2 (Promotion)"],
            key=f"format_{item['filename']}",
            horizontal=True,
        )

    if st.button("Buat Slide Endweek", type="secondary", use_container_width=True):
        status_box3 = st.empty()
        with st.spinner("Building Endweek Slide..."):
            try:
                link_endweek = jalankan_otomatisasi_endweek(
                    creds, st.session_state.processed_data, selections, status_box3
                )
                st.session_state.endweek_link = link_endweek
                st.success("Endweek Slide Complete!")
            except Exception as e:
                st.error(f"Error while building Endweek: {e}")

if st.session_state.endweek_link:
    st.markdown(f"**[Open Endweek Presentation]({st.session_state.endweek_link})**")

if not news_items:
    st.info("Upload the main image (and comment image, if any) to get started.")
