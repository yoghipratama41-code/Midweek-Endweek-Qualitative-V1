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
# 0. KONFIGURASI
# ==========================================
st.set_page_config(page_title="GNS Censor -> Endweek", page_icon="🛡️", layout="wide")

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
# 1. AUTH GOOGLE
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
# 2. GEMINI — SATU FUNGSI FALLBACK DIPAKAI UNTUK SEMUA PANGGILAN AI
#    (identifikasi nama utk sensor, MAUPUN analisis title/context/insight)
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
    """Kirim prompt+gambar ke Gemini. Retry dengan backoff kalau limit, pindah model kalau tetap gagal."""
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
                        f"⚠️ Model **{nama_model_pendek}** kena limit/sibuk "
                        f"(percobaan {attempt + 1}/{max_retry_per_model}). Menunggu {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                    delay *= 2
                else:
                    status_box.warning(f"⚠️ Model **{nama_model_pendek}** error: {err_msg[:150]}")
                    break
        status_box.info(f"➡️ Pindah dari model **{nama_model_pendek}** ke model berikutnya...")
    raise Exception(f"Semua model gagal dicoba. Error terakhir: {last_err}")


# ==========================================
# 3. TAHAP SENSOR (Gemini identifikasi nama + EasyOCR lokasi pixel)
# ==========================================
AIM_PROMPT = """
Analisis gambar tangkapan layar media sosial ini (bisa berupa postingan utama, caption, ATAU komentar).
Fokus pada SEMUA NAMA PENGGUNA (username/account name) yang dicetak TEBAL (BOLD) di mana pun posisinya di gambar ini —
baik itu nama yang memposting konten utama, maupun nama-nama akun pada tiap komentar.
Jangan mengambil nama yang hanya disebut/di-mention di dalam ISI teks komentar atau caption (bukan nama akun itu sendiri).

Kembalikan hasilnya dalam format JSON murni, tanpa teks penjelasan, tanpa markdown.
JSON harus berupa array of string, urut dari atas ke bawah gambar. Kalau tidak ada nama akun yang terlihat, kembalikan array kosong [].
Contoh:
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
        draw.rectangle([x1 - pad_ukuran, y1_adj - pad_ukuran, x2 + pad_ukuran, y2_adj + pad_ukuran], fill="black")
    return censored_image


def sensor_satu_gambar(uploaded_file, model_fallback_list, reader, match_threshold, pad_ukuran, offset_y, status_box):
    """Jalankan pipeline sensor untuk satu file upload, return (image_asli, image_disensor, name_list, unmatched_names)."""
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
# 4. TAHAP ENDWEEK (analisis AI atas gambar yg SUDAH disensor + upload Drive + generate slide)
# ==========================================
PROMPT_ANALISIS = """
Analyze this image (and comments if any) for a professional research slide.
Write the analysis in one single cohesive paragraph in English.

Strict Rules:
1. Refer to both users and drivers ONLY as "rider".
2. Do NOT mention any social media usernames, account names, or the image filename.
3. You MAY mention the name of the gig/delivery platform (e.g. Grab, Foodpanda, Lalamove, etc.) if it is visibly shown or referenced in the image or comments. Only usernames, handles, and filenames are off-limits — platform names are allowed and encouraged when relevant.
4. Never begin the paragraph with a generic opener such as "This image...", "This discussion...", "The illustration...", "This conversation...", or any variant that refers to "the image" or "the illustration" itself. Do not describe the fact that you are looking at an image at all. Instead, dive straight into the substance: open with the rider's situation, the specific complaint or sentiment, the operational issue, a concrete detail, or the context of the exchange. Vary the opening construction from one slide to the next (e.g. start with a cause, a location, a time reference, a rider's action, or a direct statement of the issue) so that consecutive outputs do not read as templated or repetitive.
5. The paragraph must consist of at least 3-4 sentences.

Output Format:
[TITLE] Write a long, specific, headline-style title, roughly 12-20 words, that reads like a mini research-slide headline capturing the core theme plus a specific supporting detail (not a short generic label). For reference, match this style and length:
"Cross-Region Operational Challenges: Detailed Suggestion from Community Regarding Working on a Different Zone Registered"
[CONTENT] Write the full paragraph here.
"""


def upload_pil_ke_drive(drive_service, pil_image, filename):
    """Upload PIL image (yang sudah disensor) ke Drive sebagai PNG, kembalikan link publik."""
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
    for slide in presentation.get("slides", []):
        txt = _get_slide_text(slide)
        if "{{IMG}}" in txt and id_main is None:
            id_main = slide["objectId"]
        if "{{CMT}}" in txt and id_comment is None:
            id_comment = slide["objectId"]
    return id_main, id_comment


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
    """
    Sama seperti jalankan_otomatisasi_midweek versi asli, tapi:
    - gambar yang dipakai adalah gambar yang SUDAH DISENSOR (bukan file upload asli)
    - setiap slide baru dibuat dengan duplicateObject() dari slide template,
      lalu placeholder-nya diisi di slide HASIL DUPLIKAT — template referensi sendiri
      tidak pernah disentuh/diedit langsung.
    Return: (link_presentasi_midweek, processed_data) — processed_data lalu dipakai
    untuk generate Endweek di tahap berikutnya.
    """
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
    id_templat_main, id_templat_comment = cari_template_slide(presentation)
    if not id_templat_main:
        raise Exception("Template Midweek tidak ditemukan! Pastikan ada slide berisi placeholder '{{IMG}}'.")

    slide_count = len(presentation.get("slides", []))
    jumlah = len(censored_items)

    processed_data = []
    sheets_append_data = []

    for index, item in enumerate(censored_items):
        fname = item["filename"]
        try:
            status_box.info(f"[{index+1}/{jumlah}] Memproses Midweek: {fname}...")

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

            link_gambar_main = upload_pil_ke_drive(drive_service, item["img_main_pil"], fname)
            link_gambar_comment = None
            if item.get("img_cmt_pil") is not None:
                link_gambar_comment = upload_pil_ke_drive(drive_service, item["img_cmt_pil"], f"comment_{fname}")

            processed_data.append({
                "filename": fname,
                "title": judul,
                "context": konteks,
                "insight_list": insight_list,
                "img_main": link_gambar_main,
                "img_cmt": link_gambar_comment,
            })

            # ---- Slide gambar utama: duplicate dari template, edit di hasil duplikat ----
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
                {"replaceAllShapesWithImage": {"imageUrl": link_gambar_main, "replaceMethod": "CENTER_INSIDE", "containsText": {"text": "{{IMG}}", "matchCase": True}, "pageObjectIds": [id_baru_main]}},
            ]
            slides_service.presentations().batchUpdate(presentationId=id_slide_baru, body={"requests": req_main}).execute()
            slide_count += 1

            # ---- Slide gambar komentar (kalau ada): duplicate juga dari template ----
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
            status_box.error(f"❌ {fname} dilewati: {e}")
        finally:
            progress_bar.progress((index + 1) / jumlah)

    if sheets_append_data:
        try:
            status_box.info("Mengirim data (Title & Kalimat Pertama) ke Spreadsheet...")
            sheets_service.spreadsheets().values().append(
                spreadsheetId=TARGET_SPREADSHEET_ID,
                range=TARGET_SHEET_RANGE,
                valueInputOption="USER_ENTERED",
                body={"values": sheets_append_data},
            ).execute()
            status_box.success("✅ Data berhasil ditambahkan ke Spreadsheet!")
        except Exception as sheet_err:
            status_box.error(f"⚠️ Slide Midweek sukses, namun gagal menambahkan ke Spreadsheet: {sheet_err}")

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
        raise Exception("Template Endweek tidak lengkap! Pastikan ada slide ber-teks 'Facebook Group' & 'Promotion'.")

    slide_count = len(presentation.get("slides", []))

    for item in processed_data:
        fname = item["filename"]
        format_pilihan = selections[fname]
        status_box.info(f"Memproses Endweek: {fname} sebagai {format_pilihan}...")

        judul, konteks = item["title"], item["context"]
        img_main, img_cmt = item["img_main"], item["img_cmt"]
        insight_endweek = " ".join(item["insight_list"])

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
                {"replaceAllShapesWithImage": {"imageUrl": img_main, "replaceMethod": "CENTER_INSIDE", "containsText": {"text": "{{IMG}}", "matchCase": True}, "pageObjectIds": [id_baru_main]}},
            ]
            slides_service.presentations().batchUpdate(presentationId=id_slide_baru, body={"requests": req_main}).execute()
            slide_count += 1

    return link_presentasi


# ==========================================
# 5. UI
# ==========================================
st.title("🛡️➡️📊 GNS Censor → Endweek (Sekali Jalan)")
st.caption("Upload → Sensor otomatis (review & approve) → Slide Midweek (gambar sensor) → Pilih Format → Slide Endweek jadi.")

try:
    creds = get_creds()
except Exception as e:
    st.error(f"Gagal autentikasi ke Google: {e}")
    st.stop()

with st.sidebar:
    st.header("🛠️ Pengaturan Sensor")
    offset_y = st.slider("Geser Vertikal (Y) px:", min_value=-20, max_value=20, value=-1, step=1)
    pad_ukuran = st.slider("Lebar Ekstra (Padding px):", min_value=0, max_value=20, value=0, step=1)
    match_threshold = st.slider(
        "Ambang Kecocokan Nama (fuzzy match):", min_value=0.5, max_value=1.0, value=0.72, step=0.02,
        help="Kalau ada nama yang gagal tersensor, coba turunkan nilai ini sedikit."
    )

for key, default in [
    ("censor_done", False), ("censored_items", []),
    ("approved", False), ("processed_data", []),
    ("midweek_link", ""), ("endweek_link", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.divider()

# ---------- TAHAP 1: UPLOAD ----------
st.subheader("1️⃣ Upload Gambar")
week_range_input = st.text_input("Masukkan Tanggal / Week Range (Bebas ketik):", value="")
main_files = st.file_uploader("Upload gambar utama", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="main_uploader")
comment_files = st.file_uploader(
    "Upload gambar komentar (opsional — nama file harus sama dengan gambar utama pasangannya)",
    type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="comment_uploader"
)
promo_files = st.file_uploader(
    "Upload gambar promo (opsional — TIDAK melalui sensor, langsung masuk Midweek apa adanya)",
    type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="promo_uploader"
)
if promo_files:
    st.caption(f"🖼️ {len(promo_files)} gambar promo siap (tidak disensor):")
    cols = st.columns(min(len(promo_files), 6))
    for i, pf in enumerate(promo_files):
        cols[i % len(cols)].image(pf, caption=pf.name, use_container_width=True)

news_items = []
if main_files:
    comment_by_name = {f.name: f for f in (comment_files or [])}
    for mf in main_files:
        news_items.append({"main": mf, "comment": comment_by_name.get(mf.name)})

# ---------- TAHAP 2: SENSOR ----------
if news_items:
    st.divider()
    st.subheader("2️⃣ Sensor Otomatis")
    mulai_sensor = st.button(f"🚀 Jalankan Sensor ({len(news_items)} pasang gambar)", type="primary", use_container_width=True)

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
                st.error(f"❌ Gagal memproses {main_file.name}: {e}")
            progress_bar.progress((idx + 1) / jumlah)

        st.session_state.censored_items = censored_items
        st.session_state.censor_done = True
        st.session_state.approved = False  # reset approval kalau sensor dijalankan ulang
        st.session_state.midweek_link = ""
        st.session_state.processed_data = []
        st.success("✅ Sensor selesai. Cek hasilnya di bawah sebelum lanjut ke Endweek.")

# ---------- REVIEW HASIL SENSOR (hanya relevan kalau ada gambar utama/komentar) ----------
if st.session_state.censor_done and st.session_state.censored_items:
    st.divider()
    st.subheader("👀 Review Hasil Sensor")
    for item in st.session_state.censored_items:
        with st.container(border=True):
            st.markdown(f"**{item['filename']}**")
            col1, col2 = st.columns(2)
            col1.image(item["preview_main_asli"], caption="Utama - Asli", use_container_width=True)
            col2.image(item["img_main_pil"], caption=f"Utama - Disensor ({len(item['names_main'])} nama)", use_container_width=True)
            if item["unmatched_main"]:
                st.warning(f"⚠️ Nama utama tidak tercocokkan: {item['unmatched_main']}")

            if item["img_cmt_pil"] is not None:
                col3, col4 = st.columns(2)
                col3.image(item["preview_cmt_asli"], caption="Komentar - Asli", use_container_width=True)
                col4.image(item["img_cmt_pil"], caption=f"Komentar - Disensor ({len(item['names_cmt'])} nama)", use_container_width=True)
                if item["unmatched_cmt"]:
                    st.warning(f"⚠️ Nama komentar tidak tercocokkan: {item['unmatched_cmt']}")

    st.info("Kurang pas? Ubah slider di sidebar lalu klik **🚀 Jalankan Sensor** lagi di atas.")

# ---------- LANJUT KE MIDWEEK ----------
# Bisa lanjut kalau: (tidak ada gambar utama sama sekali) ATAU (gambar utama sudah disensor & direview).
# Gambar promo tidak perlu menunggu apa pun — langsung ikut ke Midweek apa adanya.
butuh_sensor_dulu = bool(news_items) and not st.session_state.censor_done
ada_yang_bisa_diproses = bool(st.session_state.censored_items) or bool(promo_files)

if ada_yang_bisa_diproses and not butuh_sensor_dulu:
    st.divider()
    if not week_range_input.strip():
        st.warning("⚠️ Isi 'Tanggal / Week Range' di atas dulu sebelum lanjut ke Midweek.")
    else:
        jumlah_promo = len(promo_files or [])
        jumlah_sensor = len(st.session_state.censored_items)
        label = f"✅ Lanjut ke Midweek & Endweek ({jumlah_sensor} disensor + {jumlah_promo} promo)"
        setuju = st.button(label, type="primary", use_container_width=True)
        if setuju:
            status_box2 = st.empty()
            progress_bar2 = st.progress(0)
            with st.spinner("Membuat Slide Midweek & mengupload ke Drive..."):
                try:
                    promo_items = []
                    for pf in (promo_files or []):
                        img_promo = PIL.Image.open(pf).convert("RGB")
                        promo_items.append({"filename": pf.name, "img_main_pil": img_promo, "img_cmt_pil": None})

                    semua_items = st.session_state.censored_items + promo_items

                    link_midweek, processed_data = jalankan_otomatisasi_midweek_dari_sensor(
                        creds, semua_items, week_range_input, progress_bar2, status_box2
                    )
                    st.session_state.midweek_link = link_midweek
                    st.session_state.processed_data = processed_data
                    st.session_state.approved = True
                    st.success("🎉 Slide Midweek selesai dibuat!")
                except Exception as e:
                    st.error(f"Kesalahan saat membuat Midweek: {e}")

if st.session_state.midweek_link:
    st.markdown(f"**[📂 Buka Presentasi Midweek]({st.session_state.midweek_link})**")

# ---------- TAHAP 3: PILIH FORMAT & GENERATE ENDWEEK ----------
if st.session_state.approved and st.session_state.processed_data:
    st.divider()
    st.subheader("3️⃣ Kategori Slide untuk Endweek")
    st.info("Pilih format presentasi untuk masing-masing gambar. Gambar yang dipakai di slide sudah versi tersensor.")

    selections = {}
    for item in st.session_state.processed_data:
        selections[item["filename"]] = st.radio(
            f"Format untuk gambar: **{item['filename']}**",
            options=["Format 1 (Facebook Group)", "Format 2 (Promotion)"],
            key=f"format_{item['filename']}",
            horizontal=True,
        )

    if st.button("🚀 Buat Slide Endweek", type="secondary", use_container_width=True):
        status_box3 = st.empty()
        with st.spinner("Menyusun Slide Endweek..."):
            try:
                link_endweek = jalankan_otomatisasi_endweek(
                    creds, st.session_state.processed_data, selections, status_box3
                )
                st.session_state.endweek_link = link_endweek
                st.balloons()
                st.success("🎉 Slide Endweek Selesai!")
            except Exception as e:
                st.error(f"Kesalahan saat menyusun Endweek: {e}")

if st.session_state.endweek_link:
    st.markdown(f"**[📂 Buka Presentasi Endweek]({st.session_state.endweek_link})**")

if not news_items:
    st.info("👆 Upload gambar utama (dan komentar jika ada) untuk memulai.")
