from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance
import io
import base64
import sqlite3
import os
import datetime
import cv2

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
# CONFIG  ← updated for your device
# ─────────────────────────────────────────────────────────────
MODEL_PATH  = r"C:\Users\rajes\Desktop\Web app\ocular_cad_model.keras"
IMAGE_SIZE  = (256, 256)
CLASS_NAMES = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
DB_PATH     = r"C:\Users\rajes\Desktop\Web app\ocular_cad.db"
ESP32_IP    = "http://10.244.72.184/capture"   # ← change to your ESP32-CAM IP
# ─────────────────────────────────────────────────────────────

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print(f"✅ Model loaded!  Input shape expected: {model.input_shape}")


# ─────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT,
            age         TEXT,
            gender      TEXT,
            diagnosis   TEXT,
            confidence  REAL,
            risk_level  TEXT,
            source      TEXT,
            timestamp   TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()


def save_to_db(name, age, gender, diagnosis, confidence, risk_level, source="Upload"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO patients (name, age, gender, diagnosis, confidence, risk_level, source, timestamp) VALUES (?,?,?,?,?,?,?,?)",
        (name, age, gender, diagnosis, confidence, risk_level, source,
         datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()


def get_all_records():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, age, gender, diagnosis, confidence, risk_level, source, timestamp FROM patients ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "age": r[2], "gender": r[3],
             "diagnosis": r[4], "confidence": r[5], "risk_level": r[6],
             "source": r[7], "timestamp": r[8]}
            for r in rows]


def delete_record(record_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM patients WHERE id=?", (record_id,))
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────
# IMAGE PROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────
def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def apply_clahe(img_rgb_np):
    green = img_rgb_np[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    enhanced_rgb = cv2.merge([enhanced, enhanced, enhanced])
    return enhanced, enhanced_rgb


def apply_segmentation(img_rgb_np):
    green = img_rgb_np[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask_3ch = cv2.merge([mask, mask, mask])
    segmented = cv2.bitwise_and(img_rgb_np, mask_3ch)
    return segmented


def process_pipeline(image_bytes):
    original_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    step1_b64    = pil_to_base64(original_pil)

    resized_pil  = original_pil.resize(IMAGE_SIZE, Image.BILINEAR)
    resized_np   = np.array(resized_pil)
    step2_b64    = pil_to_base64(resized_pil)

    _, clahe_rgb = apply_clahe(resized_np)
    clahe_pil    = Image.fromarray(clahe_rgb.astype(np.uint8))
    step3_b64    = pil_to_base64(clahe_pil)

    norm_display = Image.fromarray(resized_np)
    norm_display = ImageEnhance.Contrast(norm_display).enhance(1.4)
    step4_b64    = pil_to_base64(norm_display)

    seg_np    = apply_segmentation(resized_np)
    seg_pil   = Image.fromarray(seg_np.astype(np.uint8))
    step5_b64 = pil_to_base64(seg_pil)

    img_array = np.expand_dims(resized_np, axis=0)

    pipeline_b64 = [step1_b64, step2_b64, step3_b64, step4_b64, step5_b64]
    return img_array, pipeline_b64


def run_prediction(image_bytes, name, age, gender, source="Upload"):
    img_array, pipeline_images = process_pipeline(image_bytes)
    predictions     = model.predict(img_array, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence      = float(predictions[predicted_index]) * 100

    if predicted_class.lower() == "normal":
        risk = "Normal"
    elif confidence > 80:
        risk = "High"
    else:
        risk = "Medium"

    save_to_db(name, age, gender, predicted_class, confidence, risk, source)

    all_probs = [
        {"class": CLASS_NAMES[i], "probability": round(float(p) * 100, 2)}
        for i, p in enumerate(predictions)
    ]
    return {
        "predicted_class":   predicted_class,
        "confidence":        round(confidence, 2),
        "all_probabilities": all_probs,
        "pipeline_images":   pipeline_images,
        "source":            source,
    }


# ─────────────────────────────────────────────────────────────
# HTML TEMPLATE
# ─────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>OcularAI — CAD System</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#080c14;--surface:#0f1622;--surface2:#151d2e;--border:#1a2540;--border2:#22304a;
  --accent:#00d4ff;--accent2:#7c6fff;--accent3:#00ffb3;
  --text:#e8edf5;--muted:#5a6a85;--muted2:#3a4a65;
  --success:#00e599;--warning:#ffb547;--danger:#ff5f6d;
  --font:'Space Grotesk',sans-serif;
}
body{font-family:var(--font);background:var(--bg);color:var(--text);min-height:100vh;display:flex;flex-direction:column;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;background:
  radial-gradient(ellipse 70% 50% at 10% 0%,rgba(0,212,255,0.06) 0%,transparent 60%),
  radial-gradient(ellipse 60% 60% at 90% 90%,rgba(124,111,255,0.07) 0%,transparent 60%),
  radial-gradient(ellipse 40% 40% at 50% 50%,rgba(0,255,179,0.03) 0%,transparent 70%);
  pointer-events:none;z-index:0}

header{position:relative;z-index:10;padding:1rem 2.5rem;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--border);backdrop-filter:blur(10px);background:rgba(8,12,20,0.8)}
.logo{display:flex;align-items:center;gap:0.75rem}
.logo-eye{width:36px;height:36px;border:2px solid var(--accent);border-radius:50%;display:flex;align-items:center;justify-content:center;position:relative}
.logo-eye::after{content:'';width:12px;height:12px;background:var(--accent);border-radius:50%;box-shadow:0 0 10px var(--accent)}
.logo-text{font-family:'DM Serif Display',serif;font-size:1.4rem;letter-spacing:0.02em}
.logo-text em{color:var(--accent);font-style:normal}
.nav-tabs{display:flex;gap:0.25rem;background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:0.25rem}
.nav-tab{padding:0.45rem 1.1rem;font-size:0.82rem;font-weight:600;border-radius:7px;cursor:pointer;transition:all 0.2s;color:var(--muted);border:none;background:none;font-family:var(--font)}
.nav-tab.active{background:var(--accent);color:#080c14}
.version-badge{font-size:0.7rem;font-weight:600;background:rgba(0,212,255,0.1);color:var(--accent);border:1px solid rgba(0,212,255,0.25);padding:0.3rem 0.8rem;border-radius:999px;letter-spacing:0.06em}

.page{display:none;position:relative;z-index:1;flex:1}
.page.active{display:flex;flex-direction:column}

#page-analyze{padding:2rem 2.5rem;gap:2rem}
.analyze-grid{display:grid;grid-template-columns:380px 1fr;gap:2rem;flex:1}

.input-panel{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:1.75rem;display:flex;flex-direction:column;gap:1.25rem}
.panel-label{font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:var(--muted);margin-bottom:0.25rem}

.form-row{display:grid;grid-template-columns:1fr 1fr;gap:0.75rem}
.form-group{display:flex;flex-direction:column;gap:0.35rem}
.form-group.full{grid-column:1/-1}
label{font-size:0.72rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:0.06em}
input,select{background:rgba(255,255,255,0.04);border:1px solid var(--border2);border-radius:8px;padding:0.6rem 0.9rem;font-family:var(--font);font-size:0.88rem;color:var(--text);outline:none;transition:border-color 0.2s,box-shadow 0.2s;width:100%}
input:focus,select:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(0,212,255,0.1)}
select option{background:#0f1622}
.sep{height:1px;background:var(--border);margin:0.25rem 0}

.source-toggle{display:grid;grid-template-columns:1fr 1fr;gap:0;background:var(--surface2);border:1px solid var(--border2);border-radius:10px;padding:3px;margin-bottom:0.25rem}
.src-btn{padding:0.5rem 0;font-family:var(--font);font-size:0.8rem;font-weight:600;border:none;background:none;color:var(--muted);cursor:pointer;border-radius:7px;transition:all 0.2s;display:flex;align-items:center;justify-content:center;gap:0.4rem}
.src-btn.active{background:var(--accent);color:#080c14}
.src-btn.active-cam{background:var(--accent3);color:#080c14}

.upload-zone{border:2px dashed var(--border2);border-radius:12px;padding:1.75rem 1.5rem;text-align:center;cursor:pointer;transition:all 0.2s;position:relative;background:rgba(255,255,255,0.015)}
.upload-zone:hover,.upload-zone.drag{border-color:var(--accent);background:rgba(0,212,255,0.04)}
.upload-zone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.upload-icon{font-size:1.8rem;margin-bottom:0.5rem}
.upload-label{font-size:0.88rem;font-weight:600;margin-bottom:0.25rem}
.upload-sub{font-size:0.75rem;color:var(--muted)}
#preview-wrap{display:none;margin-top:0.75rem;border-radius:10px;overflow:hidden;border:1px solid var(--border2);position:relative}
#preview-wrap img{width:100%;max-height:180px;object-fit:cover;display:block}
.preview-label{position:absolute;bottom:0;left:0;right:0;padding:0.4rem 0.75rem;background:rgba(0,0,0,0.65);font-size:0.72rem;color:#94a3b8}

.esp32-section{display:flex;flex-direction:column;gap:0.85rem}
.esp32-status{display:flex;align-items:center;gap:0.6rem;background:rgba(255,255,255,0.03);border:1px solid var(--border2);border-radius:8px;padding:0.6rem 0.9rem;font-size:0.78rem}
.esp32-dot{width:9px;height:9px;border-radius:50%;background:var(--muted);flex-shrink:0;transition:background 0.3s,box-shadow 0.3s}
.esp32-dot.online{background:var(--success);box-shadow:0 0 7px var(--success)}
.esp32-dot.offline{background:var(--danger)}
.esp32-dot.checking{background:var(--warning);animation:pulse 0.8s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
.esp32-status-text{flex:1;color:var(--muted)}
.esp32-status-text span{color:var(--text)}
.ping-btn{background:none;border:1px solid var(--border2);color:var(--muted);padding:0.25rem 0.7rem;border-radius:6px;font-family:var(--font);font-size:0.72rem;cursor:pointer;font-weight:600;transition:border-color 0.2s,color 0.2s}
.ping-btn:hover{border-color:var(--accent);color:var(--accent)}
.ip-row{display:flex;gap:0.5rem;align-items:center}
.ip-row input{flex:1;font-size:0.8rem;padding:0.5rem 0.75rem}
.ip-row button{padding:0.5rem 0.85rem;background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.25);color:var(--accent);border-radius:8px;font-family:var(--font);font-size:0.78rem;font-weight:700;cursor:pointer;white-space:nowrap;transition:background 0.2s}
.ip-row button:hover{background:rgba(0,212,255,0.18)}
.esp32-preview{display:none;border-radius:10px;overflow:hidden;border:1px solid var(--border2);position:relative}
.esp32-preview img{width:100%;max-height:170px;object-fit:cover;display:block}
.esp32-preview-lbl{position:absolute;bottom:0;left:0;right:0;padding:0.4rem 0.75rem;background:rgba(0,0,0,0.65);font-size:0.72rem;color:#94a3b8;display:flex;align-items:center;gap:0.4rem}
.live-dot{width:6px;height:6px;border-radius:50%;background:var(--success);animation:pulse 1s ease-in-out infinite}
#esp32-capture-btn{width:100%;padding:0.8rem;font-family:var(--font);font-size:0.9rem;font-weight:700;color:#080c14;background:linear-gradient(135deg,var(--accent3),var(--accent));border:none;border-radius:10px;cursor:pointer;transition:opacity 0.2s,transform 0.15s;display:flex;align-items:center;justify-content:center;gap:0.5rem}
#esp32-capture-btn:hover{opacity:0.88;transform:translateY(-1px)}
#esp32-capture-btn:disabled{opacity:0.35;cursor:not-allowed;transform:none}

.model-badge{display:flex;align-items:center;gap:0.5rem;background:rgba(0,229,153,0.07);border:1px solid rgba(0,229,153,0.2);border-radius:8px;padding:0.55rem 0.85rem;font-size:0.78rem}
.model-badge .dot{width:8px;height:8px;background:var(--success);border-radius:50%;box-shadow:0 0 6px var(--success);flex-shrink:0}
.model-badge strong{color:var(--success)}
#submit-btn{width:100%;padding:0.85rem;font-family:var(--font);font-size:0.95rem;font-weight:700;color:#080c14;background:linear-gradient(135deg,var(--accent),var(--accent2));border:none;border-radius:10px;cursor:pointer;transition:opacity 0.2s,transform 0.15s;display:flex;align-items:center;justify-content:center;gap:0.5rem;letter-spacing:0.02em}
#submit-btn:hover{opacity:0.88;transform:translateY(-1px)}
#submit-btn:disabled{opacity:0.35;cursor:not-allowed;transform:none}
.spin{width:16px;height:16px;border:2.5px solid rgba(0,0,0,0.2);border-top-color:#080c14;border-radius:50%;animation:spin 0.7s linear infinite;display:none}
@keyframes spin{to{transform:rotate(360deg)}}
.err-box{background:rgba(255,95,109,0.08);border:1px solid rgba(255,95,109,0.25);border-radius:8px;padding:0.85rem;color:var(--danger);font-size:0.82rem;display:none;margin-top:0.5rem}

.results-panel{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:1.75rem;display:flex;flex-direction:column;gap:1.5rem;overflow-y:auto}
.placeholder-state{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1rem;color:var(--muted);text-align:center;padding:3rem 1rem}
.ph-icon{font-size:3.5rem;opacity:0.2;filter:grayscale(1)}
.ph-text{font-size:0.9rem;line-height:1.7}
.pipeline-title{font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:var(--muted);margin-bottom:0.75rem}
.pipeline-steps{display:grid;grid-template-columns:repeat(5,1fr);gap:0.75rem}
.pipe-step{background:var(--surface2);border:1px solid var(--border);border-radius:10px;overflow:hidden;transition:border-color 0.2s,transform 0.2s;cursor:pointer}
.pipe-step:hover{border-color:var(--accent2);transform:translateY(-2px)}
.pipe-step img{width:100%;aspect-ratio:1;object-fit:cover;display:block}
.pipe-step-label{padding:0.35rem 0.5rem;text-align:center;font-size:0.65rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:0.06em;background:var(--surface)}
.pipe-step.active-step{border-color:var(--accent)}
.step-num{display:inline-flex;align-items:center;justify-content:center;width:16px;height:16px;background:var(--accent2);border-radius:50%;font-size:0.6rem;font-weight:700;color:white;margin-right:0.3rem}
.diag-row{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
.diag-card{background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:1.25rem}
.diag-card.highlight{border-color:rgba(0,212,255,0.35);background:rgba(0,212,255,0.04)}
.diag-tag{font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:var(--accent);margin-bottom:0.4rem}
.diag-main{font-family:'DM Serif Display',serif;font-size:1.7rem;line-height:1.1;text-transform:capitalize;margin-bottom:0.3rem}
.diag-conf{font-size:0.82rem;color:var(--muted)}
.conf-val{color:var(--success);font-weight:700}
.risk-pill{display:inline-flex;align-items:center;gap:0.35rem;font-size:0.75rem;font-weight:700;padding:0.3rem 0.8rem;border-radius:999px;margin-top:0.6rem}
.risk-normal{background:rgba(0,229,153,0.12);color:var(--success);border:1px solid rgba(0,229,153,0.3)}
.risk-medium{background:rgba(255,181,71,0.12);color:var(--warning);border:1px solid rgba(255,181,71,0.3)}
.risk-high{background:rgba(255,95,109,0.12);color:var(--danger);border:1px solid rgba(255,95,109,0.3)}
.source-badge{display:inline-flex;align-items:center;gap:0.3rem;font-size:0.68rem;font-weight:700;padding:0.2rem 0.6rem;border-radius:999px;margin-top:0.4rem;background:rgba(0,255,179,0.1);color:var(--accent3);border:1px solid rgba(0,255,179,0.25)}
.prob-item{margin-bottom:0.9rem}
.prob-head{display:flex;justify-content:space-between;font-size:0.82rem;margin-bottom:0.3rem;font-weight:500}
.prob-pct{color:var(--muted);font-size:0.78rem}
.bar-bg{height:5px;background:var(--border2);border-radius:999px;overflow:hidden}
.bar-fg{height:100%;border-radius:999px;background:linear-gradient(90deg,var(--accent),var(--accent2));transition:width 0.9s cubic-bezier(.4,0,.2,1);width:0%}
.patient-summary{background:rgba(124,111,255,0.06);border:1px solid rgba(124,111,255,0.2);border-radius:12px;padding:1rem 1.25rem;display:none}
.ps-tag{font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:var(--accent2);margin-bottom:0.75rem}
.ps-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:0.5rem}
.ps-item .ps-label{font-size:0.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.06em}
.ps-item .ps-val{font-size:0.9rem;font-weight:600}
.disclaim{font-size:0.75rem;color:var(--muted);background:rgba(255,181,71,0.05);border:1px solid rgba(255,181,71,0.15);border-radius:8px;padding:0.75rem 1rem;line-height:1.65;display:none}
.disclaim strong{color:var(--warning)}
.modal{position:fixed;inset:0;background:rgba(0,0,0,0.88);z-index:1000;display:none;align-items:center;justify-content:center;flex-direction:column;gap:1rem;padding:2rem}
.modal.open{display:flex}
.modal img{max-width:600px;max-height:70vh;border-radius:12px;border:2px solid var(--border2);object-fit:contain}
.modal-label{font-size:0.85rem;font-weight:600;color:var(--text)}
.modal-close{font-size:0.8rem;color:var(--muted);cursor:pointer;padding:0.5rem 1rem;border:1px solid var(--border2);border-radius:8px;background:var(--surface)}

#page-database{padding:2rem 2.5rem}
.db-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:1.5rem}
.db-title{font-size:1.2rem;font-weight:700}
.btn-refresh{background:var(--surface);border:1px solid var(--border2);color:var(--text);padding:0.5rem 1rem;border-radius:8px;font-family:var(--font);font-size:0.82rem;cursor:pointer;font-weight:600}
.btn-refresh:hover{border-color:var(--accent);color:var(--accent)}
.db-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1.75rem}
.stat-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1.1rem 1.25rem}
.stat-val{font-size:1.8rem;font-weight:700;margin-bottom:0.2rem}
.stat-label{font-size:0.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.07em}
.stat-card.s-total .stat-val{color:var(--accent)}
.stat-card.s-dr .stat-val{color:var(--danger)}
.stat-card.s-glaucoma .stat-val{color:var(--warning)}
.stat-card.s-cataract .stat-val{color:var(--accent2)}
.db-table-wrap{background:var(--surface);border:1px solid var(--border);border-radius:14px;overflow:hidden}
.db-table{width:100%;border-collapse:collapse}
.db-table th{padding:0.75rem 1rem;text-align:left;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:var(--muted);border-bottom:1px solid var(--border)}
.db-table td{padding:0.8rem 1rem;font-size:0.85rem;border-bottom:1px solid var(--border)}
.db-table tr:last-child td{border-bottom:none}
.db-table tr:hover td{background:rgba(255,255,255,0.02)}
.diag-badge{display:inline-flex;align-items:center;font-size:0.72rem;font-weight:700;padding:0.25rem 0.65rem;border-radius:999px;text-transform:capitalize}
.d-cataract{background:rgba(124,111,255,0.15);color:var(--accent2);border:1px solid rgba(124,111,255,0.3)}
.d-diabetic_retinopathy{background:rgba(255,95,109,0.15);color:var(--danger);border:1px solid rgba(255,95,109,0.3)}
.d-glaucoma{background:rgba(255,181,71,0.15);color:var(--warning);border:1px solid rgba(255,181,71,0.3)}
.d-normal{background:rgba(0,229,153,0.15);color:var(--success);border:1px solid rgba(0,229,153,0.3)}
.src-tag{font-size:0.68rem;padding:0.15rem 0.45rem;border-radius:999px;font-weight:600}
.src-upload{background:rgba(0,212,255,0.1);color:var(--accent);border:1px solid rgba(0,212,255,0.2)}
.src-esp32{background:rgba(0,255,179,0.1);color:var(--accent3);border:1px solid rgba(0,255,179,0.2)}
.btn-del{background:rgba(255,95,109,0.1);border:1px solid rgba(255,95,109,0.25);color:var(--danger);padding:0.3rem 0.65rem;border-radius:6px;font-family:var(--font);font-size:0.75rem;cursor:pointer;font-weight:600}
.btn-del:hover{background:rgba(255,95,109,0.2)}
.db-empty{padding:3rem;text-align:center;color:var(--muted);font-size:0.9rem}
footer{position:relative;z-index:1;text-align:center;padding:1.25rem;font-size:0.75rem;color:var(--muted);border-top:1px solid var(--border)}

@media(max-width:900px){
  .analyze-grid{grid-template-columns:1fr}
  .pipeline-steps{grid-template-columns:repeat(3,1fr)}
  .diag-row{grid-template-columns:1fr}
  .db-stats{grid-template-columns:1fr 1fr}
}
</style>
</head>
<body>

<div class="modal" id="img-modal" onclick="closeModal()">
  <img id="modal-img" src="" alt=""/>
  <div class="modal-label" id="modal-label"></div>
  <div class="modal-close">Click anywhere to close</div>
</div>

<header>
  <div class="logo">
    <div class="logo-eye"></div>
    <div class="logo-text">Ocular<em>AI</em></div>
  </div>
  <div class="nav-tabs">
    <button class="nav-tab active" onclick="showPage('analyze',event)">Analyze</button>
    <button class="nav-tab" onclick="showPage('database',event)">Database</button>
  </div>
  <span class="version-badge">CAD v2.0 · 89% Acc</span>
</header>

<!-- ═══════ ANALYZE PAGE ═══════ -->
<div class="page active" id="page-analyze">
<div class="analyze-grid">

  <div class="input-panel">
    <div>
      <div class="panel-label">Patient Details</div>
      <div class="form-row" style="margin-top:0.75rem">
        <div class="form-group full">
          <label for="p-name">Full Name</label>
          <input type="text" id="p-name" placeholder="e.g. Rahul Sharma"/>
        </div>
        <div class="form-group">
          <label for="p-age">Age</label>
          <input type="number" id="p-age" placeholder="35" min="1" max="120"/>
        </div>
        <div class="form-group">
          <label for="p-gender">Gender</label>
          <select id="p-gender">
            <option value="" disabled selected>Select</option>
            <option>Male</option><option>Female</option><option>Other</option>
          </select>
        </div>
      </div>
    </div>

    <div class="sep"></div>

    <div class="model-badge">
      <div class="dot"></div>
      <div>Model: <strong>DenseNet121</strong> &nbsp;·&nbsp; Accuracy: <strong>89%</strong> &nbsp;·&nbsp; Input: 256×256</div>
    </div>

    <div>
      <div class="panel-label">Image Source</div>
      <div class="source-toggle" style="margin-top:0.6rem">
        <button class="src-btn active" id="btn-src-upload" onclick="switchSource('upload')">
          &#x1F4C2; Upload Image
        </button>
        <button class="src-btn" id="btn-src-esp32" onclick="switchSource('esp32')">
          &#x1F4F7; ESP32-CAM
        </button>
      </div>
    </div>

    <div id="section-upload">
      <div class="panel-label">Upload Fundus Image</div>
      <div class="upload-zone" id="upload-zone" style="margin-top:0.75rem">
        <input type="file" id="file-input" accept="image/*"/>
        <div class="upload-icon">&#x1F52C;</div>
        <div class="upload-label">Drop image or click to browse</div>
        <div class="upload-sub">JPG · PNG · BMP — Retinal / Fundus scans</div>
      </div>
      <div id="preview-wrap">
        <img id="preview-img" src="" alt=""/>
        <div class="preview-label" id="preview-name"></div>
      </div>
    </div>

    <div id="section-esp32" style="display:none">
      <div class="panel-label">ESP32-CAM Live Capture</div>
      <div class="esp32-section" style="margin-top:0.75rem">
        <div class="ip-row">
          <input type="text" id="esp32-ip-input" placeholder="e.g. 192.168.43.100"
                 value="192.168.43.100"/>
          <button onclick="saveIP()">Set IP</button>
        </div>
        <div class="esp32-status">
          <div class="esp32-dot checking" id="esp32-dot"></div>
          <span class="esp32-status-text" id="esp32-status-text">Checking camera...</span>
          <button class="ping-btn" onclick="checkESP32()">Ping</button>
        </div>
        <div class="esp32-preview" id="esp32-preview">
          <img id="esp32-preview-img" src="" alt="ESP32 capture"/>
          <div class="esp32-preview-lbl">
            <div class="live-dot"></div>
            <span>Live capture from ESP32-CAM</span>
          </div>
        </div>
        <button id="esp32-capture-btn" onclick="captureFromESP32()">
          <div class="spin" id="esp32-spin"></div>
          <span id="esp32-btn-lbl">&#x1F4F7; Capture &amp; Analyze</span>
        </button>
      </div>
    </div>

    <div id="section-analyze-btn">
      <button id="submit-btn" disabled>
        <div class="spin" id="spin"></div>
        <span id="btn-lbl">&#x1FA7A; Analyze Image</span>
      </button>
    </div>

    <div class="err-box" id="err-box"></div>
  </div>

  <div class="results-panel" id="results-panel">
    <div class="panel-label">Diagnostic Report</div>
    <div class="placeholder-state" id="ph-state">
      <div class="ph-icon">&#x1F9EC;</div>
      <div class="ph-text">Upload a retinal fundus image or capture from<br>ESP32-CAM, fill patient details, then analyze.<br><br>The system will run the full preprocessing<br>pipeline and show the diagnosis.</div>
    </div>
    <div class="patient-summary" id="ps-card">
      <div class="ps-tag">Patient Information</div>
      <div class="ps-grid">
        <div class="ps-item"><div class="ps-label">Name</div><div class="ps-val" id="ps-name">—</div></div>
        <div class="ps-item"><div class="ps-label">Age</div><div class="ps-val" id="ps-age">—</div></div>
        <div class="ps-item"><div class="ps-label">Gender</div><div class="ps-val" id="ps-gender">—</div></div>
      </div>
    </div>
    <div id="pipeline-section" style="display:none">
      <div class="pipeline-title">Image Processing Pipeline — Click any step to enlarge</div>
      <div class="pipeline-steps" id="pipeline-steps"></div>
    </div>
    <div id="diag-section" style="display:none">
      <div class="diag-row">
        <div class="diag-card highlight">
          <div class="diag-tag">Detected Condition</div>
          <div class="diag-main" id="diag-main">—</div>
          <div class="diag-conf">Confidence: <span class="conf-val" id="diag-conf">—</span></div>
          <div id="risk-pill-wrap"></div>
          <div id="source-badge-wrap"></div>
        </div>
        <div class="diag-card">
          <div class="diag-tag">Class Probabilities</div>
          <div id="prob-bars" style="margin-top:0.5rem"></div>
        </div>
      </div>
    </div>
    <div class="disclaim" id="disclaim">
      <strong>&#x26A0; Medical Disclaimer:</strong> This tool is for research and educational purposes only.
      Results must not replace professional ophthalmological diagnosis. Always consult a qualified doctor.
    </div>
  </div>
</div>
</div>

<!-- ═══════ DATABASE PAGE ═══════ -->
<div class="page" id="page-database">
  <div class="db-header">
    <div class="db-title">Patient Records Database</div>
    <button class="btn-refresh" onclick="loadDB()">&#x27F3; Refresh</button>
  </div>
  <div class="db-stats" id="db-stats"></div>
  <div class="db-table-wrap">
    <table class="db-table">
      <thead>
        <tr>
          <th>#</th><th>Name</th><th>Age</th><th>Gender</th>
          <th>Diagnosis</th><th>Confidence</th><th>Risk</th>
          <th>Source</th><th>Timestamp</th><th>Action</th>
        </tr>
      </thead>
      <tbody id="db-tbody"></tbody>
    </table>
  </div>
</div>

<footer>OcularAI &copy; 2026 &middot; DenseNet121 · 89% Accuracy · AI-Based CAD System for Ocular Disorders</footer>

<script>
const STEPS_META=[
  {label:"Original",   num:"1"},
  {label:"Resized",    num:"2"},
  {label:"CLAHE",      num:"3"},
  {label:"Normalized", num:"4"},
  {label:"Segmented",  num:"5"},
];

let currentSource = 'upload';
let esp32IP = '192.168.43.100';

function showPage(p, e){
  document.querySelectorAll('.page').forEach(el=>el.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(el=>el.classList.remove('active'));
  document.getElementById('page-'+p).classList.add('active');
  if(e) e.target.classList.add('active');
  if(p==='database') loadDB();
}

function switchSource(src){
  currentSource = src;
  const isUpload = src === 'upload';
  document.getElementById('section-upload').style.display   = isUpload ? 'block' : 'none';
  document.getElementById('section-esp32').style.display    = isUpload ? 'none'  : 'block';
  document.getElementById('section-analyze-btn').style.display = isUpload ? 'block' : 'none';
  document.getElementById('btn-src-upload').className = 'src-btn' + (isUpload ? ' active' : '');
  document.getElementById('btn-src-esp32').className  = 'src-btn' + (!isUpload ? ' active-cam' : '');
  clearResults();
  document.getElementById('err-box').style.display = 'none';
  if(!isUpload) checkESP32();
}

const fileInput  = document.getElementById('file-input');
const uploadZone = document.getElementById('upload-zone');
const submitBtn  = document.getElementById('submit-btn');
const errBox     = document.getElementById('err-box');
let selectedFile = null;

uploadZone.addEventListener('dragover',e=>{e.preventDefault();uploadZone.classList.add('drag')});
uploadZone.addEventListener('dragleave',()=>uploadZone.classList.remove('drag'));
uploadZone.addEventListener('drop',e=>{
  e.preventDefault();uploadZone.classList.remove('drag');
  if(e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change',()=>{if(fileInput.files[0]) handleFile(fileInput.files[0])});

function handleFile(f){
  selectedFile = f;
  document.getElementById('preview-img').src = URL.createObjectURL(f);
  document.getElementById('preview-name').textContent = f.name;
  document.getElementById('preview-wrap').style.display = 'block';
  submitBtn.disabled = false;
  clearResults();
  errBox.style.display = 'none';
}

submitBtn.addEventListener('click', async()=>{
  if(!selectedFile) return;
  const name   = document.getElementById('p-name').value.trim()  || 'Not provided';
  const age    = document.getElementById('p-age').value.trim()   || 'Not provided';
  const gender = document.getElementById('p-gender').value       || 'Not provided';
  submitBtn.disabled = true;
  document.getElementById('spin').style.display = 'block';
  document.getElementById('btn-lbl').textContent = 'Processing…';
  errBox.style.display = 'none';
  clearResults();
  const fd = new FormData();
  fd.append('file', selectedFile);
  fd.append('name', name); fd.append('age', age); fd.append('gender', gender);
  try{
    const res  = await fetch('/predict', {method:'POST', body:fd});
    const data = await res.json();
    if(data.error) throw new Error(data.error);
    showResults(data, name, age, gender);
  } catch(e){
    errBox.textContent = '❌ ' + e.message;
    errBox.style.display = 'block';
    document.getElementById('ph-state').style.display = 'flex';
  } finally{
    submitBtn.disabled = false;
    document.getElementById('spin').style.display = 'none';
    document.getElementById('btn-lbl').textContent = '🩺 Analyze Image';
  }
});

function saveIP(){
  const val = document.getElementById('esp32-ip-input').value.trim();
  if(val){ esp32IP = val; checkESP32(); }
}

async function checkESP32(){
  const dot  = document.getElementById('esp32-dot');
  const text = document.getElementById('esp32-status-text');
  dot.className  = 'esp32-dot checking';
  text.innerHTML = 'Checking camera...';
  try{
    const res  = await fetch('/esp32/status');
    const data = await res.json();
    if(data.status === 'online'){
      dot.className  = 'esp32-dot online';
      text.innerHTML = 'Camera online &nbsp;<span>' + data.ip + '</span>';
    } else { throw new Error('offline'); }
  } catch{
    dot.className  = 'esp32-dot offline';
    text.innerHTML = 'Camera offline — check WiFi / IP';
  }
}

async function captureFromESP32(){
  const name   = document.getElementById('p-name').value.trim()  || 'Not provided';
  const age    = document.getElementById('p-age').value.trim()   || 'Not provided';
  const gender = document.getElementById('p-gender').value       || 'Not provided';
  const btn = document.getElementById('esp32-capture-btn');
  btn.disabled = true;
  document.getElementById('esp32-spin').style.display = 'block';
  document.getElementById('esp32-btn-lbl').textContent = 'Capturing...';
  errBox.style.display = 'none';
  clearResults();
  const fd = new FormData();
  fd.append('name', name); fd.append('age', age); fd.append('gender', gender);
  try{
    const res  = await fetch('/esp32/live', {method:'POST', body:fd});
    const data = await res.json();
    if(data.error) throw new Error(data.error);
    if(data.pipeline_images && data.pipeline_images[0]){
      document.getElementById('esp32-preview-img').src = data.pipeline_images[0];
      document.getElementById('esp32-preview').style.display = 'block';
    }
    showResults(data, name, age, gender);
  } catch(e){
    errBox.textContent = '📷 ESP32 Error: ' + e.message;
    errBox.style.display = 'block';
    document.getElementById('ph-state').style.display = 'flex';
  } finally{
    btn.disabled = false;
    document.getElementById('esp32-spin').style.display = 'none';
    document.getElementById('esp32-btn-lbl').textContent = '📷 Capture & Analyze';
  }
}

function clearResults(){
  document.getElementById('ph-state').style.display      = 'flex';
  document.getElementById('ps-card').style.display       = 'none';
  document.getElementById('pipeline-section').style.display = 'none';
  document.getElementById('diag-section').style.display  = 'none';
  document.getElementById('disclaim').style.display      = 'none';
}

function showResults(data, name, age, gender){
  document.getElementById('ph-state').style.display = 'none';
  document.getElementById('ps-name').textContent   = name;
  document.getElementById('ps-age').textContent    = age !== 'Not provided' ? age + ' yrs' : age;
  document.getElementById('ps-gender').textContent = gender;
  document.getElementById('ps-card').style.display = 'block';
  const pipeSteps = document.getElementById('pipeline-steps');
  pipeSteps.innerHTML = '';
  data.pipeline_images.forEach((src,i)=>{
    const meta = STEPS_META[i];
    const div  = document.createElement('div');
    div.className = 'pipe-step' + (i===4 ? ' active-step' : '');
    div.innerHTML = `<img src="${src}" alt="${meta.label}" loading="lazy"/>
      <div class="pipe-step-label"><span class="step-num">${meta.num}</span>${meta.label}</div>`;
    div.onclick = ()=>openModal(src, `Step ${meta.num}: ${meta.label}`);
    pipeSteps.appendChild(div);
  });
  document.getElementById('pipeline-section').style.display = 'block';
  document.getElementById('diag-main').textContent  = data.predicted_class.replace(/_/g,' ');
  document.getElementById('diag-conf').textContent  = data.confidence.toFixed(1) + '%';
  const isNormal = data.predicted_class.toLowerCase() === 'normal';
  const c = data.confidence;
  let cls, lbl;
  if(isNormal)   { cls='risk-normal'; lbl='✅ No Disease Detected'; }
  else if(c>80)  { cls='risk-high';   lbl='🔴 High Risk — Consult Ophthalmologist'; }
  else           { cls='risk-medium'; lbl='🟡 Further Evaluation Recommended'; }
  document.getElementById('risk-pill-wrap').innerHTML = `<span class="risk-pill ${cls}">${lbl}</span>`;
  const src = data.source || 'Upload';
  document.getElementById('source-badge-wrap').innerHTML =
    `<span class="source-badge">&#x1F4F7; Source: ${src}</span>`;
  const bars = document.getElementById('prob-bars');
  bars.innerHTML = '';
  data.all_probabilities.forEach(item=>{
    const d = document.createElement('div');
    d.className = 'prob-item';
    d.innerHTML = `<div class="prob-head"><span>${item.class.replace(/_/g,' ')}</span>
      <span class="prob-pct">${item.probability.toFixed(1)}%</span></div>
      <div class="bar-bg"><div class="bar-fg" data-w="${item.probability}"></div></div>`;
    bars.appendChild(d);
  });
  document.getElementById('diag-section').style.display = 'block';
  document.getElementById('disclaim').style.display     = 'block';
  requestAnimationFrame(()=>{
    document.querySelectorAll('.bar-fg').forEach(b=>b.style.width=b.dataset.w+'%');
  });
}

function openModal(src, label){
  document.getElementById('modal-img').src         = src;
  document.getElementById('modal-label').textContent = label;
  document.getElementById('img-modal').classList.add('open');
}
function closeModal(){ document.getElementById('img-modal').classList.remove('open'); }

async function loadDB(){
  try{
    const res  = await fetch('/db/records');
    const data = await res.json();
    renderDB(data.records);
  } catch(e){ console.error(e); }
}

function renderDB(records){
  const tbody = document.getElementById('db-tbody');
  tbody.innerHTML = '';
  if(!records.length){
    tbody.innerHTML = '<tr><td colspan="10" class="db-empty">No records yet. Analyze some images first.</td></tr>';
    renderStats([]); return;
  }
  records.forEach(r=>{
    const srcClass = (r.source||'').toLowerCase().includes('esp32') ? 'src-esp32' : 'src-upload';
    const srcLabel = (r.source||'Upload');
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${r.id}</td>
      <td><strong>${r.name}</strong></td>
      <td>${r.age}</td>
      <td>${r.gender}</td>
      <td><span class="diag-badge d-${r.diagnosis}">${r.diagnosis.replace(/_/g,' ')}</span></td>
      <td>${r.confidence.toFixed(1)}%</td>
      <td>${r.risk_level}</td>
      <td><span class="src-tag ${srcClass}">${srcLabel}</span></td>
      <td style="color:var(--muted);font-size:0.78rem">${r.timestamp}</td>
      <td><button class="btn-del" onclick="deleteRecord(${r.id})">Delete</button></td>`;
    tbody.appendChild(tr);
  });
  renderStats(records);
}

function renderStats(records){
  const total = records.length;
  const dr    = records.filter(r=>r.diagnosis==='diabetic_retinopathy').length;
  const gl    = records.filter(r=>r.diagnosis==='glaucoma').length;
  const ca    = records.filter(r=>r.diagnosis==='cataract').length;
  document.getElementById('db-stats').innerHTML = `
    <div class="stat-card s-total"><div class="stat-val">${total}</div><div class="stat-label">Total Patients</div></div>
    <div class="stat-card s-dr"><div class="stat-val">${dr}</div><div class="stat-label">Diabetic Retinopathy</div></div>
    <div class="stat-card s-glaucoma"><div class="stat-val">${gl}</div><div class="stat-label">Glaucoma</div></div>
    <div class="stat-card s-cataract"><div class="stat-val">${ca}</div><div class="stat-label">Cataract</div></div>`;
}

async function deleteRecord(id){
  if(!confirm('Delete this record?')) return;
  await fetch('/db/delete/'+id, {method:'DELETE'});
  loadDB();
}
</script>
</body>
</html>"""


@app.route("/")
def index():
    return HTML


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    name   = request.form.get("name",   "Not provided")
    age    = request.form.get("age",    "Not provided")
    gender = request.form.get("gender", "Not provided")
    try:
        result = run_prediction(file.read(), name, age, gender, source="Upload")
        return jsonify(result)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/esp32/live", methods=["POST"])
def esp32_live():
    import requests as req
    name   = request.form.get("name",   "Not provided")
    age    = request.form.get("age",    "Not provided")
    gender = request.form.get("gender", "Not provided")
    try:
        cam = req.get(ESP32_IP, timeout=8)
        if cam.status_code != 200:
            return jsonify({"error": "ESP32-CAM not responding"}), 500
        if len(cam.content) < 1000:
            return jsonify({"error": "Image too small — check camera position"}), 500
        result = run_prediction(cam.content, name, age, gender, source="ESP32-CAM")
        return jsonify(result)
    except req.exceptions.Timeout:
        return jsonify({"error": "ESP32-CAM timed out — check WiFi connection"}), 500
    except req.exceptions.ConnectionError:
        return jsonify({"error": "Cannot reach ESP32-CAM — verify IP address and WiFi"}), 500
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/esp32/status")
def esp32_status():
    import requests as req
    try:
        r = req.get(ESP32_IP, timeout=3)
        return jsonify({"status": "online", "ip": ESP32_IP})
    except:
        return jsonify({"status": "offline", "ip": ESP32_IP})


@app.route("/db/records")
def db_records():
    return jsonify({"records": get_all_records()})


@app.route("/db/delete/<int:record_id>", methods=["DELETE"])
def db_delete(record_id):
    delete_record(record_id)
    return jsonify({"status": "deleted"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
