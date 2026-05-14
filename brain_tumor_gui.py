"""
Brain Tumor Classification GUI
Models: EfficientNet-B0+SVM  |  EfficientNet-B0 End-to-End  |  ResNet50
"""
import os, json, threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSES  = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_DISPLAY = {
    'glioma':     'Glioma',
    'meningioma': 'Meningioma',
    'notumor':    'No Tumor Detected',
    'pituitary':  'Pituitary Tumor',
}

# ── Palette ──────────────────────────────────────────────────────────────────
BG      = "#0d1117"
SURFACE = "#161b22"
CARD    = "#21262d"
BORDER  = "#30363d"
ACCENT  = "#58a6ff"
PURPLE  = "#bc8cff"
GREEN   = "#3fb950"
ORANGE  = "#e3b341"
RED     = "#f85149"
TEXT    = "#c9d1d9"
DIM     = "#8b949e"
WHITE   = "#ffffff"

FT  = ("Segoe UI", 22, "bold")
FH  = ("Segoe UI", 13, "bold")
FB  = ("Segoe UI", 11)
FS  = ("Segoe UI", 9)
FM  = ("Consolas", 10)

MODEL_META = {
    "EfficientNet-B0 + SVM": {
        "key": "svm",
        "color": ACCENT,
        "desc": "Pre-trained CNN features → SVM classifier",
        "files": ["svm_brain_tumor_classifier.pkl", "feature_scaler.pkl"],
    },
    "EfficientNet-B0 End-to-End": {
        "key": "e2e",
        "color": PURPLE,
        "desc": "Frozen EfficientNet base + trained classifier head",
        "files": ["end_to_end_brain_tumor_model.h5"],
    },
    "ResNet50 Transfer Learning": {
        "key": "resnet",
        "color": GREEN,
        "desc": "Frozen ResNet50 base + trained classifier head",
        "files": ["best_model_resnet.h5"],
    },
}

VIZ_FILES = {
    "Metrics Comparison":      "model_comparison_metrics_3models.png",
    "Confusion Matrices":      "confusion_matrices_comparison_3models.png",
    "ROC Curves":              "roc_curves_comparison_3models.png",
    "EfficientNet Training":   "training_history_e2e.png",
    "ResNet50 Training":       "training_history_resnet.png",
    "SVM Confusion Matrix":    "confusion_matrix_svm.png",
    "E2E Confusion Matrix":    "confusion_matrix_e2e.png",
    "ResNet50 Confusion":      "confusion_matrix_resnet.png",
}

BAR_COLORS = [ACCENT, PURPLE, GREEN, ORANGE]


# ── Model loader (lazy, cached) ───────────────────────────────────────────────
class ModelLoader:
    def __init__(self, on_status):
        self._cache = {}
        self._on_status = on_status

    def _status(self, msg):
        self._on_status(msg)

    def predict(self, key, img_path):
        if key not in self._cache:
            self._cache[key] = self._load(key)
        m = self._cache[key]
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32)

        if m["type"] == "svm":
            from tensorflow.keras.applications.efficientnet import preprocess_input
            proc = preprocess_input(arr.copy())[np.newaxis]
            feats = m["base"].predict(proc, verbose=0)
            scaled = m["scaler"].transform(feats)
            return m["svm"].predict_proba(scaled)[0]

        if m["preprocess"] == "efficientnet":
            from tensorflow.keras.applications.efficientnet import preprocess_input
        else:
            from tensorflow.keras.applications.resnet50 import preprocess_input
        proc = preprocess_input(arr.copy())[np.newaxis]
        return m["model"].predict(proc, verbose=0)[0]

    def _load(self, key):
        if key == "svm":
            return self._load_svm()
        if key == "e2e":
            return self._load_keras("end_to_end_brain_tumor_model.h5", "efficientnet")
        if key == "resnet":
            return self._load_keras("best_model_resnet.h5", "resnet")
        raise ValueError(key)

    def _load_svm(self):
        import joblib
        import tensorflow as tf
        from tensorflow.keras.applications import EfficientNetB0
        self._status("Loading EfficientNet-B0 feature extractor…")
        base = EfficientNetB0(weights="imagenet", include_top=False,
                              pooling="avg", input_shape=(224, 224, 3))
        base.trainable = False
        self._status("Loading SVM classifier and scaler…")
        svm    = joblib.load(os.path.join(BASE_DIR, "svm_brain_tumor_classifier.pkl"))
        scaler = joblib.load(os.path.join(BASE_DIR, "feature_scaler.pkl"))
        return {"type": "svm", "base": base, "svm": svm, "scaler": scaler}

    def _load_keras(self, fname, preprocess):
        import tensorflow as tf
        fpath = os.path.join(BASE_DIR, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Model file not found: {fname}\n"
                "Please run the notebook to train the model first."
            )
        self._status(f"Loading {fname}…")
        model = tf.keras.models.load_model(fpath)
        return {"type": "keras", "model": model, "preprocess": preprocess}


# ── Main application ──────────────────────────────────────────────────────────
class App:
    def __init__(self, root):
        self.root = root
        root.title("Brain Tumor Classification System")
        root.geometry("1300x840")
        root.minsize(1000, 680)
        root.configure(bg=BG)

        self._refs = []          # keep Tkinter image refs alive
        self._img_path = None
        self._busy = False

        self._results = self._load_results()
        self.loader = ModelLoader(on_status=self._set_status)

        self._build()

    # ── Data ─────────────────────────────────────────────────────────────────
    def _load_results(self):
        try:
            with open(os.path.join(BASE_DIR, "model_comparison_results.json")) as f:
                return json.load(f)
        except Exception:
            return {}

    def _set_status(self, msg):
        try:
            self._status_var.set(msg)
            self.root.update_idletasks()
        except Exception:
            pass

    # ── Top-level layout ──────────────────────────────────────────────────────
    def _build(self):
        # Header bar
        bar = tk.Frame(self.root, bg=SURFACE, height=60)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        tk.Label(bar, text="  \U0001f9e0  Brain Tumor Classification System",
                 bg=SURFACE, fg=WHITE, font=FT).pack(side="left", pady=10)
        self._status_var = tk.StringVar(value="Ready")
        tk.Label(bar, textvariable=self._status_var,
                 bg=SURFACE, fg=DIM, font=FS).pack(side="right", padx=16)

        # Notebook
        style = ttk.Style()
        style.theme_use("default")
        style.configure("X.TNotebook", background=BG, borderwidth=0, tabmargins=0)
        style.configure("X.TNotebook.Tab", background=CARD, foreground=DIM,
                        padding=(18, 8), font=FB)
        style.map("X.TNotebook.Tab",
                  background=[("selected", SURFACE)],
                  foreground=[("selected", WHITE)])

        nb = ttk.Notebook(self.root, style="X.TNotebook")
        nb.pack(fill="both", expand=True)

        self._t_predict = tk.Frame(nb, bg=BG)
        self._t_compare = tk.Frame(nb, bg=BG)
        self._t_viz     = tk.Frame(nb, bg=BG)

        nb.add(self._t_predict, text="   Predict   ")
        nb.add(self._t_compare, text="   Model Comparison   ")
        nb.add(self._t_viz,     text="   Visualizations   ")

        self._build_predict()
        self._build_compare()
        self._build_viz()

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1 – PREDICT
    # ═════════════════════════════════════════════════════════════════════════
    def _build_predict(self):
        p = self._t_predict

        # ── Left panel (controls) ─────────────────────────────────────────────
        left = tk.Frame(p, bg=SURFACE, width=400)
        left.pack(side="left", fill="y", padx=(16, 8), pady=16)
        left.pack_propagate(False)

        self._section(left, "Input Image")

        # Image canvas
        self._img_canvas = tk.Canvas(left, bg=CARD, width=360, height=290,
                                     highlightthickness=1, highlightbackground=BORDER)
        self._img_canvas.pack(padx=16, pady=4)
        self._img_canvas.create_text(180, 145,
            text="No image selected\n\nClick  Upload Image  below",
            fill=DIM, font=FB, tags="ph", justify="center")

        self._btn(left, "Upload Image", self._upload, ACCENT).pack(
            fill="x", padx=16, pady=(8, 16))

        tk.Frame(left, bg=BORDER, height=1).pack(fill="x", padx=16)

        self._section(left, "Select Model")
        self._model_var = tk.StringVar(value=list(MODEL_META.keys())[0])
        for name, meta in MODEL_META.items():
            row = tk.Frame(left, bg=CARD)
            row.pack(fill="x", padx=16, pady=3)
            tk.Radiobutton(row, variable=self._model_var, value=name,
                           bg=CARD, fg=TEXT, selectcolor=BG,
                           activebackground=CARD, activeforeground=WHITE,
                           font=FB, text=name).pack(side="left", padx=8, pady=6)
            tk.Label(row, text=meta["desc"], bg=CARD, fg=DIM, font=FS).pack(
                side="left", padx=4)

        self._predict_btn = self._btn(left, "Run Prediction", self._predict, GREEN)
        self._predict_btn.pack(fill="x", padx=16, pady=(16, 8))

        # ── Right panel (results) ─────────────────────────────────────────────
        right = tk.Frame(p, bg=BG)
        right.pack(side="right", fill="both", expand=True, padx=(8, 16), pady=16)

        self._section(right, "Prediction Results", bg=BG)

        self._result_frame = tk.Frame(right, bg=SURFACE)
        self._result_frame.pack(fill="both", expand=True)
        self._empty_result()

    def _empty_result(self):
        for w in self._result_frame.winfo_children():
            w.destroy()
        tk.Label(self._result_frame,
                 text="Upload an image and click  Run Prediction",
                 bg=SURFACE, fg=DIM, font=FB, justify="center").pack(expand=True)

    def _show_result(self, probs, pred_idx):
        for w in self._result_frame.winfo_children():
            w.destroy()
        rf = self._result_frame

        pred_cls = CLASSES[pred_idx]
        is_tumor = pred_cls != "notumor"
        vcolor   = RED if is_tumor else GREEN
        verdict  = "TUMOR DETECTED" if is_tumor else "NO TUMOR DETECTED"

        # Verdict block
        vf = tk.Frame(rf, bg=SURFACE)
        vf.pack(fill="x", padx=24, pady=(24, 8))

        tk.Label(vf, text=verdict, bg=SURFACE, fg=vcolor,
                 font=("Segoe UI", 20, "bold")).pack(anchor="w")
        tk.Label(vf, text=f"Type: {CLASS_DISPLAY[pred_cls]}",
                 bg=SURFACE, fg=TEXT, font=("Segoe UI", 13)).pack(anchor="w", pady=(4, 0))
        tk.Label(vf, text=f"Confidence: {probs[pred_idx]*100:.1f}%",
                 bg=SURFACE, fg=ORANGE, font=("Segoe UI", 12, "bold")).pack(anchor="w")

        tk.Frame(rf, bg=BORDER, height=1).pack(fill="x", padx=24, pady=12)

        # Confidence bars
        tk.Label(rf, text="Confidence per Class", bg=SURFACE, fg=TEXT,
                 font=FH).pack(anchor="w", padx=24, pady=(0, 10))

        for i, (cls, prob) in enumerate(zip(CLASSES, probs)):
            row = tk.Frame(rf, bg=SURFACE)
            row.pack(fill="x", padx=24, pady=5)

            is_pred = (i == pred_idx)
            fg_lbl  = WHITE if is_pred else DIM

            tk.Label(row, text=CLASS_DISPLAY[cls], width=22, anchor="w",
                     bg=SURFACE, fg=fg_lbl, font=FB).pack(side="left")

            # Track + fill
            track = tk.Canvas(row, bg=CARD, height=20, width=300,
                               highlightthickness=1, highlightbackground=BORDER)
            track.pack(side="left", padx=(0, 8))
            fw = int(300 * prob)
            if fw > 0:
                track.create_rectangle(0, 0, fw, 20,
                                       fill=BAR_COLORS[i % 4], outline="")
            tk.Label(row, text=f"{prob*100:.1f}%", width=6, anchor="e",
                     bg=SURFACE, fg=fg_lbl, font=FM).pack(side="left")

        tk.Frame(rf, bg=BORDER, height=1).pack(fill="x", padx=24, pady=14)
        tk.Label(rf,
                 text="For research/educational use only. Not a clinical diagnostic tool.",
                 bg=SURFACE, fg=DIM, font=FS, justify="left").pack(anchor="w", padx=24, pady=(0, 16))

    def _upload(self):
        path = filedialog.askopenfilename(
            title="Select Brain MRI Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All", "*.*")]
        )
        if not path:
            return
        self._img_path = path
        self._preview(path)
        self._empty_result()
        self._set_status(f"Loaded: {os.path.basename(path)}")

    def _preview(self, path):
        self._img_canvas.delete("all")
        img = Image.open(path).convert("RGB")
        img.thumbnail((340, 270))
        photo = ImageTk.PhotoImage(img)
        self._refs.append(photo)
        x = (360 - img.width)  // 2
        y = (290 - img.height) // 2
        self._img_canvas.create_image(x, y, anchor="nw", image=photo)

    def _predict(self):
        if self._busy:
            return
        if not self._img_path:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return

        key  = MODEL_META[self._model_var.get()]["key"]
        self._busy = True
        self._predict_btn.config(state="disabled", text="Running…")
        self._set_status("Running prediction…")

        def _work():
            try:
                probs    = self.loader.predict(key, self._img_path)
                pred_idx = int(np.argmax(probs))
                self.root.after(0, lambda: self._show_result(probs, pred_idx))
                self.root.after(0, lambda: self._set_status(
                    f"Result: {CLASS_DISPLAY[CLASSES[pred_idx]]}  "
                    f"({probs[pred_idx]*100:.1f}% confidence)"))
            except FileNotFoundError as e:
                self.root.after(0, lambda: messagebox.showerror("Model Not Found", str(e)))
                self.root.after(0, lambda: self._set_status("Error – model not found"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
                self.root.after(0, lambda: self._set_status("Prediction failed"))
            finally:
                self._busy = False
                self.root.after(0, lambda: self._predict_btn.config(
                    state="normal", text="Run Prediction"))

        threading.Thread(target=_work, daemon=True).start()

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2 – MODEL COMPARISON
    # ═════════════════════════════════════════════════════════════════════════
    def _build_compare(self):
        p = self._t_compare

        tk.Label(p, text="Model Performance Comparison",
                 bg=BG, fg=WHITE, font=FT).pack(anchor="w", padx=20, pady=(16, 2))
        tk.Label(p, text="Test set — 1,600 brain MRI scans | 4 classes: glioma, meningioma, no tumor, pituitary",
                 bg=BG, fg=DIM, font=FB).pack(anchor="w", padx=20, pady=(0, 10))

        # Scrollable area
        outer = tk.Frame(p, bg=BG)
        outer.pack(fill="both", expand=True, padx=16, pady=4)

        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=BG)

        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self._populate_compare(inner)

    def _populate_compare(self, parent):
        rows = [
            ("model_1_svm",    "Model 1", "EfficientNet-B0 + SVM",        ACCENT),
            ("model_2_e2e",    "Model 2", "EfficientNet-B0 End-to-End",   PURPLE),
            ("model_3_resnet", "Model 3", "ResNet50 Transfer Learning",   GREEN),
        ]
        cols  = ["accuracy", "precision", "f1_score", "recall"]
        heads = ["Accuracy", "Precision", "F1-Score", "Recall"]

        # ── Overall metrics ───────────────────────────────────────────────────
        tk.Label(parent, text="Overall Test Metrics",
                 bg=BG, fg=WHITE, font=FH).pack(anchor="w", padx=4, pady=(8, 6))

        # Find best per column
        best = {}
        for m in cols:
            vals = [self._results.get(k, {}).get(m) for k, *_ in rows
                    if self._results.get(k, {}).get(m) is not None]
            if vals:
                best[m] = max(vals)

        # Header
        hdr = tk.Frame(parent, bg=CARD)
        hdr.pack(fill="x", pady=(0, 2))
        tk.Label(hdr, text="Model", width=36, bg=CARD, fg=DIM,
                 font=FB, anchor="w").grid(row=0, column=0, padx=12, pady=8, sticky="w")
        for j, h in enumerate(heads):
            tk.Label(hdr, text=h, width=12, bg=CARD, fg=DIM,
                     font=FB, anchor="center").grid(row=0, column=j+1, padx=4, pady=8)

        for key, num, name, color in rows:
            rf = tk.Frame(parent, bg=SURFACE)
            rf.pack(fill="x", pady=2)
            tk.Label(rf, text=f"  {num}: {name}", width=36, bg=SURFACE, fg=color,
                     font=FB, anchor="w").grid(row=0, column=0, padx=12, pady=10, sticky="w")
            for j, m in enumerate(cols):
                val = self._results.get(key, {}).get(m)
                if val is not None:
                    is_best = best.get(m) is not None and abs(val - best[m]) < 1e-8
                    tk.Label(rf, text=f"{val:.4f}", width=12,
                             bg=ACCENT if is_best else SURFACE,
                             fg=WHITE if is_best else TEXT,
                             font=FM, anchor="center").grid(row=0, column=j+1, padx=4, pady=10)
                else:
                    tk.Label(rf, text="N/A", width=12, bg=SURFACE, fg=DIM,
                             font=FM, anchor="center").grid(row=0, column=j+1, padx=4, pady=10)

        # ── Per-class F1 ──────────────────────────────────────────────────────
        pcm = self._results.get("per_class_metrics", {})
        if not pcm:
            return

        tk.Label(parent, text="Per-Class F1-Score",
                 bg=BG, fg=WHITE, font=FH).pack(anchor="w", padx=4, pady=(24, 6))

        model_slots = [
            ("svm",   "Model 1 (SVM)", ACCENT),
            ("e2e",   "Model 2 (E2E)", PURPLE),
            ("resnet","Model 3 (ResNet50)", GREEN),
        ]

        # Column headers
        hdr2 = tk.Frame(parent, bg=CARD)
        hdr2.pack(fill="x", pady=(0, 2))
        tk.Label(hdr2, text="Class", width=20, bg=CARD, fg=DIM,
                 font=FB, anchor="w").grid(row=0, column=0, padx=12, pady=8, sticky="w")
        for j, (_, mlabel, _) in enumerate(model_slots):
            tk.Label(hdr2, text=mlabel, width=22, bg=CARD, fg=DIM,
                     font=FS, anchor="center").grid(row=0, column=j+1, padx=4, pady=8)

        for cls in CLASSES:
            crow = tk.Frame(parent, bg=SURFACE)
            crow.pack(fill="x", pady=2)
            tk.Label(crow, text=CLASS_DISPLAY[cls], width=20, bg=SURFACE, fg=TEXT,
                     font=FB, anchor="w").grid(row=0, column=0, padx=12, pady=8, sticky="w")
            for j, (mkey, _, mcolor) in enumerate(model_slots):
                f1 = pcm.get(mkey, {}).get(cls, {}).get("f1")
                cell = tk.Frame(crow, bg=SURFACE)
                cell.grid(row=0, column=j+1, padx=4, pady=8)
                if f1 is not None:
                    track = tk.Canvas(cell, bg=CARD, height=20, width=200, highlightthickness=0)
                    track.pack()
                    fw = int(200 * f1)
                    if fw > 0:
                        track.create_rectangle(0, 0, fw, 20, fill=mcolor, outline="")
                    track.create_text(fw + 6, 10, text=f"{f1:.3f}",
                                      fill=TEXT, font=FS, anchor="w")
                else:
                    tk.Label(cell, text="N/A", bg=SURFACE, fg=DIM, font=FM).pack()

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 3 – VISUALIZATIONS
    # ═════════════════════════════════════════════════════════════════════════
    def _build_viz(self):
        p = self._t_viz

        tk.Label(p, text="Visualizations", bg=BG, fg=WHITE, font=FT).pack(
            anchor="w", padx=20, pady=(16, 2))
        tk.Label(p, text="Click any thumbnail to view full size",
                 bg=BG, fg=DIM, font=FB).pack(anchor="w", padx=20, pady=(0, 10))

        # Scrollable grid
        outer = tk.Frame(p, bg=BG)
        outer.pack(fill="both", expand=True, padx=16, pady=4)

        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        grid_frame = tk.Frame(canvas, bg=BG)

        grid_frame.bind("<Configure>",
                        lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=grid_frame, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        COLS = 4
        for idx, (label, fname) in enumerate(VIZ_FILES.items()):
            fpath = os.path.join(BASE_DIR, fname)
            r, c = divmod(idx, COLS)
            cell = tk.Frame(grid_frame, bg=CARD, cursor="hand2",
                            highlightthickness=1, highlightbackground=BORDER)
            cell.grid(row=r, column=c, padx=8, pady=8, sticky="nsew")
            grid_frame.columnconfigure(c, weight=1)

            if os.path.exists(fpath):
                try:
                    thumb = Image.open(fpath)
                    thumb.thumbnail((230, 155))
                    photo = ImageTk.PhotoImage(thumb)
                    self._refs.append(photo)
                    lbl_img = tk.Label(cell, image=photo, bg=CARD, cursor="hand2")
                    lbl_img.pack(padx=4, pady=(8, 2))
                    for w in (lbl_img, cell):
                        w.bind("<Button-1>",
                               lambda e, fp=fpath, t=label: self._fullscreen(fp, t))
                except Exception:
                    tk.Label(cell, text="[Error]", bg=CARD, fg=RED,
                             font=FS).pack(padx=8, pady=50)
            else:
                tk.Label(cell,
                         text="Not available\n(run notebook first)",
                         bg=CARD, fg=DIM, font=FS, justify="center").pack(
                             padx=8, pady=50)

            tk.Label(cell, text=label, bg=CARD, fg=TEXT,
                     font=FS, wraplength=220).pack(padx=4, pady=(0, 8))

    def _fullscreen(self, path, title):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.configure(bg=BG)
        img = Image.open(path)
        sw = self.root.winfo_screenwidth()  - 80
        sh = self.root.winfo_screenheight() - 80
        img.thumbnail((sw, sh))
        photo = ImageTk.PhotoImage(img)
        win.geometry(f"{img.width}x{img.height}")
        lbl = tk.Label(win, image=photo, bg=BG)
        lbl.image = photo
        lbl.pack()

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _btn(self, parent, text, command, color=ACCENT):
        return tk.Button(parent, text=text, command=command,
                         bg=color, fg=WHITE, activebackground=BG,
                         activeforeground=WHITE, relief="flat",
                         font=FB, cursor="hand2", padx=10, pady=9)

    def _section(self, parent, text, bg=SURFACE):
        tk.Label(parent, text=text, bg=bg, fg=TEXT, font=FH).pack(
            anchor="w", padx=16, pady=(14, 4))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()