#!/usr/bin/env python3
"""
Interactive Cell Cropper v2
===========================
- High-DPI aware
- Top: 6 representative thumbnails
- Bottom: 3 editing blocks (cell1 / cell2 / cell3)
  Each block: source canvas with crop rect | live crop preview
- Zoom (scroll), pan (right-drag), draw / move / resize rect (left mouse)
- Saves cell1.png, cell2.png, cell3.png per sample folder

Usage:
    python cell_cropper.py [--input_dir PATH]
"""

import os, sys, glob, argparse, platform
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# ── High-DPI awareness ───────────────────────
if platform.system() == "Windows":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def pick_six(paths):
    """First 2, middle 2, last 2."""
    n = len(paths)
    if n <= 6:
        return paths[:]
    first = paths[:2]
    mid = (n - 2) // 2
    middle = paths[mid : mid + 2]
    last = paths[-2:]
    return first + middle + last


# ═══════════════════════════════════════════════
#  CellEditor – one crop‑editing panel
# ═══════════════════════════════════════════════
class CellEditor:
    HANDLE_R = 5

    HANDLE_CURSORS = {
        "nw": "top_left_corner",    "n": "top_side",      "ne": "top_right_corner",
        "w":  "left_side",                                 "e":  "right_side",
        "sw": "bottom_left_corner", "s": "bottom_side",   "se": "bottom_right_corner",
    }

    def __init__(self, parent, title, idx, app):
        self.app = app
        self.idx = idx
        self.title = title

        # --- State ---
        self.pil_image = None
        self.src_path = None
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.crop_orig = None       # (x0,y0,x1,y1) original px, always x0<x1
        self._drag_mode = None      # None | 'new' | 'move' | handle-name
        self._drag_start = None
        self._drag_rect0 = None
        self._pan_start = None
        self._canvas_photo = None
        self._preview_photo = None
        self.active = False
        self._already_saved = False  # True if cell loaded from disk

        # --- UI frame ---
        self.frame = tk.LabelFrame(
            parent, text=f"  {title}  ", font=("Segoe UI", 11, "bold"),
            fg="#cdd6f4", bg="#1e1e2e", labelanchor="n",
            highlightbackground="#45475a", highlightthickness=2,
            padx=4, pady=4,
        )

        self.paned = tk.PanedWindow(
            self.frame, orient=tk.HORIZONTAL, bg="#45475a",
            sashwidth=5, sashrelief="flat", opaqueresize=True,
        )
        self.paned.pack(fill="both", expand=True)

        # Source canvas (left pane)
        src_frame = tk.Frame(self.paned, bg="#181825")
        self.src_canvas = tk.Canvas(src_frame, bg="#181825", highlightthickness=0,
                                    cursor="crosshair")
        self.src_canvas.pack(fill="both", expand=True)
        self.paned.add(src_frame, stretch="always", minsize=120)

        # Preview panel (right pane)
        prev_frame = tk.Frame(self.paned, bg="#1e1e2e")
        prev_frame.columnconfigure(0, weight=1)
        prev_frame.rowconfigure(0, weight=0)
        prev_frame.rowconfigure(1, weight=1)

        tk.Label(prev_frame, text="裁剪预览", font=("Segoe UI", 9),
                 fg="#a6adc8", bg="#1e1e2e").grid(row=0, sticky="ew", pady=(2, 0))
        self.prev_canvas = tk.Canvas(prev_frame, bg="#11111b", highlightthickness=0)
        self.prev_canvas.grid(row=1, sticky="nsew")
        self.paned.add(prev_frame, stretch="always", minsize=120)

        # Info bar
        self.info_var = tk.StringVar(value="点击缩略图加载图片")
        tk.Label(self.frame, textvariable=self.info_var, font=("Consolas", 9),
                 fg="#a6adc8", bg="#1e1e2e").pack(anchor="w")

        # --- Bindings ---
        self.src_canvas.bind("<ButtonPress-1>",   self._on_left_press)
        self.src_canvas.bind("<B1-Motion>",        self._on_left_drag)
        self.src_canvas.bind("<ButtonRelease-1>",  self._on_left_release)
        self.src_canvas.bind("<ButtonPress-3>",    self._on_right_press)
        self.src_canvas.bind("<B3-Motion>",        self._on_right_drag)
        self.src_canvas.bind("<Motion>",           self._on_motion)
        self.src_canvas.bind("<MouseWheel>",       self._on_wheel)
        self.src_canvas.bind("<Button-4>",  lambda e: self._do_zoom(e, 1.15))
        self.src_canvas.bind("<Button-5>",  lambda e: self._do_zoom(e, 1 / 1.15))

        # Click anywhere in editor to activate it
        for w in (self.frame, self.src_canvas, self.prev_canvas):
            w.bind("<ButtonPress-1>", self._activate, add="+")

    # ── Activate / Deactivate ────────────────
    def _activate(self, event=None):
        self.app.set_active_editor(self.idx)

    def init_sash(self):
        """Set initial sash position to ~55% left / 45% right."""
        try:
            w = self.paned.winfo_width()
            if w > 10:
                self.paned.sash_place(0, int(w * 0.55), 0)
        except Exception:
            pass

    def set_active(self, val: bool):
        self.active = val
        color = "#89b4fa" if val else "#45475a"
        thick = 3 if val else 2
        self.frame.config(highlightbackground=color, highlightthickness=thick)

    # ── Load image ───────────────────────────
    def set_image(self, pil_image, path):
        self.pil_image = pil_image
        self.src_path = path
        self.crop_orig = None
        self._already_saved = False
        self._fit_to_view()
        self.render()
        self._update_preview()
        self._update_info()

    def _fit_to_view(self):
        self.src_canvas.update_idletasks()
        cw = max(self.src_canvas.winfo_width(), 100)
        ch = max(self.src_canvas.winfo_height(), 100)
        if self.pil_image is None:
            return
        zw = cw / self.pil_image.width
        zh = ch / self.pil_image.height
        self.zoom = min(zw, zh)
        dw = self.pil_image.width * self.zoom
        dh = self.pil_image.height * self.zoom
        self.pan_x = (cw - dw) / 2
        self.pan_y = (ch - dh) / 2

    # ── Coord transforms ────────────────────
    def _o2d(self, ox, oy):
        return ox * self.zoom + self.pan_x, oy * self.zoom + self.pan_y

    def _d2o(self, dx, dy):
        return (dx - self.pan_x) / self.zoom, (dy - self.pan_y) / self.zoom

    # ── Rendering ────────────────────────────
    def render(self):
        c = self.src_canvas
        c.delete("all")
        if self.pil_image is None:
            c.create_text(
                c.winfo_width() // 2, c.winfo_height() // 2,
                text="无图片", fill="#6c7086", font=("Segoe UI", 12),
            )
            return

        cw = max(c.winfo_width(), 10)
        ch = max(c.winfo_height(), 10)

        # Visible region in orig coords
        ox0, oy0 = self._d2o(0, 0)
        ox1, oy1 = self._d2o(cw, ch)
        ix0 = max(0, int(ox0))
        iy0 = max(0, int(oy0))
        ix1 = min(self.pil_image.width, int(ox1) + 1)
        iy1 = min(self.pil_image.height, int(oy1) + 1)
        if ix1 <= ix0 or iy1 <= iy0:
            return

        crop = self.pil_image.crop((ix0, iy0, ix1, iy1))
        dw = max(1, int((ix1 - ix0) * self.zoom))
        dh = max(1, int((iy1 - iy0) * self.zoom))
        resized = crop.resize((dw, dh), Image.LANCZOS)
        self._canvas_photo = ImageTk.PhotoImage(resized)

        ddx, ddy = self._o2d(ix0, iy0)
        c.create_image(ddx, ddy, anchor="nw", image=self._canvas_photo, tags="img")

        # Crop rectangle
        if self.crop_orig is not None:
            rx0, ry0, rx1, ry1 = self.crop_orig
            dx0, dy0 = self._o2d(rx0, ry0)
            dx1, dy1 = self._o2d(rx1, ry1)

            # Dim overlay (4 rectangles around the selection)
            c.create_rectangle(0, 0, cw, dy0,  fill="#000000", stipple="gray25", outline="")
            c.create_rectangle(0, dy1, cw, ch,  fill="#000000", stipple="gray25", outline="")
            c.create_rectangle(0, dy0, dx0, dy1, fill="#000000", stipple="gray25", outline="")
            c.create_rectangle(dx1, dy0, cw, dy1, fill="#000000", stipple="gray25", outline="")

            # Selection border
            c.create_rectangle(dx0, dy0, dx1, dy1, outline="#f38ba8", width=2, tags="rect")

            # 8 resize handles
            hr = self.HANDLE_R
            for name, (hx, hy) in self._handle_display_pos().items():
                c.create_rectangle(
                    hx - hr, hy - hr, hx + hr, hy + hr,
                    fill="#f38ba8", outline="#cdd6f4", width=1, tags="handle",
                )

    def _handle_display_pos(self):
        if self.crop_orig is None:
            return {}
        x0, y0, x1, y1 = self.crop_orig
        dx0, dy0 = self._o2d(x0, y0)
        dx1, dy1 = self._o2d(x1, y1)
        mx, my = (dx0 + dx1) / 2, (dy0 + dy1) / 2
        return {
            "nw": (dx0, dy0), "n": (mx, dy0), "ne": (dx1, dy0),
            "w":  (dx0, my),                   "e":  (dx1, my),
            "sw": (dx0, dy1), "s": (mx, dy1), "se": (dx1, dy1),
        }

    def _update_preview(self):
        pc = self.prev_canvas
        pc.delete("all")
        if self.pil_image is None or self.crop_orig is None:
            if not self._already_saved:
                pc.create_text(
                    pc.winfo_width() // 2, pc.winfo_height() // 2,
                    text="等待选区…", fill="#6c7086", font=("Segoe UI", 10),
                )
            return

        x0, y0, x1, y1 = [int(v) for v in self.crop_orig]
        x0 = max(0, x0); x1 = min(self.pil_image.width, x1)
        y0 = max(0, y0); y1 = min(self.pil_image.height, y1)
        if x1 <= x0 or y1 <= y0:
            return

        cropped = self.pil_image.crop((x0, y0, x1, y1))
        pc.update_idletasks()
        pw = max(pc.winfo_width(), 50)
        ph = max(pc.winfo_height(), 50)
        s = min(pw / cropped.width, ph / cropped.height)
        dw = max(1, int(cropped.width * s))
        dh = max(1, int(cropped.height * s))
        display = cropped.resize((dw, dh), Image.LANCZOS)
        self._preview_photo = ImageTk.PhotoImage(display)
        pc.create_image(pw // 2, ph // 2, anchor="center", image=self._preview_photo)

    def _update_info(self):
        if self.pil_image is None:
            self.info_var.set("点击缩略图加载图片")
            return
        name = os.path.basename(self.src_path) if self.src_path else "?"
        z = f"{self.zoom * 100:.0f}%"
        if self.crop_orig:
            w = int(self.crop_orig[2] - self.crop_orig[0])
            h = int(self.crop_orig[3] - self.crop_orig[1])
            self.info_var.set(f"源: {name}  |  缩放: {z}  |  选区: {w}×{h} px")
        else:
            self.info_var.set(f"源: {name}  |  缩放: {z}")

    # ── Hit testing ──────────────────────────
    def _hit_handle(self, dx, dy):
        for name, (hx, hy) in self._handle_display_pos().items():
            if abs(dx - hx) <= self.HANDLE_R + 4 and abs(dy - hy) <= self.HANDLE_R + 4:
                return name
        return None

    def _inside_rect(self, dx, dy):
        if self.crop_orig is None:
            return False
        x0, y0, x1, y1 = self.crop_orig
        dx0, dy0 = self._o2d(x0, y0)
        dx1, dy1 = self._o2d(x1, y1)
        return (min(dx0, dx1) <= dx <= max(dx0, dx1) and
                min(dy0, dy1) <= dy <= max(dy0, dy1))

    # ── Cursor feedback ──────────────────────
    def _on_motion(self, event):
        if self.pil_image is None:
            return
        h = self._hit_handle(event.x, event.y)
        if h:
            self.src_canvas.config(cursor=self.HANDLE_CURSORS.get(h, "crosshair"))
        elif self._inside_rect(event.x, event.y):
            self.src_canvas.config(cursor="fleur")
        else:
            self.src_canvas.config(cursor="crosshair")

    # ── Left mouse: draw / move / resize ─────
    def _on_left_press(self, event):
        if self.pil_image is None:
            return
        self._drag_start = (event.x, event.y)
        h = self._hit_handle(event.x, event.y)
        if h:
            self._drag_mode = h
            self._drag_rect0 = self.crop_orig
        elif self._inside_rect(event.x, event.y):
            self._drag_mode = "move"
            self._drag_rect0 = self.crop_orig
        else:
            self._drag_mode = "new"
            ox, oy = self._d2o(event.x, event.y)
            self.crop_orig = (ox, oy, ox, oy)

    def _on_left_drag(self, event):
        if self._drag_mode is None or self.pil_image is None:
            return
        sx, sy = self._drag_start

        if self._drag_mode == "new":
            ox0, oy0 = self._d2o(sx, sy)
            ox1, oy1 = self._d2o(event.x, event.y)
            self.crop_orig = (min(ox0, ox1), min(oy0, oy1),
                              max(ox0, ox1), max(oy0, oy1))

        elif self._drag_mode == "move":
            r = self._drag_rect0
            ddx = (event.x - sx) / self.zoom
            ddy = (event.y - sy) / self.zoom
            self.crop_orig = (r[0]+ddx, r[1]+ddy, r[2]+ddx, r[3]+ddy)

        else:  # resize handle
            r = self._drag_rect0
            ox, oy = self._d2o(event.x, event.y)
            x0, y0, x1, y1 = r
            m = self._drag_mode
            if "n" in m: y0 = oy
            if "s" in m: y1 = oy
            if "w" in m: x0 = ox
            if "e" in m: x1 = ox
            self.crop_orig = (min(x0, x1), min(y0, y1),
                              max(x0, x1), max(y0, y1))

        # Clamp to image
        if self.crop_orig:
            x0, y0, x1, y1 = self.crop_orig
            x0 = max(0, x0); y0 = max(0, y0)
            x1 = min(self.pil_image.width, x1)
            y1 = min(self.pil_image.height, y1)
            self.crop_orig = (x0, y0, x1, y1)

        self.render()
        self._update_preview()
        self._update_info()

    def _on_left_release(self, event):
        if self._drag_mode == "new" and self.crop_orig:
            x0, y0, x1, y1 = self.crop_orig
            if abs(x1 - x0) < 3 or abs(y1 - y0) < 3:
                self.crop_orig = None
                self.render()
                self._update_preview()
                self._update_info()
        self._drag_mode = None

    # ── Right mouse: pan ─────────────────────
    def _on_right_press(self, event):
        self._pan_start = (event.x, event.y, self.pan_x, self.pan_y)

    def _on_right_drag(self, event):
        if self._pan_start is None:
            return
        sx, sy, px, py = self._pan_start
        self.pan_x = px + (event.x - sx)
        self.pan_y = py + (event.y - sy)
        self.render()
        self._update_info()

    # ── Scroll: zoom ─────────────────────────
    def _on_wheel(self, event):
        factor = 1.15 if event.delta > 0 else 1 / 1.15
        self._do_zoom(event, factor)

    def _do_zoom(self, event, factor):
        if self.pil_image is None:
            return
        ox, oy = self._d2o(event.x, event.y)
        self.zoom = max(0.05, min(30.0, self.zoom * factor))
        self.pan_x = event.x - ox * self.zoom
        self.pan_y = event.y - oy * self.zoom
        self.render()
        self._update_info()

    # ── Get crop result ──────────────────────
    def get_crop_image(self):
        if self.pil_image is None or self.crop_orig is None:
            return None
        x0, y0, x1, y1 = [int(v) for v in self.crop_orig]
        x0 = max(0, x0); x1 = min(self.pil_image.width, x1)
        y0 = max(0, y0); y1 = min(self.pil_image.height, y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return self.pil_image.crop((x0, y0, x1, y1))

    # ── Clear / Load existing ────────────────
    def clear(self):
        # Remove canvas items FIRST while photo refs still alive
        self.src_canvas.delete("all")
        self.prev_canvas.delete("all")
        # Now safe to drop references
        self.pil_image = None
        self.src_path = None
        self.crop_orig = None
        self.zoom = 1.0
        self.pan_x = self.pan_y = 0
        self._canvas_photo = None
        self._preview_photo = None
        self._already_saved = False
        self.info_var.set("点击缩略图加载图片")

    def load_existing(self, crop_path):
        """Show already-saved cell in preview."""
        if not os.path.exists(crop_path):
            return
        self._already_saved = True
        img = Image.open(crop_path)
        pc = self.prev_canvas
        pc.update_idletasks()
        pw = max(pc.winfo_width(), 50)
        ph = max(pc.winfo_height(), 50)
        s = min(pw / img.width, ph / img.height)
        dw = max(1, int(img.width * s))
        dh = max(1, int(img.height * s))
        display = img.resize((dw, dh), Image.LANCZOS)
        self._preview_photo = ImageTk.PhotoImage(display)
        pc.delete("all")
        pc.create_image(pw // 2, ph // 2, anchor="center", image=self._preview_photo)
        self.info_var.set(f"✅ 已有: {os.path.basename(crop_path)}  ({img.width}×{img.height})")


# ═══════════════════════════════════════════════
#  Main Application
# ═══════════════════════════════════════════════
class CellCropperApp:
    THUMB_W = 180
    THUMB_H = 150
    CELL_NAMES = ["cell1.png", "cell2.png", "cell3.png"]

    def __init__(self, root, input_dir):
        self.root = root
        self.input_dir = input_dir

        self.sample_dirs = sorted(
            [os.path.join(input_dir, d)
             for d in os.listdir(input_dir)
             if os.path.isdir(os.path.join(input_dir, d))]
        )
        self.current_idx = -1
        self.six_paths = []
        self.six_images = []
        self.thumb_photos = []
        self.active_editor = 0

        self._build_ui()
        self.root.after(250, self._init_and_advance)

    def _init_and_advance(self):
        for ed in self.editors:
            ed.init_sash()
        self.advance()

    def _build_ui(self):
        self.root.title("Cell Cropper v2")
        self.root.configure(bg="#1e1e2e")

        # === Status bar ===
        self.status_var = tk.StringVar(value="正在加载…")
        tk.Label(
            self.root, textvariable=self.status_var,
            font=("Segoe UI", 12, "bold"), fg="#cdd6f4", bg="#313244",
            anchor="w", padx=10, pady=5,
        ).pack(fill="x")

        # === Thumbnail row ===
        thumb_outer = tk.Frame(self.root, bg="#1e1e2e")
        thumb_outer.pack(fill="x", padx=6, pady=(6, 2))

        self.thumb_labels = []
        for i in range(6):
            wrapper = tk.Frame(thumb_outer, bg="#1e1e2e", padx=3)
            wrapper.pack(side="left", expand=True)
            lbl = tk.Label(wrapper, bg="#313244", width=self.THUMB_W,
                           height=self.THUMB_H, cursor="hand2", relief="flat")
            lbl.pack()
            lbl.bind("<Button-1>", lambda e, idx=i: self._on_thumb_click(idx))
            tag = tk.Label(wrapper, text="", font=("Consolas", 8),
                           fg="#a6adc8", bg="#1e1e2e")
            tag.pack()
            self.thumb_labels.append((lbl, tag))

        # === Separator ===
        tk.Frame(self.root, height=2, bg="#585b70").pack(fill="x", padx=8, pady=2)

        # === Hint ===
        tk.Label(
            self.root,
            text="操作: 左键 = 绘制/移动/调整选框  |  右键拖动 = 平移视图  |  滚轮 = 缩放  |  点击缩略图 → 加载到激活的编辑区",
            font=("Segoe UI", 9), fg="#7f849c", bg="#1e1e2e",
        ).pack(fill="x", padx=10)

        # === 3 Cell editor blocks ===
        editor_frame = tk.Frame(self.root, bg="#1e1e2e")
        editor_frame.pack(fill="both", expand=True, padx=6, pady=(2, 4))
        for col in range(3):
            editor_frame.columnconfigure(col, weight=1)
        editor_frame.rowconfigure(0, weight=1)

        self.editors = []
        for i in range(3):
            ed = CellEditor(editor_frame, f"Cell {i+1}", i, self)
            ed.frame.grid(row=0, column=i, sticky="nsew", padx=3, pady=2)
            self.editors.append(ed)
        self.editors[0].set_active(True)

        # === Bottom buttons ===
        btn_frame = tk.Frame(self.root, bg="#1e1e2e")
        btn_frame.pack(fill="x", padx=10, pady=(2, 8))

        self.btn_save = tk.Button(
            btn_frame, text="✅  保存并下一个", font=("Segoe UI", 11, "bold"),
            bg="#a6e3a1", fg="#1e1e2e", activebackground="#94e2d5",
            relief="flat", padx=18, pady=5, command=self._on_save,
        )
        self.btn_save.pack(side="left", padx=(0, 8))

        tk.Button(
            btn_frame, text="⏭  跳过", font=("Segoe UI", 11),
            bg="#f9e2af", fg="#1e1e2e", activebackground="#fab387",
            relief="flat", padx=18, pady=5, command=self.advance,
        ).pack(side="left", padx=(0, 8))

        tk.Button(
            btn_frame, text="🔄  重置全部选框", font=("Segoe UI", 11),
            bg="#89b4fa", fg="#1e1e2e", activebackground="#74c7ec",
            relief="flat", padx=18, pady=5, command=self._reset_all,
        ).pack(side="left")

        self.progress_var = tk.StringVar(value="")
        tk.Label(
            btn_frame, textvariable=self.progress_var,
            font=("Consolas", 11), fg="#a6adc8", bg="#1e1e2e",
        ).pack(side="right")

    # ── Editor activation ────────────────────
    def set_active_editor(self, idx):
        self.active_editor = idx
        for i, ed in enumerate(self.editors):
            ed.set_active(i == idx)

    # ── Thumbnail click ──────────────────────
    def _on_thumb_click(self, idx):
        if idx >= len(self.six_images):
            return
        for i, (lbl, _) in enumerate(self.thumb_labels):
            if i == idx:
                lbl.config(relief="solid", highlightbackground="#f38ba8",
                           highlightthickness=3)
            else:
                lbl.config(relief="flat", highlightthickness=0)
        ed = self.editors[self.active_editor]
        ed.set_image(self.six_images[idx], self.six_paths[idx])

    # ── Navigation ───────────────────────────
    def advance(self):
        while True:
            self.current_idx += 1
            if self.current_idx >= len(self.sample_dirs):
                messagebox.showinfo("完成", "所有样本已处理完毕！🎉")
                self.root.destroy()
                return

            sample_dir = self.sample_dirs[self.current_idx]
            all_done = all(
                os.path.exists(os.path.join(sample_dir, cn))
                for cn in self.CELL_NAMES
            )
            if all_done:
                continue

            pngs = sorted(glob.glob(os.path.join(sample_dir, "*.png")))
            pngs = [p for p in pngs
                     if os.path.basename(p).lower() not in
                     [c.lower() for c in self.CELL_NAMES]]
            if not pngs:
                continue

            self._load_sample(sample_dir, pngs)
            return

    def _load_sample(self, sample_dir, pngs):
        sample_name = os.path.basename(sample_dir)
        self.six_paths = pick_six(pngs)
        self.six_images = [Image.open(p) for p in self.six_paths]

        total = len(self.sample_dirs)
        done = sum(
            1 for d in self.sample_dirs
            if all(os.path.exists(os.path.join(d, cn)) for cn in self.CELL_NAMES)
        )
        self.status_var.set(f"📂  {sample_name}    |    共 {len(pngs)} 张原始图片，代表帧 {len(self.six_paths)} 张")
        self.progress_var.set(f"进度: {done}/{total}")

        # Update thumbnails — build new list BEFORE releasing old refs
        new_photos = []
        for i in range(6):
            lbl, tag = self.thumb_labels[i]
            if i < len(self.six_images):
                img = self.six_images[i].copy()
                img.thumbnail((self.THUMB_W, self.THUMB_H), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                new_photos.append(photo)
                lbl.config(image=photo, relief="flat", highlightthickness=0)
                tag.config(text=os.path.basename(self.six_paths[i]))
            else:
                new_photos.append(None)
                lbl.config(image="", relief="flat", highlightthickness=0)
                tag.config(text="")
        # Now safe to release old references
        self.thumb_photos = new_photos

        # Clear editors; load any existing cell crops
        for i, ed in enumerate(self.editors):
            ed.clear()
            cell_path = os.path.join(sample_dir, self.CELL_NAMES[i])
            if os.path.exists(cell_path):
                self.root.after(300, lambda e=ed, p=cell_path: e.load_existing(p))

        self.set_active_editor(0)

    # ── Save ─────────────────────────────────
    def _on_save(self):
        sample_dir = self.sample_dirs[self.current_idx]
        saved = []
        for i, ed in enumerate(self.editors):
            crop_img = ed.get_crop_image()
            if crop_img is not None:
                path = os.path.join(sample_dir, self.CELL_NAMES[i])
                crop_img.save(path)
                saved.append(f"{self.CELL_NAMES[i]}  ({crop_img.width}×{crop_img.height})")

        if not saved:
            existing = [cn for cn in self.CELL_NAMES
                        if os.path.exists(os.path.join(sample_dir, cn))]
            if not existing:
                messagebox.showwarning("提示", "请至少在一个编辑区中裁剪细胞区域再保存。")
                return

        msg = "已保存:\n" + "\n".join(saved) if saved else "无新裁剪，跳至下一样本。"
        messagebox.showinfo("保存成功", msg)
        self.advance()

    def _reset_all(self):
        for ed in self.editors:
            if ed.pil_image:
                ed.crop_orig = None
                ed.render()
                ed._update_preview()
                ed._update_info()


# ═══════════════════════════════════════════════
#  Entry
# ═══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Interactive Cell Cropper v2")
    parser.add_argument("--input_dir", type=str, default=r"F:\42858965\1\channel_1\matlabphotos2",
                        help="Root directory with sample sub-folders")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"❌ 目录不存在: {args.input_dir}")
        sys.exit(1)

    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", root.winfo_fpixels("1i") / 72.0)
    except Exception:
        pass

    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    ww = min(1600, sw - 60)
    wh = min(960, sh - 60)
    root.geometry(f"{ww}x{wh}+{(sw-ww)//2}+{(sh-wh)//2}")
    root.minsize(1100, 700)

    CellCropperApp(root, args.input_dir)
    root.mainloop()


if __name__ == "__main__":
    main()