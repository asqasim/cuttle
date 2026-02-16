import sys
import os
import time
import zipfile
import random
from datetime import datetime

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                               QProgressBar, QTextEdit, QCheckBox, QFrame, 
                               QSplitter, QScrollArea, QColorDialog, QSlider, QComboBox)
from PySide6.QtCore import Qt, QThread, Signal, QSize, QTimer
from PySide6.QtGui import QIcon, QColor, QFont, QPalette, QAction

import qtawesome as qta
import matplotlib
matplotlib.use('Qt5Agg') # Compatible with PySide6
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches
from shapely.geometry import Polygon, box

# --- CONFIGURATION & STYLES ---

class Theme:
    PRIMARY = "#000000"       # Solid Black
    SECONDARY = "#ffffff"     # Solid White
    ACCENT = "#333333"        # Dark Grey
    BG_LIGHT = "#F9F9F9"      # Very light grey for panels
    BORDER = "#E0E0E0"        # Light border
    FONT_FAMILY = "Segoe UI"  # Modern font

    # The "Premium" Stylesheet
    STYLESHEET = f"""
        QMainWindow {{ background-color: {SECONDARY}; }}
        QWidget {{ font-family: '{FONT_FAMILY}'; font-size: 14px; color: {PRIMARY}; }}
        
        /* Modern Black Buttons */
        QPushButton {{
            background-color: {PRIMARY};
            color: {SECONDARY};
            border: none;
            border-radius: 6px;
            padding: 10px 20px;
            font-weight: 600;
        }}
        QPushButton:hover {{ background-color: {ACCENT}; }}
        QPushButton:pressed {{ background-color: #555; }}
        QPushButton:disabled {{ background-color: #CCC; color: #888; }}

        /* Hollow/Secondary Buttons */
        QPushButton.secondary {{
            background-color: transparent;
            color: {PRIMARY};
            border: 2px solid {PRIMARY};
        }}
        QPushButton.secondary:hover {{ background-color: {BG_LIGHT}; }}

        /* Panels */
        QFrame.panel {{
            background-color: {BG_LIGHT};
            border-right: 1px solid {BORDER};
            border-left: 1px solid {BORDER};
        }}

        /* Inputs */
        QLineEdit, QTextEdit {{
            border: 1px solid {BORDER};
            border-radius: 4px;
            background-color: {SECONDARY};
            padding: 5px;
        }}

        /* Custom Checkbox: Black box, White tick */
        QCheckBox {{ spacing: 8px; }}
        QCheckBox::indicator {{
            width: 18px; height: 18px;
            border: 2px solid {PRIMARY};
            border-radius: 3px;
            background: {SECONDARY};
        }}
        QCheckBox::indicator:checked {{
            background: {PRIMARY};
            image: url(data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'><path d='M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z'/></svg>);
        }}

        /* Progress Bar */
        QProgressBar {{
            border: none;
            background-color: {BORDER};
            border-radius: 4px;
            height: 8px;
            text-align: center;
        }}
        QProgressBar::chunk {{
            background-color: {PRIMARY};
            border-radius: 4px;
        }}
        
        /* Sliders */
        QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; }}
        QSlider::handle:horizontal {{
            background: {PRIMARY};
            width: 14px; margin: -5px 0; border-radius: 7px;
        }}
    """

# --- WORKER THREAD FOR AI SIMULATION ---

class AIProcessor(QThread):
    progress_update = Signal(int, str) # percent, log message
    finished = Signal(object) # Returns the 'path' to the shapefile

    def __init__(self, use_full_image):
        super().__init__()
        self.use_full_image = use_full_image

    def run(self):
        steps = [
            "Initializing GIS Engine...",
            "Loading Raster Data...",
            "Preprocessing Image (Contrast Normalization)...",
            "Loading Neural Network (DroneNet-V4)...",
            "Inference: Scanning Sectors...",
            "Vectorizing Segmentation Masks...",
            "Simplifying Geometry...",
            "Writing Shapefile Attributes...",
            "Finalizing Output..."
        ]
        
        for i, step in enumerate(steps):
            time.sleep(random.uniform(0.3, 0.8)) # Simulate work
            percent = int((i / len(steps)) * 100)
            self.progress_update.emit(percent, step)
        
        self.progress_update.emit(100, "Process Complete.")
        
        # Simulate Shapefile Creation if not exists
        os.makedirs("data", exist_ok=True)
        shp_path = "data/fake.shp"
        
        # In a real app, this is where the AI writes the file.
        # We will assume it's created or we create a dummy marker file.
        with open(f"data/generation_log_{int(time.time())}.txt", "w") as f:
            f.write("Generated")
            
        self.finished.emit(shp_path)

# --- CUSTOM WIDGETS ---

class LayerControlWidget(QWidget):
    """Row widget for managing a single layer"""
    style_changed = Signal(str, dict) # layer_id, changes

    def __init__(self, layer_id, name, color, parent=None):
        super().__init__(parent)
        self.layer_id = layer_id
        self.color = color
        self.opacity = 0.5
        self.is_filled = True
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Visibility Checkbox
        self.chk_visible = QCheckBox(name)
        self.chk_visible.setChecked(True)
        self.chk_visible.stateChanged.connect(self.emit_change)
        layout.addWidget(self.chk_visible)
        
        layout.addStretch()
        
        # Color Button
        self.btn_color = QPushButton()
        self.btn_color.setFixedSize(20, 20)
        self.btn_color.setStyleSheet(f"background-color: {self.color}; border: 1px solid #ccc; border-radius: 3px;")
        self.btn_color.clicked.connect(self.pick_color)
        layout.addWidget(self.btn_color)
        
        # Opacity Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        self.slider.setFixedWidth(60)
        self.slider.valueChanged.connect(self.emit_change)
        layout.addWidget(self.slider)

        # Style Toggle (Fill/Border)
        self.btn_style = QPushButton()
        self.btn_style.setIcon(qta.icon('fa5s.square-full', color='black'))
        self.btn_style.setFlat(True)
        self.btn_style.setFixedSize(25, 25)
        self.btn_style.clicked.connect(self.toggle_fill)
        layout.addWidget(self.btn_style)
        
        # Download
        self.btn_dl = QPushButton()
        self.btn_dl.setIcon(qta.icon('fa5s.download', color='black'))
        self.btn_dl.setFlat(True)
        self.btn_dl.setFixedSize(25, 25)
        self.btn_dl.clicked.connect(self.download_zip)
        layout.addWidget(self.btn_dl)

    def pick_color(self):
        c = QColorDialog.getColor(QColor(self.color))
        if c.isValid():
            self.color = c.name()
            self.btn_color.setStyleSheet(f"background-color: {self.color}; border: 1px solid #ccc; border-radius: 3px;")
            self.emit_change()

    def toggle_fill(self):
        self.is_filled = not self.is_filled
        icon = 'fa5s.square-full' if self.is_filled else 'fa5s.border-style'
        self.btn_style.setIcon(qta.icon(icon, color='black'))
        self.emit_change()

    def emit_change(self):
        self.style_changed.emit(self.layer_id, {
            'visible': self.chk_visible.isChecked(),
            'color': self.color,
            'opacity': self.slider.value() / 100.0,
            'filled': self.is_filled
        })

    def download_zip(self):
        # Dummy zip download
        path, _ = QFileDialog.getSaveFileName(self, "Save Layer", f"{self.layer_id}.zip", "Zip Files (*.zip)")
        if path:
            with zipfile.ZipFile(path, 'w') as zf:
                zf.writestr(f"{self.layer_id}.txt", "Dummy Shapefile Data")
            print(f"Downloaded {path}")

# --- MAIN APPLICATION ---

class DroneGISApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AERO-VISION | Advanced GIS Digitizer")
        self.resize(1200, 800)
        
        # State
        self.current_image_path = None
        self.layers_data = [] # List of dicts: {'id', 'geometry', 'type'}
        
        self.setup_ui()
        self.apply_theme()

    def apply_theme(self):
        self.setStyleSheet(Theme.STYLESHEET)

    def setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0,0,0,0)
        self.main_layout.setSpacing(0)

        # HEADER
        self.header = QFrame()
        self.header.setFixedHeight(60)
        self.header.setStyleSheet(f"background-color: {Theme.SECONDARY}; border-bottom: 1px solid {Theme.BORDER};")
        header_layout = QHBoxLayout(self.header)
        
        title = QLabel("AERO-VISION AI")
        title.setFont(QFont(Theme.FONT_FAMILY, 16, QFont.Bold))
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        # STACKED VIEWS
        self.view_stack = QWidget() # We will simulate stacking by hiding/showing containers
        self.main_layout.addWidget(self.header)
        self.main_layout.addWidget(self.view_stack)
        self.stack_layout = QVBoxLayout(self.view_stack)

        # --- VIEW 1: UPLOAD & ROI ---
        self.view_upload = QWidget()
        self.setup_upload_view()
        self.stack_layout.addWidget(self.view_upload)

        # --- VIEW 2: PROCESSING ---
        self.view_processing = QWidget()
        self.view_processing.hide()
        self.setup_processing_view()
        self.stack_layout.addWidget(self.view_processing)

        # --- VIEW 3: MAP / RESULTS ---
        self.view_map = QWidget()
        self.view_map.hide()
        self.setup_map_view()
        self.stack_layout.addWidget(self.view_map)

    def setup_upload_view(self):
        layout = QVBoxLayout(self.view_upload)
        layout.setAlignment(Qt.AlignCenter)
        
        # Upload Box
        self.upload_box = QFrame()
        self.upload_box.setFixedSize(600, 400)
        self.upload_box.setStyleSheet(f"border: 2px dashed {Theme.BORDER}; border-radius: 10px; background: {Theme.BG_LIGHT};")
        box_layout = QVBoxLayout(self.upload_box)
        
        icon = QLabel()
        icon.setPixmap(qta.icon('fa5s.cloud-upload-alt', color=Theme.PRIMARY).pixmap(64, 64))
        icon.setAlignment(Qt.AlignCenter)
        
        lbl_instr = QLabel("Drag & Drop Drone Imagery\nor Click to Browse")
        lbl_instr.setAlignment(Qt.AlignCenter)
        lbl_instr.setStyleSheet("color: #666;")
        
        self.btn_browse = QPushButton("Select Image")
        self.btn_browse.setFixedWidth(200)
        self.btn_browse.clicked.connect(self.browse_image)
        
        # ROI Controls (Initially Hidden)
        self.roi_controls = QWidget()
        self.roi_controls.hide()
        roi_layout = QVBoxLayout(self.roi_controls)
        
        self.chk_full_img = QCheckBox("Use Complete File Analysis")
        self.chk_full_img.setChecked(True)
        self.chk_full_img.setStyleSheet("font-weight: bold;")
        
        self.btn_process = QPushButton("Start AI Processing")
        self.btn_process.clicked.connect(self.start_processing)
        
        roi_layout.addWidget(self.chk_full_img)
        roi_layout.addSpacing(10)
        roi_layout.addWidget(self.btn_process)
        
        # Add to Box
        box_layout.addStretch()
        box_layout.addWidget(icon)
        box_layout.addWidget(lbl_instr)
        box_layout.addWidget(self.btn_browse, alignment=Qt.AlignCenter)
        box_layout.addSpacing(20)
        box_layout.addWidget(self.roi_controls, alignment=Qt.AlignCenter)
        box_layout.addStretch()
        
        layout.addWidget(self.upload_box)

    def setup_processing_view(self):
        layout = QVBoxLayout(self.view_processing)
        layout.setAlignment(Qt.AlignCenter)
        
        container = QFrame()
        container.setFixedSize(500, 400)
        v = QVBoxLayout(container)
        
        lbl = QLabel("Processing Imagery")
        lbl.setFont(QFont(Theme.FONT_FAMILY, 14, QFont.Bold))
        
        self.pbar = QProgressBar()
        self.pbar.setValue(0)
        
        self.log_window = QTextEdit()
        self.log_window.setReadOnly(True)
        self.log_window.setStyleSheet("background: #000; color: #0f0; font-family: Consolas; font-size: 11px;")
        
        self.lbl_time = QLabel("Estimated Time: Calculating...")
        
        v.addWidget(lbl, alignment=Qt.AlignCenter)
        v.addWidget(self.pbar)
        v.addWidget(self.lbl_time, alignment=Qt.AlignCenter)
        v.addWidget(self.log_window)
        
        layout.addWidget(container)

    def setup_map_view(self):
        layout = QHBoxLayout(self.view_map)
        
        # -- LEFT PANEL (Layers) --
        self.panel_layers = QFrame()
        self.panel_layers.setFixedWidth(300)
        self.panel_layers.setProperty("class", "panel")
        pl_layout = QVBoxLayout(self.panel_layers)
        
        lbl_layers = QLabel("Layer Management")
        lbl_layers.setFont(QFont(Theme.FONT_FAMILY, 12, QFont.Bold))
        
        self.layer_list_area = QScrollArea()
        self.layer_list_area.setWidgetResizable(True)
        self.layer_list_widget = QWidget()
        self.layer_list_layout = QVBoxLayout(self.layer_list_widget)
        self.layer_list_layout.setAlignment(Qt.AlignTop)
        self.layer_list_area.setWidget(self.layer_list_widget)
        
        self.btn_add_model = QPushButton("Run New Model")
        self.btn_add_model.clicked.connect(self.go_back_to_upload)
        
        pl_layout.addWidget(lbl_layers)
        pl_layout.addWidget(self.layer_list_area)
        pl_layout.addWidget(self.btn_add_model)
        
        # -- CENTER (Map Canvas) --
        self.map_container = QWidget()
        mc_layout = QVBoxLayout(self.map_container)
        mc_layout.setContentsMargins(0,0,0,0)
        
        # Matplotlib Figure
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.figure.patch.set_facecolor('#ffffff')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.axis('off') # Hide axis for clean map look
        
        mc_layout.addWidget(self.canvas)
        
        # -- BOTTOM RIGHT CHIPS (Overlay) --
        # We simulate this by a horizontal layout at the bottom of the map container
        # Note: In a real complex layout, we'd use absolute positioning, 
        # but for simplicity, we add a toolbar row.
        
        tools_layout = QHBoxLayout()
        tools_layout.addStretch()
        
        btn_fit = QPushButton()
        btn_fit.setIcon(qta.icon('fa5s.expand', color='white'))
        btn_fit.setFixedSize(40, 40)
        btn_fit.setToolTip("Fit Screen")
        btn_fit.clicked.connect(self.map_fit)
        
        btn_zin = QPushButton()
        btn_zin.setIcon(qta.icon('fa5s.search-plus', color='white'))
        btn_zin.setFixedSize(40, 40)
        btn_zin.clicked.connect(lambda: self.map_zoom(1.2))
        
        btn_zout = QPushButton()
        btn_zout.setIcon(qta.icon('fa5s.search-minus', color='white'))
        btn_zout.setFixedSize(40, 40)
        btn_zout.clicked.connect(lambda: self.map_zoom(0.8))

        tools_layout.addWidget(btn_zout)
        tools_layout.addWidget(btn_zin)
        tools_layout.addWidget(btn_fit)
        tools_layout.setContentsMargins(10, 10, 20, 20)
        
        # Since we can't easily overlay widgets on Matplotlib widget in standard Layouts without complex stacking,
        # We will append this toolbar below the canvas.
        mc_layout.addLayout(tools_layout)

        layout.addWidget(self.panel_layers)
        layout.addWidget(self.map_container)

    # --- LOGIC ---

    def browse_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Drone Image", "", "Images (*.png *.jpg *.tif)")
        if fname:
            self.current_image_path = fname
            self.btn_browse.setText(os.path.basename(fname))
            self.roi_controls.show()

    def start_processing(self):
        self.view_upload.hide()
        self.view_processing.show()
        self.log_window.clear()
        
        self.worker = AIProcessor(self.chk_full_img.isChecked())
        self.worker.progress_update.connect(self.update_progress)
        self.worker.finished.connect(self.processing_finished)
        self.worker.start()

    def update_progress(self, val, msg):
        self.pbar.setValue(val)
        self.log_window.append(f"> [{datetime.now().strftime('%H:%M:%S')}] {msg}")
        if val < 100:
            rem = (100 - val) * 0.1
            self.lbl_time.setText(f"Estimated Time: {rem:.1f}s")
        else:
            self.lbl_time.setText("Finalizing...")

    def processing_finished(self, shp_path):
        time.sleep(1) # Pause for effect
        self.view_processing.hide()
        self.view_map.show()
        self.generate_result_layers(shp_path)

    def generate_result_layers(self, shp_path):
        # 1. Background Image Layer (only add if not present)
        if not any(l['type'] == 'raster' for l in self.layers_data):
            self.layers_data.append({
                'id': 'raster_base',
                'type': 'raster',
                'path': self.current_image_path,
                'visible': True,
                'opacity': 1.0,
                'name': 'Base Imagery'
            })
            self.add_layer_control('raster_base', 'Base Imagery', '#000000')

        # 2. Vector Layer (Mock Data logic)
        # In reality, you'd read shp_path with geopandas. 
        # Here we create a random polygon to simulate the "AI Result"
        
        # Create a dummy polygon somewhere in the middle of a 100x100 coord system
        # Assuming the image is mapped to 0-100 for demo
        cx, cy = random.randint(30, 70), random.randint(30, 70)
        poly = Polygon([
            (cx, cy), (cx+20, cy+5), (cx+15, cy+25), (cx-5, cy+20), (cx, cy)
        ])
        
        layer_id = f"model_gen_{len(self.layers_data)}"
        color = random.choice(['#FF0000', '#00FF00', '#0000FF', '#FFFF00'])
        
        self.layers_data.append({
            'id': layer_id,
            'type': 'vector',
            'geometry': poly,
            'visible': True,
            'color': color,
            'opacity': 0.5,
            'filled': True,
            'name': f"AI Detection {len(self.layers_data)}"
        })
        
        self.add_layer_control(layer_id, f"AI Detection {len(self.layers_data)}", color)
        self.refresh_map()

    def add_layer_control(self, lid, name, color):
        w = LayerControlWidget(lid, name, color)
        w.style_changed.connect(self.update_layer_style)
        self.layer_list_layout.addWidget(w)

    def update_layer_style(self, lid, styles):
        for l in self.layers_data:
            if l['id'] == lid:
                l.update(styles)
        self.refresh_map()

    def refresh_map(self):
        self.ax.clear()
        self.ax.axis('off')
        
        # 1. Draw Raster (Mocked by just setting limits or loading if real)
        # For this demo, we just draw a box representing the image
        img_layer = next((l for l in self.layers_data if l['type'] == 'raster'), None)
        if img_layer and img_layer['visible']:
            # In real code: img = plt.imread(img_layer['path']) ... imshow
            # Here: just a black border frame
            self.ax.add_patch(patches.Rectangle((0,0), 100, 100, fill=False, edgecolor='black', lw=2))
            self.ax.text(50, 50, "DRONE IMAGE", ha='center', alpha=0.3)
            self.ax.set_xlim(-10, 110)
            self.ax.set_ylim(-10, 110)

        # 2. Draw Vectors
        for l in self.layers_data:
            if l['type'] == 'vector' and l['visible']:
                poly = l['geometry']
                x, y = poly.exterior.xy
                
                fc = l['color'] if l['filled'] else 'none'
                ec = l['color']
                alpha = l['opacity']
                
                patch = patches.Polygon(list(zip(x, y)), facecolor=fc, edgecolor=ec, alpha=alpha, linewidth=2)
                self.ax.add_patch(patch)

        self.canvas.draw()

    def map_fit(self):
        self.ax.set_xlim(-10, 110)
        self.ax.set_ylim(-10, 110)
        self.canvas.draw()

    def map_zoom(self, factor):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        w = (xlim[1] - xlim[0]) * factor
        h = (ylim[1] - ylim[0]) * factor
        cx = (xlim[1] + xlim[0]) / 2
        cy = (ylim[1] + ylim[0]) / 2
        
        self.ax.set_xlim(cx - w/2, cx + w/2)
        self.ax.set_ylim(cy - h/2, cy + h/2)
        self.canvas.draw()

    def go_back_to_upload(self):
        # We don't clear layers_data, so we can stack new models
        self.view_map.hide()
        self.view_upload.show()
        # Reset upload view logic if needed, but keep file selected

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") # Better base for custom styling
    
    window = DroneGISApp()
    window.show()
    
    sys.exit(app.exec())
