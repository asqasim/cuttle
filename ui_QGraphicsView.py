import sys
import os
import time
import math
import random
from datetime import datetime

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                               QProgressBar, QTextEdit, QCheckBox, QFrame, 
                               QScrollArea, QColorDialog, QSlider, QGraphicsView, 
                               QGraphicsScene, QGraphicsPixmapItem, QGraphicsPolygonItem,
                               QGraphicsItem, QGraphicsRectItem)
from PySide6.QtCore import Qt, QThread, Signal, QPointF, QRectF
from PySide6.QtGui import QIcon, QColor, QFont, QPen, QBrush, QPolygonF, QPixmap, QPainter, QRadialGradient

import qtawesome as qta
from shapely.geometry import Polygon

# --- CONFIGURATION & THEME ---

class Theme:
    PRIMARY = "#000000"       # Solid Black
    SECONDARY = "#ffffff"     # Solid White
    ACCENT = "#333333"        # Dark Grey
    BG_LIGHT = "#F5F5F5"      # Light grey for panels
    BORDER = "#DDDDDD"        # Crisp borders
    FONT_FAMILY = "Segoe UI"  # Modern clean font

    STYLESHEET = f"""
        QMainWindow {{ background-color: {SECONDARY}; }}
        QWidget {{ font-family: '{FONT_FAMILY}'; font-size: 13px; color: {PRIMARY}; }}
        
        /* Modern Solid Buttons */
        QPushButton.primary {{
            background-color: {PRIMARY};
            color: {SECONDARY};
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        QPushButton.primary:hover {{ background-color: #444; }}
        QPushButton.primary:pressed {{ background-color: #222; }}

        /* Tool Buttons */
        QPushButton.tool {{
            background-color: {SECONDARY};
            border: 1px solid {BORDER};
            border-radius: 4px;
        }}
        QPushButton.tool:hover {{ background-color: {BG_LIGHT}; border-color: {PRIMARY}; }}

        /* Panels */
        QFrame.panel {{
            background-color: {SECONDARY};
            border-right: 1px solid {BORDER};
        }}
        
        /* Inputs */
        QTextEdit {{
            background-color: #111;
            color: #0f0;
            border: 1px solid {BORDER};
            border-radius: 4px;
            font-family: 'Consolas', monospace;
        }}

        /* Progress Bar */
        QProgressBar {{
            border: none;
            background-color: {BG_LIGHT};
            height: 6px;
            border-radius: 3px;
        }}
        QProgressBar::chunk {{
            background-color: {PRIMARY};
            border-radius: 3px;
        }}
        
        /* Checkbox */
        QCheckBox {{ spacing: 8px; font-weight: 500; }}
        QCheckBox::indicator {{
            width: 16px; height: 16px;
            border: 2px solid {PRIMARY};
            background: {SECONDARY};
            border-radius: 2px;
        }}
        QCheckBox::indicator:checked {{
            background: {PRIMARY};
            image: url(data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'><path d='M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z'/></svg>);
        }}
    """

# --- HIGH PERFORMANCE MAP VIEW (The Core Engine) ---

class InfiniteCanvas(QGraphicsView):
    """
    A High-Performance GIS-like view allowing smooth Zoom & Pan.
    Supports infinite scrolling and vector overlays.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform | QPainter.TextAntialiasing)
        self.setDragMode(QGraphicsView.NoDrag) # We handle dragging manually for "Pan" feel
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor("#e5e5e5"))) # Map background color
        self.setFrameShape(QFrame.NoFrame)
        
        # State
        self._panning = False
        self._last_pan_point = QPointF()

    def wheelEvent(self, event):
        """Smooth Zoom Logic"""
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        self.scale(zoom_factor, zoom_factor)

    def mousePressEvent(self, event):
        """Right click to pan, Left to select"""
        if event.button() == Qt.RightButton:
            self._panning = True
            self._last_pan_point = event.position()
            self.setCursor(Qt.ClosedHandCursor)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.position() - self._last_pan_point
            self._last_pan_point = event.position()
            # Scroll scrollbars
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)

# --- WORKER THREAD ---

class AIProcessor(QThread):
    progress = Signal(int, str)
    finished = Signal(object) 

    def run(self):
        logs = [
            "Initializing Tensor Core...",
            "Tiling 4K Raster Image...",
            "Running Inference (YOLOv8-Custom)...",
            "Merging Vector Polygons...",
            "Topology Correction...",
            "Calculating Geo-Spatial Metadata..."
        ]
        for i, log in enumerate(logs):
            time.sleep(random.uniform(0.4, 0.8))
            self.progress.emit(int((i+1)/len(logs)*100), log)
        
        # Simulate result generation
        self.finished.emit("dummy_path")

# --- LAYER CONTROL WIDGET ---

class LayerWidget(QWidget):
    """Controls for a single GIS layer"""
    updated = Signal(str, dict)

    def __init__(self, layer_id, name, color):
        super().__init__()
        self.layer_id = layer_id
        self.color = color
        self.fill_active = True
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5,5,5,5)
        
        self.chk = QCheckBox(name)
        self.chk.setChecked(True)
        self.chk.stateChanged.connect(self.emit_update)
        
        self.btn_color = QPushButton()
        self.btn_color.setFixedSize(16, 16)
        self.btn_color.setStyleSheet(f"background: {color}; border: 1px solid #ccc; border-radius: 3px;")
        self.btn_color.clicked.connect(self.change_color)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(40)
        self.slider.setFixedWidth(60)
        self.slider.valueChanged.connect(self.emit_update)

        self.btn_dl = QPushButton()
        self.btn_dl.setIcon(qta.icon('fa5s.download', color='black'))
        self.btn_dl.setFlat(True)
        self.btn_dl.setFixedSize(24,24)

        layout.addWidget(self.chk)
        layout.addStretch()
        layout.addWidget(self.btn_color)
        layout.addWidget(self.slider)
        layout.addWidget(self.btn_dl)

    def change_color(self):
        c = QColorDialog.getColor(QColor(self.color))
        if c.isValid():
            self.color = c.name()
            self.btn_color.setStyleSheet(f"background: {self.color}; border: 1px solid #ccc;")
            self.emit_update()

    def emit_update(self):
        self.updated.emit(self.layer_id, {
            'visible': self.chk.isChecked(),
            'color': self.color,
            'opacity': self.slider.value() / 100.0
        })

# --- MAIN APP ---

class DroneGISApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AERO-VISION | Professional GIS")
        self.resize(1280, 850)
        self.setStyleSheet(Theme.STYLESHEET)
        
        # Data
        self.layers = {} # Stores graphical items
        self.scene = QGraphicsScene()
        self.scene.setBackgroundBrush(QBrush(QColor("#f0f0f0")))
        
        self.setup_ui()

    def setup_ui(self):
        # Layouts
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0,0,0,0)
        main_layout.setSpacing(0)

        # -- LEFT SIDEBAR (Controls) --
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(320)
        self.sidebar.setProperty("class", "panel")
        sb_layout = QVBoxLayout(self.sidebar)
        sb_layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        lbl_title = QLabel("AERO-VISION")
        lbl_title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        lbl_sub = QLabel("AI Digitization Suite")
        lbl_sub.setStyleSheet("color: #666; font-size: 11px; margin-bottom: 20px;")
        
        # Mode Switcher (Stacked Widget Simulation)
        self.mode_container = QWidget()
        self.mode_layout = QVBoxLayout(self.mode_container)
        self.mode_layout.setContentsMargins(0,0,0,0)
        
        # 1. Upload Mode
        self.widget_upload = QWidget()
        u_layout = QVBoxLayout(self.widget_upload)
        
        self.btn_load = QPushButton("  Load Raster Image")
        self.btn_load.setProperty("class", "primary")
        self.btn_load.setIcon(qta.icon('fa5s.layer-group', color='white'))
        self.btn_load.clicked.connect(self.load_image)
        
        self.chk_roi = QCheckBox("Full Scene Inference")
        self.chk_roi.setChecked(True)
        
        self.btn_run = QPushButton("  Process AI Model")
        self.btn_run.setProperty("class", "primary")
        self.btn_run.setIcon(qta.icon('fa5s.microchip', color='white'))
        self.btn_run.clicked.connect(self.run_ai)
        self.btn_run.setEnabled(False) # Disabled until image loaded
        
        u_layout.addWidget(QLabel("INPUT SOURCE"))
        u_layout.addWidget(self.btn_load)
        u_layout.addSpacing(10)
        u_layout.addWidget(self.chk_roi)
        u_layout.addSpacing(20)
        u_layout.addWidget(self.btn_run)
        u_layout.addStretch()
        
        # 2. Progress Mode
        self.widget_progress = QWidget()
        self.widget_progress.hide()
        p_layout = QVBoxLayout(self.widget_progress)
        
        self.pbar = QProgressBar()
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        
        p_layout.addWidget(QLabel("PROCESSING STATUS"))
        p_layout.addWidget(self.pbar)
        p_layout.addWidget(self.log_box)
        
        # 3. Layers Mode (Result)
        self.widget_layers = QWidget()
        self.widget_layers.hide()
        l_layout = QVBoxLayout(self.widget_layers)
        
        self.layer_scroll = QScrollArea()
        self.layer_scroll.setWidgetResizable(True)
        self.layer_scroll.setFrameShape(QFrame.NoFrame)
        self.layer_container = QWidget()
        self.layer_vbox = QVBoxLayout(self.layer_container)
        self.layer_vbox.setAlignment(Qt.AlignTop)
        self.layer_scroll.setWidget(self.layer_container)
        
        btn_back = QPushButton("New Process")
        btn_back.setProperty("class", "tool")
        btn_back.clicked.connect(self.reset_ui)
        
        l_layout.addWidget(QLabel("LAYER STACK"))
        l_layout.addWidget(self.layer_scroll)
        l_layout.addWidget(btn_back)

        # Add all to sidebar
        self.mode_layout.addWidget(self.widget_upload)
        self.mode_layout.addWidget(self.widget_progress)
        self.mode_layout.addWidget(self.widget_layers)

        sb_layout.addWidget(lbl_title)
        sb_layout.addWidget(lbl_sub)
        sb_layout.addWidget(self.mode_container)
        
        # -- RIGHT SIDE (Map Canvas) --
        self.map_view = InfiniteCanvas()
        self.map_view.setScene(self.scene)
        
        # Overlay Controls on Map (Zoom buttons)
        self.overlay_container = QWidget(self.map_view)
        self.overlay_container.setGeometry(20, 20, 50, 200)
        ov_layout = QVBoxLayout(self.overlay_container)
        
        btn_zin = QPushButton()
        btn_zin.setIcon(qta.icon('fa5s.plus', color='black'))
        btn_zin.setFixedSize(36, 36)
        btn_zin.setStyleSheet("background: white; border: 1px solid #ccc; border-radius: 4px;")
        btn_zin.clicked.connect(lambda: self.map_view.scale(1.2, 1.2))
        
        btn_zout = QPushButton()
        btn_zout.setIcon(qta.icon('fa5s.minus', color='black'))
        btn_zout.setFixedSize(36, 36)
        btn_zout.setStyleSheet("background: white; border: 1px solid #ccc; border-radius: 4px;")
        btn_zout.clicked.connect(lambda: self.map_view.scale(0.8, 0.8))
        
        btn_fit = QPushButton()
        btn_fit.setIcon(qta.icon('fa5s.expand', color='black'))
        btn_fit.setFixedSize(36, 36)
        btn_fit.setStyleSheet("background: white; border: 1px solid #ccc; border-radius: 4px;")
        btn_fit.clicked.connect(self.fit_view)

        ov_layout.addWidget(btn_zin)
        ov_layout.addWidget(btn_zout)
        ov_layout.addWidget(btn_fit)
        
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.map_view, 1) # 1 stretch factor

    # --- LOGIC ---

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.jpg *.png *.tif)")
        if path:
            self.scene.clear()
            self.layers = {}
            
            # Load Pixmap
            pixmap = QPixmap(path)
            self.img_item = QGraphicsPixmapItem(pixmap)
            self.img_item.setZValue(0) # Bottom layer
            self.scene.addItem(self.img_item)
            
            # Center view
            self.scene.setSceneRect(self.img_item.boundingRect())
            self.map_view.fitInView(self.img_item, Qt.KeepAspectRatio)
            
            self.btn_run.setEnabled(True)
            self.btn_load.setText("Image Loaded")

    def run_ai(self):
        self.widget_upload.hide()
        self.widget_progress.show()
        self.log_box.clear()
        
        self.worker = AIProcessor()
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.ai_finished)
        self.worker.start()

    def update_progress(self, val, text):
        self.pbar.setValue(val)
        self.log_box.append(f"> {text}")

    def ai_finished(self, result):
        self.widget_progress.hide()
        self.widget_layers.show()
        self.generate_dummy_layer()

    def generate_dummy_layer(self):
        """Creates a vector layer on top of the image"""
        layer_id = f"layer_{len(self.layers)}"
        
        # Create a dummy Polygon relative to image size
        if not hasattr(self, 'img_item'): return
        
        rect = self.img_item.boundingRect()
        w, h = rect.width(), rect.height()
        
        # Generate random polygon coordinates
        cx, cy = w/2, h/2
        points = []
        for i in range(5):
            angle = i * (360/5) * (3.14/180)
            r = min(w, h) * 0.2
            px = cx + r * math.cos(angle) + random.randint(-50, 50)
            py = cy + r * math.sin(angle) + random.randint(-50, 50)
            points.append(QPointF(px, py))
            
        qpoly = QPolygonF(points)
        
        # Graphics Item
        poly_item = QGraphicsPolygonItem(qpoly)
        color = QColor(random.choice(["#ff0000", "#00ff00", "#0000ff", "#ff00ff"]))
        
        poly_item.setPen(QPen(color, 2))
        color.setAlpha(100) # Initial transparency
        poly_item.setBrush(QBrush(color))
        poly_item.setZValue(10) # On top
        
        self.scene.addItem(poly_item)
        
        # Store ref
        self.layers[layer_id] = poly_item
        
        # Add Control
        lw = LayerWidget(layer_id, f"AI Detection {len(self.layers)+1}", color.name())
        lw.updated.connect(self.update_layer)
        self.layer_vbox.addWidget(lw)

    def update_layer(self, lid, data):
        item = self.layers.get(lid)
        if item:
            item.setVisible(data['visible'])
            
            c = QColor(data['color'])
            item.setPen(QPen(c, 2))
            
            c.setAlpha(int(data['opacity'] * 255))
            item.setBrush(QBrush(c))

    def fit_view(self):
        if hasattr(self, 'img_item'):
            self.map_view.fitInView(self.img_item, Qt.KeepAspectRatio)

    def reset_ui(self):
        self.widget_layers.hide()
        self.widget_upload.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DroneGISApp()
    window.show()
    sys.exit(app.exec())
