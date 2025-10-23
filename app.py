import sys, os, mmap, shutil, time
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QLabel, QComboBox,
    QSpinBox, QPushButton, QSlider, QCheckBox, QLineEdit, QHBoxLayout,
    QVBoxLayout, QScrollArea, QStatusBar
)
from PySide6.QtGui import QImage, QPixmap, QIcon
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QObject

RES_PRESETS = [
    ("1024 x 768", 1024, 768),
    ("1366 x 768", 1366, 768),
    ("1600 x 900", 1600, 900),
    ("1920 x 1080", 1920, 1080),
    ("2560 x 1440", 2560, 1440),
    ("3840 x 2160", 3840, 2160),
    ("Custom", -1, -1),
]

IMAGE_FORMATS = ["RGB", "RGBA", "BGR", "BGRA", "Luma"]
RGBA_ORDERS = ["RGBA", "BGRA"]


def align(n, base):
    return ((n + base - 1) // base) * base


class ScanWorker(QObject):
    progress = Signal(int)
    finished = Signal(object, int, str, float)
    error = Signal(str)

    def __init__(self, path, file_size, w, h, start_off, end_off, step, pitches, orders, alpha_force, aligned_only=False, align_block=4096):
        super().__init__()
        self.path = path
        self.file_size = file_size
        self.w = w
        self.h = h
        self.start_off = start_off
        self.end_off = end_off
        self.step = max(1, int(step))
        self.pitches = pitches
        self.orders = orders
        self.alpha_force = alpha_force
        self.aligned_only = bool(aligned_only)
        self.align_block = max(1, int(align_block))
        self._stop = False
        self._last_emit = 0.0

    def stop(self):
        self._stop = True

    def run(self):
        try:
            f = open(self.path, 'rb')
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception as e:
            self.error.emit(f"无法打开 dump: {e}")
            self.finished.emit(None, 0, 'BGRA', 0.0)
            return
        total_steps = max(1, int((self.end_off - self.start_off) // self.step) + 1)
        done = 0
        best_score = -1.0
        best = (None, None, None)
        rowbytes = self.w * 4
        off = self.start_off
        if self.aligned_only:
            off = ((off + self.align_block - 1) // self.align_block) * self.align_block
        try:
            while off <= self.end_off and not self._stop:
                for pitch in self.pitches:
                    need = pitch * self.h
                    if off < 0 or off + need > self.file_size or pitch < rowbytes:
                        continue
                    try:
                        buf = np.frombuffer(mm, dtype=np.uint8, count=need, offset=off)
                        src_rows = buf.reshape(self.h, pitch)
                        src_pixels = src_rows[:, :rowbytes].reshape(self.h, self.w, 4)
                    except Exception:
                        continue
                    for order in self.orders:
                        if order == 'BGRA':
                            R = src_pixels[..., 2]; G = src_pixels[..., 1]; B = src_pixels[..., 0]
                        else:
                            R = src_pixels[..., 0]; G = src_pixels[..., 1]; B = src_pixels[..., 2]
                        g = ((77 * R.astype(np.uint16) + 150 * G.astype(np.uint16) + 29 * B.astype(np.uint16)) >> 8)
                        H, W = g.shape
                        stride = max(1, max(H, W) // 512)
                        gs = g[::stride, ::stride]
                        dx = np.abs(np.diff(gs, axis=1)).mean() if gs.shape[1] > 1 else 0.0
                        dy = np.abs(np.diff(gs, axis=0)).mean() if gs.shape[0] > 1 else 0.0
                        s = float((dx + dy) / 255.0)
                        if s > best_score:
                            best_score = s
                            best = (off, pitch, order)
                done += 1
                now = time.time()
                if now - self._last_emit > 0.1:
                    self.progress.emit(int(done * 100 / total_steps))
                    self._last_emit = now
                off += self.step
                if self.aligned_only:
                    off = ((off + self.align_block - 1) // self.align_block) * self.align_block
        finally:
            try:
                mm.close(); f.close()
            except:
                pass
        self.finished.emit(best[0], best[1] if best[1] else 0, best[2] if best[2] else 'BGRA', best_score if best_score >= 0 else 0.0)


class BruteforceWorker(QObject):
    progress = Signal(int)
    finished = Signal(int)
    error = Signal(str)

    def __init__(self, path, file_size, w, h, step, pitches, orders, alpha_force, threshold, out_dir, aligned_only=False, align_block=4096, export_limit=1000):
        super().__init__()
        self.path = path
        self.file_size = file_size
        self.w = w
        self.h = h
        self.step = max(1, int(step))
        self.pitches = pitches
        self.orders = orders
        self.alpha_force = alpha_force
        self.threshold = float(threshold)
        self.out_dir = out_dir
        self.aligned_only = bool(aligned_only)
        self.align_block = max(1, int(align_block))
        self.export_limit = int(export_limit)
        self._stop = False
        self._last_emit = 0.0

    def stop(self):
        self._stop = True

    def _score(self, arr):
        g = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)
        H, W = g.shape
        stride = max(1, max(H, W) // 512)
        gs = g[::stride, ::stride]
        dx = np.abs(np.diff(gs, axis=1)).mean() if gs.shape[1] > 1 else 0.0
        dy = np.abs(np.diff(gs, axis=0)).mean() if gs.shape[0] > 1 else 0.0
        return float((dx + dy) / 255.0)

    def run(self):
        try:
            f = open(self.path, 'rb')
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception as e:
            self.error.emit(f"无法打开 dump: {e}")
            self.finished.emit(0)
            return
        try:
            os.makedirs(self.out_dir, exist_ok=True)
        except Exception as e:
            self.error.emit(f"创建输出目录失败: {e}")
            self.finished.emit(0)
            return
        total_steps = max(1, int(self.file_size // max(1, self.step)))
        done = 0
        exported = 0
        tight = np.empty((self.h, self.w, 4), dtype=np.uint8)
        rowbytes = self.w * 4
        off = 0
        if self.aligned_only:
            off = ((off + self.align_block - 1) // self.align_block) * self.align_block
        try:
            while off + rowbytes * self.h <= self.file_size and not self._stop and exported < self.export_limit:
                for pitch in self.pitches:
                    need = pitch * self.h
                    if off < 0 or off + need > self.file_size or pitch < rowbytes:
                        continue
                    try:
                        buf = np.frombuffer(mm, dtype=np.uint8, count=need, offset=off)
                        src_rows = buf.reshape(self.h, pitch)
                        src_pixels = src_rows[:, :rowbytes].reshape(self.h, self.w, 4)
                    except Exception:
                        continue
                    for order in self.orders:
                        if order == 'BGRA':
                            tight[..., 0] = src_pixels[..., 2]
                            tight[..., 1] = src_pixels[..., 1]
                            tight[..., 2] = src_pixels[..., 0]
                            tight[..., 3] = src_pixels[..., 3]
                        else:
                            tight[...] = src_pixels
                        if self.alpha_force:
                            tight[..., 3] = 255
                        s = self._score(tight)
                        if s >= self.threshold:
                            name = f"off{off}_pitch{pitch}_{order}_{self.w}x{self.h}_{s:.3f}.png"
                            out_path = os.path.join(self.out_dir, name)
                            img = QImage(tight.data, self.w, self.h, self.w*4, QImage.Format_RGBA8888)
                            try:
                                img.save(out_path)
                                exported += 1
                            except Exception:
                                pass
                done += 1
                now = time.time()
                if now - self._last_emit > 0.1:
                    self.progress.emit(int(done * 100 / total_steps))
                    self._last_emit = now
                off += self.step
                if self.aligned_only:
                    off = ((off + self.align_block - 1) // self.align_block) * self.align_block
        finally:
            try:
                mm.close(); f.close()
            except:
                pass
        self.finished.emit(exported)


class DumpImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DWM Dump Viewer")
        self.resize(1200, 800)
        self.setAcceptDrops(True)
        try:
            self.setWindowIcon(QIcon(os.path.join(os.getcwd(), "avatar.png")))
        except:
            pass

        self.dump_path = None
        self.file_handle = None
        self.mm = None
        self.file_size = 0
        self.step_bytes = 4096
        self.current_arr = None
        self.update_timer = QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.update_image)
        self.playback_timer = QTimer(self)
        self.playback_timer.setSingleShot(False)
        self.playback_timer.timeout.connect(self.on_playback_tick)
        self.build_ui()

    def build_ui(self):
        open_btn = QPushButton("打开 dump")
        open_btn.clicked.connect(self.open_dump)

        export_btn = QPushButton("导出 PNG")
        export_btn.clicked.connect(self.export_png)
        self.scan_btn = QPushButton("自动扫描")
        self.scan_btn.clicked.connect(self.auto_scan)
        self.cancel_scan_btn = QPushButton("取消扫描")
        self.cancel_scan_btn.setEnabled(False)
        self.cancel_scan_btn.clicked.connect(self.cancel_scan)
        self.brute_btn = QPushButton("一键爆破导出")
        self.brute_btn.clicked.connect(self.bruteforce_export)
        self.sample_btn = QPushButton("均匀采样导出")
        self.sample_btn.clicked.connect(self.sample_export)
        self.cancel_brute_btn = QPushButton("取消爆破")
        self.cancel_brute_btn.setEnabled(False)
        self.cancel_brute_btn.clicked.connect(self.cancel_brute)
        self.cancel_sample_btn = QPushButton("取消采样")
        self.cancel_sample_btn.setEnabled(False)
        self.cancel_sample_btn.clicked.connect(self.cancel_sample)
        self.uninstall_btn = QPushButton("卸载镜像")
        self.uninstall_btn.clicked.connect(self.uninstall_output)

        self.res_combo = QComboBox()
        for name, w, h in RES_PRESETS:
            self.res_combo.addItem(name, (w, h))
        self.res_combo.currentIndexChanged.connect(self.on_res_combo_change)

        self.width_spin = QSpinBox(); self.width_spin.setRange(16, 8192); self.width_spin.setValue(1024)
        self.height_spin = QSpinBox(); self.height_spin.setRange(16, 8192); self.height_spin.setValue(768)
        self.width_spin.valueChanged.connect(self.schedule_update)
        self.width_spin.valueChanged.connect(self.update_slider_range)
        self.height_spin.valueChanged.connect(self.schedule_update)
        self.height_spin.valueChanged.connect(self.update_slider_range)

        self.pitch_spin = QSpinBox(); self.pitch_spin.setRange(64, 1_000_000); self.pitch_spin.setValue(align(1024*4, 256))
        pitch_auto_btn = QPushButton("自动计算 pitch")
        pitch_auto_btn.clicked.connect(self.auto_pitch)
        self.pitch_spin.valueChanged.connect(self.schedule_update)
        self.pitch_spin.valueChanged.connect(self.update_slider_range)

        self.format_combo = QComboBox(); self.format_combo.addItems(IMAGE_FORMATS)
        self.format_combo.setCurrentText("RGBA")
        self.format_combo.currentIndexChanged.connect(self.schedule_update)
        self.format_combo.currentIndexChanged.connect(self.update_slider_range)
        self.alpha_force_chk = QCheckBox("强制 Alpha=255"); self.alpha_force_chk.setChecked(True)
        self.alpha_force_chk.stateChanged.connect(self.schedule_update)

        self.step_spin = QSpinBox(); self.step_spin.setRange(1, 1024*1024); self.step_spin.setValue(self.step_bytes)
        self.step_spin.valueChanged.connect(self.on_step_change)
        self.scan_span_spin = QSpinBox(); self.scan_span_spin.setRange(0, 20*1024*1024); self.scan_span_spin.setValue(524288)
        self.aligned_only_chk = QCheckBox("仅对齐位置"); self.aligned_only_chk.setChecked(True)
        self.aligned_only_chk.stateChanged.connect(self.schedule_update)
        self.export_limit_spin = QSpinBox(); self.export_limit_spin.setRange(1, 10000); self.export_limit_spin.setValue(200)
        self.sample_count_spin = QSpinBox(); self.sample_count_spin.setRange(1, 10000); self.sample_count_spin.setValue(200)

        self.offset_slider = QSlider(Qt.Horizontal); self.offset_slider.setRange(0, 0)
        self.offset_slider.valueChanged.connect(self.on_slider_change)
        self.offset_edit = QLineEdit("0"); self.offset_edit.setPlaceholderText("偏移(字节)")
        self.offset_edit.editingFinished.connect(self.on_offset_edit_commit)
        self.slider_max_spin = QSpinBox(); self.slider_max_spin.setRange(0, 2_000_000_000); self.slider_max_spin.setValue(0)
        self.slider_max_spin.valueChanged.connect(self.update_slider_range)

        self.play_btn = QPushButton("播放")
        self.play_btn.clicked.connect(self.start_playback)
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.stop_playback)
        self.play_interval_spin = QSpinBox(); self.play_interval_spin.setRange(10, 5000); self.play_interval_spin.setValue(100)
        self.play_step_spin = QSpinBox(); self.play_step_spin.setRange(1, 1024); self.play_step_spin.setValue(1)
        self.play_loop_chk = QCheckBox("循环"); self.play_loop_chk.setChecked(True)

        self.fit_chk = QCheckBox("适应窗口"); self.fit_chk.stateChanged.connect(self.apply_fit)

        ctrl_row1 = QHBoxLayout()
        ctrl_row1.addWidget(open_btn)
        ctrl_row1.addWidget(export_btn)
        ctrl_row1.addStretch()

        ctrl_row2 = QHBoxLayout()
        ctrl_row2.addWidget(QLabel("分辨率:"))
        ctrl_row2.addWidget(self.res_combo)
        ctrl_row2.addWidget(QLabel("宽:")); ctrl_row2.addWidget(self.width_spin)
        ctrl_row2.addWidget(QLabel("高:")); ctrl_row2.addWidget(self.height_spin)
        ctrl_row2.addWidget(QLabel("pitch:")); ctrl_row2.addWidget(self.pitch_spin)
        ctrl_row2.addWidget(pitch_auto_btn)
        ctrl_row2.addWidget(QLabel("图像格式:")); ctrl_row2.addWidget(self.format_combo)
        ctrl_row2.addWidget(self.alpha_force_chk)
        self.adv_chk = QCheckBox("显示高级选项")
        self.adv_chk.setChecked(False)
        self.adv_chk.stateChanged.connect(self.toggle_advanced)
        ctrl_row2.addWidget(self.adv_chk)

        ctrl_row3 = QHBoxLayout()
        ctrl_row3.addWidget(QLabel("滑块步长(字节):")); ctrl_row3.addWidget(self.step_spin)
        ctrl_row3.addWidget(QLabel("偏移:")); ctrl_row3.addWidget(self.offset_edit)
        ctrl_row3.addWidget(QLabel("扫描跨度(±字节):")); ctrl_row3.addWidget(self.scan_span_spin)
        ctrl_row3.addWidget(self.aligned_only_chk)
        ctrl_row3.addWidget(QLabel("导出上限:")); ctrl_row3.addWidget(self.export_limit_spin)
        ctrl_row3.addWidget(QLabel("采样张数:")); ctrl_row3.addWidget(self.sample_count_spin)
        ctrl_row3.addWidget(self.fit_chk)
        ctrl_row3.addStretch()
        ctrl_row3.addWidget(self.scan_btn)
        ctrl_row3.addWidget(self.cancel_scan_btn)
        ctrl_row3.addWidget(self.brute_btn)
        ctrl_row3.addWidget(self.sample_btn)
        ctrl_row3.addWidget(self.cancel_brute_btn)
        ctrl_row3.addWidget(self.cancel_sample_btn)
        ctrl_row3.addWidget(self.uninstall_btn)

        ctrl_row4 = QHBoxLayout()
        ctrl_row4.addWidget(QLabel("偏移滑块:"))
        ctrl_row4.addWidget(self.offset_slider)
        ctrl_row4.addWidget(QLabel("滑块上限(字节,0自动):"))
        ctrl_row4.addWidget(self.slider_max_spin)

        ctrl_row4.addWidget(QLabel("自动播放:"))
        ctrl_row4.addWidget(self.play_btn)
        ctrl_row4.addWidget(self.pause_btn)
        ctrl_row4.addWidget(QLabel("间隔(ms):"))
        ctrl_row4.addWidget(self.play_interval_spin)
        ctrl_row4.addWidget(QLabel("步长(块):"))
        ctrl_row4.addWidget(self.play_step_spin)
        ctrl_row4.addWidget(self.play_loop_chk)

        ctrl_panel = QVBoxLayout()
        ctrl_panel.addLayout(ctrl_row1)
        ctrl_panel.addLayout(ctrl_row2)
        self.row3_widget = QWidget(); self.row3_widget.setLayout(ctrl_row3); self.row3_widget.setVisible(False)
        self.row4_widget = QWidget(); self.row4_widget.setLayout(ctrl_row4); self.row4_widget.setVisible(False)

        self.image_label = QLabel("请先打开 dwm.exe.dmp")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("background:#222; color:#aaa")

        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(self.image_label)

        root = QWidget(); layout = QVBoxLayout(root)
        layout.addLayout(ctrl_panel)
        layout.addWidget(scroll)
        layout.addWidget(self.row3_widget)
        layout.addWidget(self.row4_widget)

        self.status = QStatusBar(); self.setStatusBar(self.status)

        self.setCentralWidget(root)

    def selected_format(self):
        try:
            return self.format_combo.currentText()
        except Exception:
            return "RGBA"

    def bytes_per_pixel(self):
        fmt = self.selected_format()
        if fmt in ("RGBA", "BGRA"):
            return 4
        if fmt in ("RGB", "BGR"):
            return 3
        if fmt.lower() == "luma":
            return 1
        return 4

    def toggle_advanced(self, state):
        show = bool(state)
        try:
            self.row3_widget.setVisible(show)
            self.row4_widget.setVisible(show)
        except Exception:
            pass

    def open_dump(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 dwm.exe.dmp", os.getcwd(), "Dump (*.dmp);;All (*.*)")
        if not path:
            return
        self.open_dump_path(path)

    def open_dump_path(self, path: str):
        if not path:
            return
        try:
            if self.mm:
                try: self.mm.close()
                except: pass
            if self.file_handle:
                try: self.file_handle.close()
                except: pass
            self.file_handle = open(path, 'rb')
            self.mm = mmap.mmap(self.file_handle.fileno(), 0, access=mmap.ACCESS_READ)
            self.file_size = os.path.getsize(path)
            self.dump_path = path
            self.status.showMessage(f"已打开: {os.path.basename(path)} 大小 {self.file_size} 字节")
            self.update_slider_range()
            self.schedule_update()
        except Exception as e:
            self.status.showMessage(f"打开失败: {e}")

    def dragEnterEvent(self, event):
        try:
            md = event.mimeData()
            if md and md.hasUrls():
                for url in md.urls():
                    p = url.toLocalFile()
                    if p and p.lower().endswith('.dmp'):
                        event.acceptProposedAction()
                        return
        except Exception:
            pass
        event.ignore()

    def dropEvent(self, event):
        try:
            md = event.mimeData()
            if md and md.hasUrls():
                for url in md.urls():
                    p = url.toLocalFile()
                    if p:
                        if p.lower().endswith('.dmp'):
                            self.open_dump_path(p)
                            event.acceptProposedAction()
                            return
        except Exception:
            pass
        event.ignore()

    def update_slider_range(self):
        if self.step_bytes > 0:
            manual_max = 0
            try:
                manual_max = self.slider_max_spin.value()
            except Exception:
                manual_max = 0
            if manual_max > 0:
                max_off = int(manual_max)
            elif self.file_size > 0:
                w = self.width_spin.value()
                h = self.height_spin.value()
                pitch = self.pitch_spin.value()
                rowbytes = w * self.bytes_per_pixel()
                need = max(rowbytes, pitch) * h
                max_off = max(0, self.file_size - need)
            else:
                max_off = 0
            max_blocks = max_off // self.step_bytes
            self.offset_slider.setRange(0, int(max_blocks))

    def on_step_change(self, v):
        self.step_bytes = int(v)
        self.update_slider_range()

    def on_slider_change(self, value):
        off = int(value) * self.step_bytes
        self.offset_edit.setText(str(off))
        self.schedule_update()

    def on_offset_edit_commit(self):
        try:
            off = int(self.offset_edit.text().strip())
        except:
            off = 0
        v = max(0, off // max(1, self.step_bytes))
        self.offset_slider.blockSignals(True)
        self.offset_slider.setValue(int(v))
        self.offset_slider.blockSignals(False)
        self.schedule_update()

    def on_res_combo_change(self):
        w, h = self.res_combo.currentData()
        if w > 0 and h > 0:
            self.width_spin.setValue(w)
            self.height_spin.setValue(h)
            self.auto_pitch()
            self.update_slider_range()
        self.schedule_update()

    def auto_pitch(self):
        w = self.width_spin.value(); rowbytes = w * self.bytes_per_pixel()
        self.pitch_spin.setValue(align(rowbytes, 256))

    def apply_fit(self):
        self.image_label.setScaledContents(self.fit_chk.isChecked())

    def schedule_update(self):
        self.update_timer.start(50)

    def start_playback(self):
        if self.playback_timer.isActive():
            self.playback_timer.stop()
        self.playback_timer.start(int(self.play_interval_spin.value()))
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)

    def stop_playback(self):
        self.playback_timer.stop()
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)

    def on_playback_tick(self):
        step = max(1, int(self.play_step_spin.value()))
        cur = int(self.offset_slider.value())
        maxv = int(self.offset_slider.maximum())
        nxt = cur + step
        if nxt <= maxv:
            self.offset_slider.setValue(nxt)
        else:
            if self.play_loop_chk.isChecked():
                self.offset_slider.setValue(0)
            else:
                self.stop_playback()

    def export_png(self):
        if not self.current_arr is None:
            out, _ = QFileDialog.getSaveFileName(self, "保存 PNG", os.getcwd(), "PNG (*.png)")
            if out:
                arr = self.current_arr
                try:
                    if arr.dtype == np.uint16 and arr.ndim == 3 and arr.shape[2] == 4:
                        img = QImage(arr.data, arr.shape[1], arr.shape[0], arr.shape[1]*8, QImage.Format_RGBA64)
                    elif arr.dtype == np.uint8:
                        if arr.ndim == 2:
                            img = QImage(arr.data, arr.shape[1], arr.shape[0], arr.shape[1]*1, QImage.Format_Grayscale8)
                        elif arr.ndim == 3 and arr.shape[2] == 4:
                            img = QImage(arr.data, arr.shape[1], arr.shape[0], arr.shape[1]*4, QImage.Format_RGBA8888)
                        elif arr.ndim == 3 and arr.shape[2] == 3:
                            img = QImage(arr.data, arr.shape[1], arr.shape[0], arr.shape[1]*3, QImage.Format_RGB888)
                        else:
                            self.status.showMessage("导出失败：不支持的图像通道数")
                            return
                    else:
                        self.status.showMessage("导出失败：不支持的数据类型")
                        return
                    img.save(out)
                    self.status.showMessage(f"已保存: {out}")
                except Exception as e:
                    self.status.showMessage(f"保存失败: {e}")

    def update_image(self):
        if not self.mm:
            return
        w = self.width_spin.value(); h = self.height_spin.value(); pitch = self.pitch_spin.value()
        fmt = self.selected_format(); bpp = self.bytes_per_pixel()
        rowbytes = w * bpp
        if pitch < rowbytes or w <= 0 or h <= 0:
            self.image_label.setText("参数错误: pitch < width*bpp 或尺寸无效")
            return
        off_text = self.offset_edit.text().strip()
        try:
            off = int(off_text)
        except:
            off = 0
        need = pitch * h
        if off < 0 or off + need > self.file_size:
            self.image_label.setText("读取越界: 请调整偏移/尺寸")
            return
        try:
            buf = np.frombuffer(self.mm, dtype=np.uint8, count=need, offset=off)
            src_rows = buf.reshape(h, pitch)
            if fmt.lower() == "luma":
                src_pixels = src_rows[:, :w]
                if self.current_arr is None or self.current_arr.dtype != np.uint8 or self.current_arr.shape != (h, w):
                    self.current_arr = np.empty((h, w), dtype=np.uint8)
                tight = self.current_arr
                tight[...] = src_pixels
                img = QImage(tight.data, w, h, w*1, QImage.Format_Grayscale8)
                self.image_label.setPixmap(QPixmap.fromImage(img))
                self.image_label.adjustSize()
            elif fmt in ("RGB", "BGR"):
                src_pixels = src_rows[:, :rowbytes].reshape(h, w, 3)
                if self.current_arr is None or self.current_arr.dtype != np.uint8 or self.current_arr.shape != (h, w, 4):
                    self.current_arr = np.empty((h, w, 4), dtype=np.uint8)
                tight = self.current_arr
                if fmt == "BGR":
                    tight[..., 0] = src_pixels[..., 2]
                    tight[..., 1] = src_pixels[..., 1]
                    tight[..., 2] = src_pixels[..., 0]
                else:
                    tight[..., 0] = src_pixels[..., 0]
                    tight[..., 1] = src_pixels[..., 1]
                    tight[..., 2] = src_pixels[..., 2]
                tight[..., 3] = 255
                if self.alpha_force_chk.isChecked():
                    tight[..., 3] = 255
                img = QImage(tight.data, w, h, w*4, QImage.Format_RGBA8888)
                self.image_label.setPixmap(QPixmap.fromImage(img))
                self.image_label.adjustSize()
            elif fmt in ("RGBA", "BGRA"):
                src_pixels = src_rows[:, :rowbytes].reshape(h, w, 4)
                if self.current_arr is None or self.current_arr.dtype != np.uint8 or self.current_arr.shape != (h, w, 4):
                    self.current_arr = np.empty((h, w, 4), dtype=np.uint8)
                tight = self.current_arr
                if fmt == "BGRA":
                    tight[..., 0] = src_pixels[..., 2]
                    tight[..., 1] = src_pixels[..., 1]
                    tight[..., 2] = src_pixels[..., 0]
                    tight[..., 3] = src_pixels[..., 3]
                else:
                    tight[...] = src_pixels
                if self.alpha_force_chk.isChecked():
                    tight[..., 3] = 255
                img = QImage(tight.data, w, h, w*4, QImage.Format_RGBA8888)
                self.image_label.setPixmap(QPixmap.fromImage(img))
                self.image_label.adjustSize()
            else:
                self.image_label.setText(f"不支持的图像格式: {fmt}")
                return
            self.status.showMessage(f"偏移 {off} | {w}x{h} | pitch={pitch} | fmt={fmt}")
        except Exception as e:
            self.image_label.setText(f"渲染失败: {e}")

    def candidate_pitches(self):
        w = self.width_spin.value()
        rowbytes = w * self.bytes_per_pixel()
        candidates = [rowbytes, align(rowbytes, 128), align(rowbytes, 256), align(rowbytes, 512)]
        base = align(rowbytes, 256)
        for delta in (-512, -256, 256, 512):
            v = base + delta
            if v >= rowbytes:
                candidates.append(v)
        return sorted(set(candidates))

    def auto_scan(self):
        if not self.mm:
            self.status.showMessage("请先打开 dump")
            return
        self.scan_btn.setEnabled(False)
        self.cancel_scan_btn.setEnabled(True)
        w = self.width_spin.value(); h = self.height_spin.value()
        step = self.step_spin.value()
        span = self.scan_span_spin.value()
        try:
            base_off = int(self.offset_edit.text().strip())
        except:
            base_off = 0
        if span and base_off > 0:
            start = max(0, base_off - span)
            end = min(self.file_size - 1, base_off + span)
            scan_step = max(1, step)
        else:
            start = 0
            end = self.file_size - 1
            scan_step = max(step, 16 * 1024 * 1024)
        pitches = self.candidate_pitches()
        fmt = self.selected_format()
        if fmt not in ("RGBA", "BGRA"):
            self.scan_btn.setEnabled(True)
            self.cancel_scan_btn.setEnabled(False)
            self.status.showMessage("自动扫描仅支持 RGBA/BGRA 四通道格式，请切换后重试")
            return
        orders = [fmt] + [o for o in RGBA_ORDERS if o != fmt]
        alpha_force = self.alpha_force_chk.isChecked()
        aligned_only = self.aligned_only_chk.isChecked()
        self.scan_thread = QThread(self)
        self.scan_worker = ScanWorker(
            path=self.dump_path,
            file_size=self.file_size,
            w=w,
            h=h,
            start_off=start,
            end_off=end,
            step=scan_step,
            pitches=pitches,
            orders=orders,
            alpha_force=alpha_force,
            aligned_only=aligned_only,
            align_block=4096,
        )
        self.scan_worker.moveToThread(self.scan_thread)
        self.scan_thread.started.connect(self.scan_worker.run)
        self.scan_worker.progress.connect(self.on_scan_progress)
        self.scan_worker.finished.connect(self.on_scan_finished)
        self.scan_worker.error.connect(lambda msg: self.status.showMessage(msg))
        self.scan_thread.setPriority(QThread.LowPriority)
        self.scan_thread.start()

    def on_scan_progress(self, pct):
        self.status.showMessage(f"扫描进度: {pct}%")

    def on_scan_finished(self, best_off, best_pitch, best_order, best_score):
        try:
            self.scan_thread.quit(); self.scan_thread.wait()
        except:
            pass
        self.scan_btn.setEnabled(True)
        self.cancel_scan_btn.setEnabled(False)
        if best_off is None:
            self.status.showMessage("扫描未发现有效候选")
            return
        self.pitch_spin.setValue(best_pitch)
        idx = self.format_combo.findText(best_order)
        if idx >= 0:
            self.format_combo.setCurrentIndex(idx)
        self.offset_edit.setText(str(best_off))
        self.schedule_update()
        self.status.showMessage(f"扫描完成: 偏移={best_off} pitch={best_pitch} order={best_order} score={best_score:.4f}")

    def bruteforce_export(self):
        if not self.mm:
            self.status.showMessage("请先打开 dump")
            return
        self.brute_btn.setEnabled(False)
        self.cancel_brute_btn.setEnabled(True)
        w = self.width_spin.value(); h = self.height_spin.value()
        step = max(1, self.step_spin.value())
        pitches = self.candidate_pitches()
        fmt = self.selected_format()
        if fmt not in ("RGBA", "BGRA"):
            self.brute_btn.setEnabled(True)
            self.cancel_brute_btn.setEnabled(False)
            self.status.showMessage("爆破导出仅支持 RGBA/BGRA 四通道格式，请切换后重试")
            return
        orders = [fmt] + [o for o in RGBA_ORDERS if o != fmt]
        alpha_force = self.alpha_force_chk.isChecked()
        aligned_only = self.aligned_only_chk.isChecked()
        export_limit = self.export_limit_spin.value()
        threshold = 0.10
        out_dir = os.path.join(os.getcwd(), "output")
        self.brute_thread = QThread(self)
        self.brute_worker = BruteforceWorker(
            path=self.dump_path,
            file_size=self.file_size,
            w=w,
            h=h,
            step=step,
            pitches=pitches,
            orders=orders,
            alpha_force=alpha_force,
            threshold=threshold,
            out_dir=out_dir,
            aligned_only=aligned_only,
            align_block=4096,
            export_limit=export_limit,
        )
        self.brute_worker.moveToThread(self.brute_thread)
        self.brute_thread.started.connect(self.brute_worker.run)
        self.brute_worker.progress.connect(lambda p: self.status.showMessage(f"爆破进度: {p}%"))
        self.brute_worker.finished.connect(self.on_brute_finished)
        self.brute_worker.error.connect(lambda msg: self.status.showMessage(msg))
        self.brute_thread.setPriority(QThread.LowPriority)
        self.brute_thread.start()

    def on_brute_finished(self, count):
        try:
            self.brute_thread.quit(); self.brute_thread.wait()
        except:
            pass
        self.brute_btn.setEnabled(True)
        self.cancel_brute_btn.setEnabled(False)
        out_dir = os.path.join(os.getcwd(), "output")
        self.status.showMessage(f"爆破完成，导出 {count} 张到 {out_dir}")

    def cancel_scan(self):
        try:
            if hasattr(self, 'scan_worker') and self.scan_worker:
                self.scan_worker.stop()
        except:
            pass
        self.status.showMessage("已请求取消扫描")

    def cancel_brute(self):
        try:
            if hasattr(self, 'brute_worker') and self.brute_worker:
                self.brute_worker.stop()
        except:
            pass
        self.status.showMessage("已请求取消爆破")

    def sample_export(self):
        if not self.mm:
            self.status.showMessage("请先打开 dump")
            return
        self.sample_btn.setEnabled(False)
        self.cancel_sample_btn.setEnabled(True)
        w = self.width_spin.value(); h = self.height_spin.value()
        pitch = self.pitch_spin.value()
        fmt = self.selected_format()
        if fmt not in ("RGBA", "BGRA"):
            self.sample_btn.setEnabled(True)
            self.cancel_sample_btn.setEnabled(False)
            self.status.showMessage("均匀采样导出仅支持 RGBA/BGRA 四通道格式，请切换后重试")
            return
        alpha_force = self.alpha_force_chk.isChecked()
        count = max(1, self.sample_count_spin.value())
        aligned_only = self.aligned_only_chk.isChecked()

        max_blocks = self.offset_slider.maximum()
        max_offset = int(max_blocks) * int(self.step_bytes)
        need = int(pitch) * int(h)
        safe_max = max(0, min(max_offset, self.file_size - need))

        if count <= 1 or safe_max <= 0:
            step = max(1, int(self.step_bytes))
        else:
            step = max(1, int(safe_max // max(1, count - 1)))

        if aligned_only:
            step = align(step, int(self.step_bytes))

        out_dir = os.path.join(os.getcwd(), "output")
        self.sample_thread = QThread(self)
        self.sample_worker = BruteforceWorker(
            path=self.dump_path,
            file_size=self.file_size,
            w=w,
            h=h,
            step=step,
            pitches=[pitch],
            orders=[fmt],
            alpha_force=alpha_force,
            threshold=-1.0,  # 不做评分过滤，所有采样点均导出
            out_dir=out_dir,
            aligned_only=aligned_only,
            align_block=int(self.step_bytes),
            export_limit=count,
        )
        self.sample_worker.moveToThread(self.sample_thread)
        self.sample_thread.started.connect(self.sample_worker.run)
        self.sample_worker.progress.connect(lambda p: self.status.showMessage(f"采样进度: {p}%"))
        self.sample_worker.finished.connect(self.on_sample_finished)
        self.sample_worker.error.connect(lambda msg: self.status.showMessage(msg))
        self.sample_thread.setPriority(QThread.LowPriority)
        self.sample_thread.start()

    def on_sample_finished(self, count):
        try:
            self.sample_thread.quit(); self.sample_thread.wait()
        except:
            pass
        self.sample_btn.setEnabled(True)
        self.cancel_sample_btn.setEnabled(False)
        out_dir = os.path.join(os.getcwd(), "output")
        self.status.showMessage(f"采样完成，导出 {count} 张到 {out_dir}")

    def cancel_sample(self):
        try:
            if hasattr(self, 'sample_worker') and self.sample_worker:
                self.sample_worker.stop()
        except:
            pass
        self.status.showMessage("已请求取消采样")

    def uninstall_output(self):
        out_dir = os.path.join(os.getcwd(), "output")
        try:
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
                self.status.showMessage("已卸载镜像并删除 output 目录")
            else:
                self.status.showMessage("output 目录不存在，无需卸载")
            self.current_arr = None
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("请先打开 dwm.exe.dmp")
        except Exception as e:
            self.status.showMessage(f"卸载失败: {e}")

    def closeEvent(self, event):
        try:
            out_dir = os.path.join(os.getcwd(), "output")
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
        except Exception:
            pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    try:
        app.setWindowIcon(QIcon(os.path.join(os.getcwd(), "avatar.png")))
    except:
        pass
    win = DumpImageViewer()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()