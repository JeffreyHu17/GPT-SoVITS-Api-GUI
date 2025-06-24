import sys
import os
import re
import time
# import psutil
import shutil
import requests
import sounddevice as sd
import soundfile as sf
from datetime import datetime
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox,
                             QSpinBox, QDoubleSpinBox, QCheckBox, QTextEdit,
                             QFileDialog, QGroupBox, QMessageBox, QStatusBar, QInputDialog, QGridLayout,
                             QFormLayout, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl, QFileSystemWatcher, QTimer
from GAG_tools.config_manager import ConfigManager
import numpy as np
import pyqtgraph as pg
from pyqtgraph import PlotWidget
import resources_rc


class WaveformWidget(PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        if parent and hasattr(parent, 'config_manager'):
            self.MAX_WAVE_DISPLAY_POINTS = parent.config_manager.get_value('MAX_WAVE_DISPLAY_POINTS', 2500)
        else:
            self.MAX_WAVE_DISPLAY_POINTS = 2500
        self.setBackground(None)
        self.setStyleSheet("background: transparent;")
        self.setMouseEnabled(x=True, y=False)
        self.setMenuEnabled(False)

        self.plotItem.vb.setMouseEnabled(x=False, y=False)
        self.plotItem.vb.setMenuEnabled(False)
        self.plotItem.setMouseEnabled(x=True, y=False)

        self.plotItem.showAxis('left', False)
        self.hideButtons()
        self.plotItem.getAxis('bottom').setPen(pg.mkPen(color=(100, 100, 100, 60)))
        self.plotItem.getAxis('bottom').setTextPen(pg.mkPen(color=(100, 100, 100, 60)))

        # Waveform
        self.waveform_plot = self.plot(
            pen=pg.mkPen(
                color=(30, 144, 255, 175),
                width=1.2,
                cosmetic=True
            ),
            fillLevel=0,
            antialias=True
        )
        self.waveform_plot.setZValue(10)

        gradient = pg.ColorMap(
            [0, 0.5, 1],
            [
                (30, 144, 255, 80),
                (0, 191, 255, 120),
                (100, 149, 237, 80)
            ]
        ).getBrush(span=(0, 1))
        self.waveform_plot.setBrush(gradient)

        # Selection_region
        self.selection_region = pg.LinearRegionItem(
            brush=pg.mkBrush(30, 144, 255, 80),
            pen=pg.mkPen((30, 144, 255), width=1.5)
        )
        self.selection_region.setZValue(20)
        self.selection_region.hide()
        self.addItem(self.selection_region)

        # Cursor_line
        self.cursor_line = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen(color=(255, 69, 0, 200), width=1.5, cosmetic=True),
            hoverPen=pg.mkPen(color=(255, 69, 0, 255), width=2, cosmetic=True)
        )
        self.cursor_line.setZValue(30)
        self.addItem(self.cursor_line)

        # Zero_line
        self.zero_line = pg.InfiniteLine(
            pos=0,
            angle=0,
            pen=pg.mkPen(color=(30, 144, 255, 50), width=0.8, cosmetic=True)
        )
        self.addItem(self.zero_line)

        self.duration = 0
        self.audio_peak = 1.0
        self.selecting = False
        self.selection_start = 0
        self.mouse_pressed = False
        self.auto_scroll = True
        self.show_demo_waveform()

    def _downsample(self, data):
        n = len(data)
        if n <= self.MAX_WAVE_DISPLAY_POINTS:
            return np.linspace(0, self.duration, n), data

        step = max(1, n // self.MAX_WAVE_DISPLAY_POINTS)
        chunks = n // step
        reshaped = data[:chunks * step].reshape(chunks, step)
        avg_vals = reshaped.mean(axis=1)

        x = np.linspace(0, self.duration, chunks)
        y = avg_vals
        return x, y

    def load_audio(self, file_path, reset_view=False):
        try:
            data, sample_rate = sf.read(file_path)
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            self.duration = len(data) / sample_rate

            x, y = self._downsample(data)
            self.waveform_plot.setData(x, y)

            self.plotItem.vb.setLimits(
                xMin=0,
                xMax=self.duration,
                minXRange=min(self.duration, 0.1),
                maxXRange=self.duration
            )

            self.audio_peak = np.max(np.abs(y)) if len(y) > 0 else 1.0
            self.setYRange(-self.audio_peak, self.audio_peak, padding=0.01)

            current_view = self.plotItem.viewRange()[0]
            if reset_view:
                self.setXRange(0, self.duration, padding=0)
            else:
                self.setXRange(current_view[0], current_view[1], padding=0)

            self.cursor_line.setValue(0)
            self.selection_region.hide()
            return True
        except Exception as e:
            print(f"Error loading audio: {str(e)}")
            return False

    def mousePressEvent(self, event):
        pos = self.plotItem.vb.mapSceneToView(event.pos())
        x = max(0, min(pos.x(), self.duration))

        if event.button() == Qt.RightButton:
            event.accept()
            self.mouse_pressed = True
            self.selection_start = x
            self.selection_region.setRegion([x, x])
            self.selection_region.show()
            self.selecting = True
        elif event.button() == Qt.LeftButton:
            super().mousePressEvent(event)
            self.cursor_line.setValue(x)

            if hasattr(self.parent(), 'play_audio_from'):
                self.parent().play_audio_from(x)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.mouse_pressed = False
            region = self.selection_region.getRegion()
            if region[0] == region[1]:
                self.selection_region.hide()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.RightButton:
            pos = self.plotItem.vb.mapSceneToView(event.pos())
            end = max(0, min(pos.x(), self.duration))
            self.selection_region.setRegion([self.selection_start, end])
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def get_selection(self):
        if self.selection_region.isVisible():
            return sorted(self.selection_region.getRegion())
        return None

    def get_cursor_position(self):
        return self.cursor_line.value()

    def update_cursor_during_playback(self, position):
        self.cursor_line.setValue(position)

        if self.auto_scroll:
            view_range = self.plotItem.viewRange()[0]
            visible_width = view_range[1] - view_range[0]
            scroll_margin = visible_width * 0.2  # ~20%

            if position > view_range[1] - scroll_margin:
                new_left = position - visible_width * 0.8
                new_right = new_left + visible_width
                self.setXRange(new_left, new_right, padding=0)

    @staticmethod
    def generate_demo_waveform(duration=2.0, sample_rate=44100):
        t = np.linspace(0, duration, int(duration * sample_rate))
        wave = np.sin(2 * np.pi * 3 * t) * np.sin(np.pi * t / duration)
        return wave * 0.8

    def show_demo_waveform(self):
        self.duration = 2
        x, y = self._downsample(self.generate_demo_waveform(duration=self.duration))
        self.waveform_plot.setData(x, y)
        self.audio_peak = np.max(np.abs(y)) if len(y) > 0 else 1.0
        self.setXRange(0, self.duration, padding=0)
        self.setYRange(-self.audio_peak, self.audio_peak, padding=0.01)
        self.plotItem.vb.setLimits(xMin=0, xMax=self.duration, minXRange=min(self.duration, 0.1), maxXRange=self.duration)
        self.cursor_line.setValue(0)
        self.selection_region.hide()


class ModelSwitchThread(QThread):
    switch_signal = pyqtSignal(bool, str)

    def __init__(self, url, model_type, weights_path):
        super().__init__()
        self.url = url
        self.model_type = model_type  # 'gpt' or 'sovits'
        self.weights_path = weights_path

    def run(self):
        try:
            endpoint = '/set_gpt_weights' if self.model_type == 'gpt' else '/set_sovits_weights'
            response = requests.get(f"{self.url}{endpoint}", params={'weights_path': self.weights_path})

            if response.status_code == 200:
                self.switch_signal.emit(True, self.tr("{} 模型切换成功").format(self.model_type.upper()))
            else:
                error_msg = response.json() if response.headers.get(
                    'content-type') == 'application/json' else response.text
                self.switch_signal.emit(False,
                                        self.tr("{} 模型切换失败:\n {}").format(self.model_type.upper(), error_msg))
        except Exception as e:
            self.switch_signal.emit(False,
                                    self.tr("{} 模型切换时发生错误:\n {}").format(self.model_type.upper(), str(e)))


class TTSThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, url, params, cache_dir):
        super().__init__()
        self.url = url
        self.params = params
        self.temp_file = None
        self.cache_dir = cache_dir

    def run(self):
        try:
            print(str(self.params))
            response = requests.post(f"{self.url}/tts", json=self.params)
            if response.status_code == 200:
                temp_file_path = os.path.join(self.cache_dir, f"tmp_{int(time.time())}.wav")
                with open(temp_file_path, 'wb') as f:
                    f.write(response.content)
                self.temp_file = temp_file_path
                self.finished.emit(self.temp_file)
            else:
                self.error.emit(self.tr("执行推理时出现错误:\n {}").format(response.text))
        except Exception as e:
            self.error.emit(self.tr("执行推理时出现错误:\n {}").format(str(e)))


class APICheckThread(QThread):
    status_signal = pyqtSignal(bool, str)

    def __init__(self, url, silent=False, continuous=False, retry_interval=2000):
        super().__init__()
        self.url = url
        self.silent = silent
        self.continuous = continuous
        self.retry_interval = retry_interval
        self.should_continue = True

    def run(self):
        while self.should_continue:
            try:
                response = requests.get(f"{self.url}/control")
                if response.status_code == 400:  # API is ready
                    self.status_signal.emit(True, self.tr("API 就绪"))
                    if not self.continuous:
                        break
                elif not self.silent:
                    self.status_signal.emit(False, self.tr("API 未就绪"))
            except Exception as e:
                if not self.silent:
                    self.status_signal.emit(False, self.tr("API 不可用: {}").format(str(e)))

            # If not continuous or stopped, break the loop
            if not self.continuous or not self.should_continue:
                break

            # Sleep for retry interval before next check
            self.msleep(self.retry_interval)

    def stop(self):
        self.should_continue = False


class TTSGUI(QMainWindow):
    stop_playback_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.param_widgets = None
        self.SPLIT_METHODS = {
            'cut0': self.tr('不切'),
            'cut1': self.tr('凑四句一切'),
            'cut2': self.tr('凑50字一切'),
            'cut3': self.tr('按中文句号。切'),
            'cut4': self.tr('按英文句号.切'),
            'cut5': self.tr('按标点符号切')
        }
        self.LANGUAGES = {
            'all_zh': self.tr('中文'),
            'en': self.tr('英文'),
            'all_ja': self.tr('日文'),
            'all_yue': self.tr('粤语'),
            'all_ko': self.tr('韩文'),
            'zh': self.tr('中英混合'),
            'ja': self.tr('日英混合'),
            'yue': self.tr('粤英混合'),
            'ko': self.tr('韩英混合'),
            'auto': self.tr('多语种混合'),
            'auto_yue': self.tr('多语种混合(粤)')
        }
        self.SAMPLE_STEPS = {
            4: '4',
            8: '8',
            16: '16',
            32: '32',
            64: '64',
            128: '128'
        }
        self.GPT_DIRS = [
            ('GPT_weights', 'v1'),
            ('GPT_weights_v2', 'v2'),
            ('GPT_weights_v2Pro', 'v2p'),
            ('GPT_weights_v2ProPlus', 'v2pp'),
            ('GPT_weights_v3', 'v3'),
            ('GPT_weights_v4', 'v4')
        ]
        self.SOVITS_DIRS = [
            ('SoVITS_weights', 'v1'),
            ('SoVITS_weights_v2', 'v2'),
            ('SoVITS_weights_v2Pro', 'v2p'),
            ('SoVITS_weights_v2ProPlus', 'v2pp'),
            ('SoVITS_weights_v3', 'v3'),
            ('SoVITS_weights_v4', 'v4')
        ]
        self.MODEL_PARAM_RESTRICTIONS = {
            'sovits': {
                'v1': ['sample_steps', 'super_sampling'],
                'v2': ['sample_steps', 'super_sampling'],
                'v2p': ['sample_steps', 'super_sampling'],
                'v2pp': ['sample_steps', 'super_sampling'],
                'v3': ['aux_ref_audio_paths', 'no_prompt'],
                'v4': ['aux_ref_audio_paths', 'no_prompt', 'super_sampling'],
            },
            'gpt': {
                'v1': [],
                'v2': [],
                'v2p': [],
                'v2pp': [],
                'v3': [],
                'v4': [],
            }
        }
        self.config_manager = ConfigManager()
        self.setup_cache_directory()
        self.current_audio_file = None
        self.is_playing = False
        self.current_gpt_model = None
        self.current_sovits_model = None
        self.gpt_switching = False
        self.sovits_switching = False
        self.synthesis_pending = False
        self.edit_stack = []
        self.redo_stack = []
        self.play_position = 0
        self.audio_cache = {}
        self.cursor_update_timer = QTimer(self)
        self.cursor_update_timer.timeout.connect(self.update_playback_cursor)
        self.cursor_update_timer.setInterval(16)  # ~60fps
        self.stop_playback_signal.connect(self.handle_playback_stop)
        self.api_check_delay_timer = QTimer(self)
        self.api_check_delay_timer.setSingleShot(True)
        self.api_check_delay_timer.setInterval(800)
        self.api_check_delay_timer.timeout.connect(self.on_api_url_changed)
        self.watcher = QFileSystemWatcher(self)
        for dir_info in self.GPT_DIRS + self.SOVITS_DIRS:
            dir_name = dir_info[0]
            os.makedirs(dir_name, exist_ok=True)
            self.watcher.addPath(dir_name)
        self.watcher.directoryChanged.connect(self.update_model_lists)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAutoFillBackground(True)
        self.setup_background()
        self.initUI()
        self.setup_edit_actions()
        # self.start_system_monitoring()  # Uncomment the System Monitoring section if you want to use it.
        self.api_check_thread = None
        self.autostart_api_check()

    def initUI(self):
        self.setWindowTitle(self.tr('TTS GUI'))
        self.setGeometry(100, 100, 800, 800)

        # Create central widget
        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # API settings group
        self.api_url_input = QLineEdit(self.config_manager.get_value('api_url'))
        self.api_url_input.textChanged.connect(self.start_api_check_delay)
        self.api_status_label = QLabel(self.tr("状态: API 未就绪"))
        self.check_api_button = QPushButton(self.tr("检查"))
        self.check_api_button.clicked.connect(self.check_api_status_manually)

        # Parameter settings group
        params_group = QGroupBox(self.tr("待合成文本:"))
        params_group.setFlat(True)
        params_layout = QVBoxLayout()

        # Text input
        self.text_input = QTextEdit()
        self.text_input.setAcceptRichText(False)
        params_layout.addWidget(self.text_input)
        self.text_input.setPlaceholderText(
            self.tr("在这里输入需要合成的文本..."
                    "\n\n使用方法：\n"
                    "1.将本exe放入GPT-SoVITS-v2pro-20250604或更新的官方整合包下，双击启动，支持v1，v2，v2p，v2pp，v3，v4模型。\n"
                    "2.将读取并使用GPT_weights，_v2，_v2Pro，_v2ProPlus，_v3, _v4与SoVITS_weights，_v2，_v2Pro，_v2ProPlus，_v3, _v4下的模型，请先完成训练获得模型。\n"
                    "3.保存预设将保存当前所有合成参数设定，可视为一个说话人，后续可快速切换，亦可用于批量合成页面。\n"
                    "4.默认使用整合包自带环境来调起并使用API，也可以在API管理页面自定义。\n"
                    "\n此外，若无可用N卡并使用官方整合包，请在初次启动前修改GPT_SoVITS/configs/tts_infer.yaml中的device为cpu, is_half为false 以避免API启动失败。"
                    "\n\nGitHub开源地址: https://github.com/AliceNavigator/GPT-SoVITS-Api-GUI           by  领航员未鸟\n")
        )

        # Create parameter input widgets
        self.create_parameter_inputs(params_layout)

        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)

        # Control edit group
        control_edit_group = QGroupBox(self.tr("合成与编辑"))
        control_edit_layout = QVBoxLayout()
        control_edit_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # Delete & mute
        edit_buttons_layout = QHBoxLayout()
        self.delete_button = QPushButton(self.tr("删除选中"))
        self.delete_button.clicked.connect(self.delete_selection)
        self.mute_button = QPushButton(self.tr("静音选中"))
        self.mute_button.clicked.connect(self.mute_selection)

        # Silence insertion
        self.silence_spin = QSpinBox()
        self.silence_spin.setRange(10, 10000)  # 10ms to 10s
        self.silence_spin.setValue(300)
        self.silence_spin.setSuffix(" ms")
        self.insert_silence_button = QPushButton(self.tr("插入静音"))
        self.insert_silence_button.clicked.connect(self.insert_silence)

        # Undo/redo
        self.undo_button = QPushButton(self.tr("撤销"))
        self.undo_button.clicked.connect(self.undo_edit)
        self.redo_button = QPushButton(self.tr("重做"))
        self.redo_button.clicked.connect(self.redo_edit)

        edit_buttons_layout.addWidget(self.delete_button)
        edit_buttons_layout.addWidget(self.mute_button)
        edit_buttons_layout.addWidget(QLabel(self.tr("静音时长:")))
        edit_buttons_layout.addWidget(self.silence_spin)
        edit_buttons_layout.addWidget(self.insert_silence_button)
        edit_buttons_layout.addWidget(self.undo_button)
        edit_buttons_layout.addWidget(self.redo_button)
        control_edit_layout.addLayout(edit_buttons_layout)

        # Waveform display
        self.waveform = WaveformWidget(self)
        self.waveform.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Fixed)
        control_edit_layout.addWidget(self.waveform)

        # Save path settings
        path_layout = QHBoxLayout()
        self.save_path_input = QLineEdit(self.config_manager.get_value('save_directory'))
        self.save_path_input.setPlaceholderText(self.tr("将会把合成的音频保存在该路径下..."))
        self.browse_button = QPushButton(self.tr("浏览"))
        self.browse_button.clicked.connect(self.set_save_path)
        self.open_save_directory_button = QPushButton(self.tr("打开"))
        self.open_save_directory_button.clicked.connect(self.open_save_directory)
        path_layout.addWidget(QLabel(self.tr("保存路径:")))
        path_layout.addWidget(self.save_path_input)
        path_layout.addWidget(self.browse_button)
        path_layout.addWidget(self.open_save_directory_button)
        control_edit_layout.addLayout(path_layout)

        # Control buttons
        buttons_layout = QHBoxLayout()
        self.synthesize_button = QPushButton(self.tr("开始合成"))
        self.synthesize_button.setStyleSheet("min-height: 20px;")
        self.synthesize_button.setEnabled(False)
        self.synthesize_button.clicked.connect(self.prepare_synthesis)
        self.play_button = QPushButton(self.tr("播放"))
        self.play_button.setStyleSheet("min-height: 20px;")
        self.play_button.clicked.connect(self.play_audio)
        self.save_button = QPushButton(self.tr("保存音频"))
        self.save_button.setStyleSheet("min-height: 20px;")
        self.save_button.clicked.connect(self.save_audio)
        buttons_layout.addWidget(self.synthesize_button)
        buttons_layout.addWidget(self.play_button)
        buttons_layout.addWidget(self.save_button)
        control_edit_layout.addLayout(buttons_layout)

        control_edit_group.setLayout(control_edit_layout)
        main_layout.addWidget(control_edit_group)

        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        # Update model lists
        self.update_model_lists()

        # Load preset
        self.load_preset()

        # Sets the translucent GroupBox style
        self.setup_group_box_styles()

    def create_parameter_inputs(self, layout):
        """All parameters in param_widgets will be used as request parameters."""
        self.param_widgets = {}

        # Create main horizontal layout to utilize screen width
        main_param_layout = QHBoxLayout()

        # Left side layout (Presets, Models, and Language settings)
        left_side = QVBoxLayout()

        # Top row layout for presets and models
        top_row = QHBoxLayout()

        # Preset controls
        preset_group = QGroupBox(self.tr("预设"))
        preset_layout = QVBoxLayout()
        preset_layout.setSpacing(8)

        # Preset combo
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(self.config_manager.get_value('presets').keys())
        self.preset_combo.setCurrentText(self.config_manager.get_value('current_preset'))
        self.preset_combo.currentTextChanged.connect(self.load_preset)
        preset_layout.addWidget(self.preset_combo)

        # Preset buttons
        preset_buttons = QHBoxLayout()
        preset_buttons.setSpacing(4)
        self.save_preset_button = QPushButton(self.tr("保存"))
        self.delete_preset_button = QPushButton(self.tr("删除"))
        preset_buttons.addWidget(self.save_preset_button)
        self.save_preset_button.clicked.connect(self.save_preset)
        preset_buttons.addWidget(self.delete_preset_button)
        self.delete_preset_button.clicked.connect(self.delete_preset)
        preset_layout.addLayout(preset_buttons)

        preset_group.setLayout(preset_layout)
        top_row.addWidget(preset_group)

        # Model Selection
        model_group = QGroupBox(self.tr("模型选择"))
        model_layout = QFormLayout()
        model_layout.setSpacing(8)

        # GPT model
        self.gpt_combo = QComboBox()
        self.gpt_combo.setMinimumWidth(175)
        self.gpt_combo.activated.connect(self.update_model_lists)
        self.gpt_combo.currentIndexChanged.connect(self.update_param_restrictions)
        model_layout.addRow(self.tr("GPT 模型:"), self.gpt_combo)

        # Sovits model
        self.sovits_combo = QComboBox()
        self.sovits_combo.setMinimumWidth(175)
        self.sovits_combo.activated.connect(self.update_model_lists)
        self.sovits_combo.currentIndexChanged.connect(self.update_param_restrictions)
        model_layout.addRow(self.tr("SoVITS 模型:"), self.sovits_combo)

        model_group.setLayout(model_layout)
        top_row.addWidget(model_group)

        # Add top row to left side
        left_side.addLayout(top_row)

        # Language and Text Settings
        lang_group = QGroupBox(self.tr("语言与文本设置"))
        lang_layout = QVBoxLayout()
        lang_layout.setSpacing(8)

        # Text & Prompt lang
        lang_form = QFormLayout()
        lang_form.setSpacing(8)
        lang_row = QHBoxLayout()

        lang_row.addWidget(QLabel(self.tr("合成文本语种:")))
        self.param_widgets['text_lang'] = QComboBox()
        for key, value in self.LANGUAGES.items():
            self.param_widgets['text_lang'].addItem(value, key)
        lang_row.addWidget(self.param_widgets['text_lang'])

        lang_row.addWidget(QLabel(self.tr("参考音频语种:")))
        self.param_widgets['prompt_lang'] = QComboBox()
        for key, value in self.LANGUAGES.items():
            self.param_widgets['prompt_lang'].addItem(value, key)
        lang_row.addWidget(self.param_widgets['prompt_lang'])
        lang_form.addRow(lang_row)

        # Text split method
        self.param_widgets['text_split_method'] = QComboBox()
        for key, value in self.SPLIT_METHODS.items():
            self.param_widgets['text_split_method'].addItem(value, key)
        lang_form.addRow(self.tr("文本分割方式:"), self.param_widgets['text_split_method'])

        # Reference audio settings
        ref_audio_widget = QWidget()
        ref_audio_layout = QHBoxLayout(ref_audio_widget)
        ref_audio_layout.setContentsMargins(0, 0, 0, 0)
        ref_audio_layout.setSpacing(4)
        self.param_widgets['ref_audio_path'] = QLineEdit()
        self.param_widgets['ref_audio_path'].setPlaceholderText(self.tr("3-10秒的参考音频，超过会报错"))
        ref_audio_button = QPushButton(self.tr("浏览"))
        ref_audio_button.setFixedWidth(60)
        ref_audio_button.clicked.connect(lambda: self.browse_file('ref_audio_path'))
        ref_audio_layout.addWidget(self.param_widgets['ref_audio_path'])
        ref_audio_layout.addWidget(ref_audio_button)
        lang_form.addRow(self.tr("参考音频:"), ref_audio_widget)

        # Prompt text
        prompt_widget = QWidget()
        prompt_layout = QHBoxLayout(prompt_widget)
        prompt_layout.setContentsMargins(0, 0, 0, 0)
        prompt_layout.setSpacing(4)
        self.param_widgets['prompt_text'] = QLineEdit()
        self.param_widgets['prompt_text'].setPlaceholderText(self.tr("不填则视为使用无参考文本模式"))
        self.param_widgets['no_prompt'] = QCheckBox(self.tr("无参  "))
        prompt_layout.addWidget(self.param_widgets['prompt_text'])
        prompt_layout.addWidget(self.param_widgets['no_prompt'])
        lang_form.addRow(self.tr("参考音频文本:"), prompt_widget)

        # Auxiliary reference audio
        aux_ref_widget = QWidget()
        aux_ref_layout = QHBoxLayout(aux_ref_widget)
        aux_ref_layout.setContentsMargins(0, 0, 0, 0)
        aux_ref_layout.setSpacing(4)
        self.param_widgets['aux_ref_audio_paths'] = QLineEdit()
        self.param_widgets['aux_ref_audio_paths'].setPlaceholderText(self.tr("(可选)以额外的多个音频平均融合音色"))
        aux_ref_button = QPushButton(self.tr("浏览"))
        aux_ref_button.setFixedWidth(60)
        aux_ref_button.clicked.connect(lambda: self.browse_files('aux_ref_audio_paths'))
        aux_ref_layout.addWidget(self.param_widgets['aux_ref_audio_paths'])
        aux_ref_layout.addWidget(aux_ref_button)
        lang_form.addRow(self.tr("辅助参考音频:"), aux_ref_widget)

        lang_layout.addLayout(lang_form)
        lang_group.setLayout(lang_layout)
        left_side.addWidget(lang_group)

        # Add left side to main layout
        main_param_layout.addLayout(left_side, stretch=1)

        # Right side - Generation Parameters
        gen_group = QGroupBox(self.tr("合成参数"))
        gen_layout = QVBoxLayout()

        # API settings
        api_layout = QHBoxLayout()
        api_layout.addWidget(QLabel(self.tr("API URL:")))
        api_layout.addWidget(self.api_url_input)
        api_layout.addWidget(self.api_status_label)
        api_layout.addWidget(self.check_api_button)
        gen_layout.addLayout(api_layout)

        # Parameter grid
        param_grid = QGridLayout()
        param_grid.setVerticalSpacing(10)

        # Create parameter input widgets with original layout
        self.param_widgets['top_k'] = QSpinBox()
        self.param_widgets['top_k'].setRange(1, 100)
        self.param_widgets['top_p'] = QDoubleSpinBox()
        self.param_widgets['top_p'].setRange(0, 1)
        self.param_widgets['top_p'].setSingleStep(0.05)
        self.param_widgets['temperature'] = QDoubleSpinBox()
        self.param_widgets['temperature'].setRange(0, 1)
        self.param_widgets['temperature'].setSingleStep(0.05)

        param_grid.addWidget(QLabel(self.tr("Top K:")), 0, 0)
        param_grid.addWidget(self.param_widgets['top_k'], 0, 1)
        param_grid.addWidget(QLabel(self.tr("Top P:")), 1, 0)
        param_grid.addWidget(self.param_widgets['top_p'], 1, 1)
        param_grid.addWidget(QLabel(self.tr("Temperature:")), 2, 0)
        param_grid.addWidget(self.param_widgets['temperature'], 2, 1)

        self.param_widgets['speed_factor'] = QDoubleSpinBox()
        self.param_widgets['speed_factor'].setRange(0.1, 3)
        self.param_widgets['speed_factor'].setSingleStep(0.1)
        self.param_widgets['repetition_penalty'] = QDoubleSpinBox()
        self.param_widgets['repetition_penalty'].setRange(0, 5)
        self.param_widgets['repetition_penalty'].setSingleStep(0.05)
        self.param_widgets['seed'] = QSpinBox()
        self.param_widgets['seed'].setRange(-1, 1000000)

        param_grid.addWidget(QLabel(self.tr("语速:")), 0, 2)
        param_grid.addWidget(self.param_widgets['speed_factor'], 0, 3)
        param_grid.addWidget(QLabel(self.tr("重复惩罚:")), 1, 2)
        param_grid.addWidget(self.param_widgets['repetition_penalty'], 1, 3)
        param_grid.addWidget(QLabel(self.tr("种子:")), 2, 2)
        param_grid.addWidget(self.param_widgets['seed'], 2, 3)

        gen_layout.addLayout(param_grid)

        # Checkbox options
        options_grid = QGridLayout()
        self.param_widgets['parallel_infer'] = QCheckBox(self.tr("并行推理"))
        self.param_widgets['split_bucket'] = QCheckBox(self.tr("数据分桶"))
        self.param_widgets['super_sampling'] = QCheckBox(self.tr("音频超分"))
        self.param_widgets['sample_steps'] = QComboBox()
        for key, value in self.SAMPLE_STEPS.items():
            self.param_widgets['sample_steps'].addItem(value, key)
        self.param_widgets['sample_steps'].setCurrentIndex(3)

        sample_steps_widget = QWidget()
        sample_steps_layout = QHBoxLayout(sample_steps_widget)
        sample_steps_layout.setContentsMargins(0, 0, 0, 0)
        sample_steps_layout.addWidget(QLabel(self.tr("采样步数:")))
        sample_steps_layout.addWidget(self.param_widgets['sample_steps'])

        options_grid.addWidget(self.param_widgets['parallel_infer'], 0, 0)
        options_grid.addWidget(sample_steps_widget, 0, 1)
        options_grid.addWidget(self.param_widgets['split_bucket'], 1, 0)
        options_grid.addWidget(self.param_widgets['super_sampling'], 1, 1)

        gen_layout.addLayout(options_grid)

        # Batch settings
        batch_layout = QGridLayout()
        self.param_widgets['batch_size'] = QSpinBox()
        self.param_widgets['batch_size'].setRange(1, 1000)
        self.param_widgets['batch_threshold'] = QDoubleSpinBox()
        self.param_widgets['batch_threshold'].setRange(0, 1)
        self.param_widgets['batch_threshold'].setSingleStep(0.05)

        batch_layout.addWidget(QLabel(self.tr("批次大小:")), 0, 0)
        batch_layout.addWidget(self.param_widgets['batch_size'], 0, 1)
        batch_layout.addWidget(QLabel(self.tr("分批阈值:")), 0, 2)
        batch_layout.addWidget(self.param_widgets['batch_threshold'], 0, 3)

        gen_layout.addLayout(batch_layout)

        # # System monitoring
        # monitor_layout = QVBoxLayout()
        #
        # # CPU usage
        # cpu_layout = QGridLayout()
        # cpu_layout.addWidget(QLabel(self.tr("CPU 占用:")), 0, 0)
        # self.cpu_progress = QProgressBar()
        # self.cpu_progress.setRange(0, 100)
        # cpu_layout.addWidget(self.cpu_progress, 0, 1)
        # monitor_layout.addLayout(cpu_layout)
        #
        # # Memory usage
        # memory_layout = QGridLayout()
        # memory_layout.addWidget(QLabel(self.tr("内存占用:")), 1, 0)
        # self.memory_progress = QProgressBar()
        # self.memory_progress.setRange(0, 100)
        # self.memory_progress.setTextVisible(False)
        # memory_layout.addWidget(self.memory_progress, 1, 1)
        # self.memory_label = QLabel()
        # self.memory_label.setAlignment(Qt.AlignCenter)
        # memory_layout.addWidget(self.memory_label, 1, 1)
        # monitor_layout.addLayout(memory_layout)
        #
        # gen_layout.addLayout(monitor_layout)

        # Add right side to main layout
        gen_group.setLayout(gen_layout)
        main_param_layout.addWidget(gen_group, stretch=1)

        # Add the main layout to the parent layout
        layout.addLayout(main_param_layout)

    def open_save_directory(self):
        if os.path.exists(self.config_manager.get_value('save_directory')):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.config_manager.get_value('save_directory')))
        else:
            QMessageBox.warning(self, self.tr("错误"), self.tr("指定的保存文件夹不存在"))

    def browse_file(self, param_name):
        file_name, _ = QFileDialog.getOpenFileName(self, self.tr("选择音频文件"), "", self.tr("Audio Files (*.wav)"))
        if file_name:
            self.param_widgets[param_name].setText(file_name)

    def browse_files(self, param_name):
        files, _ = QFileDialog.getOpenFileNames(self, self.tr("选择音频文件"), "", self.tr("Audio Files (*.wav)"))
        if files:
            self.param_widgets[param_name].setText(';'.join(files))

    def autostart_api_check(self):
        self.on_api_url_changed()

    def start_api_check_delay(self):
        if self.config_manager.get_value('autostart_api', False):
            self.api_check_delay_timer.stop()
            self.api_check_delay_timer.start()

    def handle_autostart_api_check(self, is_available, message):
        if is_available:
            self.api_status_label.setText(self.tr("状态: API 就绪"))
            self.check_api_button.setEnabled(True)
            self.synthesize_button.setEnabled(True)
            if self.api_check_thread:
                self.api_check_thread.stop()
                self.api_check_thread = None

    def on_api_url_changed(self):
        if not self.config_manager.get_value('autostart_api', False):
            return

        if self.api_check_thread:
            self.api_check_thread.stop()
            self.api_check_thread = None

        current_url = self.api_url_input.text().strip()
        if not current_url:
            self.api_status_label.setText(self.tr("状态: URL为空"))
            return

        self.check_api_button.setEnabled(False)
        self.api_status_label.setText(self.tr("状态: 检查中..."))

        self.api_check_thread = APICheckThread(
            current_url,
            silent=True,
            continuous=True
        )
        self.api_check_thread.status_signal.connect(self.handle_autostart_api_check)
        self.api_check_thread.start()

    def check_api_status_manually(self):
        self.api_status_label.setText(self.tr("状态: 检查中..."))
        self.check_api_button.setEnabled(False)
        self.synthesize_button.setEnabled(False)

        self.api_check_thread_once = APICheckThread(
            self.api_url_input.text(),
            silent=False,
            continuous=False
        )
        self.api_check_thread_once.status_signal.connect(self.handle_manually_api_check)
        self.api_check_thread_once.start()

    def handle_manually_api_check(self, is_available, message):
        if is_available:
            self.api_status_label.setText(self.tr("状态: API 就绪"))
            self.check_api_button.setEnabled(True)
            self.synthesize_button.setEnabled(True)
        else:
            self.api_status_label.setText(self.tr("状态: API 未就绪"))
            self.check_api_button.setEnabled(True)
            self.synthesize_button.setEnabled(True)
            QMessageBox.warning(self, self.tr("API 错误"), message)

    # def start_system_monitoring(self):
    #     self.monitor_timer = QTimer()
    #     self.monitor_timer.timeout.connect(self.update_system_monitoring)
    #     self.monitor_timer.start(1000)  # 1000ms
    #
    # def update_system_monitoring(self):
    #     cpu_percent = psutil.cpu_percent()
    #     memory_info = psutil.virtual_memory()
    #     memory_percent = memory_info.percent
    #
    #     self.cpu_progress.setValue(int(cpu_percent))
    #
    #     memory_used_gb = memory_info.used / (1024 ** 3)
    #     memory_total_gb = memory_info.total / (1024 ** 3)
    #     self.memory_label.setText(
    #         self.tr("{:.2f} GB / {:.2f} GB").format(memory_used_gb, memory_total_gb))
    #     self.memory_progress.setValue(int(memory_percent))

    def get_model_version(self, model_type):
        combo = self.gpt_combo if model_type == 'gpt' else self.sovits_combo
        return combo.currentText().split('/')[0]

    def update_param_restrictions(self):
        sovits_ver = self.get_model_version('sovits')
        gpt_ver = self.get_model_version('gpt')

        restricted_params = set(
            self.MODEL_PARAM_RESTRICTIONS['sovits'].get(sovits_ver, []) +
            self.MODEL_PARAM_RESTRICTIONS['gpt'].get(gpt_ver, [])
        )

        for param, widget in self.param_widgets.items():
            widget.setEnabled(param not in restricted_params)

    @staticmethod
    def get_model_files(directory, extension):
        if not os.path.exists(directory):
            return []
        return [f for f in os.listdir(directory) if f.endswith(extension)]

    def update_model_lists(self):
        current_gpt = self.gpt_combo.currentData()
        current_sovits = self.sovits_combo.currentData()

        self.gpt_combo.clear()
        gpt_models = []
        for dir_name, version in self.GPT_DIRS:
            models = self.get_model_files(dir_name, '.ckpt')
            for model in models:
                display_name = f"{version}/{model}"
                full_path = os.path.join(dir_name, model)
                gpt_models.append((display_name, full_path))

        seen = set()
        unique_models = []
        for display, path in gpt_models:
            if path not in seen:
                seen.add(path)
                unique_models.append((display, path))
        unique_models.sort(key=lambda x: (x[0].split('/')[0], x[0].split('/')[1]))

        for display, path in unique_models:
            self.gpt_combo.addItem(display, path)

        if current_gpt in seen:
            index = self.gpt_combo.findData(current_gpt)
            if index >= 0:
                self.gpt_combo.setCurrentIndex(index)

        self.sovits_combo.clear()
        sovits_models = []
        for dir_name, version in self.SOVITS_DIRS:
            models = self.get_model_files(dir_name, '.pth')
            for model in models:
                display_name = f"{version}/{model}"
                full_path = os.path.join(dir_name, model)
                sovits_models.append((display_name, full_path))

        seen_sovits = set()
        unique_sovits = []
        for display, path in sovits_models:
            if path not in seen_sovits:
                seen_sovits.add(path)
                unique_sovits.append((display, path))
        unique_sovits.sort(key=lambda x: (x[0].split('/')[0], x[0].split('/')[1]))

        for display, path in unique_sovits:
            self.sovits_combo.addItem(display, path)

        if current_sovits in seen_sovits:
            index = self.sovits_combo.findData(current_sovits)
            if index >= 0:
                self.sovits_combo.setCurrentIndex(index)

    def switch_models_and_synthesize(self):
        gpt_model = self.gpt_combo.currentData()
        sovits_model = self.sovits_combo.currentData()

        self.gpt_switching = False
        self.sovits_switching = False

        self.synthesize_button.setEnabled(False)

        if gpt_model != self.current_gpt_model:
            self.gpt_switching = True
            self.gpt_switch_thread = ModelSwitchThread(self.api_url_input.text(), 'gpt', gpt_model)
            self.gpt_switch_thread.switch_signal.connect(self.handle_switch_result)
            self.gpt_switch_thread.start()

        if sovits_model != self.current_sovits_model:
            self.sovits_switching = True
            self.sovits_switch_thread = ModelSwitchThread(self.api_url_input.text(), 'sovits', sovits_model)
            self.sovits_switch_thread.switch_signal.connect(self.handle_switch_result)
            self.sovits_switch_thread.start()

        # If no models need to be switched
        if not self.gpt_switching and not self.sovits_switching:
            self.statusBar.showMessage(self.tr("未检测到模型变更"), 5000)
            self.execute_synthesis()

    def handle_switch_result(self, success, message):
        if not success:
            self.synthesis_pending = False
            self.synthesize_button.setEnabled(True)
            QMessageBox.warning(self, self.tr("模型切换时出错"), message)

            if "GPT" in message:
                self.gpt_switching = False
            elif "SOVITS" in message:
                self.sovits_switching = False
        else:
            self.statusBar.showMessage(message, 5000)
            if "GPT" in message:
                self.current_gpt_model = self.gpt_combo.currentData()
                self.gpt_switching = False
            elif "SOVITS" in message:
                self.current_sovits_model = self.sovits_combo.currentData()
                self.sovits_switching = False

        # If all requested switches are complete and there is a pending synthesis task, execute synthesis
        if not self.gpt_switching and not self.sovits_switching and self.synthesis_pending:
            self.execute_synthesis()

    def prepare_synthesis(self):
        if not self.text_input.toPlainText():
            QMessageBox.warning(self, self.tr("错误"), self.tr("请先输入需要合成的文本!"))
            return

        self.synthesis_pending = True
        self.synthesize_button.setEnabled(False)
        self.switch_models_and_synthesize()

    def execute_synthesis(self):
        self.synthesis_pending = False
        self.statusBar.showMessage(self.tr("合成中..."))

        params = self.get_current_parameters()
        api_url = self.api_url_input.text()

        self.tts_thread = TTSThread(api_url, params, self.cache_dir)
        self.tts_thread.finished.connect(self.synthesis_finished)
        self.tts_thread.error.connect(self.synthesis_error)
        self.tts_thread.start()

    def synthesis_finished(self, audio_file):
        self.current_audio_file = audio_file
        self.audio_cache = {}
        self.edit_stack = []
        self.redo_stack = []

        if self.waveform.load_audio(audio_file, reset_view=True):
            self.synthesize_button.setEnabled(True)
            self.statusBar.showMessage(self.tr("合成成功!"))
            self.play_button.setEnabled(True)
            self.save_button.setEnabled(True)
            self.delete_button.setEnabled(True)
            self.mute_button.setEnabled(True)
            self.insert_silence_button.setEnabled(True)
            self.undo_button.setEnabled(False)
            self.redo_button.setEnabled(False)
        else:
            self.synthesis_error(self.tr("无法加载波形显示"))
            self.synthesize_button.setEnabled(True)
            self.play_button.setEnabled(True)
            self.save_button.setEnabled(True)

    def synthesis_error(self, error_message):
        self.synthesize_button.setEnabled(True)
        self.statusBar.showMessage(self.tr("合成失败!"))
        QMessageBox.critical(self, self.tr("合成语音失败"), error_message)

    def setup_edit_actions(self):
        self.delete_button.setEnabled(False)
        self.mute_button.setEnabled(False)
        self.insert_silence_button.setEnabled(False)
        self.undo_button.setEnabled(False)
        self.redo_button.setEnabled(False)

    def delete_selection(self):
        selection = self.waveform.get_selection()
        if not selection:
            QMessageBox.warning(self, self.tr("警告"), self.tr("请先选择要删除的区域\n（右键拖拽选择区域）"))
            return
        try:
            if self.current_audio_file in self.audio_cache:
                data, sr = self.audio_cache[self.current_audio_file]
            else:
                data, sr = sf.read(self.current_audio_file)
            start_sample = int(selection[0] * sr)
            end_sample = int(selection[1] * sr)
            new_data = np.concatenate([data[:start_sample], data[end_sample:]])
            self.apply_edit(new_data, sr)

        except Exception as e:
            QMessageBox.critical(self, self.tr("错误"), self.tr("删除选区时出错: {}").format(str(e)))

    def mute_selection(self):
        selection = self.waveform.get_selection()
        if not selection:
            QMessageBox.warning(self, self.tr("警告"), self.tr("请先选择要静音的区域\n（右键拖拽选择区域）"))
            return
        try:
            if self.current_audio_file in self.audio_cache:
                data, sr = self.audio_cache[self.current_audio_file]
            else:
                data, sr = sf.read(self.current_audio_file)
            start_sample = int(selection[0] * sr)
            end_sample = int(selection[1] * sr)
            if len(data.shape) > 1:
                data[start_sample:end_sample] = 0
            else:
                data[start_sample:end_sample] = 0
            self.apply_edit(data, sr)

        except Exception as e:
            QMessageBox.critical(self, self.tr("错误"), self.tr("静音选区时出错: {}").format(str(e)))

    def insert_silence(self):
        duration_ms = self.silence_spin.value()
        duration_sec = duration_ms / 1000.0
        cursor_pos = self.waveform.get_cursor_position()

        try:
            if self.current_audio_file in self.audio_cache:
                data, sr = self.audio_cache[self.current_audio_file]
            else:
                data, sr = sf.read(self.current_audio_file)
            silence_samples = int(duration_sec * sr)
            silence = np.zeros(silence_samples)
            insert_pos = int(cursor_pos * sr)

            if len(data.shape) > 1:  # Stereo - need 2D silence
                silence = np.zeros((silence_samples, data.shape[1]))

            new_data = np.concatenate([data[:insert_pos], silence, data[insert_pos:]])
            self.apply_edit(new_data, sr)

        except Exception as e:
            QMessageBox.critical(self, self.tr("错误"), self.tr("插入静音时出错: {}").format(str(e)))

    def apply_edit(self, new_data, sample_rate):
        try:
            temp_file = os.path.join(self.cache_dir, f"edit_{int(time.time())}.wav")
            sf.write(temp_file, new_data, sample_rate)
            if self.current_audio_file in self.audio_cache:
                del self.audio_cache[self.current_audio_file]
            self.audio_cache[temp_file] = (new_data, sample_rate)
            self.push_undo_state()
            self.current_audio_file = temp_file
            self.waveform.load_audio(temp_file, reset_view=False)
            self.undo_button.setEnabled(len(self.edit_stack) > 0)
            self.redo_button.setEnabled(False)

        except Exception as e:
            QMessageBox.critical(self, self.tr("错误"), self.tr("应用编辑时出错: {}").format(str(e)))

    def push_undo_state(self):
        if os.path.exists(self.current_audio_file):
            self.edit_stack.append(self.current_audio_file)
            self.redo_stack.clear()  # Clear redo stack on new action

    def undo_edit(self):
        if len(self.edit_stack) > 0:
            self.redo_stack.append(self.current_audio_file)
            prev_file = self.edit_stack.pop()
            self.current_audio_file = prev_file
            self.waveform.load_audio(prev_file, reset_view=False)
            self.undo_button.setEnabled(len(self.edit_stack) > 0)
            self.redo_button.setEnabled(True)
        else:
            QMessageBox.information(self, self.tr("信息"), self.tr("没有更多操作可以撤销"))

    def redo_edit(self):
        if len(self.redo_stack) > 0:
            self.edit_stack.append(self.current_audio_file)
            next_file = self.redo_stack.pop()
            self.current_audio_file = next_file
            self.waveform.load_audio(next_file, reset_view=False)
            self.undo_button.setEnabled(True)
            self.redo_button.setEnabled(len(self.redo_stack) > 0)
        else:
            QMessageBox.information(self, self.tr("信息"), self.tr("没有更多操作可以重做"))

    def play_audio(self):
        if not self.current_audio_file:
            return

        if self.is_playing:
            self.stop_playback()
            return

        cursor_pos = self.waveform.get_cursor_position()
        self.play_audio_from(cursor_pos)

    def handle_playback_stop(self):
        if self.cursor_update_timer.isActive():
            self.cursor_update_timer.stop()

        if hasattr(self, 'stream') and self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
            delattr(self, 'stream')

        self.is_playing = False
        self.play_button.setText(self.tr("播放"))

    def stop_playback(self):
        if self.is_playing:
            self.stop_playback_signal.emit()

    def __del__(self):
        # Clean up resources
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop()
            self.stream.close()

    def save_audio(self):
        if not self.current_audio_file:
            return

        save_directory = self.save_path_input.text()

        if not save_directory:
            QMessageBox.warning(self, self.tr("警告"), self.tr("请先设置音频保存的路径。"))
            return

        if not os.path.exists(save_directory):
            reply = QMessageBox.question(self, self.tr("目录不存在"),
                                         self.tr(" {} 不存在， 是否尝试新建该文件夹？").format(save_directory),
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                os.makedirs(save_directory)
            else:
                return

        try:
            filename = self.generate_filename()
            file_path = os.path.join(save_directory, filename)
            shutil.copy2(self.current_audio_file, file_path)
            self.statusBar.showMessage(self.tr("音频保存至 {}").format(file_path))

        except Exception as e:
            QMessageBox.critical(self, self.tr("错误"), self.tr("保存音频时出错: {}").format(str(e)))

    def restart_playback_from(self, position):
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop()
            self.stream.close()
            delattr(self, 'stream')
        self.play_audio_from(position)

    def play_audio_from(self, position):
        if not self.current_audio_file:
            return

        selection = self.waveform.get_selection()
        if selection:
            position = selection[0]
            end_position = selection[1]
        else:
            end_position = None

        if hasattr(self, 'stream') and self.stream:
            self.stop_playback()

        try:
            if self.current_audio_file in self.audio_cache:
                data, samplerate = self.audio_cache[self.current_audio_file]
            else:
                data, samplerate = sf.read(self.current_audio_file)
                self.audio_cache[self.current_audio_file] = (data, samplerate)

            if len(data.shape) == 1:
                data = data.reshape(-1, 1)

            self.play_position = int(position * samplerate)
            self.play_end_position = int(end_position * samplerate) if end_position else len(data)

            def callback(outdata, frames, time, status):
                if status:
                    print('Status:', status)

                if self.play_position + frames >= self.play_end_position:
                    remaining = self.play_end_position - self.play_position
                    if remaining > 0:
                        outdata[:remaining] = data[self.play_position:self.play_position + remaining]
                        outdata[remaining:] = 0
                    self.is_playing = False
                    self.play_button.setText(self.tr("播放"))
                    self.play_position = int(position * samplerate)
                    self.stop_playback_signal.emit()
                    self.waveform.update_cursor_during_playback(position)
                    raise sd.CallbackStop()

                current_chunk = data[self.play_position:self.play_position + frames]
                if len(current_chunk.shape) == 1:
                    current_chunk = current_chunk.reshape(-1, 1)
                outdata[:] = current_chunk
                self.play_position += frames

            self.cursor_update_timer.start()
            self.stream = sd.OutputStream(
                samplerate=samplerate,
                channels=data.shape[1] if len(data.shape) > 1 else 1,
                callback=callback
            )
            self.stream.start()
            self.is_playing = True
            self.play_button.setText(self.tr("停止"))
            self.waveform.update_cursor_during_playback(position)

        except Exception as e:
            QMessageBox.critical(self, self.tr("错误"), self.tr("尝试播放时出错: {}").format(str(e)))
            self.is_playing = False
            self.play_button.setText(self.tr("播放"))
            if hasattr(self, 'stream') and self.stream:
                self.stream.stop()
                self.stream.close()
                delattr(self, 'stream')

    def update_playback_cursor(self):
        if not hasattr(self, 'stream') or not self.stream or not self.is_playing:
            return
        current_time = self.play_position / self.audio_cache[self.current_audio_file][1]
        self.waveform.update_cursor_during_playback(current_time)

    def set_save_path(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            self.tr("选择保存目录"),
            self.config_manager.get_value('save_directory', '')
        )
        if directory:
            self.save_path_input.setText(directory)

    def generate_filename(self):
        # Get first 10 characters of text, remove special characters
        text = self.text_input.toPlainText().strip()
        clean_text = re.sub(r'[^\w\s]', '', text)
        clean_text = clean_text.replace('\n', '_').replace('\r', '_').replace(' ', '_').replace('_', '')
        prefix = clean_text[:10] if clean_text else "audio"

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{prefix}_{timestamp}.wav"

    def load_preset(self):
        preset_name = self.preset_combo.currentText()
        presets = self.config_manager.get_value('presets', {})
        preset_data = presets.get(preset_name, {})

        if 'gpt_model' in preset_data and preset_data['gpt_model']:
            index = self.gpt_combo.findData(preset_data['gpt_model'])
            if index >= 0:
                self.gpt_combo.setCurrentIndex(index)

        if 'sovits_model' in preset_data and preset_data['sovits_model']:
            index = self.sovits_combo.findData(preset_data['sovits_model'])
            if index >= 0:
                self.sovits_combo.setCurrentIndex(index)

        for param, value in preset_data.items():
            if param in self.param_widgets:
                widget = self.param_widgets[param]
                if isinstance(widget, QLineEdit):
                    if isinstance(value, list):
                        widget.setText(';'.join(value))
                    else:
                        widget.setText(str(value))
                elif isinstance(widget, QComboBox):
                    index = widget.findData(value)
                    if index >= 0:
                        widget.setCurrentIndex(index)
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.setValue(value)
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(value)

    def save_preset(self):
        preset_name, ok = QInputDialog.getText(self, self.tr("保存预设"), self.tr("输入预设名:"))
        if ok and preset_name:
            preset_data = {
                'gpt_model': self.gpt_combo.currentData(),
                'sovits_model': self.sovits_combo.currentData(),
            }

            for param, widget in self.param_widgets.items():
                if isinstance(widget, QLineEdit):
                    if param == 'aux_ref_audio_paths':
                        value = widget.text().split(';') if widget.text() else []
                    else:
                        value = widget.text()
                elif isinstance(widget, QComboBox):
                    value = widget.currentData()
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    value = widget.value()
                elif isinstance(widget, QCheckBox):
                    value = widget.isChecked()
                preset_data[param] = value

            update_data = {
                'presets': {
                    preset_name: preset_data
                }
            }
            self.config_manager.update_config(update_data)

            current_items = [self.preset_combo.itemText(i) for i in range(self.preset_combo.count())]
            if preset_name not in current_items:
                self.preset_combo.addItem(preset_name)
                self.preset_combo.setCurrentText(preset_name)

            QMessageBox.information(self, self.tr("成功"),
                                    self.tr("已成功将当前参数保存为 '{}' !").format(preset_name))

    def delete_preset(self):
        current_preset = self.preset_combo.currentText()
        if current_preset == 'Default':
            QMessageBox.warning(self, self.tr('警告'),
                                self.tr('默认预设不能被删除!'))
            return

        reply = QMessageBox.question(self, self.tr('删除预设'),
                                     self.tr('确认删除当前预设 "{}"?').format(current_preset),
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            if self.config_manager.delete_value(f'presets.{current_preset}'):
                # 更新UI
                self.preset_combo.removeItem(self.preset_combo.currentIndex())
                QMessageBox.information(self, self.tr("成功"),
                                        self.tr("预设 '{}' 已被成功删除!").format(current_preset))
            else:
                QMessageBox.warning(self, self.tr("错误"),
                                    self.tr("删除预设 '{}' 失败!").format(current_preset))

    def get_current_parameters(self):
        params = {}
        params['text'] = self.text_input.toPlainText()

        for param, widget in self.param_widgets.items():
            if isinstance(widget, QLineEdit):
                if param == 'aux_ref_audio_paths':
                    value = widget.text().split(';') if widget.text() else []
                elif param == 'prompt_text' and self.param_widgets['no_prompt'].isChecked() and self.get_model_version('sovits') not in {'v3', 'v4'}:
                    value = ""
                else:
                    value = widget.text()
            elif isinstance(widget, QComboBox):
                value = widget.currentData()  # Get current item's data (key)
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                value = widget.value()
            elif isinstance(widget, QCheckBox):
                if param == 'no_prompt':
                    continue
                else:
                    value = widget.isChecked()
            params[param] = value

        return params

    def setup_cache_directory(self):
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        else:
            for file in os.listdir(self.cache_dir):
                try:
                    file_path = os.path.join(self.cache_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(self.tr("清理缓存时出错 {}: {}").format(file, str(e)))

    def setup_background(self):
        background_path = ":/GAG_tools/images/GAG_background.png"
        style = f"""
            QMainWindow {{
                background-image: url({background_path});
            }}
            QWidget#centralWidget {{
                background-color: rgba(255, 255, 255, 180);
            }}
            QGroupBox {{
                background-color: rgba(255, 255, 255, 200);
            }}
            QGroupBox::title {{
                background-color: rgba(255, 255, 255, 200);
            }}
        """
        self.setStyleSheet(style)

    def setup_group_box_styles(self):
        style = """
            QGroupBox {
                background-color: rgba(255, 255, 255, 150);
            }
            QGroupBox::title {
                background-color: rgba(255, 255, 255, 200);
            }
            QTextEdit{
                background-color: rgba(255, 255, 255, 150);
            }

        """
        for widget in self.findChildren(QGroupBox):
            widget.setStyleSheet(style)

    def cleanup(self):
        # Save current configuration
        updates = {
            'api_url': self.api_url_input.text(),
            'current_preset': self.preset_combo.currentText(),
            'save_directory': self.save_path_input.text()
        }
        self.config_manager.update_config(updates)

        # Stop API check thread if running
        if self.api_check_thread:
            self.api_check_thread.stop()

    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    gui = TTSGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
