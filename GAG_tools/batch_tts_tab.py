import os
import time
import shutil
import requests
import threading
import subprocess
from subprocess import PIPE
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
                             QPushButton, QComboBox, QListWidget,
                             QLabel, QFileDialog, QMessageBox, QSizePolicy)
from PyQt5.QtCore import pyqtSignal, QUrl
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QColor, QDesktopServices
from charset_normalizer import from_path
from GAG_tools.config_manager import ConfigManager


class EnterableComboBox(QComboBox):
    entered = pyqtSignal()

    def enterEvent(self, event):
        self.entered.emit()
        super().enterEvent(event)


@dataclass
class TTSTask:
    file_path: str
    preset_name: str
    segment_count: int = 0
    completed_segments: int = 0
    status: str = 'pending'
    average_time_per_segment: float = 0.0

    @property
    def progress(self) -> int:
        return int(self.completed_segments * 100 / self.segment_count) if self.segment_count else 0

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


class BatchTTS(QWidget):
    status_updated = pyqtSignal(str, str, int, str, float)  # file_path, status, progress, color, remaining_time

    def __init__(self):
        super().__init__()
        self.tasks = {}
        self.current_gpt = ''
        self.current_sovits = ''
        self.config_manager = ConfigManager()
        self.setAcceptDrops(True)
        self.processing = False
        self.worker_thread = None
        self.executor = ThreadPoolExecutor(max_workers=1)

        self.initUI()
        self.setup_connections()
        self.load_saved_state()

    def initUI(self):
        self.setGeometry(100, 100, 800, 800)
        layout = QVBoxLayout()

        # API URL
        url_layout = QHBoxLayout()
        self.url_edit = QLineEdit()
        url_layout.addWidget(QLabel(self.tr("API URL:")), 0)
        url_layout.addWidget(self.url_edit, 1)
        layout.addLayout(url_layout)

        # Preset and format selector
        preset_layout = QHBoxLayout()
        self.preset_combo = EnterableComboBox()
        self.format_combo = QComboBox()
        self.format_combo.addItems(['wav', 'mp3', 'flac', 'ogg', 'aac'])

        preset_label = QLabel(self.tr("预设:   "))
        format_label = QLabel(self.tr("输出格式:"))

        preset_layout.addWidget(preset_label, 0)
        preset_layout.addWidget(self.preset_combo, 4)
        preset_layout.addWidget(format_label, 0)
        preset_layout.addWidget(self.format_combo, 2)
        layout.addLayout(preset_layout)

        # File list with vertical buttons on the right
        file_section = QHBoxLayout()

        # File list on the left
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        file_section.addWidget(self.file_list)

        # Create a container widget for buttons to match file_list height
        btn_container = QWidget()
        btn_layout = QVBoxLayout(btn_container)
        btn_layout.setSpacing(4)  # Remove spacing between buttons
        btn_layout.setContentsMargins(0, 1, 0, 1)  # Remove margins

        # Create vertical buttons
        self.clear_btn = QPushButton('\n'.join(self.tr("清空列表")))
        self.clear_btn.setFixedWidth(30)
        self.clear_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        btn_layout.addWidget(self.clear_btn)

        self.move_up_btn = QPushButton('\n'.join(self.tr("▲‖‖‖")))
        self.move_up_btn.setFixedWidth(30)
        self.move_up_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        btn_layout.addWidget(self.move_up_btn)

        self.move_down_btn = QPushButton('\n'.join(self.tr("‖‖‖▼")))
        self.move_down_btn.setFixedWidth(30)
        self.move_down_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        btn_layout.addWidget(self.move_down_btn)

        self.remove_btn = QPushButton('\n'.join(self.tr("删除选中")))
        self.remove_btn.setFixedWidth(30)
        self.remove_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        btn_layout.addWidget(self.remove_btn)

        file_section.addWidget(btn_container)
        layout.addLayout(file_section)

        # Output directory
        out_layout = QHBoxLayout()
        self.out_dir_edit = QLineEdit()
        self.out_dir_edit.setPlaceholderText(self.tr("在这里设置批量合成的保存路径..."))
        self.browse_btn = QPushButton(self.tr("浏览"))
        self.open_output_button = QPushButton(self.tr("打开"))
        out_layout.addWidget(QLabel(self.tr("批量合成保存路径:")), 0)
        out_layout.addWidget(self.out_dir_edit, 1)
        out_layout.addWidget(self.browse_btn)
        out_layout.addWidget(self.open_output_button)
        layout.addLayout(out_layout)

        # Synthesis controls
        control_layout = QHBoxLayout()
        self.synth_btn = QPushButton(self.tr("开始合成"))
        self.synth_btn.setStyleSheet("min-height: 30px;")
        control_layout.addWidget(self.synth_btn)
        layout.addLayout(control_layout)

        self.setLayout(layout)


    def setup_connections(self):
        self.clear_btn.clicked.connect(self.clear_list)
        self.move_up_btn.clicked.connect(self.move_up)
        self.move_down_btn.clicked.connect(self.move_down)
        self.remove_btn.clicked.connect(self.remove_selected)
        self.browse_btn.clicked.connect(self.browse_output_dir)
        self.open_output_button.clicked.connect(self.open_output_folder)
        self.synth_btn.clicked.connect(self.start_synthesis)
        self.preset_combo.entered.connect(self._update_preset)
        self.status_updated.connect(self._update_item_status)

    def open_output_folder(self):
        output_folder = self.out_dir_edit.text()
        if os.path.isdir(output_folder):
            QDesktopServices.openUrl(QUrl.fromLocalFile(output_folder))
        else:
            QMessageBox.warning(self, self.tr('警告'), self.tr('输出文件夹无效'))

    def _update_preset(self):
        presets = self.config_manager.get_value('presets', {})
        current = self.preset_combo.currentText()
        self.preset_combo.clear()
        self.preset_combo.addItems(presets.keys())
        if current in presets:
            self.preset_combo.setCurrentText(current)

    def load_saved_state(self):
        self.url_edit.setText(self.config_manager.get_value('api_url', ''))
        self.out_dir_edit.setText(self.config_manager.get_value('batch_tts_save_directory', ''))

        presets = self.config_manager.get_value('presets', {})
        self.preset_combo.clear()
        self.preset_combo.addItems(presets.keys())
        self.preset_combo.setCurrentText(self.config_manager.get_value('current_preset', ''))

        saved_format = self.config_manager.get_value('output_format', 'wav')
        self.format_combo.setCurrentText(saved_format)

        saved_tasks = self.config_manager.get_value('batch_tts_tasks', [])
        for task_dict in saved_tasks:
            file_path = task_dict.get('file_path', '')
            if os.path.exists(file_path):
                task = TTSTask.from_dict(task_dict)
                self.file_list.addItem(file_path)
                self.tasks[file_path] = task

                cache_dir = os.path.join(self.out_dir_edit.text(), 'cache',
                                         os.path.splitext(os.path.basename(file_path))[0])
                if os.path.exists(cache_dir):
                    segments = [f for f in os.listdir(cache_dir) if f.endswith('.txt') and f != 'files.txt']
                    wavs = [f for f in os.listdir(cache_dir) if f.endswith('.wav')]
                    if segments:
                        task.segment_count = len(segments)
                        task.completed_segments = len(wavs)
                        status = 'completed' if task.completed_segments == task.segment_count else 'processing'
                        self.status_updated.emit(
                            file_path,
                            self.tr("已完成") if status == 'completed' else self.tr("已暂停"),
                            task.progress,
                            'green' if status == 'completed' else 'blue',
                            0.0
                        )

    def save_current_state(self):
        self.config_manager.set_value('api_url', self.url_edit.text())
        self.config_manager.set_value('batch_tts_save_directory', self.out_dir_edit.text())
        self.config_manager.set_value('output_format', self.format_combo.currentText())

        tasks_data = []
        for file_path, task in self.tasks.items():
            if os.path.exists(file_path):
                tasks_data.append(asdict(task))
        self.config_manager.set_value('batch_tts_tasks', tasks_data)

    def cleanup(self):
        if self.processing:
            self.processing = False

            for file_path, task in self.tasks.items():
                if task.status == 'processing':
                    task.status = 'pending'

            if self.worker_thread:
                self.worker_thread.join(timeout=0.5)

        self.save_current_state()
        if self.executor:
            self.executor.shutdown(wait=False)

    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().endswith('.txt'):
                    event.acceptProposedAction()
                    return

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.endswith('.txt') and os.path.exists(file_path):
                base_path = file_path.split(" （")[0]
                if base_path not in self.tasks:
                    self.file_list.addItem(base_path)
                    self.tasks[base_path] = TTSTask(
                        file_path=base_path,
                        preset_name=self.preset_combo.currentText()
                    )
                    self.status_updated.emit(base_path, self.tr("待合成"), -1, 'gray', 0.0)
        self.save_current_state()

    def clear_list(self):
        try:
            cache_dir = os.path.join(self.out_dir_edit.text(), 'cache')
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=False)
            self.file_list.clear()
            self.tasks.clear()
            self.save_current_state()
        except Exception as e:
            QMessageBox.warning(self, self.tr("警告"),
                                self.tr("清空列表时出错: {}").format(str(e)))

    def move_up(self):
        current = self.file_list.currentRow()
        if current > 0:
            item = self.file_list.takeItem(current)
            self.file_list.insertItem(current - 1, item)
            self.file_list.setCurrentRow(current - 1)
            self.save_current_state()

    def move_down(self):
        current = self.file_list.currentRow()
        if current < self.file_list.count() - 1:
            item = self.file_list.takeItem(current)
            self.file_list.insertItem(current + 1, item)
            self.file_list.setCurrentRow(current + 1)
            self.save_current_state()

    def remove_selected(self):
        items = self.file_list.selectedItems()
        for item in items:
            try:
                file_path = item.text().split(" （")[0]  # Extract file path
                self.file_list.takeItem(self.file_list.row(item))
                if file_path in self.tasks:
                    del self.tasks[file_path]
                    cache_dir = os.path.join(self.out_dir_edit.text(), 'cache',
                                             os.path.splitext(os.path.basename(file_path))[0])
                    if os.path.exists(cache_dir):
                        shutil.rmtree(cache_dir, ignore_errors=False)
            except Exception as e:
                QMessageBox.warning(self, self.tr("警告"),
                                    self.tr("删除文件 {} 时出错: {}").format(file_path, str(e)))
                continue
        self.save_current_state()

    def browse_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, self.tr("选择批量合成保存路径"))
        if directory:
            self.out_dir_edit.setText(directory)
            self.save_current_state()

    def start_synthesis(self):
        if hasattr(self, '_last_click_time') and time.time() - self._last_click_time < 1.0:
            return
        self._last_click_time = time.time()
        if self.processing:
            self.processing = False
            self.clear_btn.setEnabled(True)
            self.move_up_btn.setEnabled(True)
            self.move_down_btn.setEnabled(True)
            self.remove_btn.setEnabled(True)
            for file_path, task in self.tasks.items():
                if task.status == 'processing':
                    self._update_item_status(file_path, self.tr("已暂停"), -1, 'blue', 0)
            self.synth_btn.setText(self.tr("开始合成"))
            return

        if not self.file_list.count():
            QMessageBox.warning(self, self.tr("警告"),
                                self.tr("请先添加需要合成的文件！"))
            return

        for file_path, task in self.tasks.items():
            if task.status == 'failed':
                task.status = 'pending'
                task.completed_segments = 0
                self._update_item_status(file_path, self.tr("准备合成"), -1, 'blue', 0)

        self.processing = True
        self.clear_btn.setEnabled(False)
        self.move_up_btn.setEnabled(False)
        self.move_down_btn.setEnabled(False)
        self.remove_btn.setEnabled(False)
        self.synth_btn.setText(self.tr("停止合成"))
        self.worker_thread = threading.Thread(target=self._synthesis_worker)
        self.worker_thread.start()

    def _synthesis_worker(self):
        for i in range(self.file_list.count()):
            if not self.processing:
                break

            file_path = self.file_list.item(i).text().split(" （")[0]
            if file_path not in self.tasks:
                continue

            task = self.tasks[file_path]
            if task.status == 'completed':
                continue

            task.status = 'processing'

            try:
                self._process_file(task)
                if not self.processing:
                    break
                task.status = 'completed'
            except Exception as e:
                if self.processing:
                    self.processing = False
                    QMessageBox.critical(self, self.tr("错误"),
                                         self.tr("处理文件时出错: {}: {}").format(file_path, str(e)))
                task.status = 'failed'
                break

        self.processing = False
        self.clear_btn.setEnabled(True)
        self.move_up_btn.setEnabled(True)
        self.move_down_btn.setEnabled(True)
        self.remove_btn.setEnabled(True)
        self.synth_btn.setText(self.tr("开始合成"))

    def _process_file(self, task: TTSTask):
        try:
            try:
                with open(task.file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                text = str(from_path(task.file_path).best())
                if not text:
                    raise Exception(
                        self.tr("无法检测文件编码。该文件可能不是文本文件或编码未知: {}").format(task.file_path))

            self.status_updated.emit(task.file_path, self.tr("正在准备文本..."), -1, 'blue', 0)
            base_name = os.path.splitext(os.path.basename(task.file_path))[0]
            cache_dir = os.path.join(self.out_dir_edit.text(), 'cache', base_name)
            os.makedirs(cache_dir, exist_ok=True)

            segment_size = self.config_manager.get_value('batch_tts_segment_size', 100)
            segments = self._split_text(text, segment_size)
            task.segment_count = len(segments)

            existing_txt_files = [f for f in os.listdir(cache_dir) if f.endswith('.txt')]
            if len(existing_txt_files) != task.segment_count:
                for f in os.listdir(cache_dir):
                    if f.endswith(('.txt', '.wav')):
                        os.remove(os.path.join(cache_dir, f))

                for i, segment in enumerate(segments):
                    segment_txt = os.path.join(cache_dir, f"{i:09d}.txt")
                    with open(segment_txt, 'w', encoding='utf-8') as f:
                        f.write(segment)

            existing_wav_files = [f for f in os.listdir(cache_dir) if f.endswith('.wav')]
            task.completed_segments = len(existing_wav_files)

            self.status_updated.emit(task.file_path, self.tr("正在合成"), -1, 'blue', 0)
            txt_files = sorted([f for f in os.listdir(cache_dir) if f.endswith('.txt')])
            for txt_file in txt_files:
                if not self.processing:
                    return

                segment_index = int(txt_file.split('.')[0])
                segment_wav = os.path.join(cache_dir, f"{segment_index:09d}.wav")

                if not os.path.exists(segment_wav):
                    segment_start_time = time.time()
                    segment_txt = os.path.join(cache_dir, txt_file)
                    with open(segment_txt, 'r', encoding='utf-8') as f:
                        segment_text = f.read()

                    response = self._synthesize_segment(segment_text)
                    if not response:
                        raise Exception(self.tr("合成片段 {} 失败，请检查API状态").format(segment_index))

                    with open(segment_wav, 'wb') as f:
                        f.write(response)

                    segment_end_time = time.time()
                    segment_time = segment_end_time - segment_start_time
                    task.average_time_per_segment = ((task.average_time_per_segment * task.completed_segments) + segment_time) / (task.completed_segments + 1)
                    task.completed_segments += 1

                remaining_time = task.average_time_per_segment * (task.segment_count - task.completed_segments)

                self.status_updated.emit(task.file_path, self.tr("正在合成"), task.progress, 'blue', remaining_time)

            if task.completed_segments == task.segment_count:
                output_format = self.format_combo.currentText()
                output_file = os.path.join(
                    self.out_dir_edit.text(),
                    f"{base_name}_{task.preset_name}.{output_format}"
                )
                self.status_updated.emit(task.file_path, self.tr("正在压制音频"), -1, 'blue', 0)
                self._merge_audio_files(cache_dir, output_file, output_format)
                self.status_updated.emit(task.file_path, self.tr("已完成"), 100, 'green', 0)

        except Exception as e:
            self.status_updated.emit(task.file_path, self.tr("失败"), -1, 'red', 0)
            raise Exception(self.tr("处理文件时出错: {}").format(str(e)))

    def _update_item_status(self, file_path: str, status: str, progress: int, color: str, remaining_time: float):
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.text().startswith(file_path):
                status_text = (f"{file_path} （{status}"
                               f"{f'：{progress}%' if progress != -1 else ''}"
                               f"{f' - 剩余时间：{int(remaining_time)}秒' if remaining_time > 0.5 else ''}"
                               f"）")
                item.setText(status_text)
                item.setForeground(QColor(
                    {'blue': '#0000FF', 'green': '#008000', 'red': '#FF0000', 'black': '#000000', 'gray': '#555555'}[
                        color]
                ))
                break

    def _split_text(self, text: str, segment_size: int) -> List[str]:
        if not text or segment_size <= 0:
            return []

        delimiters = ['。', '！', '？', '；', '\n', '.', '!', '?', ';']
        segments = []
        current_pos = 0

        while current_pos < len(text):
            min_end = current_pos + segment_size

            if min_end >= len(text):
                segment = text[current_pos:]
                if segment.strip():
                    segments.append(self._remove_empty_lines(segment))
                break

            delimiter_pos = -1
            search_pos = min_end

            while search_pos < len(text):
                for delimiter in delimiters:
                    pos = text.find(delimiter, search_pos, search_pos + 1)
                    if pos != -1:
                        delimiter_pos = pos + 1
                        break
                if delimiter_pos != -1:
                    break
                search_pos += 1

            if delimiter_pos == -1:
                delimiter_pos = len(text)

            segment = text[current_pos:delimiter_pos]
            if segment.strip():
                segments.append(self._remove_empty_lines(segment))

            current_pos = delimiter_pos

        return segments

    def _remove_empty_lines(self, text: str) -> str:
        lines = text.splitlines()
        non_empty_lines = [line for line in lines if line.strip()]
        return '\n'.join(non_empty_lines)

    def _synthesize_segment(self, text: str) -> Optional[bytes]:
        if not self.processing:  # Check if stopped
            return None

        base_url = self.url_edit.text().rstrip('/')
        preset_name = self.preset_combo.currentText()
        preset_config = self.config_manager.get_value('presets', {}).get(preset_name, {})

        # Prepare request parameters
        params = {
            'text': text,
            'text_lang': preset_config.get('text_lang', 'all_zh'),
            'ref_audio_path': preset_config.get('ref_audio_path', ''),
            'prompt_lang': preset_config.get('prompt_lang', 'all_zh'),
            'prompt_text': '' if preset_config.get('no_prompt', False) else preset_config.get('prompt_text', ''),
            'aux_ref_audio_paths': preset_config.get('aux_ref_audio_paths', []),
            'top_k': preset_config.get('top_k', 5),
            'top_p': preset_config.get('top_p', 1.0),
            'temperature': preset_config.get('temperature', 1.0),
            'text_split_method': preset_config.get('text_split_method', 'cut1'),
            'batch_size': preset_config.get('batch_size', 1),
            'batch_threshold': preset_config.get('batch_threshold', 0.75),
            'split_bucket': preset_config.get('split_bucket', True),
            'speed_factor': preset_config.get('speed_factor', 1.0),
            'streaming_mode': False,  # 强制关闭流式传输
            'seed': preset_config.get('seed', -1),
            'parallel_infer': preset_config.get('parallel_infer', True),
            'repetition_penalty': preset_config.get('repetition_penalty', 1.35)
        }

        gpt_model = preset_config.get('gpt_model')
        sovits_model = preset_config.get('sovits_model')

        for _ in range(3):  # Retry 3 times
            if not self.processing:
                return None
            try:
                if gpt_model != self.current_gpt:
                    if gpt_model:
                        requests.get(f"{base_url}/set_gpt_weights", params={'weights_path': gpt_model})
                    self.current_gpt = gpt_model

                if sovits_model != self.current_sovits:
                    if sovits_model:
                        requests.get(f"{base_url}/set_sovits_weights", params={'weights_path': sovits_model})
                    self.current_sovits = sovits_model

                response = requests.post(f"{base_url}/tts", json=params)

                if response.status_code == 200:
                    return response.content
                elif response.status_code == 400:
                    error = response.json()
                    raise Exception(self.tr("API错误: {}").format(error))

            except Exception as e:
                if not self.processing:  # Check if stopped
                    return None
                print(self.tr("合成尝试失败: {}").format(str(e)))
                time.sleep(1)

        return None

    def _merge_audio_files(self, cache_dir: str, output_file: str, format: str):
        # Create file list
        file_list = os.path.join(cache_dir, 'files.txt')
        with open(file_list, 'w', encoding='utf-8') as f:
            for audio in sorted(os.listdir(cache_dir)):
                if audio.endswith('.wav'):
                    f.write(f"file '{os.path.join(cache_dir, audio)}'\n")

        # Map formats to FFmpeg encoders
        format_encoders = {
            'wav': ['-c', 'copy'],
            'mp3': ['-c:a', 'libmp3lame'],
            'ogg': ['-c:a', 'libvorbis'],
            'aac': ['-c:a', 'aac'],
            'flac': ['-c:a', 'flac']
        }
        encoder_args = format_encoders.get(format, ['-c', 'copy'])

        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        process = subprocess.Popen(
            [
                'ffmpeg', '-f', 'concat', '-safe', '0',
                '-i', file_list, *encoder_args,
                '-y', output_file
            ],
            stdout=PIPE,
            stderr=PIPE,
            startupinfo=startupinfo,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )

        # Wait for completion and check for errors
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode,
                process.args,
                stdout,
                stderr
            )
