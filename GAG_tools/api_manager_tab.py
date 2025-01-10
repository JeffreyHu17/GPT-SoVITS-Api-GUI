import os
import io
import re
import psutil
import logging
import subprocess
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTextEdit, QCheckBox, QFileDialog,
    QGroupBox, QMessageBox
)
from PyQt5.QtGui import QTextCursor, QTextCharFormat, QColor, QFont
from GAG_tools.config_manager import ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessOutputReader(QThread):
    output_received = pyqtSignal(str, bool)  # text, is_progress_update

    def __init__(self, process):
        super().__init__()
        self.process = process
        self.is_running = True

    def run(self):
        text_stream = io.TextIOWrapper(
            self.process.stdout,
            encoding='utf-8',
            errors='replace',
            line_buffering=True
        )

        progress_pattern = re.compile(r'^\r.*it/s\]|.*it/s\]')

        while self.is_running and self.process:
            try:
                line = text_stream.readline()
                if not line:
                    if self.process.poll() is not None:
                        break
                    continue

                line = line.rstrip()
                is_progress = bool(progress_pattern.search(line))
                self.output_received.emit(line, is_progress)

            except UnicodeDecodeError:
                # If UTF-8 fails, try GBK
                text_stream = io.TextIOWrapper(
                    self.process.stdout,
                    encoding='gbk',
                    errors='replace',
                    line_buffering=True
                )
            except Exception as e:
                self.output_received.emit(f'Error reading output: {str(e)}', False)

    def stop(self):
        self.is_running = False


class APIManager(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.last_was_progress = False
        self.config_manager = ConfigManager()
        self.process = None
        self.output_reader = None
        self.is_running = False
        self.initUI()

        # Autostart if configured
        if self.config_manager.get_value('autostart_api'):
            self.start_api()

    def initUI(self):
        self.setGeometry(100, 100, 800, 800)
        layout = QVBoxLayout(self)

        # API URL Group
        url_group = QGroupBox(self.tr("API 配置"))
        url_layout = QVBoxLayout()

        # URL input
        url_input_layout = QHBoxLayout()
        url_label = QLabel(self.tr("API URL:"))
        self.url_input = QLineEdit(self.config_manager.get_value('api_url'))
        url_input_layout.addWidget(url_label)
        url_input_layout.addWidget(self.url_input)
        url_layout.addLayout(url_input_layout)

        # Python path selection
        python_path_layout = QHBoxLayout()
        python_path_label = QLabel(self.tr("Python 环境:"))
        self.python_path_input = QLineEdit(self.config_manager.get_value('python_path'))
        self.python_path_input.setReadOnly(True)
        python_path_button = QPushButton(self.tr("浏览"))
        python_path_button.clicked.connect(self.browse_python_path)
        python_path_layout.addWidget(python_path_label)
        python_path_layout.addWidget(self.python_path_input)
        python_path_layout.addWidget(python_path_button)
        url_layout.addLayout(python_path_layout)

        # Autostart checkbox
        self.autostart_checkbox = QCheckBox(self.tr("随软件启动API"))
        self.autostart_checkbox.setChecked(self.config_manager.get_value('autostart_api'))
        url_layout.addWidget(self.autostart_checkbox)

        url_group.setLayout(url_layout)
        layout.addWidget(url_group)

        # Control buttons
        control_layout = QHBoxLayout()
        self.start_button = QPushButton(self.tr("启动 API"))
        self.start_button.setStyleSheet("min-height: 30px;")
        self.stop_button = QPushButton(self.tr("停止 API"))
        self.stop_button.setStyleSheet("min-height: 30px;")
        self.start_button.clicked.connect(self.start_api)
        self.stop_button.clicked.connect(self.stop_api)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)

        # Output console
        output_label = QLabel(self.tr("控制台输出"))
        layout.addWidget(output_label)
        self.output_console = QTextEdit()
        self.output_console.setReadOnly(True)
        self.output_console.setStyleSheet("""
                QTextEdit {
                    background-color: #1e1e1e;
                    color: #ffffff;
                    font-family: "Microsoft YaHei Mono", Consolas, Monaco, monospace;
                }
            """)
        layout.addWidget(self.output_console)

        layout.addLayout(control_layout)

    def process_inference_output(self, text, is_progress_update):
        cursor = self.output_console.textCursor()
        cursor.movePosition(QTextCursor.End)

        if hasattr(self, 'last_was_progress'):
            if not self.last_was_progress and is_progress_update:
                pass
            elif self.last_was_progress and not is_progress_update:
                text = '\n' + text
        self.last_was_progress = is_progress_update

        if is_progress_update:
            cursor.movePosition(QTextCursor.StartOfLine, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
            self.update_output(text, color='#ffa500', auto_newline=False)
        else:
            if "info:" in text.lower():
                self.update_output(text, color='green')
            elif "debug:" in text.lower():
                self.update_output(text, color='#87CEEB')
            elif "error:" in text.lower():
                self.update_output(text, color='red')
            elif "warning:" in text.lower():
                self.update_output(text, color='yellow')
            else:
                self.update_output(text)

    def update_output(self, text, color='white', bold=False, italic=False, auto_newline=True):
        cursor = self.output_console.textCursor()
        format = QTextCharFormat()
        format.setForeground(QColor(color))
        if bold:
            format.setFontWeight(QFont.Bold)
        if italic:
            format.setFontItalic(True)
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text + ('\n' if auto_newline else ''), format)
        self.output_console.setTextCursor(cursor)
        self.output_console.ensureCursorVisible()

    def print_separator(self, char='-', count=50):
        self.update_output(char * count)

    def browse_python_path(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("选择 Python Executable"),
            os.path.dirname(self.python_path_input.text()),
            "Python Executable (python.exe)"
        )
        if file_path:
            self.python_path_input.setText(file_path)
            self.config_manager.set_value('python_path', file_path)

    def start_api(self):
        if self.is_running:
            return

        python_path = self.python_path_input.text()
        self.output_console.clear()
        if not os.path.exists(python_path):
            QMessageBox.warning(
                self,
                self.tr("错误"),
                self.tr("指定的Python环境不存在.")
            )
            return

        try:
            # Add Python directory to PATH
            env = os.environ.copy()
            python_dir = os.path.dirname(python_path)
            env["PATH"] = python_dir + os.pathsep + env.get("PATH", "")

            # Start the API process
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            # Create process
            self.process = subprocess.Popen(
                [python_path, "api_v2.py", "-a", "127.0.0.1", "-p", "9880"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env={**env, "PYTHONIOENCODING": "utf-8"},
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                startupinfo=startupinfo
            )

            # Start output reader
            self.output_reader = ProcessOutputReader(self.process)
            self.output_reader.output_received.connect(self.process_inference_output)
            self.output_reader.start()

            self.is_running = True
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)

            logger.info("API process started")
            self.update_output("API process started...", color='#90EE90', bold=True)

        except Exception as e:
            QMessageBox.critical(
                self,
                self.tr("错误"),
                self.tr("尝试启动API时出错: ") + str(e)
            )
            logger.error(f"Failed to start API process: {e}")

    def stop_api(self):
        if not self.is_running:
            return

        try:
            self.terminate_process()
            self.is_running = False
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

            if self.output_reader:
                self.output_reader.stop()
                self.output_reader = None

            logger.info("API process stopped")
            self.update_output("API process stopped.", color='#90EE90', bold=True)

        except Exception as e:
            QMessageBox.critical(
                self,
                self.tr("错误"),
                self.tr("尝试停止API时出错: ") + str(e)
            )
            logger.error(f"Failed to stop API process: {e}")

    def terminate_process(self):
        logger.info("Terminating API process")
        if self.process:
            try:
                parent = psutil.Process(self.process.pid)
                children = parent.children(recursive=True)
                for child in children:
                    child.terminate()
                parent.terminate()
                gone, still_alive = psutil.wait_procs(children + [parent], timeout=5)
                for p in still_alive:
                    p.kill()
            except psutil.NoSuchProcess:
                pass
            self.process = None

    def cleanup(self):
        # Save current configuration
        updates = {
            'api_url': self.url_input.text(),
            'autostart_api': self.autostart_checkbox.isChecked()
        }
        self.config_manager.update_config(updates)

        # Stop API if running
        if self.is_running:
            self.stop_api()

    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)


# For testing purposes
if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    window = APIManager()
    window.show()
    sys.exit(app.exec_())