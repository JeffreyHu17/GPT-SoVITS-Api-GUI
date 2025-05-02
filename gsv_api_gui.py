import sys
import os

from PyQt5.QtGui import QPixmap, QPainter, QColor, QIcon
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout,
                             QTabWidget, QLabel)
from PyQt5.QtCore import Qt, QTranslator, QLocale
import tempfile
import qdarktheme
from GAG_tools.tts_gui_tab import TTSGUI
from GAG_tools.api_manager_tab import APIManager
from GAG_tools.batch_tts_tab import BatchTTS
import resources_rc


def get_language():
    locale = QLocale.system().name()
    if locale.startswith('zh'):
        return 'zh'  # Chinese (Simplified and Traditional)
    else:
        return 'en'  # English (default for all other languages)


def get_base_path():
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # PyInstaller
        print('at PyInstaller')
        return sys._MEIPASS
    elif hasattr(get_base_path, '__compiled__'):
        # Nuitka
        print('at Nuitka')
        return os.path.dirname(os.path.abspath(__file__))
    else:
        # dev
        print('at dev')
        return ''

def get_translator_path():
    language = get_language()
    base_path = get_base_path()

    possible_paths = [
        os.path.join(base_path, "translations", f"GAG_{language}.qm"),
        os.path.join(base_path, f"GAG_{language}.qm"),
    ]

    for translations_path in possible_paths:
        print(f"Attempting to load translations from: {translations_path}")
        if os.path.exists(translations_path):
            print("translations found!")
            return translations_path

    print("Failed to load translator")

def remove_screen_splash():
    if "NUITKA_ONEFILE_PARENT" in os.environ:
        splash_filename = os.path.join(
            tempfile.gettempdir(),
            "onefile_%d_splash_feedback.tmp" % int(os.environ["NUITKA_ONEFILE_PARENT"]),
        )

        if os.path.exists(splash_filename):
            os.unlink(splash_filename)


class GSVApiGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setWindowIcon(QIcon(":/GAG_tools/images/GAG_icon.ico"))
        self.setStyleSheet("""     
                    QPushButton {
                        background-color: #42A5F5;
                        min-height: 15px;
                        color: white;
                    }

                    QPushButton:hover {
                        background-color: #1E88E5;
                    }

                    QPushButton:pressed {
                        background-color: #1565C0;
                    }

                    QPushButton:disabled {
                        background-color: #BDBDBD;
                        color: #757575;
                    }      
                         
                    QTabWidget::tab-bar {
                        alignment: left;
                    }

                    QTabBar::tab {
                        background-color: #D3DDE6;
                        color: #424242;
                        padding: 8px 16px;
                        margin-right: 2px;
                        border-top-left-radius: 4px;
                        border-top-right-radius: 4px;
                    }

                    QTabBar::tab:selected {
                        background-color: #2196F3;
                        color: white;
                    }

                    QTabBar::tab:hover:!selected {
                        background-color: #90CAF9;
                    }
                    
                    QGroupBox {background-color: rgba(255, 255, 255, 220);font-weight: Normal;}                 
                    QTextEdit{background-color: rgba(255, 255, 255, 220);}            
                    QHBoxLayout{background-color: rgba(255, 255, 255, 220);}
                    QLineEdit{background-color: rgba(255, 255, 255, 220);}
                    QListWidget{background-color: rgba(255, 255, 255, 220);}
                    QHBoxLayout{background-color: rgba(255, 255, 255, 220);}
                    QVBoxLayout{background-color: rgba(255, 255, 255, 220);}
                    QTabWidget{background-color: rgba(255, 255, 255, 220);}
                """)

    def initUI(self):
        # Get the primary screen scaling_factor
        screen = QApplication.primaryScreen()
        dpi = screen.logicalDotsPerInch()
        scaling_factor = max(1.0, dpi / 96.0)  # assuming 96 DPI as the base

        # Scale the window size
        base_width, base_height = 800, 800
        scaled_width = int(base_width * scaling_factor)
        scaled_height = int(base_height * scaling_factor)
        self.setWindowTitle(self.tr('GSV Api GUI v0.4.0   by  领航员未鸟'))
        self.setGeometry(100, 100, scaled_width, scaled_height)

        # Scale the font, in theory this is redundant but there are strange special cases
        font = QApplication.font()
        font.setPointSize(int(font.pointSize()))
        QApplication.setFont(font)

        main_layout = QVBoxLayout()

        # Create a tab
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Main function tab
        self.tts_gui_tab = TTSGUI()
        self.tab_widget.addTab(self.tts_gui_tab, self.tr("语音合成"))

        # API management tab
        self.api_manager_tab = APIManager()
        self.tab_widget.addTab(self.api_manager_tab, self.tr("API管理"))
        self.set_tab_background(self.api_manager_tab, ":/GAG_tools/images/GAG_background.png")

        # Batch TTS tab
        self.batch_tts_tab = BatchTTS()
        self.tab_widget.addTab(self.batch_tts_tab, self.tr("批量合成"))
        self.set_tab_background(self.batch_tts_tab, ":/GAG_tools/images/GAG_background.png")

        self.setLayout(main_layout)

    def set_tab_background(self, tab_widget, image_path):
        background = QPixmap(image_path)
        if background.isNull():
            print(f"Failed to load background image: {image_path}")
            return

        overlay = QPixmap(background.size())
        overlay.fill(QColor(255, 255, 255, 128))

        painter = QPainter(background)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        painter.drawPixmap(0, 0, overlay)
        painter.end()

        background_label = QLabel(tab_widget)
        background_label.setPixmap(background)
        background_label.setScaledContents(True)
        background_label.resize(tab_widget.size())
        background_label.lower()

        background_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        background_label.setStyleSheet("background-color: transparent;")

        background_label.lower()

    def closeEvent(self, event):
        if hasattr(self, 'tts_gui_tab'):
            self.tts_gui_tab.cleanup()
        if hasattr(self, 'batch_tts_tab'):
            self.batch_tts_tab.cleanup()
        if hasattr(self, 'api_manager_tab'):
            self.api_manager_tab.cleanup()
        super().closeEvent(event)

if __name__ == '__main__':
    qdarktheme.enable_hi_dpi()
    app = QApplication(sys.argv)

    translator = QTranslator()
    if translator.load(get_translator_path()):
        app.installTranslator(translator)
    else:
        print('Use default')

    ex = GSVApiGUI()
    qdarktheme.setup_theme("light")
    remove_screen_splash()
    ex.show()
    sys.exit(app.exec_())
