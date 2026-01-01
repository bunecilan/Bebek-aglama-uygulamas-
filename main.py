
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.core.audio import SoundLoader
from android.permissions import request_permissions, Permission
import numpy as np
import librosa
from scipy import signal
import threading
import queue

request_permissions([Permission.RECORD_AUDIO, Permission.WRITE_EXTERNAL_STORAGE])

class BabyCryAnalyzer:
    def __init__(self):
        # Dunstan metodu - 5 ağlama tipi
        self.cry_types = {
            'neh': {'name': 'ACIKMIS', 'freq_range': (250, 400), 'desc': '🍼 Bebeğiniz aç'},
            'owh': {'name': 'UYKUSU VAR', 'freq_range': (200, 300), 'desc': '😴 Bebeğiniz uyumak istiyor'},
            'heh': {'name': 'RAHATSIZ', 'freq_range': (300, 450), 'desc': '😣 Bez değiştirin veya pozisyon'},
            'eairh': {'name': 'GAZI VAR', 'freq_range': (350, 500), 'desc': '💨 Bebeğinizin gazı var'},
            'eh': {'name': 'GAZ CIKARACAK', 'freq_range': (280, 380), 'desc': '🤱 Bebeği gazını çıkarması için tutun'}
        }
        self.buffer_size = 2048
        self.sample_rate = 16000  # Düşük sample rate = hızlı işleme
        self.audio_queue = queue.Queue(maxsize=5)

    def extract_features_fast(self, audio_data):
        """Hızlı özellik çıkarma - minimum gecikme"""
        try:
            # Zero Crossing Rate (çok hızlı)
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data))

            # RMS Energy (çok hızlı)
            rms = np.sqrt(np.mean(audio_data**2))

            # Dominant frekans (FFT ile)
            fft = np.fft.rfft(audio_data)
            freqs = np.fft.rfftfreq(len(audio_data), 1/self.sample_rate)
            magnitude = np.abs(fft)
            dominant_freq = freqs[np.argmax(magnitude)]

            # Spectral Centroid (hızlı)
            spec_cent = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate))

            return {
                'zcr': zcr,
                'rms': rms,
                'dominant_freq': dominant_freq,
                'spectral_centroid': spec_cent
            }
        except:
            return None

    def analyze_cry_type(self, features):
        """Ağlama tipini analiz et"""
        if not features or features['rms'] < 0.01:
            return None, "🔇 Ses algılanmadı"

        dominant_freq = features['dominant_freq']

        # Frekans aralığına göre sınıflandırma
        best_match = None
        min_distance = float('inf')

        for cry_type, info in self.cry_types.items():
            freq_min, freq_max = info['freq_range']

            if freq_min <= dominant_freq <= freq_max:
                distance = abs(dominant_freq - (freq_min + freq_max) / 2)
                if distance < min_distance:
                    min_distance = distance
                    best_match = cry_type

        if best_match:
            return best_match, self.cry_types[best_match]['desc']

        # Yakın eşleşme bul
        for cry_type, info in self.cry_types.items():
            freq_min, freq_max = info['freq_range']
            center = (freq_min + freq_max) / 2
            distance = abs(dominant_freq - center)

            if distance < min_distance and distance < 100:
                min_distance = distance
                best_match = cry_type

        if best_match:
            return best_match, self.cry_types[best_match]['desc']

        return None, f"📊 Analiz ediliyor... (Frekans: {dominant_freq:.0f} Hz)"

class BabyCryApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.analyzer = BabyCryAnalyzer()
        self.is_listening = False
        self.audio_thread = None

    def build(self):
        self.layout = BoxLayout(orientation='vertical', padding=20, spacing=15)

        # Başlık
        title = Label(
            text='👶 Bebek Ağlama Analizi',
            font_size='24sp',
            size_hint=(1, 0.15),
            bold=True
        )

        # Durum göstergesi
        self.status_label = Label(
            text='Başlatmak için butona basın',
            font_size='18sp',
            size_hint=(1, 0.2)
        )

        # Sonuç etiketi
        self.result_label = Label(
            text='',
            font_size='32sp',
            size_hint=(1, 0.3),
            bold=True
        )

        # Açıklama etiketi
        self.desc_label = Label(
            text='',
            font_size='20sp',
            size_hint=(1, 0.2)
        )

        # Buton
        self.toggle_btn = Button(
            text='🎤 Dinlemeyi Başlat',
            font_size='20sp',
            size_hint=(1, 0.15),
            background_color=(0.2, 0.8, 0.2, 1)
        )
        self.toggle_btn.bind(on_press=self.toggle_listening)

        self.layout.add_widget(title)
        self.layout.add_widget(self.status_label)
        self.layout.add_widget(self.result_label)
        self.layout.add_widget(self.desc_label)
        self.layout.add_widget(self.toggle_btn)

        return self.layout

    def toggle_listening(self, instance):
        if not self.is_listening:
            self.start_listening()
        else:
            self.stop_listening()

    def start_listening(self):
        self.is_listening = True
        self.toggle_btn.text = '⏸ Durdur'
        self.toggle_btn.background_color = (0.8, 0.2, 0.2, 1)
        self.status_label.text = '🎤 Dinleniyor...'

        # Gerçek zamanlı dinleme başlat
        Clock.schedule_interval(self.process_audio, 0.5)  # Her 0.5 saniyede bir analiz

    def stop_listening(self):
        self.is_listening = False
        self.toggle_btn.text = '🎤 Dinlemeyi Başlat'
        self.toggle_btn.background_color = (0.2, 0.8, 0.2, 1)
        self.status_label.text = 'Durduruldu'
        Clock.unschedule(self.process_audio)

    def process_audio(self, dt):
        """Ses işleme - gerçek zamanlı"""
        try:
            # Android mikrofon erişimi simülasyonu
            # Gerçek uygulamada PyAudio veya Android API kullanılır

            # Simüle edilmiş ses verisi (test için)
            duration = 0.5  # 500ms
            samples = int(self.analyzer.sample_rate * duration)

            # Gerçek uygulamada mikrofon verisini al
            # audio_data = microphone.read(samples)

            # Test verisi - rastgele ağlama benzeri ses
            t = np.linspace(0, duration, samples)
            freq = np.random.choice([280, 300, 350, 400, 280])  # Dunstan frekansları
            audio_data = 0.3 * np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(samples)

            # Özellikleri çıkar
            features = self.analyzer.extract_features_fast(audio_data)

            if features:
                # Ağlama tipini tespit et
                cry_type, description = self.analyzer.analyze_cry_type(features)

                # UI güncelle
                if cry_type:
                    self.result_label.text = self.analyzer.cry_types[cry_type]['name']
                    self.desc_label.text = description
                    self.result_label.color = (1, 0.3, 0.3, 1)  # Kırmızı
                else:
                    self.result_label.text = '⏳ Analiz ediliyor...'
                    self.desc_label.text = description
                    self.result_label.color = (0.5, 0.5, 0.5, 1)  # Gri

        except Exception as e:
            self.status_label.text = f'Hata: {str(e)}'

    def on_stop(self):
        self.stop_listening()

if __name__ == '__main__':
    BabyCryApp().run()
