# Rozpoznawanie Języka Migowego (ASL)

Serwis webowy API (FastAPI) do rozpoznawania liter amerykańskiego języka migowego (ASL) na podstawie zdjęć dłoni przy użyciu zoptymalizowanego modelu ONNX.

## Zakres zrealizowanych prac

1. **Model bazowy i Transfer Learning:** Wykorzystano architekturę MobileNetV2 z biblioteki PyTorch, dotrenowaną na zbiorze danych z Kaggle (ASL Alphabet).
2. **Optymalizacja (ONNX):** Model wyeksportowano do formatu `.onnx`, co uniezależniło inferencję od PyTorcha i przyspieszyło działanie aplikacji.
3. **Benchmark wydajności:** ONNX Runtime pozwolił na osiągnięcie 5.34-krotnego przyspieszenia czasu inferencji na CPU względem natywnego PyTorcha.
4. **Center Crop:** Zastosowano dynamiczne przycinanie zdjęć do kwadratu w fazie preprocessingu, co rozwiązało problem zniekształcania proporcji przy pionowych zdjęciach z telefonów.
5. **API i Interfejs:** Stworzono aplikację w FastAPI z prostym interfejsem użytkownika HTML obsługiwanym przez silnik Jinja2.
6. **Konteneryzacja:** Całość została skonteneryzowana za pomocą narzędzi Docker i Docker Compose w celu łatwego wdrażania.

## Struktura plików

```text
ASL_Recognition_PyTorch/
├── templates/
│   └── index.html          # Interfejs graficzny aplikacji (HTML)
├── app.py                  # Główny kod serwera FastAPI i preprocessing
├── asl_classifier.onnx     # Graf obliczeniowy modelu ONNX
├── asl_classifier.onnx.data # Wagi wyeksportowanego modelu ONNX
├── benchmark.py            # Skrypt porównujący wydajność PyTorch vs ONNX
├── docker-compose.yml      # Konfiguracja usług Docker Compose
├── Dockerfile              # Instrukcja budowania obrazu kontenera
├── export_onnx.py          # Skrypt konwertujący model .pth do .onnx
├── mapping.json            # Mapowanie indeksów numerycznych na litery ASL
├── mobilenet_v2_asl.pth    # Plik wag bazowego modelu PyTorch
├── requirements.txt        # Wykaz wymaganych bibliotek wraz z wersjami
├── train.py                # Skrypt treningowy modelu
└── README.md               # Dokumentacja projektu
```

## Technologie

* PyTorch, Torchvision, ONNX, ONNX Runtime
* Python, FastAPI, Uvicorn, Jinja2
* Pillow, NumPy
* Docker, Docker Compose

## Instrukcja uruchomienia lokalnego (bez Dockera)

Do uruchomienia wymagany jest zainstalowany Python (wersja 3.11 lub nowsza).

1. **Pobranie repozytorium:**
```bash
git clone <LINK_DO_TWOJEGO_REPOZYTORIUM>
cd ASL_Recognition_PyTorch
```

2. **Tworzenie i aktywacja środowiska wirtualnego:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

3. **Instalacja wymaganych zależności:**
```powershell
pip install -r requirements.txt
```

4. **Uruchomienie serwera aplikacji:**
```powershell
uvicorn app:app --reload
```

5. **Testowanie:**
Otwórz przeglądarkę internetową i przejdź pod adres: **http://127.0.0.1:8000**

## Instrukcja uruchomienia (Docker Compose)

Do uruchomienia wymagany jest zainstalowany i włączony program Docker Desktop.

1. **Uruchomienie aplikacji:**
W głównym folderze projektu wpisz:
```bash
docker compose up -d --build
```

2. **Testowanie:**
Otwórz przeglądarkę internetową i przejdź pod adres: **http://127.0.0.1:8000**

3. **Zatrzymywanie aplikacji:**
Aby zatrzymać działanie serwera i usunąć kontener, wpisz:
```bash
docker compose down
```
