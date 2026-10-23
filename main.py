import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# Configuração da página
st.set_page_config(
    page_title="Detector Otimizado - YOLOv8", layout="centered"
)

st.title("🚀 Detector de Marketing Otimizado (YOLOv8)")
st.write(
    "Ajuste os parâmetros na barra lateral para refinar a detecção de"
    " **carros**, **celulares** e **casas/edificações**."
)

# --- Barra Lateral de Ajustes de Parâmetros ---
st.sidebar.header("⚙️ Ajustes de Parâmetros")
conf_threshold = st.sidebar.slider(
    "Confiança Mínima (conf)", 0.1, 0.9, 0.4, 0.05
)
iou_threshold = st.sidebar.slider(
    "Limiar de Supressão (iou)", 0.1, 0.9, 0.45, 0.05
)
img_size = st.sidebar.selectbox(
    "Resolução de Entrada (imgsz)", [416, 640, 960, 1280], index=1
)
usar_tta = st.sidebar.checkbox(
    "Ativar TTA (Test Time Augmentation)",
    value=False,
    help="Melhora a acurácia em objetos difíceis, mas consome mais CPU.",
)


# Carregamento otimizado do modelo com cache
@st.cache_resource
def carregar_modelo():
  return YOLO("yolov8n.pt")


model = carregar_modelo()

# Upload de arquivo pelo usuário
arquivo_imagem = st.file_uploader(
    "Escolha uma imagem (JPG, PNG)", type=["jpg", "jpeg", "png"]
)

if arquivo_imagem is not None:
  # Leitura eficiente dos bytes da imagem para o OpenCV
  bytes_imagem = arquivo_imagem.read()
  np_img = np.frombuffer(bytes_imagem, np.uint8)
  imagem = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

  # Mapeamento de classes do COCO de interesse
  classes_interesse = {2: "Carro", 67: "Celular", 62: "Casa/Edificio"}

  with st.spinner("Processando com hiperparâmetros ajustados..."):
    # Executa a inferência aplicando os novos parâmetros
    resultados = model(
        imagem,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=img_size,
        augment=usar_tta,
        classes=list(classes_interesse.keys()),
        verbose=False,
    )[0]

    # Renderização limpa das detecções
    for box in resultados.boxes:
      cls_id = int(box.cls[0])
      conf = float(box.conf[0])
      x1, y1, x2, y2 = map(int, box.xyxy[0])

      nome_classe = classes_interesse.get(cls_id, "Objeto")
      label = f"{nome_classe}: {conf:.2f}"

      # Desenhar box com espessura dinâmica baseada no tamanho da imagem
      cv2.rectangle(imagem, (x1, y1), (x2, y2), (0, 255, 0), 2)

      # Fundo do texto para legibilidade
      (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
      cv2.rectangle(imagem, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
      cv2.putText(
          imagem, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
      )

  # Conversão de BGR (OpenCV) para RGB (Streamlit)
  imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

  # Métricas rápidas na tela
  total_detectados = len(resultados.boxes)
  st.success(f"🔍 Total de objetos encontrados: {total_detectados}")

  st.image(
      imagem_rgb,
      caption="Imagem Otimizada com YOLOv8",
      use_container_width=True,
  )