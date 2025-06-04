# YOLOv8 Video Tracker

Projeto profissional desenvolvido sob demanda, com foco em **detecção e rastreamento de objetos em vídeo** utilizando **YOLOv8** e **Deep SORT**.

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/notfarchi/track_counts
cd track_count
```
### 2. (Opcional) Crie e ative um ambiente virtual
```bash
# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate
```
### 3. Instale as dependências
```bash
pip install ultralytics opencv-python deep_sort_realtime
```
### Como Usar
1. Adicione o vídeo de entrada
   
Coloque seu vídeo na raiz do projeto com o nome video.mp4.

Importante: o arquivo de entrada deve obrigatoriamente se chamar video.mp4. Outros nomes ou formatos não serão reconhecidos pelo script.

2. Execute o script principal
```bash
python track.py
```
3. Verifique o resultado
   
O resultado será salvo automaticamente no arquivo:
```bash
resultado_contagem.txt
```
Atenção: Sempre renomeie o vídeo de entrada para video.mp4 antes de executar o script.
