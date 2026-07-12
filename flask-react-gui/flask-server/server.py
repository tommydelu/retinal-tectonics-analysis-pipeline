import os

from flask import Flask, request, jsonify
from common.paths import PROJECT_ROOT

app = Flask(__name__)

path = os.path.join(PROJECT_ROOT,'flask-react-gui','flask-server','loaded_imgs')
os.makedirs(path, exist_ok=True)

# 1) From the front end, ideally, a clinician click a botton to load some images from its local files. Frontend fetch a request to this end point and wait for an answer
@app.route("/load_dataset", methods=['POST'])    # riceve file e li carica --> azione fatta --> metodo POST
def load_dataset():

    # 'immagini' corrisponde alla chiave usata dal front end
    images = request.files.getlist('immagini') # usato per ottenere una lista di tutti i file caricati (ogni elemento della lista è un oggetto FileStorage con attributi come .filename, .save(percorso) e .content_type)
    fname_list = []
    for img in images:
        img_path = os.path.join(path,img.filename)
        img.save(img_path)
        fname_list.append(img.filename)


    return jsonify({"state": "Immagini salvate con successo!", "filenames": fname_list})




# nell'URL chiamato dal frontend potrei specificare il nome dell'img su cui eseguire le azioni --> diventa un parametro della funzione eseguita
@app.route("/analyze_image/<file_name>", methods=['POST'])   # riceve immagine e la elabora con funzioni opencv --> POST
def analyze_img(file_name):
    pass

@app.route("/compute_optical_flow",methods=['POST'])     # calcola optical flow sulle immagini --> POST
def compute_optical_flow():
    pass

@app.route("/download_results", methods=['GET']) # deve solo mandare i risultati al front end, no modifiche --> GET
def download_results():
    pass


if __name__ == '__main__':

    app.run(debug=True)


