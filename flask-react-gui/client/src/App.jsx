import { useState } from 'react';


function LoadDataset(){

    const [files, setFiles] = useState([]);

    
    {/* quando clicco il bottone voglio chiamare questa funzione --> riempio formData e mando richiesta POST all'end point */}
    const handleUpload = async () => {
        const formData = new FormData();
            files.forEach( (file) => {
                formData.append('immagini',file)
            });
        
        // content type del formData impostato da solo
        const res = await fetch("http://127.0.0.1:5000/load_dataset",{method: 'POST', body: formData});
        console.log("dataset caricato con successo")

    };

    return (
        
        <div>

                <h1 className="titolo-principale"> Analisi optical flow retinico </h1>
                {/*onChange: quando succede questo evento triggera la funzione e (e) => {corpo funzione}*/}
                <input type="file" onChange={ (e) => {setFiles(Array.from(e.target.files))} }/>
                <button onClick={handleUpload}>Carica</button> {/*Carica è il testo che appare dentro il bottone */}
                
        </div>

    );

}

export default LoadDataset;
