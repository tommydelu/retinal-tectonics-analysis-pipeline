import { useState } from 'react'; // permette di definire un contenitore per un tipo di dato è una funzione da chiamare per aggionrare questo tipo di dato. useState() definisce il valore iniziale del contenitore


function LoadDataset(){

    const [files, setFiles] = useState([]);
    const [fase, setFase] = useState("loading")

    
    {/* quando clicco il bottone voglio chiamare questa funzione --> riempio formData e mando richiesta POST all'end point */}
    const handleUpload = async () => {
        const formData = new FormData();
            files.forEach( (file) => {
                formData.append('immagini',file)
            });
        
        // content type del formData impostato da solo
        const res = await fetch("http://127.0.0.1:5000/load_dataset",{method: 'POST', body: formData});
        console.log("dataset caricato con successo")
        setFase("processing")

    };

    if (fase == "loading"){
        return (
            
            <div className='title-container'>

                    <h1 className="titolo-principale"> ANALISI OPTICAL FLOW RETINICO </h1>
                    {/*onChange: quando succede questo evento triggera la funzione e (e) => {corpo funzione}*/}
                    <input multiple type="file" onChange={ (e) => {setFiles(Array.from(e.target.files))} }/>
                    {files.length != 0 && <p> Hai inserito {files.length} file </p>}

                    {files.length != 0 && <ul className='lista-file'> {files.map((file, index) => <li key={index}>{file.name}</li>)} </ul>}

                    {files.length != 0 && <button className='bottone-carica' onClick={handleUpload}>Carica</button> } {/*Carica è il testo che appare dentro il bottone */}
                    
            </div>

        );
    }

    if (fase == "processing"){
        return(
            <p> CIAO SIAMO NEL PROCESSING</p>

        );


    }

}




export default LoadDataset;
