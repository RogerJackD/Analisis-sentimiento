let model = null;
let metadata = null;
let wordIndex = null;
let maxSequenceLength = 9; // valor por defecto en caso de fallo

// Función para cargar el modelo, metadatos y vocabulario
async function loadResources() {
    try {
        // Cargar el modelo TensorFlow.js
        model = await tf.loadLayersModel('model_sentiment_tfjs/model.json');

        // Cargar metadatos del modelo
        const metadataResponse = await fetch('model_sentiment_tfjs/metadata.json');
        metadata = await metadataResponse.json();

        if (metadata && metadata.max_sequence_length) {
            maxSequenceLength = metadata.max_sequence_length;
        } else {
            console.warn("No se encontró max_sequence_length en metadata.json. Usando valor por defecto:", maxSequenceLength);
        }

        // Cargar el vocabulario (word_index)
        const wordIndexResponse = await fetch('word_index.json');
        wordIndex = await wordIndexResponse.json();

        console.log("Modelo, metadatos y vocabulario cargados correctamente");
        console.log("Longitud máxima de secuencia:", maxSequenceLength);

        return true;
    } catch (error) {
        console.error("Error al cargar recursos:", error);
        return false;
    }
}

// Función para preprocesar el texto de entrada
function preprocessText(text) {
    const cleanedText = text.toLowerCase().replace(/[.,?!]/g, '').trim();
    const tokens = cleanedText.split(/\s+/);
    const sequence = tokens.map(token => wordIndex[token] || 0);

    if (sequence.length < maxSequenceLength) {
        const padding = Array(maxSequenceLength - sequence.length).fill(0);
        return padding.concat(sequence);
    } else {
        return sequence.slice(-maxSequenceLength);
    }
}

// Función principal para predecir el sentimiento
async function predictSentiment() {
    const inputElement = document.getElementById('text-input');
    const resultElement = document.getElementById('prediction-result');
    const text = inputElement.value.trim();

    if (!text) {
        resultElement.textContent = "Por favor ingresa un texto";
        return;
    }

    try {
        resultElement.textContent = "Procesando...";

        const sequence = preprocessText(text);
        console.log("Secuencia generada:", sequence);

        if (sequence.length !== maxSequenceLength) {
            console.warn(`La secuencia generada tiene longitud ${sequence.length}, pero se esperaba ${maxSequenceLength}`);
        }

        const inputTensor = tf.tensor2d([sequence], [1, maxSequenceLength]);

        const prediction = model.predict(inputTensor);
        const score = prediction.dataSync()[0];

        inputTensor.dispose();
        prediction.dispose();

        const sentiment = score >= 0.5 ? "POSITIVO 😊" : "NEGATIVO 😞";
        const confidence = (score >= 0.5 ? score : 1 - score) * 100;

        resultElement.innerHTML = `
            <p>Texto: "${text}"</p>
            <p>Sentimiento: <strong>${sentiment}</strong></p>
            <p>Confianza: ${confidence.toFixed(2)}%</p>
            <p>Puntuación: ${score.toFixed(4)}</p>
        `;

    } catch (error) {
        console.error("Error en la predicción:", error);
        resultElement.textContent = "Error al procesar el texto. Por favor intenta nuevamente.";
    }
}

// Inicializar la aplicación
async function initApp() {
    const statusElement = document.getElementById('model-status');
    statusElement.textContent = "Cargando modelo...";

    const success = await loadResources();

    if (success) {
        statusElement.textContent = "Modelo cargado correctamente ✅";
        document.getElementById('predict-button').disabled = false;

        // Conectar el botón a la función predictiva
        document.getElementById('predict-button').addEventListener('click', predictSentiment);
    } else {
        statusElement.textContent = "Error al cargar el modelo ❌";
    }
}

// Ejecutar al cargar la página
document.addEventListener('DOMContentLoaded', initApp);
