function updateTemp() {
    const inputC = document.getElementById("tempInputC");
    const mercury = document.querySelector('.mercury');
    
    // Valores límite
    const MIN_TEMP = -273.15;
    const MAX_TEMP = 100;
    const RANGE = MAX_TEMP - MIN_TEMP;   // 373.15

    // Validar entrada
    let temp = parseFloat(inputC.value) || 0;
    temp = Math.max(MIN_TEMP, Math.min(MAX_TEMP, temp)); 

    // Cálculo de altura (0% a 66%)
    const height = ((temp - MIN_TEMP) / RANGE) * 100;
    mercury.style.height = `${height}%`;

    // Cambiar color según la temperatura
    const perc = temp / 100;
    const colorIni = '#9FD7F9';
    const red = Math.round(255 * perc);
    const blue = Math.round(255 * (1 - perc));

    // Colores para el gradiente
    const colorFin = `rgb(${red * 0.8}, 0, ${blue * 0.8})`;
    mercury.style.background = `linear-gradient(to top, ${colorIni}, ${colorFin})`;
}

function updateFahr(celsius, fahrenheit) {
    const mercury = document.querySelector('.mercury-f');
    const tempDisplay = document.getElementById("currentTemp");
    
    // Valores límite 
    const MIN_TEMP = -459.67;
    const MAX_TEMP = 212;
    const RANGE = MAX_TEMP - MIN_TEMP;   // 671.67

    // Cálculo de altura (0% a 63%)
    const height = ((fahrenheit - MIN_TEMP) / RANGE) * 100;
    const heightLimit = Math.max(0, Math.min(100, height));
    mercury.style.height = `${heightLimit}%`;

    // Calcular color
    const perc = fahrenheit / 100;
    const colorIni = '#9FD7F9';
    const red = Math.round(255 * perc);
    const blue = Math.round(255 * (1 - perc));

    // Colores para el gradiente
    const colorFin = `rgb(${red * 0.8}, 0, ${blue * 0.8})`;
    mercury.style.background = `linear-gradient(to top, ${colorIni}, ${colorFin})`;

    console.log(`C: ${celsius.toFixed(2)} | F: ${fahrenheit.toFixed(2)}`);
    tempDisplay.textContent = `${fahrenheit.toFixed(2)} °F`;
}

// -------------------------- MODELO -------------------------- //
let model;

// Función para crear y entrenar el modelo
async function trainModel() {
    // Datos de entrenamiento como tensores unidimensionales
    const xCelcius = tf.tensor1d([-40, -30, -20, -10, 0, 10, 20, 30, 40]);
    const yFahrenheit = tf.tensor1d([-40, -22, -4, 14, 32, 50, 68, 86, 104]);

    // Configuración del modelo
    model = tf.sequential({
        layers: [
            tf.layers.dense({units: 3, inputShape: [1]}),
            tf.layers.dense({units: 3}),
            tf.layers.dense({units: 1})
        ]
    });

    // Compilar el modelo
    model.compile({
        optimizer: tf.train.adam(0.5),
        loss: 'meanSquaredError'
    });

    // Entrenamiento
    console.log("Entrenando modelo...");
    await model.fit(xCelcius, yFahrenheit, {
        epochs: 1000,
        verbose: 0,
    });
    console.log("Modelo entrenado!!");
}

// Función de predicción
async function predictTemp() {
    const inputC = parseFloat(document.getElementById("tempInputC").value);
    const mercuryF = document.querySelector('.mercury-f');
    const mercuryC = document.querySelector('.mercury');
    const temp = document.getElementById("currentTemp");

    if(isNaN(inputC)) {
        console.log("No hay valor")
        mercuryF.style.height = `0%`;
        mercuryC.style.height = `0%`;
        temp.textContent = "°F";
        return;
    }

    // Gestiona automáticamente la memoria de los tensores temporales
    const prediction = tf.tidy(() => {
        const inputTensor = tf.tensor1d([inputC]);
        return model.predict(inputTensor);
    });

    const fahrenheit = (await prediction.data())[0];
    prediction.dispose();

    updateFahr(inputC, fahrenheit);
}

document.addEventListener('DOMContentLoaded', async () => {
    await trainModel();
});