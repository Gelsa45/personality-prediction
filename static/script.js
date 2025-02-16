document.addEventListener("DOMContentLoaded", function () {
    const aiButton = document.getElementById("ai-button");
    const personalityButton = document.getElementById("personality-button");
    const inputText = document.getElementById("input-text");
    const resultDiv = document.getElementById("result");

    if (!aiButton || !personalityButton || !inputText || !resultDiv) {
        console.error("One or more elements are missing. Check your HTML structure.");
        return;
    }

    aiButton.addEventListener("click", function () {
        fetch("/predict_ai", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: inputText.value }),
        })
        .then(response => response.json())
        .then(data => {
            resultDiv.innerHTML = `AI Text Prediction: ${data.prediction}`;
        })
        .catch(error => {
            console.error("Error:", error);
            resultDiv.innerHTML = "Error detecting AI text!";
        });
    });

    personalityButton.addEventListener("click", function () {
        fetch("/predict_personality", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: inputText.value }),
        })
        .then(response => response.json())
        .then(data => {
            resultDiv.innerHTML = `Personality Prediction: ${data.personality}`;
        })
        .catch(error => {
            console.error("Error:", error);
            resultDiv.innerHTML = "Error predicting personality!";
        });
    });
});
