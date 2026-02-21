function startRecognition() {

    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = "en-US";

    recognition.start();

    recognition.onresult = function(event) {
        const text = event.results[0][0].transcript;
        document.getElementById("speechText").innerText = text;

        // send to backend
        fetch("/predict_voice", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                symptoms: text
            })
        })
        .then(res => res.json())
        .then(data => {
            document.getElementById("result").innerText =
                JSON.stringify(data.predictions, null, 2);
        });
    };
}