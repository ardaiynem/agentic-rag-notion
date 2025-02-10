document.getElementById("uploadButton").addEventListener("click", function() {
    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === "success") {
            addMessage("File uploaded successfully!");
        } else {
            addMessage("File upload failed.");
        }
    })
    .catch(error => {
        console.error("Error:", error);
        addMessage("An error occurred during file upload.");
    });
});

document.getElementById("queryButton").addEventListener("click", function() {
    const queryInput = document.getElementById("queryInput");
    const query = queryInput.value;

    fetch("http://localhost:5000/query", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ query: query })
    })
    .then(response => response.json())
    .then(data => {
        addMessage("Bot: " + data.answer);
    })
    .catch(error => {
        console.error("Error:", error);
        addMessage("An error occurred while processing your query.");
    });
});

document.getElementById("statsButton").addEventListener("click", function() {
    fetch("http://localhost:5000/count")
    .then(response => response.json())
    .then(data => {
        addMessage("Total documents in collection: " + data.count);
    })
    .catch(error => {
        console.error("Error:", error);
        addMessage("An error occurred while retrieving statistics.");
    });
});

function addMessage(message) {
    const chatDisplay = document.getElementById("chatDisplay");
    const messageElement = document.createElement("div");
    messageElement.textContent = message;
    chatDisplay.appendChild(messageElement);
    chatDisplay.scrollTop = chatDisplay.scrollHeight; // Scroll to the bottom
}