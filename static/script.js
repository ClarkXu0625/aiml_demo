document.getElementById('uploadForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        if (response.ok) {
            document.getElementById('result').innerText = `Predicted Hemoglobin Level: ${result.hemoglobin_level}`;
        } else {
            document.getElementById('result').innerText = `Error: ${result.error}`;
        }
    } catch (error) {
        document.getElementById('result').innerText = 'An error occurred. Please try again.';
    }
});
