document.addEventListener('DOMContentLoaded', function() {
    const imageInput = document.getElementById('image-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const form = document.getElementById('segmentation-form');

    imageInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            const fileName = file.name;
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                previewContainer.style.display = 'block';
                document.getElementById('image-preview').textContent = file.name;
            };
            reader.readAsDataURL(file);
        }
    });

    form.addEventListener('submit', function(e) {
        e.preventDefault();

        const formData = new FormData();
        formData.append('file', imageInput.files[0]);

        showLoading();

        fetch('/segment', { method: 'POST', body: formData })
        .then(res => res.json())
        .then(data => {
          if (data.error) {
            showAlert(data.error, 'error');
            return;
          }
          document.getElementById('roi_image').src = data.roi_image;
          document.getElementById('roi_image').style.display = 'block';
          document.getElementById('roi text').style.display = 'block';

          document.getElementById('mask_image').src = data.mask_image;
          document.getElementById('mask_image').style.display = 'block';
          document.getElementById('seg text').style.display = 'block';

          document.getElementById('overlay_image').src = data.overlay_image;
          document.getElementById('overlay_image').style.display = 'block';
          document.getElementById('overlay text').style.display = 'block';

          document.getElementById('diagnosis_text').innerText = `Diagnosis: ${data.diagnosis} (Confidence: ${data.confidence})`;
          document.getElementById('results').style.display = 'block';
          hideLoading();
        })
        .catch(err => {
//          showAlert("Failed to get results", "error");
          hideLoading();
        });


        fetch('/original', {
            method: 'POST',
            body: formData
        })
        .then(response => response.blob())
        .then(blob => {
            const imageUrl = URL.createObjectURL(blob);
            document.getElementById('original-display').src = imageUrl;
        })
        .catch(error => {
            console.error('Error loading original image:', error);
        });
    });

    function showLoading() {
        const button = document.querySelector('.analyze-btn');
        button.innerHTML = '<div class="loading-spinner"></div> Processing...';
        button.disabled = true;
    }

    function hideLoading() {
        const button = document.querySelector('.analyze-btn');
        button.innerHTML = 'Analyze Image';
        button.disabled = false;
    }

    function displayResults(segmentedSrc) {
        document.getElementById('segmented-result').src = segmentedSrc;
        document.getElementById('results-section').style.display = 'block';
        hideLoading();
    }

    function showAlert(message, type) {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type}`;
        alertDiv.textContent = message;
        document.querySelector('.upload-section').prepend(alertDiv);

        setTimeout(() => {
            alertDiv.remove();
        }, 5000);
    }
});
