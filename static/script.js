document.addEventListener("DOMContentLoaded", function () {
  const videoUpload = document.getElementById("videoUpload");
  const videoElement = document.getElementById("videoElement");
  const predictButton = document.getElementById("predictButton");
  const predictionResult = document.getElementById("predictionResult");
  const startButton = document.getElementById("startButton");
  const stopButton = document.getElementById("stopButton");
  const toggleButton = document.getElementById("toggleButton");
  const cameraSection = document.getElementById("cameraSection");
  const timerElement = document.getElementById("timer");
  let mediaRecorder;
  let recordedChunks = [];
  let timer;
  let seconds = 0;

  // Hiển thị/Ẩn camera section
  toggleButton.addEventListener("click", function () {
    if (cameraSection.style.display === "none") {
      cameraSection.style.display = "block";
      toggleButton.textContent = "Hide";
    } else {
      cameraSection.style.display = "none";
      toggleButton.textContent = "Show";
    }
  });

  // Xử lý tải video
  videoUpload.addEventListener("change", function (event) {
    const file = event.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      videoElement.src = url;
    }
  });

  // Quay video
  startButton.addEventListener("click", async function () {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });
      videoElement.srcObject = stream;
      videoElement.play();
      recordedChunks = [];
      mediaRecorder = new MediaRecorder(stream);

      mediaRecorder.ondataavailable = function (event) {
        if (event.data.size > 0) {
          recordedChunks.push(event.data);
        }
      };

      mediaRecorder.onstop = function () {
        const blob = new Blob(recordedChunks, { type: "video/webm" });
        const url = URL.createObjectURL(blob);
        videoElement.srcObject = null;
        videoElement.src = url;
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      startButton.disabled = true;
      stopButton.disabled = false;
      seconds = 0;
      timer = setInterval(() => {
        seconds++;
        timerElement.textContent = seconds;
      }, 1000);
    } catch (error) {
      console.error("Error accessing camera:", error);
    }
  });

  stopButton.addEventListener("click", function () {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
      clearInterval(timer);
      startButton.disabled = false;
      stopButton.disabled = true;
    }
  });

  // Gửi video để dự đoán hành động
  predictButton.addEventListener("click", function () {
    const file = videoUpload.files[0];
    if (!file) {
      alert("Vui lòng tải lên một video trước khi dự đoán.");
      return;
    }

    const formData = new FormData();
    formData.append("video", file);

    fetch("/predict", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        predictionResult.innerHTML = `<b>Hành động dự đoán:</b> ${data.action}`;
      })
      .catch((error) => {
        console.error("Error:", error);
        predictionResult.innerHTML = "Lỗi khi dự đoán hành động.";
      });
  });
});
