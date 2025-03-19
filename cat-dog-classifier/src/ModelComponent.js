import React, { useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import { useDropzone } from "react-dropzone";
import "./App.css";

const ModelComponent = () => {
  const [model, setModel] = useState(null);
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);

  useEffect(() => {
    const loadModel = async () => {
      const loadedModel = await tf.loadLayersModel("/tfjs_model/model.json");
      setModel(loadedModel);
    };
    loadModel();
  }, []);

  const onDrop = (acceptedFiles) => {
    const file = acceptedFiles[0];
    const reader = new FileReader();
    reader.onload = () => {
      setImage(reader.result);
      classifyImage(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const classifyImage = async (imageSrc) => {
    if (!model) return;
    const img = new Image();
    img.src = imageSrc;
    img.onload = async () => {
      const tensor = tf.browser.fromPixels(img)
        .resizeNearestNeighbor([128, 128])
        .toFloat()
        .expandDims();
      const predictions = model.predict(tensor);
      const scores = await predictions.data();
      setPrediction(scores[0] > 0.5 ? "Dog" : "Cat");
      setConfidence((scores[0] * 100).toFixed(2) + "%");
    };
  };

  const { getRootProps, getInputProps } = useDropzone({ onDrop, accept: "image/*" });

  return (
    <div className="container">
      <h1 className="title"><span className="highlight">Cat & Dog Classifier</span> 🐱🐶</h1>
      <p className="description">Enter the image you want to classify or predict.</p>

      <div className="dropzone-container">
        <div {...getRootProps()} className="dropzone">
          <input {...getInputProps()} />
          <p>📂 Drag & drop an image here, or click to select one</p>
        </div>
      </div>

      <div className="image-prediction-container">
        <div className="image-container">
          {image && <img src={image} alt="Uploaded" className="uploaded-image" />}
        </div>
        {prediction && (
          <div className="prediction-box">
            <h2 className="result">Prediction: {prediction}</h2>
            <h3 className="confidence">Confidence: {confidence}</h3>
          </div>
        )}
      </div>
    </div>
  );
};

export default ModelComponent;
