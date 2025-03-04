import React, { useState } from "react";
import { FaMicrophone, FaPaperPlane, FaSeedling, FaUser, FaRobot } from "react-icons/fa";
import "../styles/Chatbot.css";

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);

  // Function to add messages to chat UI
  const addMessage = (text, type) => {
    setMessages((prev) => [...prev, { text, type }]);
  };

  // Send text input to backend (Mistral)
  const handleSend = async () => {
    if (!input.trim()) return;
    addMessage(input, "user");
  
    try {
      const response = await fetch("http://127.0.0.1:8000/api/mistral", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input }),
      });
  
      const data = await response.json();
      addMessage(data.reply, "bot");
  
      // Play Google TTS response if available
      if (data.tts_audio_url) {
        setAudioUrl(data.tts_audio_url);
      }
    } catch (error) {
      console.error("❌ Error sending text:", error);
      addMessage("⚠️ Could not connect to backend!", "bot");
    }
  
    setInput(""); // ✅ Clears input field
  };
  
  // Record voice and send to backend for WhisperX STT
  const handleVoiceInput = async () => {
    setIsRecording(true);
  
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true }); // ✅ Request mic permission
      const mediaRecorder = new MediaRecorder(stream);
      const audioChunks = [];
  
      mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
      };
  
      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
        const formData = new FormData();
        formData.append("file", audioBlob, "voice_input.wav");
  
        try {
          const response = await fetch("http://127.0.0.1:8000/api/transcribe", {
            method: "POST",
            body: formData,
          });
  
          const data = await response.json();
          addMessage(`🎤 You said: ${data.text}`, "user");
          setInput(data.text); // Auto-fill transcribed text
  
          // Auto-send transcribed text to Mistral
          handleSend();
        } catch (error) {
          console.error("❌ Error transcribing voice:", error);
          addMessage("⚠️ Voice input failed!", "bot");
        }
      };
  
      mediaRecorder.start();
      setTimeout(() => {
        mediaRecorder.stop();
        setIsRecording(false);
      }, 5000); // Record for 5 seconds
    } catch (error) {
      console.error("❌ Error accessing microphone:", error);
      alert("Microphone access denied. Please allow access in browser settings.");
      setIsRecording(false);
    }
  };
  

  return (
    <div className="chat-container">
      <h1 className="chat-header">
        <FaSeedling /> Sowing Advisory Chatbot
      </h1>
      <p className="chat-subtitle">Ask for sowing and farming advice!</p>

      <div className="chat-box">
        {messages.map((msg, index) => (
          <div key={index} className={`chat-message ${msg.type}`}>
            {msg.type === "user" ? <FaUser /> : <FaRobot />} {msg.text}
          </div>
        ))}
      </div>

      <div className="chat-input">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder="Type your query..."
        />
        <button onClick={handleSend} className="send-button" title="Send">
          <FaPaperPlane />
        </button>
        <button
          onClick={handleVoiceInput}
          className={`speak-button ${isRecording ? "recording" : ""}`}
          title="Hold to Speak"
        >
          <FaMicrophone />
        </button>
      </div>

      {/* Play Google TTS Response */}
      {audioUrl && <audio src={audioUrl} autoPlay />}

      <footer className="chat-footer">© 2025 Sowing Advisory Chatbot</footer>
    </div>
  );
};

export default Chatbot;
