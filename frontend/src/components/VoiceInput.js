// VoiceInput.jsx
import React, { useState } from "react";
import { useSpeechRecognition } from "react-speech-recognition";
import "../styles/Chatbot.css";

const VoiceInput = ({ onSend, onVoiceInput }) => {
  const [input, setInput] = useState("");
  const { transcript, resetTranscript, listening, startListening, stopListening } = useSpeechRecognition();

  const handleSend = () => {
    if (input.trim()) {
      onSend(input);
      setInput("");
    }
  };

  const handleVoice = () => {
    if (!listening) {
      startListening();
    } else {
      stopListening();
      onVoiceInput(transcript);
      resetTranscript();
    }
  };

  return (
    <div className="input-container">
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Type your query..."
      />
      <button onClick={handleSend}>Send</button>
      <button onClick={handleVoice}>{listening ? "🎤 Listening..." : "🎙️ Speak"}</button>
    </div>
  );
};

export default VoiceInput;