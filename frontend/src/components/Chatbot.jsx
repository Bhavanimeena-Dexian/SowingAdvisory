import React, { useState } from "react";
import {
  FaMicrophone,
  FaPaperPlane,
  FaSeedling,
  FaUser,
  FaRobot,
} from "react-icons/fa";
import "../styles/Chatbot.css";

const API_URL = "http://127.0.0.1:8000"; // FastAPI backend URL

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  // eslint-disable-next-line no-unused-vars
  const [isListening, setIsListening] = useState(false);

  // Delay messages for smooth display
  const delayMessage = (newMessage) => {
    setTimeout(() => {
      setMessages((prev) => [...prev, newMessage]);
      if (newMessage.type === "bot") {
        speakMessage(newMessage.text);
      }
    }, 1000 * (messages.length + 1));
  };

  // Send Text Input to Chatbot API
  const handleSend = async (overrideInput) => {
    const messageText = overrideInput || input;
    if (!messageText.trim()) return;

    const userMessage = { text: messageText, type: "user" };
    setMessages((prev) => [...prev, userMessage]);

    try {
      const response = await fetch(`${API_URL}/api/mistral`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: messageText }),
      });

      const data = await response.json();
      const botResponse = { text: data.reply, type: "bot" };

      delayMessage(botResponse);
    } catch (error) {
      console.error("❌ Error sending message:", error);
      const botResponse = {
        text: "⚠️ Error: Could not connect to backend!",
        type: "bot",
      };
      delayMessage(botResponse);
    }

    setInput("");
  };

  // Text-to-Speech: Make the bot speak
  const speakMessage = (text) => {
    if ("speechSynthesis" in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = "en-US";
      speechSynthesis.speak(utterance);
    } else {
      alert("Speech synthesis not supported in this browser.");
    }
  };

  // Speech-to-Text: Capture user voice input
  const handleSpeechRecognition = () => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Speech recognition not supported in this browser.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";

    recognition.onstart = () => {
      setIsListening(true);
    };

    recognition.onresult = async (event) => {
      const transcript = event.results[0][0].transcript;
      console.log("🎤 Recognized Speech:", transcript);

      // Send transcript to FastAPI STT API
      try {
        const formData = new FormData();
        formData.append("text", transcript);

        const response = await fetch(`${API_URL}/api/transcribe`, {
          method: "POST",
          body: formData,
        });

        const data = await response.json();
        const recognizedText = data.text;

        console.log("📜 Transcribed Text:", recognizedText);
        handleSend(recognizedText);
      } catch (error) {
        console.error("❌ Error transcribing audio:", error);
      }
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    recognition.start();
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
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              handleSend(); // Calls the send function when Enter is pressed
            }
          }}
          placeholder="Type your query..."
        />

        <button
          onClick={() => handleSend()}
          className="send-button"
          title="Send"
        >
          <FaPaperPlane />
        </button>
        <button
          onClick={handleSpeechRecognition}
          className="speak-button"
          title="Speak"
        >
          <FaMicrophone />
        </button>
      </div>

      <footer className="chat-footer">© 2025 Sowing Advisory Chatbot</footer>
    </div>
  );
};

export default Chatbot;
