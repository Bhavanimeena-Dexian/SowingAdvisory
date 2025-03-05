// src/components/InputBox.js
import { useState } from "react";
import { Send, Mic } from "lucide-react";
import VoiceInput from "./VoiceInput";

const InputBox = ({ onSend }) => {
    const [input, setInput] = useState("");

    const handleSubmit = () => {
        if (input.trim()) {
            onSend(input);
            setInput("");
        }
    };

    const handleVoiceInput = (voiceText) => {
        setInput(voiceText);
        onSend(voiceText);
    };

    return (
        <div className="flex items-center gap-4 p-4">
            <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type your query..."
                className="flex-1 p-3 border rounded-lg focus:outline-none"
            />
            <button
                onClick={handleSubmit}
                className="bg-green-600 text-white p-3 rounded-lg hover:bg-green-800 transition"
            >
                <Send size={20} />
            </button>
            <VoiceInput onVoiceInput={handleVoiceInput} />
        </div>
    );
};

export default InputBox;
