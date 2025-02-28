// src/components/SpeechRecognition.js
import { useEffect } from "react";
import { useSpeechRecognition } from "react-speech-recognition";

const SpeechRecognitionComponent = ({ onResult }) => {
    const {
        transcript,
        listening,
        resetTranscript,
        browserSupportsSpeechRecognition,
        startListening,
        stopListening,
    } = useSpeechRecognition();

    useEffect(() => {
        if (transcript) {
            onResult(transcript);
            resetTranscript(); // Clear the transcript after sending
        }
    }, [transcript, onResult, resetTranscript]);

    if (!browserSupportsSpeechRecognition) {
        return <span>Speech recognition not supported in your browser.</span>;
    }

    return (
        <button
            onClick={listening ? stopListening : startListening}
            className={`bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-700 ${
                listening ? "opacity-75" : ""
            }`}
        >
            🎤 {listening ? "Listening..." : "Speak"}
        </button>
    );
};

export default SpeechRecognitionComponent;
