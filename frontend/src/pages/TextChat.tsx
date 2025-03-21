// src/pages/TextChat.tsx

import { useEffect, useRef, useState } from "react";
import { Send, Trash2, ArrowDown } from "lucide-react";
import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:8000"; // ✅ Ensure backend URL is correct

interface Message {
  text: string;
  isUser: boolean;
}

export const TextChat = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState("");
  const [showScrollArrow, setShowScrollArrow] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const chatRef = useRef<HTMLDivElement>(null);

  const exampleQuestions = [
    "I want to sow black gram in my rainfed field in October. Which variety will give me the best yield, and should I apply any organic fertilizers before sowing?",
    "My banana plantation suffered from strong winds during the last monsoon. Is there a better season to plant new banana saplings to avoid damage?",
    "I have 2 acres of land with limited water availability. Which millet should I sow during the Navarai season for a good yield with less irrigation?",
  ];


  // Auto-scroll and show arrow logic
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleScroll = () => {
    if (chatRef.current) {
      const isAtBottom =
        chatRef.current.scrollHeight - chatRef.current.scrollTop ===
        chatRef.current.clientHeight;
      setShowScrollArrow(!isAtBottom);
    }
  };

  const scrollToBottom = () => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
      setShowScrollArrow(false);
    }
  };

  // ✅ Function to Send Query to Backend
  const handleSendMessage = async (text: string) => {
    if (!text.trim()) return;
    const userMessage: Message = { text, isUser: true };
    setMessages((prev) => [...prev, userMessage]);
    setInputText("");
    setIsThinking(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/query/`, { query: text });

      if (response.status === 200 && response.data.answer) {
        const formattedText = response.data.answer
          .replace(/\*\*(.*?)\*\*/g, "<b>$1</b>") // ✅ Bold text
          .replace(/__(.*?)__/g, "<i>$1</i>") // ✅ Italic text
          .replace(/\n/g, "<br>"); // ✅ Preserve paragraph spacing

        const aiResponse: Message = { text: formattedText, isUser: false };
        setMessages((prev) => [...prev, aiResponse]);
      } else {
        setMessages((prev) => [...prev, { text: "⚠️ AI did not return a valid response.", isUser: false }]);
      }
    } catch (error) {
      console.error("❌ API Error:", error);
      setMessages((prev) => [...prev, { text: "❌ Backend not responding. Check API logs.", isUser: false }]);
    }

    setIsThinking(false);
  };

  return (
    <div className="relative min-h-screen flex justify-center items-center overflow-hidden">
      {/* Background Video */}
      <video autoPlay muted loop className="absolute inset-0 w-full h-full object-cover">
        <source src="/assets/t1.mp4" type="video/mp4" />
        Your browser does not support the video tag.
      </video>

      {/* Chat Container */}
      <div className="relative z-10 max-w-4xl w-full bg-white bg-opacity-80 rounded-lg shadow-lg flex">
        {/* Left - Instructions */}
        <div className="p-6 w-1/4 border-r">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Example Questions:</h3>
          <div className="space-y-3">
            {exampleQuestions.map((question, index) => (
              <button
                key={index}
                onClick={() => handleSendMessage(question)}
                className="text-sm bg-gray-200 hover:bg-gray-300 rounded-lg px-4 py-2 w-full text-left"
              >
                {question}
              </button>
            ))}
          </div>
        </div>

        {/* Middle - Chat Section */}
        <div className="flex-1 flex flex-col relative">
          {/* Header */}
          <div className="p-4 border-b flex justify-between items-center">
            <h2 className="text-xl font-semibold text-gray-800">Text Chat</h2>
            <button onClick={() => setMessages([])} className="p-2 text-gray-600 hover:text-red-500">
              <Trash2 className="w-5 h-5" />
            </button>
          </div>

          {/* Chat Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4" ref={chatRef} onScroll={handleScroll}>
            {messages.map((message, index) => (
              <div key={index} className={`flex w-full ${message.isUser ? "justify-end" : "justify-start"}`}>
                <div
                  className={`p-3 rounded-lg max-w-[70%] ${message.isUser ? "bg-green-500 text-white" : "bg-gray-100 text-gray-800"}`}
                  dangerouslySetInnerHTML={{ __html: message.text }} // ✅ Render formatted response
                ></div>
              </div>
            ))}
            {isThinking && (
              <div className="flex justify-center">
                <div className="loader"></div>
              </div>
            )}
          </div>

          {/* Scroll-to-Bottom Arrow */}
          {showScrollArrow && (
            <button onClick={scrollToBottom} className="absolute bottom-20 right-6 bg-green-500 text-white p-3 rounded-full shadow-lg hover:bg-green-600">
              <ArrowDown className="w-6 h-6" />
            </button>
          )}

          {/* Input Area */}
          <div className="p-4 border-t bg-white bg-opacity-90">
            <div className="flex items-center gap-4">
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSendMessage(inputText)}
                placeholder="Type your message..."
                className="flex-1 bg-gray-100 rounded-lg p-4 focus:outline-none focus:ring-2 focus:ring-green-500"
              />
              <button onClick={() => handleSendMessage(inputText)} className="p-4 bg-green-500 text-white rounded-full hover:bg-green-600">
                <Send className="w-6 h-6" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
