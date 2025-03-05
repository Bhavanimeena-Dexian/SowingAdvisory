// src/components/MessageBubble.js
const MessageBubble = ({ message, sender }) => {
    return (
        <div className={`flex ${sender === "user" ? "justify-end" : "justify-start"}`}>
            <div
                className={`p-3 rounded-lg max-w-xs sm:max-w-md ${
                    sender === "user" ? "bg-green-300 text-right" : "bg-gray-300 text-left"
                } shadow-md mb-2`}
            >
                {message}
            </div>
        </div>
    );
};

export default MessageBubble;
