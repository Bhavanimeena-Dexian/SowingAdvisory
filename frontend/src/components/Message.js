// src/components/Message.js

// src/components/Message.js
const Message = ({ text, sender }) => {
    return (
        <div
            className={`flex ${
                sender === "user" ? "justify-end" : "justify-start"
            } mb-2`}
        >
            <div
                className={`p-3 rounded-xl max-w-xs ${
                    sender === "user"
                        ? "bg-green-300 text-right"
                        : "bg-gray-300 text-left"
                }`}
            >
                {text}
            </div>
        </div>
    );
};

export default Message;

