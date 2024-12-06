// src/components/ChatBot.js
"use client";

import { useState, useRef, useEffect } from "react";
import { Moon, Sun, Send } from "lucide-react";
import { FiPaperclip } from "react-icons/fi"; // Importing paperclip icon from react-icons library
import { marked } from "marked";

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [darkMode, setDarkMode] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [isUploading, setIsUploading] = useState(false); // New state for handling file upload status
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const queries = input
      .split(/[?;]/)
      .map((q) => q.trim())
      .filter((q) => q);
    let botResponses = [];

    for (let query of queries) {
      if (!query) continue;

      const newMessage = {
        id: Date.now() + Math.random(),
        text: query,
        sender: "user",
      };
      setMessages((prev) => [...prev, newMessage]);
      setInput("");
      setIsTyping(true);

      try {
        const response = await fetch("http://127.0.0.1:5000/query", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ user_input: query }),
        });

        if (!response.ok)
          throw new Error("Failed to fetch response from server");

        const data = await response.json();
        const formattedResponse = marked(data.response);

        const botResponse = {
          id: Date.now() + Math.random(),
          text: formattedResponse,
          sender: "bot",
          sources: data.sources || "No source available",
        };
        botResponses.push(botResponse);
      } catch (error) {
        console.error("Error fetching response:", error);
        const errorMessage = {
          id: Date.now() + Math.random(),
          text: "Error: Unable to connect to the server. Please try again later.",
          sender: "bot",
        };
        botResponses.push(errorMessage);
      } finally {
        setIsTyping(false);
      }
    }

    setMessages((prev) => [...prev, ...botResponses]);
  };

  const handleInputChange = (e) => {
    setInput(e.target.value);
  };

  // Handle file upload for adding PDF files
  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (file) {
      console.log("File selected:", file.name);

      setIsUploading(true); // Set uploading state to true

      const formData = new FormData();
      formData.append("file", file); // Changed key to "file" to match backend

      try {
        const response = await fetch("http://127.0.0.1:5000/upload", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) throw new Error("Failed to upload file");

        console.log("File uploaded successfully");
        setMessages((prev) => [
          ...prev,
          {
            id: Date.now() + Math.random(),
            text: `Successfully uploaded: ${file.name}`,
            sender: "bot",
          },
        ]);
      } catch (error) {
        console.error("Error uploading file:", error);
        setMessages((prev) => [
          ...prev,
          {
            id: Date.now() + Math.random(),
            text: `Error uploading file: ${file.name}`,
            sender: "bot",
          },
        ]);
      } finally {
        setIsUploading(false); // Set uploading state to false after completion
        e.target.value = ""; // Clear the input field so users can re-upload without issues
      }
    }
  };

  return (
    <div
      className={`h-screen flex flex-col ${
        darkMode ? "bg-neutral-900 text-white" : "bg-neutral-100 text-gray-800"
      } transition-colors duration-300`}
    >
      <header
        className={`flex justify-between items-center p-4 ${
          darkMode ? "bg-neutral-800" : "bg-white"
        } border-b border-neutral-300 dark:border-neutral-700`}
      >
        <h1 className="text-xl font-semibold">RAG Bot</h1>
        <button
          onClick={() => setDarkMode(!darkMode)}
          className="p-2 hover:bg-neutral-700 dark:hover:bg-neutral-600 rounded-full transition-transform duration-300"
        >
          {darkMode ? (
            <Sun className="h-5 w-5 text-yellow-400" />
          ) : (
            <Moon className="h-5 w-5 text-gray-600" />
          )}
        </button>
      </header>

      <main className="flex-grow overflow-auto p-4">
        <div className="max-w-2xl mx-auto space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${
                message.sender === "user" ? "justify-end" : "justify-start"
              } ${
                message.sender === "user"
                  ? "animate-subtle-slide-in-right"
                  : "animate-subtle-slide-in-left"
              }`}
            >
              <div
                className={`max-w-xs md:max-w-md p-3 rounded-lg ${
                  message.sender === "user"
                    ? "bg-neutral-700 text-white rounded-br-none"
                    : darkMode
                    ? "bg-neutral-800 text-white rounded-bl-none"
                    : "bg-neutral-200 text-gray-800 rounded-bl-none"
                } shadow-sm`}
              >
                {message.sender === "bot" ? (
                  <div
                    className="markdown-content"
                    dangerouslySetInnerHTML={{ __html: message.text }}
                  ></div>
                ) : (
                  <p>{message.text}</p>
                )}
                {message.sender === "bot" &&
                  message.sources !== "No source available" && (
                    <div className="text-xs text-neutral-500 mt-2">
                      <strong>Sources:</strong> {message.sources}
                    </div>
                  )}
              </div>
            </div>
          ))}
          {isTyping && (
            <div className="flex justify-start animate-fade-in">
              <div className="bg-neutral-200 dark:bg-neutral-800 p-3 rounded-lg rounded-bl-none shadow-sm">
                <div className="flex space-x-2">
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
                  <div
                    className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.2s" }}
                  ></div>
                  <div
                    className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.4s" }}
                  ></div>
                </div>
              </div>
            </div>
          )}
          {isUploading && (
            <div className="flex justify-start animate-fade-in">
              <div className="bg-neutral-200 dark:bg-neutral-800 p-3 rounded-lg rounded-bl-none shadow-sm">
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Uploading file...
                </p>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>

      <footer
        className={`p-4 ${
          darkMode ? "bg-neutral-800" : "bg-white"
        } border-t border-neutral-300 dark:border-neutral-700`}
      >
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
          className="flex items-center space-x-3 max-w-2xl mx-auto bg-neutral-200 dark:bg-neutral-800 p-3 rounded-full shadow-md"
        >
          <input
            type="text"
            placeholder="Type your message..."
            value={input}
            onChange={handleInputChange}
            className="flex-grow bg-transparent border-none focus:outline-none focus:ring-2 focus:ring-neutral-600 rounded-full px-3"
          />
          <button
            type="button"
            className="w-10 h-10 flex items-center justify-center rounded-full hover:bg-neutral-300 dark:hover:bg-neutral-700 transition-transform duration-300 hover:scale-110"
          >
            <label className="cursor-pointer flex items-center">
              <FiPaperclip className="h-5 w-5 text-gray-600 dark:text-white" />
              <input
                type="file"
                accept=".pdf"
                className="hidden"
                onChange={handleFileUpload}
              />
            </label>
          </button>
          <button
            type="submit"
            className="w-10 h-10 flex items-center justify-center rounded-full bg-black hover:bg-gray-800 text-white transition-transform duration-300 hover:scale-110"
          >
            <Send className="h-5 w-5" />
            <span className="sr-only">Send</span>
          </button>
        </form>
      </footer>
    </div>
  );
};

export default Chatbot;
