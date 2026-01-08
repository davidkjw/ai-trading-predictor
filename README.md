# RAG PDF Assistant with Ollama

A privacy-first desktop application that allows you to chat with your PDF documents using local AI. No cloud services, no API keys, no data leaving your computer.

## ✨ Key Features

- **🔒 100% Private** - All processing happens locally on your machine
- **💰 Zero Ongoing Costs** - No API fees or subscription charges
- **📄 PDF Intelligence** - Ask questions about your documents in natural language
- **⚡ Local AI** - Uses Ollama to run LLMs like Llama 3, Mistral, and more
- **🎯 Accurate Answers** - Retrieval-Augmented Generation (RAG) for context-aware responses
- **🖥️ User-Friendly** - Designed for non-technical users with intuitive interface

## 🎯 Who Is This For?

- **Researchers & Students** analyzing academic papers and textbooks
- **Business Professionals** reviewing contracts, reports, and proposals
- **Anyone with sensitive documents** who values privacy
- **Users who want AI document analysis** without recurring costs

## 🚀 Quick Start

### Prerequisites
- **Windows 10/11** (macOS/Linux support planned)
- **8GB+ RAM** (16GB recommended for larger models)
- **Python 3.8+** (will be installed automatically)
- **10GB free disk space** for AI models

### Installation

1. **Download the installer** from [Releases](https://github.com/yourusername/rag-pdf-assistant/releases)
2. **Run the installer** (`RAG_PDF_Assistant_Setup.exe`)
3. **Follow the on-screen setup wizard** - it will install everything automatically!

The installer will:
- Install Python (if not present)
- Install required Python packages
- Download and install Ollama
- Guide you through downloading your first AI model
- Launch the application

### First-Time Setup Time
- **First run:** 15-30 minutes (includes model downloads)
- **Subsequent runs:** 30 seconds

## 📖 How to Use

### 1. Upload a PDF
- Drag and drop your PDF file into the app
- Or click "Browse" to select a file
- Support for files up to 200MB

### 2. Ask Questions
- Type questions about your document in natural language
- Examples:
  - "What are the main findings of this research?"
  - "Summarize the contract terms on page 5"
  - "What methods were used in this study?"

### 3. Get Answers
- Receive accurate answers with citations to source text
- View which parts of your document were used
- Ask follow-up questions without re-uploading

## 🛠️ Configuration Options

### AI Models
Choose from various models (all run locally):
- **Llama 3.2** (7B) - Balanced speed and quality (recommended)
- **Mistral** (7B) - Fast and efficient
- **Phi-2** (2.7B) - Very fast, lower resource usage
- **Llama 2** (7B/13B) - Established, reliable

### Document Processing
- **Chunk Size**: Adjust how the document is split (500-2000 characters)
- **Top-K Retrieval**: Control how many document sections are used (1-10)

## 📊 Performance

| Task | Target Performance |
|------|-------------------|
| PDF Upload (10MB) | <5 seconds |
| Document Processing | <30 seconds for 100 pages |
| Answer Generation | <30 seconds |
| Memory Usage | <4GB RAM |

## 🔧 Technical Details

### Architecture
Streamlit UI → Application Logic → Ollama (Local LLM)
↓
FAISS Vector Database
↓
Sentence Transformers (Embeddings)

text

### Technology Stack
- **Frontend**: Streamlit (Python web framework)
- **PDF Processing**: PyPDF2
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Database**: FAISS
- **LLM Runtime**: Ollama
- **HTTP Client**: requests

### Data Privacy Guarantee
- ✅ No data sent to external servers
- ✅ No telemetry or usage tracking
- ✅ All processing happens locally
- ✅ Temporary files deleted on exit
- ✅ Works in air-gapped environments

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Report bugs** in the [Issues](https://github.com/yourusername/rag-pdf-assistant/issues) section
2. **Suggest features** that would help your workflow
3. **Submit pull requests** for improvements
4. **Improve documentation** for other users

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 🐛 Troubleshooting

### Common Issues

**"Ollama is not running"**
- Make sure Ollama service is started
- Run `ollama serve` in command prompt

**"No AI models found"**
- Download a model via: `ollama pull llama3.2`
- Or use the in-app download interface

**"File too large"**
- PDFs are limited to 200MB in v1.0
- Consider splitting large documents

**"Slow performance"**
- Try a smaller model (Phi-2 or Mistral 7B)
- Close other memory-intensive applications
- Ensure you have at least 8GB free RAM

**More help?** Check our [Troubleshooting Guide](docs/TROUBLESHOOTING.md)

## ❓ FAQ

**Q: Is my data safe?**
A: Yes! Everything runs locally on your computer. No data ever leaves your device.

**Q: How much does this cost?**
A: The software is free. The only "cost" is disk space for AI models (4-7GB each).

**Q: What PDFs are supported?**
A: Text-based PDFs up to 200MB. Scanned PDFs (OCR) support is planned for v2.0.

**Q: Can I use this offline?**
A: Yes! Once installed, the application works completely offline.

**Q: What computer do I need?**
A: Windows 10/11 with 8GB RAM minimum, 16GB recommended for best experience.

**Q: Can I use my own models?**
A: Yes! Any model supported by Ollama can be used.

## 📈 Roadmap

### v1.1 (Next Release)
- [ ] Multi-document support
- [ ] Conversation history
- [ ] Export answers to text file
- [ ] GPU acceleration support

### v2.0 (Planned)
- [ ] Support for Word, Excel, PowerPoint files
- [ ] OCR for scanned documents
- [ ] Batch processing
- [ ] Question templates

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.com) for making local LLMs accessible
- [Hugging Face](https://huggingface.co) for sentence-transformers
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- [Streamlit](https://streamlit.io) for the amazing UI framework

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/rag-pdf-assistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/rag-pdf-assistant/discussions)
- **Email**: support@example.com

---

**⭐ If you find this useful, please consider giving it a star on GitHub! ⭐**

[![GitHub Stars](https://img.shields.io/github/stars/yourusername/rag-pdf-assistant?style=social)](https://github.com/yourusername/rag-pdf-assistant)
[![GitHub Issues](https://img.shields.io/github/issues/yourusername/rag-pdf-assistant)](https://github.com/yourusername/rag-pdf-assistant/issues)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
