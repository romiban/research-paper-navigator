# Research Paper Navigator

A conversational retrieval-augmented generation (RAG) system that answers questions about a collection of research papers. I'm building it step by step as the capstone project for my AI engineering program.

## Purpose

The goal is a system that can answer questions from individual papers, compare ideas across several papers, spot trends over time, and back up its answers with direct quotations from the source documents. I'm focusing on reliability and transparency: every answer should show where the information came from.

## Status

Work in progress. Each module adds a checkpoint, and each checkpoint has its own folder:

- `checkpoint-1.1/`: testing how an LLM performs without retrieval, and when retrieval is needed

## Setup

1. Clone the repo and create a virtual environment (Python 3.11+).
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your API key: `OPENROUTER_API_KEY=your-key-here`
4. Run the Python file in the checkpoint folder you want to try.

## License

MIT
