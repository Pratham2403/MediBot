# Configuration Guide for Setting Up MediBot

Follow these steps to set up and configure MediBot on your local machine.

## Setting Up Client and Server

1. **Open Terminal / Command Prompt:**

   - For **Windows**: Use Command Prompt or PowerShell.
   - For **Linux and macOS**: Use Terminal.

2. **Navigate to Client and Server Directories:**

   ```bash
   cd path/to/client
   npm i
   cd path/to/server
   npm i
   ```

## Setting Up Server Environment

1. **Create Virtual Environment (Python 3.12 and above preferred):**

   ```bash
   cd path/to/server
   python3 -m venv venv
   ```

   Replace `python3` with `python` or `python3.12` as appropriate.

2. **Activate the Virtual Environment:**

   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   
   - **Linux / macOS:**
     ```bash
     source venv/bin/activate
     ```

3. **Install Python Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Adding Chroma Library

1. **Download and Extract Chroma Zip Folder:**

   Download the Chroma zip folder from the provided [drive link](https://drive.google.com/file/d/1qehnf8V7SsARDanK-SIf2bAdUUMIwJh6/view?usp=sharing).
   
2. **Put Chroma Inside Server Folder:**

   Extract the Chroma folder and place it inside the `server` directory.

## Setting Up .env File

1. **Rename .env.example:**

   In both `client` and `server` directories, rename `.env.example` files to `.env`.

2. **Configure .env File:**

   Edit the `.env` files in both directories to set up API keys, ports, and tokens as per your requirements. Change the Backend PORT from `main.jsx` in `client` directory according to your requirement.

## How to Run

1. **Start Server:**

   ```bash
   cd path/to/server
   python main.py
   ```

2. **Start Client:**

   Open a new terminal or command prompt window.

   ```bash
   cd path/to/client
   npm start
   ```

3. **Access MediBot:**

   Open the provided localhost URL in your web browser as shown in the client terminal.

---

Make sure to replace `path/to/client` and `path/to/server` with the actual paths to your client and server directories. Adjust Python version (`python3`) and other commands based on your system setup.
