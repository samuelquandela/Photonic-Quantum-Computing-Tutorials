# Photonic Quantum Computing Tutorials

## Overview

This repository contains practical tutorials demonstrating the use of Quandela’s photonic quantum computing technology. The material is designed to provide hands-on examples ranging from foundational photonic circuit simulations to quantum machine learning workflows.

The tutorials are organised into two main areas:

* **Photonics-Perceval** — Tutorials using the Perceval library for modelling and simulating photonic quantum circuits.
* **Quantum_Machine_Learning-MerLin** — Tutorials combining MerLin and PyTorch for quantum machine learning applications built on photonic quantum computing principles.

---

## Repository Structure

```
Photonic-Quantum-Computing-Tutorials/
│
├── Photonics-Perceval/
│   └── Tutorials using the Perceval library
│
├── Quantum_Machine_Learning-MerLin/
│   └── Tutorials using MerLin and PyTorch
│
├── img/
│   └── Images and figures used throughout the tutorials
│
└── requirements.txt
```

### Photonics-Perceval

This folder contains tutorials focused on:

* Building and simulating photonic quantum circuits
* Modelling linear optical components
* Running simulations using Perceval
* Exploring photonic quantum computation concepts in practice

### Quantum_Machine_Learning-MerLin

This folder contains tutorials covering:

* Integration of MerLin with PyTorch
* Hybrid quantum-classical workflows
* Photonic quantum models for machine learning tasks
* Training and evaluating quantum-enhanced models

### img

This directory contains images, diagrams, and visual assets referenced within the tutorials.

### requirements.txt

This file lists the Python dependencies required to run the tutorials. It ensures a reproducible environment across systems.

---

## Installation

It is recommended to use a virtual environment.

### macOS / Linux

Create a virtual environment:

python3 -m venv venv

Activate it:

source venv/bin/activate

### Windows (PowerShell)

Create a virtual environment:

python -m venv venv

Activate it:

venv\Scripts\Activate.ps1

### Windows (Command Prompt)

Create a virtual environment:

python -m venv venv

Activate it:

venv\Scripts\activate.bat

### Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Using a `.env` File for Cloud Access

Some tutorials require access to Quandela Cloud services via a Cloud Token.
To avoid hard-coding credentials, the token is stored in a `.env` file.

### 1. Create a `.env` File

In the root of the repository, create a file named:

```
.env
```

Add your Cloud Token:

```
QUANDELA_CLOUD_TOKEN=your_token_here
```

### 2. Load the Environment Variables

If required, install `python-dotenv`:

```bash
pip install python-dotenv
```

The tutorials load the token from the `.env` file at runtime, making it available as an environment variable.

### Important

* Do not commit the `.env` file to version control.
* Keep your Cloud Token confidential.
* Ensure `.env` is listed in your `.gitignore`.


## Prerequisites

* Python 3.9 or newer (unless otherwise specified in `requirements.txt`)
* Basic familiarity with Python
* Introductory knowledge of quantum computing concepts
* PyTorch familiarity for the quantum machine learning tutorials

---

## Usage

Navigate to the relevant tutorial folder and follow the instructions provided within each notebook or script.

For example:

```bash
cd Photonics-Perceval
```

or

```bash
cd Quantum_Machine_Learning-MerLin
```

Open the tutorials in Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

---

## Purpose

This repository is intended for:

* Researchers exploring photonic quantum computing
* Developers building applications with Quandela’s technology
* Students learning practical quantum photonics
* Practitioners interested in quantum machine learning with photonic systems

---

## Licence

Please refer to the appropriate licence terms associated with the included libraries (Perceval, MerLin) and any additional project-specific licensing information.
