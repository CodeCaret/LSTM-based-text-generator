# Generative AI with LSTM Text Generation

This project is a **character-level LSTM text generator** trained on Shakespeare's works.  
It demonstrates how an LSTM model can learn patterns in text sequences and generate new, coherent text based on a seed input.

---

## Dataset

The model is trained on Shakespeare’s complete works:

- [Shakespeare Dataset](https://www.gutenberg.org/files/100/100-0.txt)


---

## Setup / Requirements

### 1. Clone the repository:

```
git clone <your_repo_url>
cd text_generator
```

### 2. Create a virtual environment and activate it:

```
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install dependencies:

```
pip install -r requirements.txt
```

---

## Usage

- Run the main script:

```
python lstm_text_generator.py
```

- The script will train the model (3 epochs) and then generate text based on a few predefined seed phrases.

- Training is CPU-based, so it may take a few minutes to complete.

---

## Sample Outputs

### Seed: to be or not to be

to be or not to be eioesoioest there me this giving moye and the clown
who paat croubond
in a avach to the gost wasty
borse gooncemer usme


### Seed: all the worlds a stage

all the worlds a stage eiaeui
where hearld
and thie by the reather dost the ganged
groof
the king and to i stand go and comphith in me the fai


### Seed: love looks not with the eyes

love looks not with the eyes it the wair bowned with boundsuly
scene i mepine pary
lions
pay from here the bastarca
prot illiil is to thee anceade

> Note: As this is a character-level LSTM trained on a small number of epochs, some words may appear garbled.

---

## Bonus – Model Experiments

I experimented with different training configurations by reducing the number of LSTM units and increasing the step size during sequence generation:

- **LSTM units:** Reduced from 128 → 64
- **Step size:** Increased from 3 → 8
- **Epochs:** Reduced to 3 for fast CPU training

---

## Observations:

- Reducing LSTM units significantly reduced training time with minor reduction in text coherence.

- Increasing the step size reduced the number of training sequences, allowing faster convergence while still preserving overall structure.

- These experiments demonstrate the trade-off between computational efficiency and output quality.

---