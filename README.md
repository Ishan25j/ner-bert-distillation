# Knowledge Distillation for BERT Models

This repository contains code for performing knowledge distillation on BERT models using the CoNLL-2003 dataset. The goal is to transfer knowledge from a finetuned BERT teacher model to a smaller student model using knowledge distillation techniques.

## Overview

### Problem

The CoNLL-2003 dataset for Named Entity Recognition (NER) has an issue where the tokenizer input and NER tags do not align perfectly due to the use of subwords and special tokens. To address this:

- Subword tokens are masked, and special tokens are set to label `-100`, which is ignored during training.
- Labels are aligned with token IDs using a custom function.

### Solution

1. **Custom Tokenization and Label Alignment**

   - `tokenize_and_align_labels()`: Masks special tokens and subwords beyond the first subword in the input sequence.
2. **Model Training**

   - **Teacher Model**: Finetuned `bert-base-uncased` on the CoNLL-2003 dataset.
   - **Student Model**: Trained `distilbert-base-uncased` using Knowledge Distillation (KD) with KL Divergence.
3. **Custom Transformer Student Model**

   - Implemented a custom transformer model in PyTorch with components such as LayerNormalization, FeedForwardBlock, InputEmbeddings, PositionalEncoding, ResidualConnection, MultiHeadAttention, EncoderBlock, and Encoder.

### Training and Evaluation

1. **Knowledge Distillation**

   - The loss function combines Cross-Entropy Loss (CE) and Knowledge Distillation Loss (KD) with KL Divergence:
     ```python
     teacher_logits = teacher_outputs.logits
     student_logits = student_outputs.logits

     loss_fct = nn.KLDivLoss(reduction='batchmean')

     loss_kd = loss_fct(
         F.log_softmax(student_logits / self.temperature, dim=-1),
         F.softmax(teacher_logits / self.temperature, dim=-1)
     )

     loss_ce = student_outputs.loss

     loss = (1 - self.alpha) * loss_ce + self.alpha * self.temperature ** 2 * loss_kd
     ```
2. **Custom Transformer Model**

   - Implemented using PyTorch with the following architecture:
     - **LayerNormalization**
     - **FeedForwardBlock**
     - **InputEmbeddings**
     - **PositionalEncoding**
     - **ResidualConnection**
     - **MultiHeadAttention**
     - **EncoderBlock**
     - **Encoder**
3. **Evaluation Metrics**

   - **Parameters**:

     - Teacher: 108.899M
     - Student: 66.370M
     - Custom Student: 67.561M
   - **Evaluation Loss**:

     - Teacher: 0.058
     - Student: 2.500
     - Custom Student: 0.450

## Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/Ishan25j/ner-bert-distillation.git
cd ner-bert-distillation
```
