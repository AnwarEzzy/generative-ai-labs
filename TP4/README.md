# TP4 — Sequence models and Transformers

Completed practical work for the RNN, LSTM and Transformer notebook. The deliverable is
[`TP4_Transformers_reponses.ipynb`](TP4_Transformers_reponses.ipynb), which covers sine-wave
forecasting, hourly electricity forecasting, a causal character-level mini-GPT, and a
bidirectional masked-language mini-BERT.

## Run

1. Place the teacher-provided `individual+household+electric+power+consumption.zip` in this folder.
2. From this folder, install `requirements.txt` in a Python environment and open the answer notebook in VS Code or Jupyter.
3. Select that Python kernel and run all cells in order. Training on CPU may take several minutes.

The ZIP, teacher notebook, screenshots and generated model files are excluded from Git. The
small `corpus.txt` is included because the mini-GPT uses it as its training data. The answer
notebook is self-contained apart from these data and Python dependencies. The committed
notebook contains the output of a complete execution with the provided dataset.

## What is checked

- Chronological train, validation and test split for the electricity series, with statistics fitted on train only.
- Test MAE and RMSE in kW against persistence and the same hour on the previous day.
- Causal-mask invariance for mini-GPT and padding-mask invariance for mini-BERT.
- Masked-token validation accuracy and example top-3 word suggestions.

The small GPT and BERT models are teaching examples trained locally; their generated text and
masked-word predictions are not expected to match pretrained language models.

## Observed results (CPU, seed 42)

| Task | Result |
| --- | --- |
| Sine RNN / LSTM / Transformer | Test MSE: 0.000050 / 0.000012 / 0.000083 |
| Electricity Transformer | Test MAE 0.3437 kW; RMSE 0.4946 kW |
| Electricity persistence | Test MAE 0.3892 kW; RMSE 0.5884 kW |
| Electricity previous day | Test MAE 0.5133 kW; RMSE 0.7639 kW |
| Mini-BERT | 57.7% accuracy on 52 selected validation tokens |

The mini-GPT causal check and mini-BERT padding check both passed. The complete outputs and
plots are saved in the notebook.
