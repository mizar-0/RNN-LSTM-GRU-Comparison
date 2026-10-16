# Word Level Limerick Generation with RNN LSTM and GRU

## Objective

This project implements and compares four word-level autoregressive language models for the ICT4403 Foundations of Generative AI Assignment 2: a Simple RNN, a Bidirectional RNN, an LSTM, and a GRU. Each model predicts the next token from a 64-token context and then generates limerick-like text sequentially from shared seed prompts.

The final recommendation is **GRU at temperature 0.5**. GRU achieved the best held-out prediction metrics. Temperature 0.5 retained nearly all of its strict rhyme performance at 0.3 while sharply reducing repeated trigrams and increasing distinct bigrams.

## Dataset

The experiments use the [Limerick Dataset (OEDILF v3)](https://doi.org/10.5281/zenodo.5722527), hosted on Zenodo under [CC BY 3.0 US](https://creativecommons.org/licenses/by/3.0/us/). The final run filtered the full source to 65,886 five-line limericks from 1,296 authors, split by poem into 80% train, 10% validation, and 10% test sets. Poems contain a mean of 29.76 words (median 30; 10th-90th percentile 26-34 words). With the 10,000-token training vocabulary, 10.16% of lexical training tokens and 9.92% of lexical test tokens map to `<unk>`. The token distribution is strongly concentrated in function words: `the`, `a`, `and`, `to`, and `in` are the five most frequent lexical tokens. See [data/README.md](data/README.md) for download instructions and a content note about crude or adult user-submitted material.

## Models and common training protocol

All models used the same full dataset split, seed (42), vocabulary cap (10,000), 128-dimensional embeddings, 256 hidden units, dropout 0.3, Adam learning rate 0.001, batch size 512, and a 64-token context window. Early stopping used validation loss with patience 5 and a maximum of 40 epochs. The best validation checkpoint was evaluated once on the held-out test set.

| Model | Parameters | Epochs | Test loss | Test perplexity | Top-1 | Top-5 | Seconds per epoch | Training time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Smoothed unigram | - | - | 5.702 | 299.59 | 11.46% | 31.60% | - | - |
| Simple RNN | 3.95M | 22 | 4.140 | 62.79 | 31.94% | 50.56% | 59.5 | 21.8 min |
| BiRNN | 6.61M | 33 | 4.140 | 62.81 | 32.00% | 50.61% | 93.9 | 51.7 min |
| LSTM | 4.25M | 26 | 3.913 | 50.03 | 34.14% | 53.55% | 68.9 | 29.9 min |
| GRU | 4.15M | 40 | **3.813** | **45.30** | **34.84%** | **54.79%** | 70.5 | 47.0 min |

The GRU reached the 40-epoch cap while its validation loss was still improving. The Simple RNN, BiRNN, and LSTM stopped earlier under the shared early-stopping rule; this is a limitation when interpreting total training time and fixed-epoch comparisons.

## Results and figures

The training histories, test metrics, generated samples, and figures are committed under `results/`.

![Learning curves](results/figures/learning_curves.svg)

![Held-out prediction metrics](results/figures/test_metrics.svg)

![Capacity and cost](results/figures/compute_cost.svg)

## Temperature study and generated-text evaluation

The generation sweep contains 7,200 poems: 4 models x 6 temperatures (0.3, 0.5, 0.8, 1.0, 1.3, 1.5) x 300 samples. Five fixed prompts were balanced within each model-temperature cell. Evaluation includes termination, exact five-line form, CMU-dictionary rhyme coverage and strict AABBA rhyme score, repeated-trigram rate, immediate repeats, training-line copying, distinct-1/2/3, and unique-poem share. Confidence intervals are 95% bootstrap percentile intervals (2,000 resamples).

Strict AABBA rhyme is the mean of four pair checks: lines 1-2, 3-4, 1-5, and 2-5. A pair scores one only when both end words are distinct, both are covered by CMUdict, and CMUdict lists them as rhymes. Identical endings, uncovered pairs, and malformed five-line outputs score zero. Under this definition, GRU scores 34.25% at temperature 0.3 and 33.67% at 0.5; the held-out human reference scores 94.25%, with 94.33% eligible-pair coverage.

The selection rule prioritises held-out prediction quality first, then balances rhyme, diversity, and repetition within the chosen model. GRU at 0.5 is preferable to 0.3 because strict rhyme falls by only 0.58 percentage points, while repeated trigrams fall from 3.85% to 0.78% and distinct-2 rises from 0.241 to 0.415. GRU and LSTM both produced five lines for every sample at 0.3, so five-line completion does not distinguish them. Across all models, distinct-2 rises from roughly 0.21-0.25 at temperature 0.3 to 0.88-0.90 at 1.5, while rhyme and form generally weaken.

![Temperature effects](results/sweep/temperature_effects.svg)

![Quality-diversity frontier](results/sweep/quality_diversity_frontier.svg)

## Blinded LLM judge scoring

An 80-poem blinded sample (4 models x 4 temperatures x 5 poems) was scored by Claude as an LLM judge on 1-5 coherence and contextual consistency without access to model or temperature labels. GRU at temperature 0.3 had the highest combined LLM-judge score (3.0/5, 95% bootstrap CI 2.6-3.4), tied in point estimate with LSTM at temperature 0.3. GRU at 0.5 scored 2.7/5. No independent human rating was collected, so these scores are supplementary rather than definitive evidence.

![Manual scores](results/sweep/manual_score_results.svg)

## Generated examples

The report compares the same prompt and sample index across all four models at temperatures 0.3, 0.5, and 1.0. Additional fixed-seed showcase outputs are available for the [Simple RNN](results/final_context64_rnn/generated_samples.md), [BiRNN](results/final_context64_birnn/generated_samples.md), [LSTM](results/final_context64_lstm/generated_samples.md), and [GRU](results/final_context64_gru/generated_samples.md). The complete 7,200-poem sweep is stored in `results/sweep/sweep.jsonl`.

## Reproduction

1. Create a Python environment and install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

2. Download the dataset as described in `data/README.md`.
3. Train one model (example):

   ```powershell
   python src/train_simple_rnn.py --data data/limerick_dataset_oedilf_v3.json --model-kind gru --context 64 --epochs 40 --patience 5 --output results/final_context64_gru
   ```

4. Verify checkpoints, recreate figures, run the sweep, and evaluate it:

   ```powershell
   python src/verify_final_checkpoints.py --data data/limerick_dataset_oedilf_v3.json
   python src/create_final_figures.py
   python src/run_generation_sweep.py --data data/limerick_dataset_oedilf_v3.json
   python src/evaluate_generation_sweep.py --data data/limerick_dataset_oedilf_v3.json
   python src/create_sweep_figures.py
   ```

Checkpoints and the raw dataset are intentionally excluded from version control because of their size. The committed JSON, CSV, SVG, and Markdown artefacts contain the reported evidence.

## Limitations

- One random seed was used.
- Models trained for different numbers of epochs because of validation-based early stopping.
- GRU still improved at the maximum epoch cap.
- The strict rhyme metric depends on CMU Pronouncing Dictionary coverage and treats uncovered or identical end-word pairs as failures.
- Manual scoring used one Claude LLM judge and five poems per cell; no independent human agreement check was collected.
- Perplexity measures next-token prediction rather than complete poetic quality.

## Repository contents

- `src/`: training, verification, sweep, evaluation, and figure scripts.
- `results/final_context64_*`: final metrics, histories, vocabulary files, and showcase samples.
- `results/figures/`: quantitative training figures and table.
- `results/sweep/`: raw generations, automatic metrics, LLM-judge outputs, and temperature figures.
- `data/README.md`: dataset provenance and download instructions.

