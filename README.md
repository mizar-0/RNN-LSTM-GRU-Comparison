# Word Level Limerick Generation with RNN LSTM and GRU

## Problem statement and objective

The assignment requires a word-level autoregressive text-generation system that implements Simple RNN, Bidirectional RNN, LSTM, and GRU models under comparable experimental conditions. The models must be evaluated for next-word prediction and generated-text quality, and the effect of at least three sampling temperatures must be analysed using the same seed sequences.

This project applies that comparison to limerick generation. Each model predicts the next token from a 64-token context and generates text sequentially from shared seed prompts. The objective is to compare predictive performance, training cost, coherence, contextual consistency, diversity, repetition, five-line form, and rhyme, then identify the most effective model and sampling temperature for this dataset.

The final recommendation is **GRU at temperature 0.5**. GRU achieved the best held-out prediction metrics. Temperature 0.5 retained nearly all of its strict rhyme performance at 0.3 while sharply reducing repeated trigrams and increasing distinct bigrams.

## Dataset

The experiments use the [Limerick Dataset (OEDILF v3)](https://doi.org/10.5281/zenodo.5722527), hosted on Zenodo under [CC BY 3.0 US](https://creativecommons.org/licenses/by/3.0/us/). The final run filtered the full source to 65,886 five-line limericks from 1,296 authors, split by poem into 80% train, 10% validation, and 10% test sets. Poems contain a mean of 29.76 words (median 30; 10th-90th percentile 26-34 words). With the 10,000-token training vocabulary, 10.16% of lexical training tokens and 9.92% of lexical test tokens map to `<unk>`. The token distribution is strongly concentrated in function words: `the`, `a`, `and`, `to`, and `in` are the five most frequent lexical tokens. See [data/README.md](data/README.md) for download instructions and a content note about crude or adult user-submitted material.

## Models and common training protocol

All models used the same full dataset split, seed (42), vocabulary cap (10,000), 128-dimensional embeddings, 256 hidden units, dropout 0.3, Adam learning rate 0.001, batch size 512, and a 64-token context window. Early stopping used validation loss with patience 5 and a maximum of 40 epochs. The best validation checkpoint was evaluated once on the held-out test set.

- **Simple RNN:** a unidirectional recurrent baseline using its final hidden state for next-word prediction.
- **Bidirectional RNN:** processes each input context in both directions and concatenates the final forward and backward states.
- **LSTM:** uses input, output, and forget gates with a cell state to preserve longer-range information.
- **GRU:** uses update and reset gates to control recurrent memory with fewer parameters than the LSTM configuration.

All four recurrent representations feed a linear projection over the same 10,000-token output vocabulary.

| Model | Parameters | Epochs | Test loss | Test perplexity | Top-1 | Top-5 | Seconds per epoch | Training time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Smoothed unigram | - | - | 5.702 | 299.59 | 11.46% | 31.60% | - | - |
| Simple RNN | 3.95M | 22 | 4.140 | 62.79 | 31.94% | 50.56% | 59.5 | 21.8 min |
| BiRNN | 6.61M | 33 | 4.140 | 62.81 | 32.00% | 50.61% | 93.9 | 51.7 min |
| LSTM | 4.25M | 26 | 3.913 | 50.03 | 34.14% | 53.55% | 68.9 | 29.9 min |
| GRU | 4.15M | 40 | **3.813** | **45.30** | **34.84%** | **54.79%** | 70.5 | 47.0 min |

The GRU reached the 40-epoch cap while its validation loss was still improving. The Simple RNN, BiRNN, and LSTM stopped earlier under the shared early-stopping rule; this is a limitation when interpreting total training time and fixed-epoch comparisons.

### Why the architectures behaved differently

The results are consistent with the memory mechanisms of the four recurrent layers, although the experiment does not establish architecture as the sole cause. The Simple RNN repeatedly compresses the 64-token history into one tanh hidden state. Gradients and older information can decay across many recurrent steps, so it learned common local word transitions but was less effective at dependencies spanning several limerick lines. This is consistent with its relatively high perplexity and near-chance rhyme score.

The BiRNN reads the same completed context window in both directions and concatenates two 256-unit states. It does not see the target word or any future generated text, so bidirectionality supplies a second ordering of the existing past rather than the future context that normally makes bidirectional networks useful. Its doubled recurrent representation increases the output projection and total parameter count substantially. The additional computation therefore produced almost the same perplexity as the Simple RNN, while the reverse state may place less emphasis on the most recent words that are especially important for autoregressive prediction. This also helps explain its weaker five-line completion.

The LSTM's cell state and input, forget, and output gates provide a more stable path for information and gradients than a vanilla RNN. It consequently modelled the 64-token context more effectively and improved perplexity and the blinded coherence scores. However, an LSTM does not explicitly plan an AABBA rhyme scheme: average next-word loss is dominated by common tokens, whereas rhyming line endings are sparse decisions separated by many steps. Its better token prediction therefore did not translate into strong strict rhyme.

The GRU uses reset and update gates to retain useful history while adapting its candidate state, but has no separate cell state and fewer parameters than the LSTM used here. That simpler gated path may have been easier to optimise on this dataset and may have retained line-position and earlier-ending cues strongly enough to reproduce rhyme patterns. The same tendency becomes overconfident at low temperature: GRU's strong rhyme at 0.3 coincides with its highest repeated-trigram rate and repeated endings such as *man*. Temperature 0.5 reduces this architectural-plus-decoding tendency without removing most of the learned rhyme. These explanations are hypotheses supported by the observed pattern; one seed, unequal stopping epochs, and GRU's continued improvement at epoch 40 prevent a clean causal attribution.

## Results and figures

The training histories, test metrics, generated samples, and figures are committed under `results/`.

![Learning curves](results/figures/learning_curves.svg)

![Held-out prediction metrics](results/figures/test_metrics.svg)

![Capacity and cost](results/figures/compute_cost.svg)

## Temperature study and generated-text evaluation

The generation sweep contains 7,200 poems: 4 models x 6 temperatures (0.3, 0.5, 0.8, 1.0, 1.3, 1.5) x 300 samples. Five fixed prompts were balanced within each model-temperature cell. Evaluation includes termination, exact five-line form, CMU-dictionary rhyme coverage and strict AABBA rhyme score, repeated-trigram rate, immediate repeats, training-line copying, distinct-1/2/3, and unique-poem share. Confidence intervals are 95% bootstrap percentile intervals (2,000 resamples).

Strict AABBA rhyme is the mean of four pair checks: lines 1-2, 3-4, 1-5, and 2-5. A pair scores one only when both end words are distinct, both are covered by CMUdict, and CMUdict lists them as rhymes. Identical endings, uncovered pairs, and malformed five-line outputs score zero. Under this definition, GRU scores 34.25% at temperature 0.3 and 33.67% at 0.5; the held-out human reference scores 94.25%, with 94.33% eligible-pair coverage.

Distinct-2 is the number of unique lexical bigrams divided by all lexical bigrams pooled within a model-temperature cell. Control tokens (`<bos>`, `<eol>`, `<eos>`, `<pad>`, and `<unk>`) and punctuation are excluded. Repeated-trigram rate is computed within each poem as excess occurrences of repeated lexical trigrams divided by all lexical trigrams, then averaged across the 300 poems. End-of-line tokens are therefore excluded from both metrics.

GRU is selected on two converging grounds: it has the best held-out next-word prediction, and it is the only architecture with substantial strict rhyme in the generation sweep. Temperature is then selected within GRU by balancing rhyme, diversity, and repetition. GRU at 0.5 is preferable to 0.3 because strict rhyme falls by only 0.58 percentage points, while repeated trigrams fall from 3.85% to 0.78% and distinct-2 rises from 0.241 to 0.415. GRU and LSTM both produced five lines for every sample at 0.3, so five-line completion does not distinguish them. Across all models, distinct-2 rises from roughly 0.21-0.25 at temperature 0.3 to 0.88-0.90 at 1.5, while rhyme and form generally weaken.

| Model | Temperature | Five-line rate | Strict rhyme | Distinct-2 | Repeated trigrams |
|---|---:|---:|---:|---:|---:|
| Simple RNN | 0.3 | 100.0% | 1.50% | 0.212 | 1.01% |
| Simple RNN | 0.5 | 99.7% | 0.92% | 0.403 | 0.22% |
| Simple RNN | 1.0 | 96.7% | 1.17% | 0.769 | 0.00% |
| BiRNN | 0.3 | 87.3% | 1.33% | 0.248 | 1.91% |
| BiRNN | 0.5 | 90.0% | 1.25% | 0.433 | 0.31% |
| BiRNN | 1.0 | 91.7% | 0.92% | 0.768 | 0.01% |
| LSTM | 0.3 | 100.0% | 2.58% | 0.237 | 1.64% |
| LSTM | 0.5 | 100.0% | 2.25% | 0.426 | 0.25% |
| LSTM | 1.0 | 99.3% | 0.58% | 0.767 | 0.00% |
| GRU | 0.3 | 100.0% | 34.25% | 0.241 | 3.85% |
| GRU | 0.5 | 100.0% | 33.67% | 0.415 | 0.78% |
| GRU | 1.0 | 98.7% | 10.67% | 0.759 | 0.00% |

![Temperature effects](results/sweep/temperature_effects.svg)

![Quality-diversity frontier](results/sweep/quality_diversity_frontier.svg)

## Blinded LLM judge scoring

An 80-poem blinded sample (4 models x 4 temperatures x 5 poems) was scored by Claude as an LLM judge on 1-5 coherence and contextual consistency without model or temperature labels in the scoring sheet. At temperature 0.3, coherence/contextual-consistency scores were 2.6/2.0 for Simple RNN, 2.4/2.2 for BiRNN, 3.0/3.0 for LSTM, and 2.6/3.4 for GRU. At 0.5, they were 2.0/2.0, 2.4/2.4, 2.6/2.6, and 2.6/2.8 respectively. No independent human rating was collected. The judge had also previously seen some labelled examples from the same sweep, so recognition cannot be ruled out and the blinding was incomplete. These scores are supplementary rather than definitive evidence.

![Manual scores](results/sweep/manual_score_results.svg)

## Generated examples

The report compares the same prompt and sample index across all four models at temperatures 0.3, 0.5, and 1.0. Additional fixed-seed showcase outputs are available for the [Simple RNN](results/final_context64_rnn/generated_samples.md), [BiRNN](results/final_context64_birnn/generated_samples.md), [LSTM](results/final_context64_lstm/generated_samples.md), and [GRU](results/final_context64_gru/generated_samples.md). The complete 7,200-poem sweep is stored in `results/sweep/sweep.jsonl`.

The models were trained from scratch on about 66,000 task-specific poems, without pretrained language knowledge or an explicit planning objective for rhyme, narrative setup, and punchline. The next-token objective can learn local phrasing and form, but the 10,000-token vocabulary maps about one tenth of lexical tokens to `<unk>`, removing many rare names and content words. These constraints help explain why the outputs are often locally fluent but weak as complete limericks.

## Reproduction

The required packages are listed in [`requirements.txt`](requirements.txt): PyTorch, NumPy, pandas, Matplotlib, tqdm, and `pronouncing` for CMU-dictionary rhyme evaluation.

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
- Manual scoring used one Claude LLM judge and five poems per cell; no independent human agreement check was collected, and prior exposure to labelled samples made the blinding incomplete.
- Perplexity measures next-token prediction rather than complete poetic quality.

## Conclusion

GRU is the best-performing architecture in this experiment because it achieved the lowest test loss and perplexity, the highest top-1 and top-5 accuracy, and substantially stronger strict rhyme than the other models. Temperature 0.5 is the recommended generation setting: compared with 0.3, it retained nearly all of the GRU's rhyme score while increasing distinct-2 and sharply reducing repeated trigrams. The generated poems still lack reliable narrative coherence and punchlines, so this conclusion is limited to the evaluated from-scratch models, dataset, seed, and metrics.

## Repository contents

- `src/`: training, verification, sweep, evaluation, and figure scripts.
- `results/final_context64_*`: final metrics, histories, vocabulary files, and showcase samples.
- `results/figures/`: quantitative training figures and table.
- `results/sweep/`: raw generations, automatic metrics, LLM-judge outputs, and temperature figures.
- `data/README.md`: dataset provenance and download instructions.

