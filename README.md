1. Prerequisites

Python Environment: Python 3.x.
Required Libraries: torch, transformers (for GPT-2), and others listed at the top of true_poetry.py.
Linguistic Data Files: The program requires the following files in the execution directory to define meter and rhyme constraints:

pronounce.txt (CMU Pronouncing Dictionary)

rhyming_tokens.p

syllable_tokens.p

stress_tokens.p

rhymesets.p

2 .How to Run
```bash
python true_poetry.py
```
The program will prompt you for two inputs:

Starting Prompt: A phrase or line to begin the poem (e.g., "The mist descended, soft and cool").

Scheme: Choose one of the predefined poetic forms: ballad, limerick, couplets, mini-couplets, or sonnet.

