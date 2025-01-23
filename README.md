# Henry_NLP
I make a decoder-only transformer based on the 2017 paper "Attention is all you need" by Vaswani el al.

## Architecture
I created a decoder-only transformer, mostly following implementation details in the 2017 paper "Attention is all you need" by Vaswani el al..

### Data
I used the CMU book summary dataset found [Here](https://www.cs.cmu.edu/~dbamman/booksummaries.html). This dataset was chosen because it is relatively small (compared to the internet) and focused on a subset of English, which I hoped the transformer could pick up.

### Tokenization
A pretrained HuggingFace tokenizer was used. I also tried with other tokenizers, but due to the limited size of the data, I beleive the pretrained tokenzier would more accurately describe English than a tokenizer trained specially on the dataset.

### Positional Encoding + Masked Multi-headed Attention
I used the positional encoding and masked multi-headed attention laid out in the 2017 paper "Attention is all you need" by Vaswani el al..

### Training
The model was trained using a context window of 128 tokens, thus the following results are all constrained to that window. Data was tokenized prior to training, and random token sequences of length 128 was used to train the transformer using each token in the sequence. This was done to allow the transformer to train on sequences of varying lengths and be more efficient than redrawing data for each length (1-128). All the hyperparameters used in training could be found in the code.

### Inference
I used top-p sampling to generate text using the trained transformer.

## Some Generated Results
a single AI fighter, is now being set on a mission to destroy the Enterprise. The group, trapped in a desperate battle with the Havenite forces, causing the survivors of the Vong arsenal system to keep the war from all of the fleet, leaving them stranded in the asteroid belt of battle. The Animorphs head back to their homes, but after a defeated Captain and the Taxxonts are killed. The next day, the Yeerks are chased by pirates and ...

of the mind of the Seeker - like combatants from the Voluntars - a ring that the Koest are unaffected by the Emperor. The Emperor is a prince who was previously presumed dead, though only a child who had a heart attack. Despite the warnings of the King, Axler states that he should not reveal the matter - she could die and, as his secret, he could not recover the Kraken and help her return. He chooses to go to the ...

of losing control over their skull, Kandeya and Bet are forced to reestablish the spacecraft and morphs the entire surface. It is not stated that the comet - like robots ( or giant ) are suddenly identified, and it is hinted that the organism is carrying the space shuttle. The people of the few that has suffered. However, the aliens receive the fact that they are experiencing an alien invasion force - the shuttle and terraforming its biocha, rendering it in their ...