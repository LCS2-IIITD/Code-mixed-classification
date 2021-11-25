## Hierarchical Transformer (HIT)

This repository contains the source code for HIT (Hierarchical Transformer). It uses Fused Attention Mechanism (FAME) for learning representation learning from code-mixed texts. We evaluate HIT on code-mixed sequence classification, token classification and generative tasks. We also experiment with pre-training objectives such as Masked Language Modeling and Zero Shot Learning. 

![HIT](https://github.com/LCS2-IIITD/HIT-ACL2021-Codemixed-Representation/blob/main/image/model.png)

We publish the datasets (publicly available) and the experimental setup used for different tasks.

### Installation for experiments

	$ pip install -r requirements.txt

### Commands to run

#### Sentiment Analysis

	$ cd experiments && python experiments_hindi_sentiment.py \
			--train_data ../data/hindi_sentiment/IIITH_Codemixed.txt \
			--model_save_path ../models/model_hindi_sentiment/

#### PoS (Parts-of-Speech) Tagging 

	$ cd experiments && python experiments_hindi_POS.py \
			--train_data '../data/POS Hindi English Code Mixed Tweets/POS Hindi English Code Mixed Tweets.tsv' \
			--model_save_path ../models/model_hindi_pos/

#### Named Entity Recognition (NER)

    $ cd experiments && python experiments_hindi_NER.py\
    		--train_data '../data/NER/NER Hindi English Code Mixed Tweets.tsv' \
			--model_save_path ../models/model_hindi_NER/

#### Machine Translation (MT)

	$ cd experiments && python nmt.py \
			--data_path '../data/IITPatna-CodeMixedMT' \
			--model_save_path ../models/model_hindi_NMT/
			
#### Sarcasm Detection

	$ cd experiments && python experiments_hindi_SH.py \
			--data_path '../data/MSH-Comics-Sarcasm/hindi_sarcasm.txt' \
			--model_save_path ../models/model_hindi_sarcasm/
			
#### Humour Classification

	$ cd experiments && python experiments_hindi_SH.py \
			--data_path '../data/MSH-Comics-Sarcasm/hindi_humour.txt' \
			--model_save_path ../models/model_hindi_humour/
			
#### Response Prediction

	$ cd experiments && python experiments_response_prediction.py \
			--data_path '../data/IITMadras-CodeMixResponse/hindi' \
			--model_save_path ../models/model_hindi_response/
			
#### Intent Detection

	$ cd experiments && python experiments_intent_detection.py \
			--data_path '../data/IITMadras-CodeMixIntent/GCN-SeA/hi-dstc2' \
			--model_save_path ../models/model_hindi_intents/

#### Slot Filling

	$ cd experiments && python experiments_slot_filling.py \
			--data_path '../data/IITMadras-CodeMixIntent/GCN-SeA/hi-dstc2' \
			--model_save_path ../models/model_hindi_slots/

#### MLM pre-training

	$ cd experiments && python experiments_hindi_MLM.py \
			--ismlm True
			--model_save_path ../models/model_hindi_mlm/

#### ZSL pre-training
			
	$ cd experiments && python experiments_hindi_ZSL.py \
			--model_save_path ../models/model_hindi_zsl/				
				


### Citation
If you find this repo useful, please cite our paper:
```BibTex
@inproceedings{,
  author    = {Ayan Sengupta and
               Tharun Suresh and
               Tanmoy Chakraborty and
               Md. Shad Akhtar},
  title     = {A Comprehensive Understanding of Code-mixed Language Semantics using Hierarchical Transformer},
  booktitle = {},
  publisher = {},
  year      = {},
  url       = {},
  doi       = {},
}
```
