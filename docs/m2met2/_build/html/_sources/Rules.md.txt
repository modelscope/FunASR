# Rules
All participants should adhere to the following rules to be eligible for the challenge.

- Data augmentation is allowed on the original training dataset, including, but not limited to, adding noise or reverberation, speed perturbation and tone change.

- Participants are permitted to use the Eval set for model training, but it is not allowed to use the Test set for this purpose. Instead, the Test set should only be utilized for parameter tuning and model selection. Any use of the Test-2023 dataset that violates these rules is strictly prohibited, including but not limited to the use of the Test set for fine-tuning or training the model.
  
- If the cpCER of the two systems on the Test dataset are the same, the system with lower computation complexity will be judged as the superior one.
  
- If the forced alignment is used to obtain the frame-level classification label, the forced alignment model must be trained on the basis of the data allowed by the corresponding sub-track.
  
- Shallow fusion is allowed to the end-to-end approaches, e.g., LAS, RNNT and Transformer, but the training data of the shallow fusion language model can only come from the transcripts of the allowed training dataset.
  
- The right of final interpretation belongs to the organizer. In case of special circumstances, the organizer will coordinate the interpretation.