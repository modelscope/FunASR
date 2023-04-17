# Rules
All participants should adhere to the following rules to be eligible for the challenge.

- Data augmentation is allowed on the original training dataset, including, but not limited to, adding noise or reverberation, speed perturbation and tone change.

- The use of Test dataset in any form of non-compliance is strictly prohibited, including but not limited to use the Test dataset to fine-tune or train the model.

- Multi-system fusion is allowed, but the systems with same structure and different parameters is not encouraged.
  
- If the cpCER of the two systems on the Test dataset are the same, the system with lower computation complexity will be judged as the superior one.
  
- If the forced alignment is used to obtain the frame-level classification label, the forced alignment model must be trained on the basis of the data allowed by the corresponding sub-track.
  
- Shallow fusion is allowed to the end-to-end approaches, e.g., LAS, RNNT and Transformer, but the training data of the shallow fusion language model can only come from the transcripts of the allowed training dataset.
  
- The right of final interpretation belongs to the organizer. In case of special circumstances, the organizer will coordinate the interpretation.