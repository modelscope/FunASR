import json

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
import yaml


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.model_config = model_config = json.loads(args['model_config'])

        output0_config = pb_utils.get_output_config_by_name(model_config, 'OUTPUT0')
        self.out0_dtype = pb_utils.triton_string_to_numpy(output0_config['data_type'])

        parameters = self.model_config['parameters']
        self.vocab_dict = self.load_vocab(parameters['vocabulary']['string_value'])
        assert len(self.vocab_dict) == 8404, len(self.vocab_dict)

        try:
            import torch_blade
        except Exception:
            print('Failed to load torch_blade')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.model = torch.jit.load(parameters['model_path']['string_value'])
        self.device = 'cuda' if next(self.model.parameters()).is_cuda else 'cpu'

    def load_vocab(self, vocab_file):
        """
        load lang_char.txt
        """
        with open(str(vocab_file), 'rb') as f:
            config = yaml.load(f, Loader=yaml.Loader)
        return config['token_list']

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.

        speech = []
        speech_len = []
        for request in requests:
            # Perform inference on the request and append it to responses list...
            _speech = pb_utils.get_input_tensor_by_name(request, 'speech')
            _speech_len = pb_utils.get_input_tensor_by_name(request, 'speech_lengths')
            _speech = _speech.as_numpy()
            _speech = torch.tensor(_speech, dtype=torch.float32).to(self.device)
            _speech_len = int(_speech_len.as_numpy())
            while _speech.dim() > 2:
                _speech = _speech.squeeze(0)
            speech.append(_speech)
            speech_len.append(int(_speech_len))

        max_len = max(speech_len)
        for i, _speech in enumerate(speech):
            pad = (0, 0, 0, max_len - speech_len[i])
            speech[i] = torch.nn.functional.pad(_speech, pad, mode='constant', value=0)
        feats = torch.stack(speech)
        feats_len = torch.tensor(speech_len, dtype=torch.int32).to(self.device)

        with torch.no_grad():
            logits = self.model(feats, feats_len)[0]

        def replace_space(tokens):
            return [i if i != '<space>' else ' ' for i in tokens]

        yseq = logits.argmax(axis=-1).tolist()
        token_int = [list(filter(lambda x: x not in (0, 2), y)) for y in yseq]
        tokens = [[self.vocab_dict[i] for i in t] for t in token_int]
        hyps = [''.join(replace_space(t)).encode('utf-8') for t in tokens]
        responses = []
        for i in range(len(requests)):
            sents = np.array(hyps[i: i + 1])
            out0 = pb_utils.Tensor('OUTPUT0', sents.astype(self.out0_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out0])
            responses.append(inference_response)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
