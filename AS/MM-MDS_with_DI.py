import os
import numpy as np
import pandas as pd
import json
import warnings
import logging
import gc
import random
import math
import pickle
import re
import ast
import torch
from tqdm import tqdm
from typing import Optional
from datetime import datetime
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
#from evaluate import load
from transformers import AutoTokenizer, AutoModel
#bertscore = load("bertscore")
from collections import UserDict

from rouge_score.rouge_scorer import RougeScorer
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
#from torchmetrics.text.bert import BERTScore
from torchmetrics.functional.text.bert import bert_score
import random
from sentence_transformers import SentenceTransformer
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import warnings
from torch import Tensor

Sem_model = SentenceTransformer('bert-base-nli-mean-tokens')
print("------------------------------------------------------------------------------------")

def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999
    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_random_seed(42)



warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

print(torch.cuda.is_available())
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:3")
    print("Using GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

from transformers.models.bart.modeling_bart import (
    BartPretrainedModel,
    BartDecoder,
    BartLearnedPositionalEmbedding,
    BartEncoderLayer,
    shift_tokens_right,
    _expand_mask
)
#_make_causal_mask,

from transformers.models.bart.configuration_bart import BartConfig

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, MSELoss

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from transformers.modeling_utils import PreTrainedModel, unwrap_model

from transformers import (
    BartTokenizerFast,
    AdamW
)

from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput
)


from transformer_encoder import TransformerEncoder


SOURCE_COLUMN = 'Conversation'
TARGET_COLUMN_1 = "Doctor_Summary" #change
# TARGET_COLUMN_2 = "Doctor_Summary" #change
TARGET_COLUMN_3 =  "Overall_Summary"
VISUAL_INPUT_PATH = 'Image_features.json'

VISUAL_DIM = 768
VISUAL_MAX_LEN = 1 

KG_DIM = 768
KG_MAX_LEN = 1


SOURCE_MAX_LEN = 360
# TARGET_MAX_LEN = 50
TARGET_MAX_LEN_1 = 111
# TARGET_MAX_LEN_2 = 111
TARGET_MAX_LEN_3 = 111
MAX_UTTERANCES = 25

# BATCH_SIZE = 32
BATCH_SIZE = 8
MAX_EPOCHS = 30

BASE_LEARNING_RATE = 5e-6
NEW_LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4

NUM_BEAMS = 5
EARLY_STOPPING = True
NO_REPEAT_NGRAM_SIZE = 3

Encoder_Cls_dim = 768
Num_labels = 9
EARLY_STOPPING_THRESHOLD = 5


MODEL_OUTPUT_DIR = '/home/jupyter/MMCSG/MMCS/'
RESULT_OUTPUT_DIR = '/home/jupyter/MMCSG/MMCS/'

def read_json_data(path):
    f = open(path)
    data = json.load(f)
    f.close()
    del f
    gc.collect()
    return data

def preprocess_dataset(dataset):

    source = list(dataset[SOURCE_COLUMN].values)
    # print(type(source))
    model_inputs = TOKENIZER(source,
                                    max_length=SOURCE_MAX_LEN,
                                    padding='max_length',
                                    truncation=True)


    target_1 = list(dataset[TARGET_COLUMN_1].values) #change
    with TOKENIZER.as_target_tokenizer():
        labels_1 = TOKENIZER(target_1,               #change
                            max_length=TARGET_MAX_LEN_1,
                            padding='max_length',
                            truncation=True)  
       
        # IMP:
        # Replace all tokenizer.pad_token_id in the labels by -100 to ignore padding tokens in the loss.
        labels_1['input_ids'] = [[(l if l != TOKENIZER.pad_token_id else -100) for l in label] for label in labels_1["input_ids"]] #change

    # target_2 = list(dataset[TARGET_COLUMN_2].values) #change
    # with TOKENIZER.as_target_tokenizer():
    #     labels_2 = TOKENIZER(target_2,               #change
    #                         max_length=TARGET_MAX_LEN_2,
    #                         padding='max_length',
    #                         truncation=True)  
       
    #     # IMP:
    #     # Replace all tokenizer.pad_token_id in the labels by -100 to ignore padding tokens in the loss.
    #     labels_2['input_ids'] = [[(l if l != TOKENIZER.pad_token_id else -100) for l in label] for label in labels_2["input_ids"]] #change

    target_3 = list(dataset[TARGET_COLUMN_3].values) #change
    with TOKENIZER.as_target_tokenizer():
        labels_3 = TOKENIZER(target_3,               #change
                            max_length=TARGET_MAX_LEN_3,
                            padding='max_length',
                            truncation=True)  
       
        # IMP:
        # Replace all tokenizer.pad_token_id in the labels by -100 to ignore padding tokens in the loss.
        labels_3['input_ids'] = [[(l if l != TOKENIZER.pad_token_id else -100) for l in label] for label in labels_3["input_ids"]] #change



    print(model_inputs.keys())
    model_inputs['input_ids'] = torch.tensor([i for i in model_inputs['input_ids']], dtype=torch.long, device=DEVICE)
    model_inputs['attention_mask'] = torch.tensor([a for a in model_inputs['attention_mask']], dtype=torch.long, device=DEVICE)
    model_inputs['visual_input'] = torch.tensor(dataset["Img_features"]).to(DEVICE)
    model_inputs['kg_input'] = torch.tensor(dataset["Concept"]).to(DEVICE)
    model_inputs['labels_1'] = torch.tensor([l for l in labels_1['input_ids']], dtype=torch.long, device=DEVICE) #change
    # model_inputs['labels_2'] = torch.tensor([l for l in labels_2['input_ids']], dtype=torch.long, device=DEVICE) #change
    model_inputs['labels_3'] = torch.tensor([l for l in labels_3['input_ids']], dtype=torch.long, device=DEVICE) #change


    del target_1
    # del target_2
    del labels_1
    # del labels_2
    del labels_3
    gc.collect()
    return model_inputs


def set_up_data_loader(dataset):
    dataset = preprocess_dataset(dataset)
    dataset = TensorDataset(dataset['input_ids'],
                            dataset['attention_mask'],
                            dataset['visual_input'],
                            dataset['kg_input'],
                            dataset['labels_1'], #change
                            # dataset['labels_2'], #change
                            dataset['labels_3'], #change
                            )
    print(len(dataset))
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

import transformers

print("Transformers version", transformers.__version__)  
# print(1/0)  

# from transformers import generation

import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput
from transformers.models.auto import (
    MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    MODEL_FOR_VISION_2_SEQ_MAPPING,
)

from transformers.utils import (
        ModelOutput,
    logging,
    # GENERATION_CONFIG_NAME,
        PushToHubMixin,
        # cached_file,
        # download_url,
        # extract_commit_hash,
        # is_remote_url,
    )
from transformers import __version__
from transformers.configuration_utils import PretrainedConfig

from transformers.configuration_utils import PretrainedConfig

from abc import ABC, abstractmethod
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import (
    # ClassifierFreeGuidanceLogitsProcessor,
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    LogitsProcessor,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    # SequenceBiasLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
)
from transformers.generation.stopping_criteria import (
    # MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)

# class ClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):
#     r"""Logits processor for classifier free guidance (CFG). The scores are split over the batch dimension,
#     where the first half correspond to the conditional logits (predicted from the input prompt) and the second half
#     correspond to the unconditional logits (predicted from an empty or 'null' prompt). The processor computes a
#     weighted average across the conditional and unconditional logits, parameterised by the `guidance_scale`.

#     Args:
#         guidance_scale (float):
#             The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
#             Higher guidance scale encourages the model to generate samples that are more closely linked to the input
#             prompt, usually at the expense of poorer quality.
#     """

#     def __init__(self, guidance_scale):
#         if guidance_scale > 1:
#             self.guidance_scale = guidance_scale
#         else:
#             raise ValueError(
#                 "Require guidance scale >1 to use the classifier free guidance processor, got guidance scale "
#                 f"{guidance_scale}."
#             )

#     # @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
#         # simple check to make sure we have compatible batch sizes between our
#         # logits scores (cond + uncond) and input ids (cond only)
#         if scores.shape[0] != 2 * input_ids.shape[0]:
#             raise ValueError(
#                 f"Logits should have twice the batch size of the input ids, the first half of batches corresponding to "
#                 f"the conditional inputs, and the second half of batches corresponding to the unconditional inputs. Got "
#                 f"batch size {scores.shape[0]} for the logits and {input_ids.shape[0]} for the input ids."
#             )
#         unguided_bsz = scores.shape[0] // 2
#         cond_logits, uncond_logits = scores.split(unguided_bsz, dim=0)
#         scores = uncond_logits + (cond_logits - uncond_logits) * self.guidance_scale
#         return scores

# class SequenceBiasLogitsProcessor(LogitsProcessor):
#     """
#     [`LogitsProcessor`] that applies an additive bias on sequences. The bias is applied to the last token of a sequence
#     when the next generated token can complete it. Consequently, to take the most of biasing sequences with more than
#     one token, consider using beam methods (to gracefully work around partially completed sequences that have a
#     negative bias) and applying the bias to their prefixes (to ensure the bias is applied earlier).

#     <Tip>

#     In order to get the token ids of the sequences that you want to bias, make sure to set `add_prefix_space=True` when
#     initializing the tokenizer, and use `tokenizer(bad_words, add_special_tokens=False).input_ids`. The
#     `add_prefix_space` argument is only supported for some slow tokenizers, as fast tokenizers' prefixing behaviours
#     come from `pre tokenizers`. Read more [here](https://huggingface.co/docs/tokenizers/api/pre-tokenizers).

#     </Tip>

#     Args:
#         sequence_bias (`Dict[Tuple[int], float]`):
#             Dictionary that maps a sequence of tokens to its bias term. Positive biases increase the odds of the
#             sequence being selected, while negative biases do the opposite. If a sequence has a length of 1, its bias
#             will always be applied. Otherwise, the bias will only be applied if the sequence in question is about to be
#             completed (in the token selection step after this processor is applied).

#     Examples:

#     ```python
#     >>> from transformers import AutoTokenizer, AutoModelForCausalLM

#     >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
#     >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
#     >>> inputs = tokenizer(["The full name of Donald is Donald"], return_tensors="pt")

#     >>> summary_ids = model.generate(inputs["input_ids"], max_new_tokens=4)
#     >>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0])
#     The full name of Donald is Donald J. Trump Jr

#     >>> # Now let's control generation through a bias. Please note that the tokenizer is initialized differently!
#     >>> tokenizer_with_prefix_space = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)


#     >>> def get_tokens_as_tuple(word):
#     ...     return tuple(tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0])


#     >>> # If we add a negative bias without beam search, it may become "stuck" in a prefix without good continuations
#     >>> sequence_bias = {get_tokens_as_tuple("Trump"): -10.0}
#     >>> biased_ids = model.generate(inputs["input_ids"], max_new_tokens=4, sequence_bias=sequence_bias)
#     >>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
#     The full name of Donald is Donald J. Donald,

#     >>> biased_ids = model.generate(inputs["input_ids"], max_new_tokens=4, num_beams=4, sequence_bias=sequence_bias)
#     >>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
#     The full name of Donald is Donald Rumsfeld,

#     >>> # We can also add a positive bias to nudge the model towards specific tokens or continuations
#     >>> sequence_bias = {get_tokens_as_tuple("Donald Duck"): 10.0}
#     >>> biased_ids = model.generate(inputs["input_ids"], max_new_tokens=4, num_beams=4, sequence_bias=sequence_bias)
#     >>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
#     The full name of Donald is Donald Duck.
#     ```
#     """

#     def __init__(self, sequence_bias: Dict[Tuple[int], float]):
#         self.sequence_bias = sequence_bias
#         self._validate_arguments()

#         # Bias variables that will be populated on the first call (for retrocompatibility purposes, the vocabulary size
#         # is infered in the first usage, which inhibits initializing here)
#         self.sequences_length_greater_than_1 = []
#         self.length_1_bias = None
#         self.length_greather_than_1_bias = None
#         self.prepared_bias_variables = False

#     # @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
#         # 1 - Prepares the bias tensors. This is only needed the first time the logit processor is called.
#         if not self.prepared_bias_variables:
#             self._prepare_bias_variables(scores)

#         # 2 - prepares an empty bias to add
#         bias = torch.zeros_like(scores)

#         # 3 - include the bias from length = 1
#         bias += self.length_1_bias

#         # 4 - include the bias from length > 1, after determining which biased sequences may be completed.
#         # `matching_mask` is a (batch_size, vocab_size) boolean mask that is True for all tokens whose corresponding
#         # bias should be applied. The bias is applied on the last token of the sequence, if (and only if) the sequence
#         # may become complete this iteration.
#         matching_mask = torch.zeros_like(scores, dtype=torch.bool)
#         for sequence_ids in self.sequences_length_greater_than_1:
#             if len(sequence_ids) > input_ids.shape[1]:  # the sequence is longer than the context, ignore
#                 continue
#             prefix_length = len(sequence_ids) - 1
#             last_token = sequence_ids[-1]
#             matching_rows = torch.eq(
#                 input_ids[:, -prefix_length:],
#                 torch.tensor(sequence_ids[:-1], dtype=input_ids.dtype, device=input_ids.device),
#             ).prod(dim=1)
#             matching_mask[:, last_token] |= matching_rows.bool()
#         bias += torch.where(
#             matching_mask,
#             self.length_greather_than_1_bias,
#             torch.tensor(0.0, device=self.length_greather_than_1_bias.device),
#         )

#         # 5 - apply the bias to the scores
#         scores = scores + bias
#         return scores

#     def _prepare_bias_variables(self, scores: torch.FloatTensor):
#         vocabulary_size = scores.shape[-1]
#         sequence_bias = self.sequence_bias
#         tokens_with_bias = []

#         # Check biased tokens out of bounds
#         invalid_biases = []
#         for sequence_ids in sequence_bias:
#             for token_id in sequence_ids:
#                 if token_id >= vocabulary_size:
#                     invalid_biases.append(token_id)
#         if len(invalid_biases) > 0:
#             raise ValueError(
#                 f"The model vocabulary size is {vocabulary_size}, but the following tokens were being biased: "
#                 f"{invalid_biases}"
#             )

#         # Precompute the bias tensors to be applied. Sequences of length 1 are kept separately, as they can be applied
#         # with simpler logic.
#         self.length_1_bias = torch.zeros((vocabulary_size,), dtype=torch.float).to(scores.device)
#         self.length_greather_than_1_bias = torch.zeros((vocabulary_size,), dtype=torch.float).to(scores.device)
#         for sequence_ids, bias in sequence_bias.items():
#             if len(sequence_ids) == 1:
#                 self.length_1_bias[sequence_ids[-1]] = bias
#             else:
#                 self.sequences_length_greater_than_1.append(sequence_ids)
#                 if self.length_greather_than_1_bias[sequence_ids[-1]] != 0.0:
#                     raise ValueError(
#                         "Setting a bias on sequences that share a common token termination is not yet supported. "
#                         "Please open an issue if you see this error message (after checking that it doesn't already "
#                         "exist)."
#                     )
#                 self.length_greather_than_1_bias[sequence_ids[-1]] = bias
#             tokens_with_bias.append(sequence_ids[-1])

#         self.prepared_bias_variables = True

#     def _validate_arguments(self):
#         sequence_bias = self.sequence_bias
#         if not isinstance(sequence_bias, dict) or len(sequence_bias) == 0:
#             raise ValueError(f"`sequence_bias` has to be a non-empty dictionary, but is {sequence_bias}.")
#         if any(not isinstance(sequence_ids, tuple) for sequence_ids in sequence_bias.keys()):
#             raise ValueError(f"`sequence_bias` has to be a dict with tuples as keys, but is {sequence_bias}.")
#         if any(
#             any((not isinstance(token_id, (int, np.integer)) or token_id < 0) for token_id in sequence_ids)
#             or len(sequence_ids) == 0
#             for sequence_ids in sequence_bias.keys()
#         ):
#             raise ValueError(
#                 f"Each key in `sequence_bias` has to be a non-empty tuple of positive integers, but is "
#                 f"{sequence_bias}."
#             )
#         if any(not isinstance(bias, float) for bias in sequence_bias.values()):
#             raise ValueError(f"`sequence_bias` has to be a dict with floats as values, but is {sequence_bias}.")


class MaxLengthCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generated number of tokens exceeds `max_length`. Keep
    in mind for decoder-only type of transformers, this will include the initial prompted tokens.

    Args:
        max_length (`int`):
            The maximum length that the output sequence can have in number of tokens.
        max_position_embeddings (`int`, `optional`):
            The maximum model length, as defined by the model's `config.max_position_embeddings` attribute.
    """

    def __init__(self, max_length: int, max_position_embeddings: Optional[int] = None):
        self.max_length = max_length
        self.max_position_embeddings = max_position_embeddings

    # @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        cur_len = input_ids.shape[-1]
        is_done = cur_len >= self.max_length
        if self.max_position_embeddings is not None and not is_done and cur_len >= self.max_position_embeddings:
            logger.warning_once(
                "This is a friendly reminder - the current text generation call will exceed the model's predefined "
                f"maximum length ({self.max_position_embeddings}). Depending on the model, you may observe "
                "exceptions, performance degradation, or nothing at all."
            )
        return is_done


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.generation.streamers import BaseStreamer



logger = logging.get_logger(__name__)

class BeamScorer(ABC):
    """
    Abstract base class for all beam scorers that are used for [`~PreTrainedModel.beam_search`] and
    [`~PreTrainedModel.beam_sample`].
    """

    @abstractmethod
    # @add_start_docstrings(PROCESS_INPUTS_DOCSTRING)
    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        raise NotImplementedError("This is an abstract method.")

    @abstractmethod
    # @add_start_docstrings(FINALIZE_INPUTS_DOCSTRING)
    def finalize(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        max_length: int,
        **kwargs,
    ) -> torch.LongTensor:
        raise NotImplementedError("This is an abstract method.")


@dataclass
class BeamSearchEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using beam search. Hidden states and attention weights
    of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the encoder_hidden_states
    attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        sequences_scores (`torch.FloatTensor` of shape `(batch_size*num_return_sequences)`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Final beam scores of the generated `sequences`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Beam transition scores for each vocabulary token at each generation step. Beam transition scores consisting
            of log probabilities of tokens conditioned on log softmax of previously generated tokens in this beam.
            Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token),
            with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.
        beam_indices (`torch.LongTensor`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Beam indices of generated token id at each generation step. `torch.LongTensor` of shape
            `(batch_size*num_return_sequences, sequence_length)`.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer of the decoder) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size*num_beams*num_return_sequences, sequence_length, hidden_size)`.
        decoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size*num_beams*num_return_sequences, num_heads, generated_length,
            sequence_length)`.
        cross_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size*num_beams*num_return_sequences, generated_length, hidden_size)`.
    """

    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    beam_indices: Optional[torch.LongTensor] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


# BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput]
# GenerateOutput = Union[ BeamSearchOutput]


class BeamHypotheses:
    def __init__(self, num_beams: int, length_penalty: float, early_stopping: bool, max_length: Optional[int] = None):
        """
        Initialize n-best list of hypotheses.
        """
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.max_length = max_length
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

        if not isinstance(self.early_stopping, bool) and self.max_length is None:
            raise ValueError(
                "When `do_early_stopping` is set to a string, `max_length` must be defined. Ensure it is passed to the"
                " BeamScorer class instance at initialization time."
            )

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp: torch.LongTensor, sum_logprobs: float, beam_indices: Optional[torch.LongTensor] = None):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, beam_indices))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False

        # `True`: stop as soon as at least `num_beams` hypotheses are finished
        if self.early_stopping is True:
            return True
        # `False`: heuristic -- compute best possible score from `cur_len`, even though it is not entirely accurate
        #  when `length_penalty` is positive. See the discussion below for more details.
        # https://github.com/huggingface/transformers/pull/20901#issuecomment-1369845565
        elif self.early_stopping is False:
            highest_attainable_score = best_sum_logprobs / cur_len**self.length_penalty
            ret = self.worst_score >= highest_attainable_score
            return ret
        # `"never"`: compute the best possible score, depending on the signal of `length_penalty`
        else:
            # `length_penalty` > 0.0 -> max denominator is obtaned from `max_length`, not from `cur_len` -> min
            # abs(`highest_attainable_score`) is obtained -> `highest_attainable_score` is negative, hence we obtain
            # its max this way
            if self.length_penalty > 0.0:
                highest_attainable_score = best_sum_logprobs / self.max_length**self.length_penalty
            # the opposite logic applies here (max `highest_attainable_score` from `cur_len`)
            else:
                highest_attainable_score = best_sum_logprobs / cur_len**self.length_penalty
            ret = self.worst_score >= highest_attainable_score
            return ret

class BeamSearchScorer(BeamScorer):
    r"""
    [`BeamScorer`] implementing standard beam search decoding.

    Adapted in part from [Facebook's XLM beam search
    code](https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529).

    Reference for the diverse beam search algorithm and implementation [Ashwin Kalyan's DBS
    implementation](https://github.com/ashwinkalyan/dbs/blob/master/dbs/beam_utils.lua)

    Args:
        batch_size (`int`):
            Batch Size of `input_ids` for which standard beam search decoding is run in parallel.
        num_beams (`int`):
            Number of beams for beam search.
        device (`torch.device`):
            Defines the device type (*e.g.*, `"cpu"` or `"cuda"`) on which this instance of `BeamSearchScorer` will be
            allocated.
        length_penalty (`float`, *optional*, defaults to 1.0):
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
            the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
            likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
            `length_penalty` < 0.0 encourages shorter sequences.
        do_early_stopping (`bool` or `str`, *optional*, defaults to `False`):
            Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values:
            `True`, where the generation stops as soon as there are `num_beams` complete candidates; `False`, where an
            heuristic is applied and the generation stops when is it very unlikely to find better candidates;
            `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical
            beam search algorithm).
        num_beam_hyps_to_keep (`int`, *optional*, defaults to 1):
            The number of beam hypotheses that shall be returned upon calling
            [`~transformer.BeamSearchScorer.finalize`].
        num_beam_groups (`int`):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            See [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
        max_length (`int`, *optional*):
            The maximum length of the sequence to be generated.
    """

    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        device: torch.device,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[Union[bool, str]] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
        num_beam_groups: Optional[int] = 1,
        max_length: Optional[int] = None,
    ):
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups

        self._is_init = False
        # self._beam_hyps[i*self.num_beam_groups+j] is the beam_hyps of the j-th group in the i-th mini-batch.
        # If group_beam_search is not used, the list consists of `batch_size` beam_hyps.
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.group_size,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
                max_length=max_length,
            )
            for _ in range(batch_size * self.num_beam_groups)
        ]
        # self._done[i*self.num_beam_groups+j] indicates whether the generation of the beam_hyps of the j-th group
        # in the i-th mini-batch is complete.
        self._done = torch.tensor(
            [False for _ in range(batch_size * self.num_beam_groups)], dtype=torch.bool, device=self.device
        )

        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1,"
                " one should make use of `greedy_search` instead."
            )

        if not isinstance(num_beam_groups, int) or (num_beam_groups > num_beams) or (num_beams % num_beam_groups != 0):
            raise ValueError(
                "`num_beam_groups` has to be an integer smaller or equal than `num_beams` and `num_beams` has to be"
                f" divisible by `num_beam_groups`, but is {num_beam_groups} with `num_beams` being {num_beams}."
            )

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices: Optional[torch.LongTensor] = None,
        group_index: Optional[int] = 0,
    ) -> Dict[str, torch.Tensor]:
        cur_len = input_ids.shape[-1] + 1  # add up to the length which the next_scores is calculated on
        batch_size = len(self._beam_hyps) // self.num_beam_groups

        if not (batch_size == (input_ids.shape[0] // self.group_size)):
            if self.num_beam_groups > 1:
                raise ValueError(
                    f"A group beam size of {input_ids.shape[0]} is used as the input, but a group beam "
                    f"size of {self.group_size} is expected by the beam scorer."
                )
            else:
                raise ValueError(
                    f"A beam size of {input_ids.shape[0]} is used as the input, but a beam size of "
                    f"{self.group_size} is expected by the beam scorer."
                )

        device = input_ids.device
        next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        for batch_idx in range(batch_size):
            batch_group_idx = batch_idx * self.num_beam_groups + group_index
            if self._done[batch_group_idx]:
                if self.num_beams < len(self._beam_hyps[batch_group_idx]):
                    raise ValueError(f"Batch can only be done if at least {self.num_beams} beams have been generated")
                if eos_token_id is None or pad_token_id is None:
                    raise ValueError("Generated beams >= num_beams -> eos_token_id and pad_token have to be defined")
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.item() in eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    if beam_indices is not None:
                        beam_index = beam_indices[batch_beam_idx]
                        beam_index = beam_index + (batch_beam_idx,)
                    else:
                        beam_index = None

                    self._beam_hyps[batch_group_idx].add(
                        input_ids[batch_beam_idx].clone(),
                        next_score.item(),
                        beam_indices=beam_index,
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            if beam_idx < self.group_size:
                raise ValueError(
                    f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id:"
                    f" {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                )

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_group_idx] = self._done[batch_group_idx] or self._beam_hyps[batch_group_idx].is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
            }
        )

    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        max_length: int,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.LongTensor]:
        batch_size = len(self._beam_hyps) // self.num_beam_groups

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_group_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_group_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for index_per_group in range(self.group_size):
                batch_beam_idx = batch_group_idx * self.group_size + index_per_group
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_index = beam_indices[batch_beam_idx] if beam_indices is not None else None
                beam_hyp.add(final_tokens, final_score, beam_indices=beam_index)

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []
        best_indices = []
        best_scores = torch.zeros(batch_size * self.num_beam_hyps_to_keep, device=self.device, dtype=torch.float32)

        # retrieve best hypotheses
        for i in range(batch_size):
            beam_hyps_in_batch = self._beam_hyps[i * self.num_beam_groups : (i + 1) * self.num_beam_groups]
            candidate_beams = [beam for beam_hyp in beam_hyps_in_batch for beam in beam_hyp.beams]
            sorted_hyps = sorted(candidate_beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                best_index = best_hyp_tuple[2]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

                # append hyp to lists
                best.append(best_hyp)

                # append indices to list
                best_indices.append(best_index)

                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

        # prepare for adding eos
        sent_lengths_max = sent_lengths.max().item() + 1
        sent_max_len = min(sent_lengths_max, max_length) if max_length is not None else sent_lengths_max
        decoded: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)

        if len(best_indices) > 0 and best_indices[0] is not None:
            indices: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)
        else:
            indices = None

        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            if pad_token_id is None:
                raise ValueError("`pad_token_id` has to be defined")
            decoded.fill_(pad_token_id)

        if indices is not None:
            indices.fill_(-1)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, (hypo, best_idx) in enumerate(zip(best, best_indices)):
            decoded[i, : sent_lengths[i]] = hypo

            if indices is not None:
                indices[i, : len(best_idx)] = torch.tensor(best_idx)

            if sent_lengths[i] < sent_max_len:
                # inserting only the first eos_token_id
                decoded[i, sent_lengths[i]] = eos_token_id[0]

        return UserDict(
            {
                "sequences": decoded,
                "sequence_scores": best_scores,
                "beam_indices": indices,
            }
        )

class ContextAwareAttention(nn.Module):

    def __init__(self,
                 dim_model: int,
                 dim_context: int,
                 dropout_rate: Optional[float]=0.0):
        super(ContextAwareAttention, self).__init__()
        
        self.dim_model = dim_model
        self.dim_context = dim_context
        self.dropout_rate = dropout_rate
        self.attention_layer = nn.MultiheadAttention(embed_dim=self.dim_model, 
                                                     num_heads=1, 
                                                     dropout=self.dropout_rate, 
                                                     bias=True,
                                                     add_zero_attn=False,
                                                     batch_first=True,
                                                     device=DEVICE)


        self.u_k = nn.Linear(self.dim_context, self.dim_model, bias=False)
        self.w1_k = nn.Linear(self.dim_model, 1, bias=False)
        self.w2_k = nn.Linear(self.dim_model, 1, bias=False)
        
        self.u_v = nn.Linear(self.dim_context, self.dim_model, bias=False)
        self.w1_v = nn.Linear(self.dim_model, 1, bias=False)
        self.w2_v = nn.Linear(self.dim_model, 1, bias=False)
        




    def forward(self,
                q: torch.Tensor, 
                k: torch.Tensor,
                v: torch.Tensor,
                context: Optional[torch.Tensor]=None):
        
        key_context = self.u_k(context)
        value_context = self.u_v(context)

        lambda_k = F.sigmoid(self.w1_k(k) + self.w2_k(key_context))
        lambda_v = F.sigmoid(self.w1_v(v) + self.w2_v(value_context))

        k_cap = (1 - lambda_k) * k + lambda_k * key_context
        v_cap = (1 - lambda_v) * v + lambda_v * value_context

        attention_output, _ = self.attention_layer(query=q,
                                                   key=k_cap,
                                                   value=v_cap)
        return attention_output

class MultimodalBartForConditionalGeneration(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = MultimodalBartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        
        self.patient_summary1 = nn.Linear(config.d_model, config.d_model, bias = False)
        self.patient_summary2 = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias = False)

        # self.doctor_summary1 = nn.Linear(config.d_model, config.d_model, bias = False)
        # self.doctor_summary2 = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias = False)

        self.overall_summary1 = nn.Linear(config.d_model, config.d_model, bias = False)
        self.overall_summary2 = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias = False)


        # self.classifier = nn.Sequential(nn.Linear(Encoder_Cls_dim,Num_labels),
        #                                 nn.Softmax(-1))
        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        visual_input=None,      # New addition of visual_input
        kg_input=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels_patient =None,
        # labels_doctor = None,
        labels_overall = None, 
        is_generate = None,
        test = True
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )


        # print("input ids shape : ", input_ids.shape)
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            visual_input=visual_input,      # New addition of visual_input
            kg_input=kg_input,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_generate = is_generate
           
        )
        # print('ddd:', outputs[0].shape)
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        # print('dfd:', lm_logits.shape)

        patient1 = self.patient_summary1(outputs[0]) 
        patient2 = self.patient_summary2(patient1) 
        # print('cc:', patient2.shape)

        # doctor1 = self.doctor_summary1(outputs[0])
        # doctor2 = self.doctor_summary2(doctor1)
        # print('dd:', doctor2.shape)

        # print("Patient 1 shape : ", patient1.shape)

        overall1 = self.overall_summary1(patient1)
        overall2 = self.overall_summary2(overall1)
        # print('oo:', overall2.shape)

        ce_loss = CrossEntropyLoss()
        loss_patient = 0
        if labels_patient is not None:
            # print('nn:', patient2.view(-1, self.config.vocab_size).shape)
            # print('mm', labels_patient.view(-1).shape)
            loss_patient = ce_loss(patient2.view(-1, self.config.vocab_size), labels_patient.view(-1))

        # loss_doctor = 0
        # if labels_doctor is not None:
        #     loss_doctor = ce_loss(doctor2.view(-1, self.config.vocab_size), labels_doctor.view(-1))
        #     loss_overall = 0
            # print('nn:', doctor2.view(-1, self.config.vocab_size).shape)
            # print('nnff:', labels_doctor.view(-1).shape)
        loss_overall = 0
        if labels_overall is not None:
            # print('mm', labels_overall.view(-1).shape)
            # print('nndf:', overall2.view(-1, self.config.vocab_size).shape)
            loss_overall = ce_loss(overall2.view(-1, self.config.vocab_size), labels_overall.view(-1))


        # L = loss_patient + loss_doctor + loss_overall

        masked_lm_loss = 0
        # masked_lm_loss = 0.2*loss_patient + 0.1*loss_doctor + 0.7*loss_overall
        masked_lm_loss = 0.1*loss_patient + 0.9*loss_overall

        # print('1', labels_patient.shape)
        # print('2', labels_doctor.shape)
        # print('3', labels_overall.shape)
        # if labels is not None:
        #     print('dfg:', labels.shape)
        #     loss_fct = CrossEntropyLoss()
        #     masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        # loss_cls = CrossEntropyLoss()
       
        # classification_loss = 0
        # if intent_labels is not None:
        #     print(outputs.keys())
        #     # print(outputs.encoder_last_hidden_state.shape)
        #     cls_logits = self.classifier(outputs.encoder_last_hidden_state.mean(dim = 1)) # New addition
        #     # print(cls_logits.shape,intent_labels.shape)
        #     classification_loss = loss_cls(cls_logits.view(-1,Num_labels),intent_labels.view(-1))
        #     Total_loss = classification_loss #+ masked_lm_loss
        # cls_logits = self.classifier(outputs.encoder_last_hidden_state.mean(dim = 1)) # New addition


        # print("return_dict : ", return_dict)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
       
       
        if test :
            return Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            ), patient2, overall2
        else:
            return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            # cls_logits = cls_logits, # New addition
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        ) , patient2, overall2 


    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        # print("decoder input ids shappe : ", decoder_input_ids.shape)
        if past is not None:
            
            decoder_input_ids = decoder_input_ids[:, -1:]
            
        # print("decoder input ids shappe : ", decoder_input_ids.shape)
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }



    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    # def prepare_inputs_for_generation(self, *args, **kwargs):
    #     raise NotImplementedError(
    #         "A model class needs to define a `prepare_inputs_for_generation` method in order to use `.generate()`."
    #     )

    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        """
        This function extracts the model-specific `inputs` for generation.
        """
        # 1. retrieve all kwargs that are non-None or non-model input related.
        # some encoder-decoder models have different names for model and encoder
        if (
            self.config.is_encoder_decoder
            and hasattr(self, "encoder")
            and self.encoder.main_input_name != self.main_input_name
        ):
            input_name = self.encoder.main_input_name
        else:
            input_name = self.main_input_name

        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None or k != input_name}

        # 2. check whether model_input_name is passed as kwarg
        # if yes and `inputs` is None use kwarg inputs
        inputs_kwarg = model_kwargs.pop(input_name, None)
        if inputs_kwarg is not None and inputs is not None:
            raise ValueError(
                f"`inputs`: {inputs}` were passed alongside {input_name} which is not allowed."
                f"Make sure to either pass {inputs} or {input_name}=..."
            )
        elif inputs_kwarg is not None:
            inputs = inputs_kwarg

        # 3. In the presence of `inputs_embeds` for text models:
        # - decoder-only models should complain if the user attempts to pass `inputs_embeds`, but the model
        # doesn't have its forwarding implemented. `inputs_embeds` is kept in `model_kwargs` and can coexist with
        # input_ids (`inputs_embeds` will be used in the 1st generation step, as opposed to `input_ids`)
        # - encoder-decoder models should complain if the user attempts to pass `inputs_embeds` and `input_ids`, and
        # pull the former to inputs. It will be used in place of `input_ids` to get the encoder hidden states.
        if input_name == "input_ids" and "inputs_embeds" in model_kwargs:
            if not self.config.is_encoder_decoder:
                has_inputs_embeds_forwarding = "inputs_embeds" in set(
                    inspect.signature(self.prepare_inputs_for_generation).parameters.keys()
                )
                if not has_inputs_embeds_forwarding:
                    raise ValueError(
                        f"You passed `inputs_embeds` to `.generate()`, but the model class {self.__class__.__name__} "
                        "doesn't have its forwarding implemented. See the GPT2 implementation for an example "
                        "(https://github.com/huggingface/transformers/pull/21405), and feel free to open a PR with it!"
                    )
                # In this case, `input_ids` is moved to the `model_kwargs`, so a few automations (like the creation of
                # the attention mask) can rely on the actual model input.
                model_kwargs["input_ids"] = self._maybe_initialize_input_ids_for_generation(
                    inputs, bos_token_id, model_kwargs=model_kwargs
                )
            else:
                if inputs is not None:
                    raise ValueError("You passed `inputs_embeds` and `input_ids` to `.generate()`. Please pick one.")
            inputs, input_name = model_kwargs["inputs_embeds"], "inputs_embeds"

        # 4. if `inputs` is still None, try to create `input_ids` from BOS token
        inputs = self._maybe_initialize_input_ids_for_generation(inputs, bos_token_id, model_kwargs)
        return inputs, input_name, model_kwargs

    def adjust_logits_during_generation(self, logits: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        """
        Implement in subclasses of [`PreTrainedModel`] for custom behavior to adjust the logits in the generate method.
        """
        return logits

    def _maybe_initialize_input_ids_for_generation(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.LongTensor:
        """Initializes input ids for generation, if necessary."""
        if inputs is not None:
            return inputs

        encoder_outputs = model_kwargs.get("encoder_outputs")
        if self.config.is_encoder_decoder and encoder_outputs is not None:
            # make dummy input_ids with value -100, as a sanity check ensuring that they won't be used for encoding
            shape = encoder_outputs.last_hidden_state.size()[:-1]
            return torch.ones(shape, dtype=torch.long, device=self.device) * -100

        if bos_token_id is None:
            raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")

        # If there is some tensor in `model_kwargs`, we can infer the batch size from it. This is helpful with
        # soft-prompting or in multimodal implementations built on top of decoder-only language models.
        batch_size = 1
        for value in model_kwargs.values():
            if isinstance(value, torch.Tensor):
                batch_size = value.shape[0]
                break
        return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * bos_token_id

    def _prepare_attention_mask_for_generation(
        self,
        inputs: torch.Tensor,
        pad_token_id: Optional[int],
        eos_token_id: Optional[Union[int, List[int]]],
    ) -> torch.LongTensor:
        is_input_ids = len(inputs.shape) == 2 and inputs.dtype in [torch.int, torch.long]
        is_pad_token_in_inputs = (pad_token_id is not None) and (pad_token_id in inputs)
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (pad_token_id not in eos_token_id)

        # Check if input is input_ids and padded -> only then is attention_mask defined
        if is_input_ids and is_pad_token_in_inputs and is_pad_token_not_equal_to_eos_token_id:
            return inputs.ne(pad_token_id).long()
        else:
            return torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()
        # Compatibility with Accelerate big model inference: we need the encoder to outputs stuff on the same device
        # as the inputs.
        if hasattr(encoder, "_hf_hook"):
            encoder._hf_hook.io_same_device = True

        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

        return model_kwargs

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, torch.Tensor],
        decoder_start_token_id: int = None,
        bos_token_id: int = None,
        device: torch.device = None,
    ) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        """Prepares `decoder_input_ids` for generation with encoder-decoder models"""
        # 1. Check whether the user has defined `decoder_input_ids` manually. To facilitate in terms of input naming,
        # we also allow the user to pass it under `input_ids`, if the encoder does not use it as the main input.
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        elif "input_ids" in model_kwargs and model_input_name != "input_ids":
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            decoder_input_ids = None

        # 2. Encoder-decoder models expect the `decoder_input_ids` to start with a special token. Let's ensure that.
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        if device is None:
            device = self.device
        decoder_input_ids_start = torch.ones((batch_size, 1), dtype=torch.long, device=device) * decoder_start_token_id

        # no user input -> use decoder_start_token_id as decoder_input_ids
        if decoder_input_ids is None:
            decoder_input_ids = decoder_input_ids_start
        # exception: Donut checkpoints have task-specific decoder starts and don't expect a BOS token
        elif self.config.model_type == "vision-encoder-decoder" and "donut" in self.name_or_path.lower():
            pass
        # user input but doesn't start with decoder_start_token_id -> prepend decoder_start_token_id (and adjust
        # decoder_attention_mask if provided)
        elif (decoder_input_ids[:, 0] != decoder_start_token_id).all().item():
            decoder_input_ids = torch.cat([decoder_input_ids_start, decoder_input_ids], dim=-1)
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = torch.cat(
                    (torch.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask),
                    dim=-1,
                )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        return decoder_input_ids, model_kwargs

    def _get_decoder_start_token_id(self, decoder_start_token_id: int = None, bos_token_id: int = None) -> int:
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.generation_config.decoder_start_token_id
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.generation_config.bos_token_id

        if decoder_start_token_id is not None:
            return decoder_start_token_id
        elif bos_token_id is not None:
            return bos_token_id
        raise ValueError(
            "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
        )

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], torch.Tensor):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

    def _extract_past_from_model_output(self, outputs: ModelOutput, standardize_cache_format: bool = False):
        past_key_values = None
        if "past_key_values" in outputs:
            past_key_values = outputs.past_key_values
        elif "mems" in outputs:
            past_key_values = outputs.mems
        elif "past_buckets_states" in outputs:
            past_key_values = outputs.past_buckets_states

        # Bloom fix: standardizes the cache format when requested
        if standardize_cache_format and hasattr(self, "_convert_to_standard_cache"):
            batch_size = outputs.logits.shape[0]
            past_key_values = self._convert_to_standard_cache(past_key_values, batch_size=batch_size)
        return past_key_values

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        return model_kwargs

    # def _reorder_cache(self, past_key_values, beam_idx):
    #     raise NotImplementedError(
    #         f"Make sure that a `_reorder_cache` function is correctly implemented in {self.__class__.__module__} to"
    #         f" enable beam search for {self.__class__}"
    #     )

    def _get_logits_warper(
        self,
        generation_config: GenerationConfig,
    ) -> LogitsProcessorList:
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsWarper`] instances
        used for multinomial sampling.
        """

        # instantiate warpers list
        warpers = LogitsProcessorList()

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if generation_config.temperature is not None and generation_config.temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(generation_config.temperature))
        min_tokens_to_keep = 2 if generation_config.num_beams > 1 else 1
        if generation_config.top_k is not None and generation_config.top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=min_tokens_to_keep))
        if generation_config.top_p is not None and generation_config.top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=min_tokens_to_keep))
        if generation_config.typical_p is not None and generation_config.typical_p < 1.0:
            warpers.append(
                TypicalLogitsWarper(mass=generation_config.typical_p, min_tokens_to_keep=min_tokens_to_keep)
            )
        if generation_config.epsilon_cutoff is not None and 0.0 < generation_config.epsilon_cutoff < 1.0:
            warpers.append(
                EpsilonLogitsWarper(epsilon=generation_config.epsilon_cutoff, min_tokens_to_keep=min_tokens_to_keep)
            )
        if generation_config.eta_cutoff is not None and 0.0 < generation_config.eta_cutoff < 1.0:
            warpers.append(
                EtaLogitsWarper(epsilon=generation_config.eta_cutoff, min_tokens_to_keep=min_tokens_to_keep)
            )
        # `LogitNormalization` should always be the last logit processor, when present
        if generation_config.renormalize_logits is True:
            warpers.append(LogitNormalization())
        return warpers

    def _get_logits_processor(
        self,
        generation_config: GenerationConfig,
        input_ids_seq_length: int,
        encoder_input_ids: torch.LongTensor,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        logits_processor: Optional[LogitsProcessorList],
    ) -> LogitsProcessorList:
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsProcessor`]
        instances used to modify the scores of the language model head.
        """
        # instantiate processors list
        processors = LogitsProcessorList()

        # print("generation_config.sequence_bias", generation_config.sequence_bias)
        # print(1/0)
        # if generation_config.sequence_bias is not None:
        #     processors.append(SequenceBiasLogitsProcessor(sequence_bias=generation_config.sequence_bias))

        if generation_config.diversity_penalty is not None and generation_config.diversity_penalty > 0.0:
            processors.append(
                HammingDiversityLogitsProcessor(
                    diversity_penalty=generation_config.diversity_penalty,
                    num_beams=generation_config.num_beams,
                    num_beam_groups=generation_config.num_beam_groups,
                )
            )
        if (
            generation_config.encoder_repetition_penalty is not None
            and generation_config.encoder_repetition_penalty != 1.0
        ):
            processors.append(
                EncoderRepetitionPenaltyLogitsProcessor(
                    penalty=generation_config.encoder_repetition_penalty, encoder_input_ids=encoder_input_ids
                )
            )
        if generation_config.repetition_penalty is not None and generation_config.repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=generation_config.repetition_penalty))
        if generation_config.no_repeat_ngram_size is not None and generation_config.no_repeat_ngram_size > 0:
            processors.append(NoRepeatNGramLogitsProcessor(generation_config.no_repeat_ngram_size))
        if (
            generation_config.encoder_no_repeat_ngram_size is not None
            and generation_config.encoder_no_repeat_ngram_size > 0
        ):
            if self.config.is_encoder_decoder:
                processors.append(
                    EncoderNoRepeatNGramLogitsProcessor(
                        generation_config.encoder_no_repeat_ngram_size, encoder_input_ids
                    )
                )
            else:
                raise ValueError(
                    "It's impossible to use `encoder_no_repeat_ngram_size` with decoder-only architecture"
                )
        if generation_config.bad_words_ids is not None:
            processors.append(
                NoBadWordsLogitsProcessor(generation_config.bad_words_ids, generation_config.eos_token_id)
            )
        if (
            generation_config.min_length is not None
            and generation_config.eos_token_id is not None
            and generation_config.min_length > 0
        ):
            processors.append(MinLengthLogitsProcessor(generation_config.min_length, generation_config.eos_token_id))
        if (
            generation_config.min_new_tokens is not None
            and generation_config.eos_token_id is not None
            and generation_config.min_new_tokens > 0
        ):
            processors.append(
                MinNewTokensLengthLogitsProcessor(
                    input_ids_seq_length, generation_config.min_new_tokens, generation_config.eos_token_id
                )
            )
        if prefix_allowed_tokens_fn is not None:
            processors.append(
                PrefixConstrainedLogitsProcessor(
                    prefix_allowed_tokens_fn, generation_config.num_beams // generation_config.num_beam_groups
                )
            )
        if generation_config.forced_bos_token_id is not None:
            processors.append(ForcedBOSTokenLogitsProcessor(generation_config.forced_bos_token_id))
        if generation_config.forced_eos_token_id is not None:
            processors.append(
                ForcedEOSTokenLogitsProcessor(generation_config.max_length, generation_config.forced_eos_token_id)
            )
        if generation_config.remove_invalid_values is True:
            processors.append(InfNanRemoveLogitsProcessor())
        if generation_config.exponential_decay_length_penalty is not None:
            processors.append(
                ExponentialDecayLengthPenalty(
                    generation_config.exponential_decay_length_penalty,
                    generation_config.eos_token_id,
                    input_ids_seq_length,
                )
            )
        if generation_config.suppress_tokens is not None:
            processors.append(SuppressTokensLogitsProcessor(generation_config.suppress_tokens))
        if generation_config.begin_suppress_tokens is not None:
            begin_index = input_ids_seq_length
            begin_index = (
                begin_index
                if (input_ids_seq_length > 1 or generation_config.forced_bos_token_id is None)
                else begin_index + 1
            )
            if generation_config.forced_decoder_ids is not None:
                # generation starts after the last token that is forced
                begin_index += generation_config.forced_decoder_ids[-1][0]
            processors.append(
                SuppressTokensAtBeginLogitsProcessor(generation_config.begin_suppress_tokens, begin_index)
            )
        if generation_config.forced_decoder_ids is not None:
            processors.append(ForceTokensLogitsProcessor(generation_config.forced_decoder_ids))
        # if generation_config.guidance_scale is not None and generation_config.guidance_scale > 1:
        #     processors.append(ClassifierFreeGuidanceLogitsProcessor(generation_config.guidance_scale))
        processors = self._merge_criteria_processor_list(processors, logits_processor)
        # `LogitNormalization` should always be the last logit processor, when present
        if generation_config.renormalize_logits is True:
            processors.append(LogitNormalization())
        return processors

    def _get_stopping_criteria(
        self, generation_config: GenerationConfig, stopping_criteria: Optional[StoppingCriteriaList]
    ) -> StoppingCriteriaList:
        criteria = StoppingCriteriaList()
        if generation_config.max_length is not None:
            max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
            criteria.append(
                MaxLengthCriteria(
                    max_length=generation_config.max_length,
                    max_position_embeddings=max_position_embeddings,
                )
            )
        if generation_config.max_time is not None:
            criteria.append(MaxTimeCriteria(max_time=generation_config.max_time))
        criteria = self._merge_criteria_processor_list(criteria, stopping_criteria)
        return criteria

    def _merge_criteria_processor_list(
        self,
        default_list: Union[LogitsProcessorList, StoppingCriteriaList],
        custom_list: Union[LogitsProcessorList, StoppingCriteriaList],
    ) -> Union[LogitsProcessorList, StoppingCriteriaList]:
        if len(custom_list) == 0:
            return default_list
        for default in default_list:
            for custom in custom_list:
                if type(custom) is type(default):
                    object_type = "stopping criteria" if isinstance(custom, StoppingCriteria) else "logits processor"
                    raise ValueError(
                        f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to"
                        f" `.generate()`, but it has already been created with the values {default}. {default} has been"
                        " created by passing the corresponding arguments to generate or by the model's config default"
                        f" values. If you just want to change the default values of {object_type} consider passing"
                        f" them as arguments to `.generate()` instead of using a custom {object_type}."
                    )
        default_list.extend(custom_list)
        return default_list

    def compute_transition_scores(
        self,
        sequences: torch.Tensor,
        scores: Tuple[torch.Tensor],
        beam_indices: Optional[torch.Tensor] = None,
        normalize_logits: bool = False,
    ) -> torch.Tensor:
      
        # 1. In absence of `beam_indices`, we can assume that we come from e.g. greedy search, which is equivalent
        # to a beam search approach were the first (and only) beam is always selected
        if beam_indices is None:
            beam_indices = torch.arange(scores[0].shape[0]).view(-1, 1).to(sequences.device)
            beam_indices = beam_indices.expand(-1, len(scores))

        # 2. reshape scores as [batch_size*vocab_size, # generation steps] with # generation steps being
        # seq_len - input_length
        scores = torch.stack(scores).reshape(len(scores), -1).transpose(0, 1)

        # 3. Optionally normalize the logits (across the vocab dimension)
        if normalize_logits:
            scores = scores.reshape(-1, self.config.vocab_size, scores.shape[-1])
            scores = torch.nn.functional.log_softmax(scores, dim=1)
            scores = scores.reshape(-1, scores.shape[-1])

        # 4. cut beam_indices to longest beam length
        beam_indices_mask = beam_indices < 0
        max_beam_length = (1 - beam_indices_mask.long()).sum(-1).max()
        beam_indices = beam_indices.clone()[:, :max_beam_length]
        beam_indices_mask = beam_indices_mask[:, :max_beam_length]

        # 5. Set indices of beams that finished early to 0; such indices will be masked correctly afterwards
        beam_indices[beam_indices_mask] = 0

        # 6. multiply beam_indices with vocab size to gather correctly from scores
        beam_sequence_indices = beam_indices * self.config.vocab_size

        # 7. Define which indices contributed to scores
        cut_idx = sequences.shape[-1] - max_beam_length
        indices = sequences[:, cut_idx:] + beam_sequence_indices

        # 8. Compute scores
        transition_scores = scores.gather(0, indices)

        # 9. Mask out transition_scores of beams that stopped early
        transition_scores[beam_indices_mask] = 0

        return transition_scores

    def _validate_model_class(self):
        """
        Confirms that the model class is compatible with generation. If not, raises an exception that points to the
        right class to use.
        """
        if not self.can_generate():
            generate_compatible_mappings = [
                MODEL_FOR_CAUSAL_LM_MAPPING,
                MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
                MODEL_FOR_VISION_2_SEQ_MAPPING,
                MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
                MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
            ]
            generate_compatible_classes = set()
            for model_mapping in generate_compatible_mappings:
                supported_models = model_mapping.get(type(self.config), default=None)
                if supported_models is not None:
                    generate_compatible_classes.add(supported_models.__name__)
            exception_message = (
                f"The current model class ({self.__class__.__name__}) is not compatible with `.generate()`, as "
                "it doesn't have a language model head."
            )
            if generate_compatible_classes:
                exception_message += f" Please use one of the following classes instead: {generate_compatible_classes}"
            raise TypeError(exception_message)

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        """Validates model kwargs for generation. Generate argument typos will also be caught here."""
        # Excludes arguments that are handled before calling any model function
        if self.config.is_encoder_decoder:
            for key in ["decoder_input_ids"]:
                model_kwargs.pop(key, None)

        unused_model_args = []
        model_args = set(inspect.signature(self.prepare_inputs_for_generation).parameters)
        # `kwargs`/`model_kwargs` is often used to handle optional forward pass inputs like `attention_mask`. If
        # `prepare_inputs_for_generation` doesn't accept them, then a stricter check can be made ;)
        if "kwargs" in model_args or "model_kwargs" in model_args:
            model_args |= set(inspect.signature(self.forward).parameters)
        for key, value in model_kwargs.items():
            if value is not None and key not in model_args:
                unused_model_args.append(key)

        if unused_model_args:
            raise ValueError(
                f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
                " generate arguments will also show up in this list)"
            )

    @torch.no_grad()
    def generate1(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        summary_type = None,
        **kwargs,
    ):
    #  -> Union[GenerateOutput, torch.LongTensor]:
        

        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation -- update the generation config
            # model attribute accordingly, if it was created from the model config
            if self.generation_config._from_model_config:
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use a generation configuration file (see"
                        " https://huggingface.co/docs/transformers/main_classes/text_generation )"
                    )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        # 3. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        # 4. Define other model kwargs
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        # decoder-only models should use left-padding for generation
        if not self.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config.pad_token_id is not None
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                bos_token_id=generation_config.bos_token_id,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_seq_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
                "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            if not has_default_max_length:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length

        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            raise ValueError(
                f"Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than"
                f" the maximum length ({generation_config.max_length})"
            )
        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        # print("\n--------------------\n")
        # print("Its time for generation config: ")  
        # print("generation_config.constraints: ", generation_config.constraints)
        # print("generation_config.force_words_ids: ", generation_config.force_words_ids)
        # print("generation_config.num_beams: ", generation_config.num_beams)
        # print("generation_config.top_k: ", generation_config.top_k)
        # print("generation_config.do_sample: ", generation_config.do_sample)
        # print("generation_config.penalty_alpha: ", generation_config.penalty_alpha)  
        # print("generation_config.num_beam_groups: ", generation_config.num_beam_groups)
        # print()

        # 7. determine generation mode
        is_constraint_gen_mode = (
            generation_config.constraints is not None or generation_config.force_words_ids is not None
        )

        is_contrastive_search_gen_mode = (
            (generation_config.num_beams == 1)
            and generation_config.top_k is not None
            and generation_config.top_k > 1
            and generation_config.do_sample is False
            and generation_config.penalty_alpha is not None
            and generation_config.penalty_alpha > 0
        )

        is_greedy_gen_mode = (
            (generation_config.num_beams == 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is False
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_sample_gen_mode = (
            (generation_config.num_beams == 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is True
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_beam_gen_mode = (
            (generation_config.num_beams > 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is False
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_beam_sample_gen_mode = (
            (generation_config.num_beams > 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is True
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_group_beam_gen_mode = (
            (generation_config.num_beams > 1)
            and (generation_config.num_beam_groups > 1)
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_assisted_gen_mode = False
        if assistant_model is not None:
            if not (is_greedy_gen_mode or is_sample_gen_mode):
                raise ValueError(
                    "You've set `assistant_model`, which triggers assisted generate. Currently, assisted generate "
                    "is only supported with Greedy Search and Sample."
                )
            is_assisted_gen_mode = True

        # print("\n Generation mode\n")
        # print("is_constraint_gen_mode: ",is_constraint_gen_mode )
        # print("is_contrastive_search_gen_mode: ",is_contrastive_search_gen_mode)
        # print("is_greedy_gen_mode: ", is_greedy_gen_mode)
        # print("is_sample_gen_mode: ", is_sample_gen_mode)
        # print("is_beam_gen_mode: ", is_beam_gen_mode)
        # print("is_beam_sample_gen_mode: ", is_beam_sample_gen_mode)
        # print("is_group_beam_gen_mode: ", is_group_beam_gen_mode)    
        # print("is_assisted_gen_mode: ", is_assisted_gen_mode)

        # print(1/0)

        if generation_config.num_beam_groups > generation_config.num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        # if is_group_beam_gen_mode and generation_config.do_sample is True:
        #     raise ValueError(
        #         "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
        #     )

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        # 9. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        # 10. go into different generation modes
        # if is_assisted_gen_mode:
        #     if generation_config.num_return_sequences > 1:
        #         raise ValueError(
        #             "num_return_sequences has to be 1 when doing assisted generate, "
        #             f"but is {generation_config.num_return_sequences}."
        #         )
        #     if batch_size > 1:
        #         raise ValueError("assisted generate is only supported for batch_size = 1")
        #     if not model_kwargs["use_cache"]:
        #         raise ValueError("assisted generate requires `use_cache=True`")

        #     # 11. If the assistant model is an encoder-decoder, prepare its encoder outputs
        #     if assistant_model.config.is_encoder_decoder:
        #         assistant_model_kwargs = copy.deepcopy(model_kwargs)
        #         inputs_tensor, model_input_name, assistant_model_kwargs = assistant_model._prepare_model_inputs(
        #             inputs_tensor, assistant_model.generation_config.bos_token_id, assistant_model_kwargs
        #         )
        #         assistant_model_kwargs = assistant_model._prepare_encoder_decoder_kwargs_for_generation(
        #             inputs_tensor, assistant_model_kwargs, model_input_name
        #         )
        #         model_kwargs["assistant_encoder_outputs"] = assistant_model_kwargs["encoder_outputs"]

        #     # 12. run assisted generate
        #     return self.assisted_decoding(
        #         input_ids,
        #         assistant_model=assistant_model,
        #         do_sample=generation_config.do_sample,
        #         logits_processor=logits_processor,
        #         logits_warper=self._get_logits_warper(generation_config) if generation_config.do_sample else None,
        #         stopping_criteria=stopping_criteria,
        #         pad_token_id=generation_config.pad_token_id,
        #         eos_token_id=generation_config.eos_token_id,
        #         output_scores=generation_config.output_scores,
        #         return_dict_in_generate=generation_config.return_dict_in_generate,
        #         synced_gpus=synced_gpus,
        #         streamer=streamer,
        #         **model_kwargs,
        #     )
        # if is_greedy_gen_mode:
        #     if generation_config.num_return_sequences > 1:
        #         raise ValueError(
        #             "num_return_sequences has to be 1 when doing greedy search, "
        #             f"but is {generation_config.num_return_sequences}."
        #         )

        #     # 11. run greedy search
        #     return self.greedy_search(
        #         input_ids,
        #         logits_processor=logits_processor,
        #         stopping_criteria=stopping_criteria,
        #         pad_token_id=generation_config.pad_token_id,
        #         eos_token_id=generation_config.eos_token_id,
        #         output_scores=generation_config.output_scores,
        #         return_dict_in_generate=generation_config.return_dict_in_generate,
        #         synced_gpus=synced_gpus,
        #         streamer=streamer,
        #         **model_kwargs,
        #     )

        # elif is_contrastive_search_gen_mode:
        #     if generation_config.num_return_sequences > 1:
        #         raise ValueError(
        #             "num_return_sequences has to be 1 when doing contrastive search, "
        #             f"but is {generation_config.num_return_sequences}."
        #         )
        #     if not model_kwargs["use_cache"]:
        #         raise ValueError("Contrastive search requires `use_cache=True`")

        #     return self.contrastive_search(
        #         input_ids,
        #         top_k=generation_config.top_k,
        #         penalty_alpha=generation_config.penalty_alpha,
        #         logits_processor=logits_processor,
        #         stopping_criteria=stopping_criteria,
        #         pad_token_id=generation_config.pad_token_id,
        #         eos_token_id=generation_config.eos_token_id,
        #         output_scores=generation_config.output_scores,
        #         return_dict_in_generate=generation_config.return_dict_in_generate,
        #         synced_gpus=synced_gpus,
        #         streamer=streamer,
        #         **model_kwargs,
        #     )

        # elif is_sample_gen_mode:
        #     # 11. prepare logits warper
        #     logits_warper = self._get_logits_warper(generation_config)

        #     # 12. expand input_ids with `num_return_sequences` additional sequences per batch
        #     input_ids, model_kwargs = self._expand_inputs_for_generation(
        #         input_ids=input_ids,
        #         expand_size=generation_config.num_return_sequences,
        #         is_encoder_decoder=self.config.is_encoder_decoder,
        #         **model_kwargs,
        #     )

        #     # 13. run sample
        #     return self.sample(
        #         input_ids,
        #         logits_processor=logits_processor,
        #         logits_warper=logits_warper,
        #         stopping_criteria=stopping_criteria,
        #         pad_token_id=generation_config.pad_token_id,
        #         eos_token_id=generation_config.eos_token_id,
        #         output_scores=generation_config.output_scores,
        #         return_dict_in_generate=generation_config.return_dict_in_generate,
        #         synced_gpus=synced_gpus,
        #         streamer=streamer,
        #         **model_kwargs,
        #     )

 
        if is_beam_gen_mode:
        # elif is_beam_gen_mode:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            # print("stopping criteria : ", stopping_criteria)    

            # if stopping_criteria.max_length is None:
            #     raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                summary_type = summary_type,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                
                **model_kwargs,
            )



    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        summary_type = None,
        **model_kwargs,
    ):
    # -> Union[BeamSearchOutput, torch.LongTensor] :
        
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # print("input ids shape : ", input_ids.shape)
            # print("input ids : ", input_ids)
            # print("batch decode input ids : ",TOKENIZER.batch_decode(input_ids[0]))
            # print("prepare inputs start")
            # print("model kwargs type : ", type(model_kwargs))
            # print("model kwargs keys : ", model_kwargs.keys())
            # print(1/0)
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # print("prepare inputs finished")

            # outputs = self(
            #     **model_inputs,
            #     return_dict=True,
            #     output_attentions=output_attentions,
            #     output_hidden_states=output_hidden_states,
            # )

            # print("model input ids type : ", type(model_inputs['input_ids']))
            # print("encoder_outputs shape : ", model_inputs['encoder_outputs'])
            # print("decoder input ids shape : ",model_inputs['decoder_input_ids'])
        #     print("encoder_outputs shape : ", model_inputs['encoder_outputs']['last_hidden_state'].shape)
        #     print("decoder input ids shape : ",model_inputs['decoder_input_ids'].shape)
        #     print("batch decode decoder ids : ",TOKENIZER.batch_decode(model_inputs['decoder_input_ids'][0]))
        # #     {
        #     "input_ids": None,  # encoder_outputs is defined. input_ids not needed
        #     "encoder_outputs": encoder_outputs,
        #     "past_key_values": past,
        #     "decoder_input_ids": decoder_input_ids,
        #     "attention_mask": attention_mask,
        #     "head_mask": head_mask,
        #     "decoder_head_mask": decoder_head_mask,
        #     "cross_attn_head_mask": cross_attn_head_mask,
        #     "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        # }
            # print(1/0)

            outputs, patient, overall = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )


            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            # print("next_token_logits shape : ", outputs.logits.shape)        
            # next_token_logits = outputs.logits[:, -1, :]
            
            # print(1/0)

            if(summary_type == "patient"):
                # print("inside patient")
                # print(1/0)
                # print("patient shape : ", patient.shape)
                # print(1/0)
                next_token_logits = patient[:, -1, :]
            # elif(summary_type == "doctor"):
            #     next_token_logits = doctor[:, -1, :]
            elif(summary_type == "overall"):
                next_token_logits = overall[:, -1, :]       


            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            # print("bottom input ids shape : ", input_ids.shape)
            # print("bottom batch input ids : ", TOKENIZER.batch_decode(input_ids[0]))

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]











class MultimodalBartModel(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = MultimodalBartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        visual_input=None,      # New addition of visual_input
        kg_input=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_generate = None
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None: #change
            decoder_input_ids = shift_tokens_right(     #change
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id #change
            )


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # print('input ids : ', input_ids.shape)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                visual_input=visual_input,      # New addition of visual_input
                kg_input=kg_input,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids, #change
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=None, #change
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class MAF(nn.Module):
    
    def __init__(self,
                 dim_model: int,
                 dropout_rate: int):
        super(MAF, self).__init__()
        self.dropout_rate = dropout_rate

        # self.kg_context_transform = nn.Linear(KG_MAX_LEN, SOURCE_MAX_LEN, bias=False) 

        # self.kg_context_attention = ContextAwareAttention(dim_model=dim_model,
        #                                                         dim_context=KG_DIM,
        #                                                         dropout_rate=dropout_rate)
        
        self.visual_context_transform = nn.Linear(VISUAL_MAX_LEN, SOURCE_MAX_LEN, bias=False)


        self.visual_context_attention = ContextAwareAttention(dim_model=dim_model,
                                                              dim_context=VISUAL_DIM,
                                                              dropout_rate=dropout_rate) 

        # self.kg_gate = nn.Linear(2*dim_model, dim_model)
        self.visual_gate = nn.Linear(2*dim_model, dim_model)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.final_layer_norm = nn.LayerNorm(dim_model)

        
        
        
        
    def forward(self,
                text_input: torch.Tensor,
                visual_context: Optional[torch.Tensor]=None,
                kg_context: Optional[torch.Tensor]=None,):
                    
 
        # Video as Context for Attention
        visual_context = visual_context.permute(0, 2, 1)
        visual_context = self.visual_context_transform(visual_context)
        visual_context = visual_context.permute(0, 2, 1)
        
        video_out = self.visual_context_attention(q=text_input,
                                                  k=text_input,
                                                  v=text_input,
                                                  context=visual_context)


         # kg as Context for Attention
        # kg_context = kg_context.permute(0, 2, 1)
        # kg_context = self.kg_context_transform(kg_context)
        # kg_context = kg_context.permute(0, 2, 1)
        
        # kg_out = self.kg_context_attention(q=text_input,
        #                                             k=text_input,
        #                                             v=text_input,
        #                                             context=kg_context)
        

        
        # Global Information Fusion Mechanism
        # weight_kg = F.sigmoid(self.kg_gate(torch.cat((kg_out, text_input), dim=-1)))
        weight_v = F.sigmoid(self.visual_gate(torch.cat((video_out, text_input), dim=-1)))

        output = self.final_layer_norm(text_input +
                                       weight_v * video_out)

        return output

# ---------------------------------------------- Modality Aware Fusion ----------------------------------------------

class MAF_KG(nn.Module):
    
    def __init__(self,
                 dim_model: int,
                 dropout_rate: int):
        super(MAF_KG, self).__init__()
        self.dropout_rate = dropout_rate

        self.kg_context_transform = nn.Linear(KG_MAX_LEN, SOURCE_MAX_LEN, bias=False) 

        self.kg_context_attention = ContextAwareAttention(dim_model=dim_model,
                                                                dim_context=KG_DIM,
                                                                dropout_rate=dropout_rate)
        
        # self.visual_context_transform = nn.Linear(VISUAL_MAX_LEN, SOURCE_MAX_LEN, bias=False)


        # self.visual_context_attention = ContextAwareAttention(dim_model=dim_model,
        #                                                       dim_context=VISUAL_DIM,
        #                                                       dropout_rate=dropout_rate) 

        self.kg_gate = nn.Linear(2*dim_model, dim_model)
        # self.visual_gate = nn.Linear(2*dim_model, dim_model)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.final_layer_norm = nn.LayerNorm(dim_model)

        
        
        
        
    def forward(self,
                text_input: torch.Tensor,
                visual_context: Optional[torch.Tensor]=None,
                kg_context: Optional[torch.Tensor]=None,):
                    
 
        # Video as Context for Attention
        # visual_context = visual_context.permute(0, 2, 1)
        # visual_context = self.visual_context_transform(visual_context)
        # visual_context = visual_context.permute(0, 2, 1)
        
        # video_out = self.visual_context_attention(q=text_input,
        #                                           k=text_input,
        #                                           v=text_input,
        #                                           context=visual_context)


         # kg as Context for Attention
        kg_context = kg_context.permute(0, 2, 1)
        kg_context = self.kg_context_transform(kg_context)
        kg_context = kg_context.permute(0, 2, 1)
        
        kg_out = self.kg_context_attention(q=text_input,
                                                    k=text_input,
                                                    v=text_input,
                                                    context=kg_context)
        

        
        # Global Information Fusion Mechanism
        weight_kg = F.sigmoid(self.kg_gate(torch.cat((kg_out, text_input), dim=-1)))
        # weight_v = F.sigmoid(self.visual_gate(torch.cat((video_out, text_input), dim=-1)))

        output = self.final_layer_norm(text_input + weight_kg * kg_out)

        return output

# ---------------------------------------------- Multimodal BartEncoder ----------------------------------------------

class MultimodalBartEncoder(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BartEncoderLayer`.

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.init_weights()
        self.gradient_checkpointing = False

        # ================================ Modifications ================================ #
        # self.fusion_at_layer = [4]
        # self.visual_transformer = TransformerEncoder(d_model=VISUAL_DIM, 
        #                                              num_layers=4,
        #                                              num_heads=8, 
        #                                              dim_feedforward=VISUAL_DIM)

        # self.MAF_layer = MAF(dim_model=embed_dim,
        #                      dropout_rate=0.2)

        # =============================================================================== #

        self.fusionVis_at_layer = [3]
        self.fusionKG_at_layer = [3]
        self.visual_transformer = TransformerEncoder(d_model=VISUAL_DIM, 
                                                     num_layers=6,
                                                     num_heads=8, 
                                                     dim_feedforward=VISUAL_DIM)
        self.kg_transformer = TransformerEncoder(d_model=KG_DIM, 
                                                       num_layers=6,
                                                       num_heads=8, 
                                                       dim_feedforward=KG_DIM)

        self.MAF_layer = MAF(dim_model=embed_dim,
                             dropout_rate=0.2)
        self.MAF_layer_KG = MAF_KG(dim_model=embed_dim,
                             dropout_rate=0.2)

       

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        visual_input=None,      # New addition of visual_input
        kg_input=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(encoder_layers, encoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        #embed_pos = self.embed_positions(input_shape)
        embed_pos = self.embed_positions(input_ids)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (len(self.layers)), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):
           
             # ================================ Modifications ================================ #
            # if idx in self.fusion_at_layer:
            #     visual_input = self.visual_transformer(visual_input)[-1]
            #     hidden_states = self.MAF_layer(text_input=hidden_states,
            #                                    visual_context=visual_input)

            # =============================================================================== #
            # print("visual input shape : ", visual_input.shape)
            if idx in self.fusionVis_at_layer:
                
                visual_input = self.visual_transformer(visual_input)[-1]
                # kg_input = self.kg_transformer(kg_input)[-1]
                hidden_states = self.MAF_layer(text_input=hidden_states,
                                               visual_context=visual_input)

            # print("kg input shape : ", kg_input.shape)
            if idx in self.fusionKG_at_layer:
                kg_input = self.kg_transformer(kg_input)[-1]
                hidden_states = self.MAF_layer_KG(text_input=hidden_states,
                                               kg_context=kg_input)
                                               #visual_context=visual_input)

              
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]
                                 

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

def _save(model, 
          output_dir: str,
          tokenizer=None,
          state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(model, PreTrainedModel):
            if isinstance(unwrap_model(model), PreTrainedModel):
                if state_dict is None:
                    state_dict = model.state_dict()
                unwrap_model(model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                print("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            model.save_pretrained(output_dir, state_dict=state_dict)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)


def save_model(model, 
               output_dir: str,
               tokenizer=None, 
               state_dict=None):
        """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.

        Will only save from the main process.
        """
        _save(model,output_dir, tokenizer=tokenizer, state_dict=state_dict)


def val_epoch(model,
              data_loader,
              optimizer):
    model.eval()
    epoch_val_loss = 0.0

    actuals = []
    predictions_1 = [] #change
    predictions_2 = [] #change

    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader, desc="Validation Loss Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, attention_mask, visual_input, kg_input, labels_1, labels_3 = batch #change
            visual_input = visual_input.unsqueeze(dim = 1)
            kg_input = kg_input.unsqueeze(dim = 1)
            outputs_1, logits_1, logits_3 = model(input_ids=input_ids,
                                                      attention_mask=attention_mask,
                                                      visual_input=visual_input,
                                                      kg_input=kg_input,
                                                      labels = labels_3,
                                                      labels_patient=labels_1, #change
                                                    #   labels_doctor=labels_2, #change
                                                      labels_overall=labels_3, #change
                                                      test = False
                                                      )
            # outputs_2, logits_2 =  model(input_ids=input_ids,
            #                                           attention_mask=attention_mask,
            #                                           labels=labels_2, #change
            #                                           test = False
            #                                           )
            # outputs_3, logits_3 =  model(input_ids=input_ids,
            #                                           attention_mask=attention_mask,
            #                                           labels=labels_3, #change
            #                                           test = False
            #                                           )
            loss = outputs_1['loss'] #change
            # print("Val_loss_1: ",loss)
            # loss_2 = outputs_2['loss'] #change
            # print("Val_loss_2: ",loss_2)
            # loss_2 = outputs_2['loss'] #change
            # print("Val_loss_2: ",loss_2)

            # loss = (loss_1 + loss_2) / 2 #change

            pred_1 = list(torch.argmax(logits_1,dim  = -1).cpu().detach().numpy()) #change
            #pred_2 = list(torch.argmax(logits_2,dim  = -1).cpu().detach().numpy()) #change

            # act = list(intent_labels.cpu().detach().numpy())

            predictions_1.extend(pred_1) #change
            #predictions_2.extend(pred_2) #change

            # actuals.extend(act)
            # print(logits_1.shape)
            # print(logits_2.shape)
            # loss =loss_1

            epoch_val_loss += loss.item()



    # accuracy = f1_score(actuals,predictions, average = "macro")
   


    del pred_1 #change
    #del pred_2 #change
    # del act
    del batch
    del input_ids
    del attention_mask
    # del acoustic_input
    # del visual_input
    del visual_input
    del kg_input
    del labels_1
    # del labels_2
    del outputs_1
    # del outputs_2
    del loss
    # del loss_2
    gc.collect()
    torch.cuda.empty_cache()
   
    return epoch_val_loss


def train(model,
          tokenizer,
          train_data_loader,
          val_data_loader,
          test_data_loader,
          base_learning_rate,
          new_learning_rate,
          weight_decay,
          **gen_kwargs):
   
    optimizer = prepare_for_training(model=model,
                                     base_learning_rate=base_learning_rate,
                                     new_learning_rate=new_learning_rate,
                                     weight_decay=weight_decay)
    # print(optimizer)
   
    train_losses = []

    val_losses = []
    val_1_rouge_2 = []
    # val_2_rouge_2 = []
    val_3_rouge_2 = []

    patience = 1
   
    for epoch in range(MAX_EPOCHS):
        # train_loss, train_acc_1, train_acc_2 = train_epoch(model, #change
        #                          train_data_loader,
        #                          optimizer)
        train_loss, train_acc_1 = train_epoch(model, #change
                                 train_data_loader,
                                 optimizer)


        # print(1/0)

        # print("TRAIN_ACC")
        # print("train_acc_1: ",train_acc_1)
        # print("train_acc_2: ",train_acc_2)
        # print("TRAIN_LOSS")
        # print("Train_loss: ", train_loss)

        train_losses.append(train_loss)
        # print(train_losses)

       
        val_loss = val_epoch(model,
                             val_data_loader,
                             optimizer)
        # print("VAL_LOSS")
        # print(val_loss)
        val_losses.append(val_loss)

        val_results_1, val_results_3= get_val_scores(model, #change
                                     tokenizer,
                                     val_data_loader,
                                     desc="Validation Generation Iteration",
                                     epoch=epoch,
                                     **gen_kwargs)
        # print("Val_results_1---------------------------------------------------")
        # print(val_results_1)
        # print("Val_results_2---------------------------------------------------")
        # print(val_results_2)

        val_1_rouge_2.append(val_results_1['rouge_2']) #change
        # val_2_rouge_2.append(val_results_2['rouge_2']) #change
        val_3_rouge_2.append(val_results_3['rouge_2']) #change

       
        test_results_1, test_results_3 = get_val_scores(model, #change
                                      tokenizer,
                                      test_data_loader,
                                      desc="Test Generation Iteration",
                                      epoch=epoch,
                                      **gen_kwargs)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
   
        print("Epoch: {}\ttrain_loss: {}\tval_loss: {}\tmin_validation_loss: {}".format(epoch+1, train_loss, val_loss, min(val_losses)))
       
        print("PS: \nval_rouge_1: {}\tval_rouge_2: {}\tval_rouge_L: {}\tval_bleu_1: {}\tval_bleu_2: {}\tval_bleu_3: {}\tval_bleu_4: {}\tval_meteor: {}".format(
        val_results_1['rouge_1'], val_results_1['rouge_2'], val_results_1['rouge_L'], val_results_1['bleu_1'], val_results_1['bleu_2'], val_results_1['bleu_3'], val_results_1['bleu_4'], val_results_1['meteor']))

        # print("DS: \nval_rouge_1: {}\tval_rouge_2: {}\tval_rouge_L: {}\tval_bleu_1: {}\tval_bleu_2: {}\tval_bleu_3: {}\tval_bleu_4: {}\tval_meteor: {}".format(
        # val_results_2['rouge_1'], val_results_2['rouge_2'], val_results_2['rouge_L'], val_results_2['bleu_1'], val_results_2['bleu_2'], val_results_2['bleu_3'], val_results_2['bleu_4'], val_results_2['meteor']))

        print("OS: \nval_rouge_1: {}\tval_rouge_2: {}\tval_rouge_L: {}\tval_bleu_1: {}\tval_bleu_2: {}\tval_bleu_3: {}\tval_bleu_4: {}\tval_meteor: {}".format(
        val_results_3['rouge_1'], val_results_3['rouge_2'], val_results_3['rouge_L'], val_results_3['bleu_1'], val_results_3['bleu_2'], val_results_3['bleu_3'], val_results_3['bleu_4'], val_results_3['meteor']))

        print("PS: \ntest_rouge_1: {}\ttest_rouge_2: {}\ttest_rouge_L: {}\ttest_bleu_1: {}\ttest_bleu_2: {}\ttest_bleu_3: {}\ttest_bleu_4: {}\ttest_meteor: {}\ttest_BS_1: {}\test_JS: {}".format(
        test_results_1['rouge_1'], test_results_1['rouge_2'], test_results_1['rouge_L'], test_results_1['bleu_1'], test_results_1['bleu_2'], test_results_1['bleu_3'], test_results_1['bleu_4'], test_results_1['meteor'],test_results_1['BS'], test_results_1['JS'] ))

        # print("DS: \ntest_rouge_1: {}\ttest_rouge_2: {}\ttest_rouge_L: {}\ttest_bleu_1: {}\ttest_bleu_2: {}\ttest_bleu_3: {}\ttest_bleu_4: {}\ttest_meteor: {}\ttest_BS_1: {}\test_JS: {}".format(
        # test_results_2['rouge_1'], test_results_2['rouge_2'], test_results_2['rouge_L'], test_results_2['bleu_1'], test_results_2['bleu_2'], test_results_2['bleu_3'], test_results_2['bleu_4'], test_results_2['meteor'],test_results_2['BS'], test_results_2['JS'] ))
      
        print("OS: \ntest_rouge_1: {}\ttest_rouge_2: {}\ttest_rouge_L: {}\ttest_bleu_1: {}\ttest_bleu_2: {}\ttest_bleu_3: {}\ttest_bleu_4: {}\ttest_meteor: {}\ttest_BS_1: {}\test_JS: {}".format(
        test_results_3['rouge_1'], test_results_3['rouge_2'], test_results_3['rouge_L'], test_results_3['bleu_1'], test_results_3['bleu_2'], test_results_3['bleu_3'], test_results_3['bleu_4'], test_results_3['meteor'],test_results_3['BS'], test_results_3['JS'] ))
        #if epoch == MAX_EPOCHS - 1:
        path = "MTMDS/Ablation_3"
        save_model(model,path,tokenizer)
        print("Model saved at path: ", path)
       
        if val_loss > min(val_losses):
            patience = patience + 1          
            if patience == EARLY_STOPPING_THRESHOLD:
                break
               
        else:
            patience = 1

        del train_loss
        del val_loss
        #del path
        gc.collect()
        torch.cuda.empty_cache()
def get_val_scores(model,
                   tokenizer,
                   data_loader,
                   desc,
                   epoch,
                   **gen_kwargs):
    predictions_1, gold_1, predictions_3, gold_3 = test_epoch(model, #change
                                   tokenizer,
                                   data_loader,
                                   desc=desc,
                                   **gen_kwargs)
   
    result_1 = get_scores(predictions_1, gold_1) #change
   
    # result_2 = get_scores(predictions_2, gold_2) #change

    result_3 = get_scores(predictions_3, gold_3) #change


    
    # if "Validation" in desc and epoch == MAX_EPOCHS - 1:
    #     val_df = pd.DataFrame(list(zip(gold, predictions)), columns=['actual', 'predicted'])
    #     file_name = "./Gen/" + str(epoch+1) + "_val_results.csv"
    #     val_df.to_csv(file_name, index=False) 
        # print("Validation File saved")
        
    if "Test" in desc:
        test_df = pd.DataFrame(list(zip(gold_1, predictions_1, gold_3, predictions_3)), columns=['actual_1', 'predicted_1', 'actual_3', 'predicted_3'])
        file_name = "MTMDS/Ablation_3/pred_file.csv"
        test_df.to_csv(file_name, index=False)  
    print("Test File saved")
    
    del predictions_1
    # del predictions_2
    del predictions_3
    del gold_1
    # del gold_2
    del gold_3
    gc.collect()
    torch.cuda.empty_cache() 
    
    return result_1, result_3
def test_epoch(model,
               tokenizer,
               data_loader,
               desc,
               **gen_kwargs):
    model.eval()
    predictions1 = [] #change
    # predictions2 = [] #cc
    predictions3 = [] #cc
    cls_pred_1 = [] #change
    cls_pred_2 = [] #change
    actuals=[]
    gold_1 = [] #change
    # gold_2 = [] #change
    gold_3 = [] #change

    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader, desc=desc)):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, attention_mask, visual_input, kg_input, labels_1, labels_3= batch #change
            # summary_type = "patient"
            visual_input = visual_input.unsqueeze(dim = 1)
            kg_input = kg_input.unsqueeze(dim = 1)
            generated_ids1 = model.generate1(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           visual_input=visual_input, 
                                           kg_input=kg_input, 
                                           summary_type = "patient",
                                           **gen_kwargs)
            # generated_ids2 = model.generate1(input_ids=input_ids,
            #                                visual_input=visual_input, 
            #                                kg_input=kg_input,
            #                                attention_mask=attention_mask, 
            #                                summary_type = "doctor",
            #                                **gen_kwargs)
            generated_ids3 = model.generate1(input_ids=input_ids,
                                           visual_input=visual_input, 
                                           kg_input=kg_input,
                                           attention_mask=attention_mask, 
                                           summary_type = "overall",
                                           **gen_kwargs)
            # generated_ids = model.generate(input_ids=input_ids,
            #                                attention_mask=attention_mask, 
                                           
            #                                **gen_kwargs)
            # print(generated_ids)
            
            outputs_1, logits_1, logits_3 = model(input_ids=input_ids, #change
                            attention_mask=attention_mask,
                            visual_input=visual_input,
                            kg_input=kg_input,
                            labels=labels_1,
                            test = False
                            )
            # outputs_2, logits_2 = model(input_ids=input_ids, #change
            #                 attention_mask=attention_mask,
            #                 labels=labels_2,
            #                 test = False
            #                 )      
             
            # pred_1 = list(torch.argmax(logits_1,dim  = -1).cpu().detach().numpy()) #change
            # pred_2 = list(torch.argmax(logits_2,dim  = -1).cpu().detach().numpy()) #change

            # #act = list(intent_labels.cpu().detach().numpy())

            # cls_pred_1.extend(pred_1) #change
            # cls_pred_2.extend(pred_2) #change

            #actuals.extend(act)
            generated_ids1 = generated_ids1.detach().cpu().numpy()
            generated_ids1 = np.where(generated_ids1 != -100, generated_ids1, tokenizer.pad_token_id)
            decoded_preds1 = tokenizer.batch_decode(generated_ids1, skip_special_tokens=True)

            ##CC    
            # generated_ids2 = generated_ids2.detach().cpu().numpy()
            # generated_ids2 = np.where(generated_ids2 != -100, generated_ids2, tokenizer.pad_token_id)
            # decoded_preds2 = tokenizer.batch_decode(generated_ids2, skip_special_tokens=True)

             ##CC    
            generated_ids3 = generated_ids3.detach().cpu().numpy()
            generated_ids3 = np.where(generated_ids3 != -100, generated_ids3, tokenizer.pad_token_id)
            decoded_preds3 = tokenizer.batch_decode(generated_ids3, skip_special_tokens=True)
            ##cc

            labels_1 = labels_1.detach().cpu().numpy() #change
            labels_1 = np.where(labels_1 != -100, labels_1, tokenizer.pad_token_id) #change
            decoded_labels_1 = tokenizer.batch_decode(labels_1, skip_special_tokens=True) #change
            
            # labels_2 = labels_2.detach().cpu().numpy() #change
            # labels_2 = np.where(labels_2 != -100, labels_2, tokenizer.pad_token_id) #change
            # decoded_labels_2 = tokenizer.batch_decode(labels_2, skip_special_tokens=True) #change   

            labels_3 = labels_3.detach().cpu().numpy() #change
            labels_3 = np.where(labels_3 != -100, labels_3, tokenizer.pad_token_id) #change
            decoded_labels_3 = tokenizer.batch_decode(labels_3, skip_special_tokens=True) #change         
            
            predictions1.extend(decoded_preds1)
            # predictions2.extend(decoded_preds2)
            predictions3.extend(decoded_preds3)
            # print("----------------------------------PREDICTIONS----------------------------------------")
            gold_1.extend(decoded_labels_1) #change
            # gold_2.extend(decoded_labels_2) #change
            gold_3.extend(decoded_labels_3) #change
            # print('prediction 1', predictions1[0])
            # print('gold 1:', gold_1[0])
            # print('prediction 2:', predictions2[0])
            # print('gold 2:', gold_2[0])
            # print('prediction 3:', predictions3[0])
            # print('gold 3:', gold_3[0])
            # print("-------------------------------------------")
            # print(1/0)

    #accuracy = f1_score(actuals,cls_pred, average = "macro")
    
    
    del batch
    del input_ids
    del attention_mask
    del labels_1 #change
    # del labels_2 #change
    del labels_3 #change
    del visual_input
    del kg_input
    del generated_ids1
    # del generated_ids2
    del generated_ids3
    del decoded_preds1
    # del decoded_preds2
    del decoded_preds3
    gc.collect()
    torch.cuda.empty_cache() 
    
    return predictions1, gold_1, predictions3, gold_3 #change

def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def get_scores(reference_list: list,
               hypothesis_list: list):
    print()
    count=0
    met=0
    bleu_1=0
    bleu_2=0
    bleu_3=0
    bleu_4=0
    rouge1=0
    bs = 0
    J = 0
    rouge2=0
    rougel = 0
    weights_1 = (1./1.,)
    weights_2 = (1./2. , 1./2.)
    weights_3 = (1./3., 1./3., 1./3.)
    weights_4 = (1./4., 1./4., 1./4., 1./4.)
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    for reference, hypothesis in list(zip(reference_list, hypothesis_list)):
        scores = rouge_scorer.score(reference, hypothesis)
        rouge1 += scores['rouge1'].fmeasure
        rouge2 += scores['rouge2'].fmeasure
        rougel += scores['rougeL'].fmeasure

        met += meteor_score([word_tokenize(reference)], word_tokenize(hypothesis))

        # print('ref:', reference)
        # print('hypothesis:', hypothesis)

        # print(1/0)

        Ref_E = Sem_model.encode(reference)
        Hyp_E = Sem_model.encode(hypothesis)

        bs += cosine_similarity([Ref_E],[Hyp_E])

        # print('ref:',reference)
        # print('hyp:',hypothesis)
        # print('co sine:',bs)


        reference = reference.split()
        hypothesis = hypothesis.split()
        
        # results = bertscore.compute(predictions=hypothesis, references=reference)

        J +=  jaccard(reference, hypothesis)


        bleu_1 += sentence_bleu([reference], hypothesis, weights_1) 
        bleu_2 += sentence_bleu([reference], hypothesis, weights_2)
        bleu_3 += sentence_bleu([reference], hypothesis, weights_3)
        bleu_4 += sentence_bleu([reference], hypothesis, weights_4)
        count += 1

    return {
        "rouge_1": rouge1*100/count,
        "rouge_2": rouge2*100/count,
        "rouge_L": rougel*100/count,
        "bleu_1": bleu_1*100/count,
        "bleu_2": bleu_2*100/count,
        "bleu_3": bleu_3*100/count,
        "bleu_4": bleu_4*100/count,
        "meteor": met*100/count,
        "JS": J/count,
        "BS": bs/count
    }

       
def prepare_for_training(model,
                         base_learning_rate: float,
                         new_learning_rate: float,
                         weight_decay: float):
    base_params_list = []
    new_params_list = []
    for name, param in model.named_parameters():
        if "acoustic_transformer" or "visual_transformer" or "MAF_layer" in name:
            new_params_list.append(param)
        else:
            base_params_list.append(param)
           
    optimizer = AdamW(
        [
            {'params': base_params_list,'lr': base_learning_rate, 'weight_decay': weight_decay},
            {'params': new_params_list,'lr': new_learning_rate, 'weight_decay': weight_decay}            
        ],
        lr=base_learning_rate,
        weight_decay=weight_decay
    )
   
    del base_params_list
    del new_params_list
    gc.collect()
    torch.cuda.empty_cache()
   
    return optimizer


def train_epoch(model,
                data_loader,
                optimizer):
    model.train()
    epoch_train_loss = 0.0
    actuals = []
    predictions_1 = [] #change
    predictions_2 = [] #change

    for step, batch in enumerate(tqdm(data_loader, desc="Training Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        # input_ids, attention_mask, labels_1, labels_2, labels_3 = batch #change
        input_ids, attention_mask, visual_input, kg_input, labels_1, labels_3 = batch
        visual_input = visual_input.unsqueeze(dim = 1)
        kg_input = kg_input.unsqueeze(dim = 1)
        # print("-----------------------------------labels_1----------------------------------------")
        # print(labels_1)
        # print("-----------------------------------labels_2----------------------------------------")
        # print(labels_2)
        optimizer.zero_grad()
        # print(acoustic_input.shape,ACOUSTIC_DIM)
        outputs_1 , logits_1, logits_3= model(input_ids=input_ids, #change
                                                    attention_mask=attention_mask,
                                                    visual_input=visual_input,
                                                    kg_input=kg_input,
                                                    labels = labels_3,
                                                    labels_patient=labels_1,  #change
                                                    # labels_doctor=labels_2,  #change
                                                    labels_overall=labels_3,  #change
                                                    test = False
                                                    )
        loss = outputs_1['loss'] #change
        # print("Loss_1: ",loss)
        pred_1 = list(torch.argmax(logits_1,dim  = -1).cpu().detach().numpy()) #change
        predictions_1.extend(pred_1) #change



        # outputs_2 , logits_2= model(input_ids=input_ids, #change
        #                                     attention_mask=attention_mask,
        #                                     labels=labels_2,  #change
        #                                     test = False
        #                                     )
        # loss_2 = outputs_2['loss'] #change
        # print("Loss_2: ",loss_2)

        # pred_2 = list(torch.argmax(logits_2,dim  = -1).cpu().detach().numpy()) #change
        # predictions_2.extend(pred_2) #change



        # loss = (loss_1 + loss_2) / 2 #change

        epoch_train_loss += loss.item()

        #act = list(intent_labels.cpu().detach().numpy())

        # actuals.extend(act)
           
        loss.backward()
        optimizer.step()
   
    # accuracy = f1_score(actuals,predictions, average = "macro")
   
    del pred_1 #change
    # del pred_2 #change
    # del act
    del batch
    del input_ids
    del attention_mask
    del outputs_1
    # del outputs_2
    del loss
    # del loss_1
    # del loss_2
    gc.collect()
    torch.cuda.empty_cache()
   
    return epoch_train_loss/ step , predictions_1[0] #change predictions_2[0]






MODEL = MultimodalBartForConditionalGeneration.from_pretrained('facebook/bart-base')
print("Model loaded...\n")
MODEL.to(DEVICE)

print("Config\n", BartConfig)

Sem_model = SentenceTransformer('bert-base-nli-mean-tokens')
Sem_model.to(DEVICE)

TOKENIZER = BartTokenizerFast.from_pretrained('facebook/bart-base')
print("Tokenizer loaded...\n")
print("specials tokens : ", TOKENIZER.all_special_tokens)
print(TOKENIZER.all_special_ids)
# print(1/0)

#bertscore = BERTScore()

SOURCE_PREFIX = ''
TARGET_PREFIX = ''

print('TARGET_COLUMN_1:',TARGET_COLUMN_1)
# print('TARGET_COLUMN_2:',TARGET_COLUMN_2)
print('MODEL_OUTPUT_DIR',MODEL_OUTPUT_DIR)
print('RESULT_OUTPUT_DIR',RESULT_OUTPUT_DIR)
print('SOURCE_PREFIX',SOURCE_PREFIX)
print('TARGET_PREFIX',TARGET_PREFIX)


gc.collect()

pytorch_total_params = sum(p.numel() for p in MODEL.parameters())
print("Total parameters: ", pytorch_total_params)
pytorch_total_train_params = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)
print("Total trainable parameters: ", pytorch_total_train_params)

# for name, param in MODEL.named_parameters():
#     if "acoustic_transformer" or "visual_transformer" or "MAF_layer" in name:
#         print(name)





NewDataset_path = 'Dataset(sep_cols).json'
Dataset = pd.DataFrame(read_json_data(NewDataset_path))
print(len(Dataset))
print(Dataset.keys())


train_data , test_data = train_test_split(Dataset,test_size = 0.2)
valid_data , test_data = train_test_split(test_data,test_size = 0.7)
test_data.to_csv("MTMDS/Ablation_3/test_data.csv")
# print(type(train_data["concated_transcript"].values))
# print(1/0)
# ------------------------------ READ DATASET ------------------------------ #

train_dataset = set_up_data_loader(train_data)
print("\nTraining Data Loaded...")
print(len(train_dataset))
val_dataset = set_up_data_loader(valid_data)
print(len(val_dataset))

print("\nValidation Data Loaded...")

test_dataset = set_up_data_loader(test_data)
print(len(test_dataset))

print("\nTest Data Loaded...")

# print(1/0)

gc.collect()

# print(next(enumerate(train_dataset)))

# ------------------------------ TRAINING SETUP ------------------------------ #

gen_kwargs = {
    'num_beams': NUM_BEAMS,
    'max_length': TARGET_MAX_LEN_1,
    'early_stopping': EARLY_STOPPING,
    'no_repeat_ngram_size': NO_REPEAT_NGRAM_SIZE
}

train(model=MODEL,
      tokenizer=TOKENIZER,
      train_data_loader=train_dataset,
      val_data_loader=val_dataset,
      test_data_loader=test_dataset,
      base_learning_rate=BASE_LEARNING_RATE,
      new_learning_rate=NEW_LEARNING_RATE,
      weight_decay=WEIGHT_DECAY,
      **gen_kwargs)

print("Model Trained!")

