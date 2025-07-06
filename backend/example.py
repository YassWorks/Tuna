# from backend.tools.helpers.utils import Model
# from backend.tools.helpers.logger import get_logger
from backend.tools import *
from datasets import load_dataset
from backend.full_params import *
from backend.PEFT import *

logger = get_logger()

text_gen = Model("distilgpt2")

print(text_gen.generate("the sky looks clear today"))

# dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

# trainer = LoRATrainer(
#     model=text_gen,
#     dataset=dataset,
#     logger=logger,
# )

