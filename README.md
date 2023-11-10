### Test out the chat quickly with Your Hugging Face Gradio Space
* create gradio space 
* put this into your directory for `app.py`
```
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
def generate_prompt(example: dict) -> str:
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    if example["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:"
    )
tokenizer = AutoTokenizer.from_pretrained("mehrdad-es/legalLLM-hf")
model = AutoModelForCausalLM.from_pretrained("mehrdad-es/legalLLM-hf", torch_dtype=torch.float16)
model = model.to('cuda:0')

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [30, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def predict(message, history):
    prompt,userInput = message.split('!!')
    message=generate_prompt({"instruction": prompt, "input": userInput})
    history_transformer_format = history + [[message, ""]]
    stop = StopOnTokens()

    messages = "".join(["".join(["\n<USER>:"+item[0], "\n<ASSISTANT>:"+item[1]])  #curr_system_message +
                for item in history_transformer_format])

    model_inputs = tokenizer([messages], return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=400,
        do_sample=True,
        top_p=0.85,
        top_k=500,
        temperature=0.1,
        num_beams=1,
        stopping_criteria=StoppingCriteriaList([stop])
        )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    partial_message  = ""
    for new_token in streamer:
        if new_token != '<':
            partial_message += new_token
            yield partial_message


gr.ChatInterface(predict).queue().launch()
```
* Put this in `requirements.txt`
```
gradio
torch
transformers 
```
* You are ready to build your space with a10g gpu with 46GB ram. Note this setup is recommended. After that you have a chatbot to talk to. Please use {instruction} !! {text} to talk to the chatbot.
### Finetuning StableLM-tuned-alpha-3b to legalLLM-hf
1. Go to `data/alpaca/BillSum` and download the datasets as so:
  ```
  from huggingface_hub import hf_hub_download
  hf_hub_download(repo_id="lighteval/legal-summarization", filename="train.jsonl", repo_type="dataset")
  hf_hub_download(repo_id="lighteval/legal-summarization", filename="test.jsonl", repo_type="dataset")
  ```
2. run `python create_pkl.py`
3. Return to main directory `cd ../../..`
4. Run the following commands in the terminal, assuming you have created a venv and enabled it. Please change the `devices` in `finetune/lora.py` according to your setup. I used 8X Nvidia L4's which ran 6400 iterations in ~7 hours with max GPU utilization of 22GB.
   ```
    pip install -r requirements-all.txt
    python scripts/download.py --repo_id stabilityai/stablelm-tuned-alpha-3b
    python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/stabilityai/stablelm-tuned-alpha-3b
    python scripts/prepare_alpaca.py \
      --destination_path data/alpaca \
      --checkpoint_dir checkpoints/stabilityai/stablelm-tuned-alpha-3b
    python finetune/lora.py \
    --precision bf16-true \
    --data_dir data/alpaca \
    --checkpoint_dir checkpoints/stabilityai/stablelm-tuned-alpha-3b \
    --out_dir out/lora/alpaca
   ```
5. Refer to `tutorials/inference.md` for how to play with the new finetuned model. In order to combine lora weights with orginal model to create legalLLM refer to `tutorials/finetune_lora.py`.
6. In order to upload model to Hugging Face use the following code (assuming you have logined with `huggingface-cli login`) to upload entire folder to newly created empty model directory in Hugging Face
   ```
    from huggingface_hub import HfApi
    api = HfApi()
    
    api.upload_folder(
        folder_path="./path/to/model_directory/",
        repo_id="hugging_face_username/model_repo_name",
        repo_type="model"
    )
   ```
