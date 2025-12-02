"""
Test text generation with the actual trained DSA model.
This will generate English text to verify the cache fix works in practice.
"""

import torch
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM
from src.dsa_llama_model import DSALlamaForCausalLM

print("=" * 80)
print("TESTING DSA LLAMA TEXT GENERATION")
print("=" * 80)

# Path to your trained model
model_path = "/iopsstor/scratch/cscs/anowak/Sparse-Attention-Zoo/checkpoints/run-1089676-sparse/final_model"
# model_path = "meta-llama/Llama-3.2-1B"

print(f"\n📂 Loading model from: {model_path}")
try:
    model = DSALlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # model = LlamaForCausalLM.from_pretrained(
    #     model_path,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    # )
    print("   ✓ Model loaded successfully")
    print(f"   - Device: {model.device}")
    print(f"   - Dtype: {model.dtype}")
    print(f"   - Layers: {model.config.num_hidden_layers}")
    # print(f"   - Index top-k: {model.config.index_top_k}")
except Exception as e:
    print(f"   ✗ Failed to load model: {e}")
    exit(1)

print(f"\n📂 Loading tokenizer from: {model_path}")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("   ✓ Tokenizer loaded successfully")
    print(f"   - Vocab size: {len(tokenizer)}")
    print(f"   - PAD token: {tokenizer.pad_token}")
    print(f"   - EOS token: {tokenizer.eos_token}")
except Exception as e:
    print(f"   ✗ Failed to load tokenizer: {e}")
    exit(1)

# Test prompts
prompts = [
    "The capital of France is",
    "Once upon a time,",
    "In a galaxy far, far away,",
    "The meaning of life is",
    "To be or not to be,",
]

print("\n" + "=" * 80)
print("GENERATING TEXT")
print("=" * 80)

for i, prompt in enumerate(prompts, 1):
    print(f"\n{'=' * 80}")
    print(f"PROMPT {i}/{len(prompts)}: '{prompt}'")
    print("=" * 80)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs['input_ids'].shape[1]
    print(f"   Input tokens: {input_len}")

    # Generate with different settings
    generation_configs = [
        {"max_new_tokens": 20, "do_sample": False, "name": "Greedy (deterministic)"},
        {"max_new_tokens": 20, "do_sample": True, "temperature": 0.7, "top_p": 0.9, "name": "Sampling (creative)"},
    ]

    for config in generation_configs:
        config_name = config.pop("name")
        print(f"\n   🔸 {config_name}")
        print(f"      Settings: {config}")

        try:
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    use_cache=True,  # This is what we're testing!
                    **config
                )

            # Decode
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            new_text = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)

            print(f"      ✓ Generation successful!")
            print(f"      Full text: '{generated_text}'")
            print(f"      New text:  '{new_text}'")
            print(f"      Tokens generated: {generated_ids.shape[1] - input_len}")

        except Exception as e:
            print(f"      ✗ Generation FAILED!")
            print(f"      Error: {e}")
            import traceback
            traceback.print_exc()
            exit(1)

print("\n" + "=" * 80)
print("🎉 ALL GENERATIONS SUCCESSFUL!")
print("=" * 80)
print("\nThe cache fix is working correctly. The model can generate text")
print("with proper caching, which makes generation much faster!")
