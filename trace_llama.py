import argparse
import json
import sys
import time
import datetime
import inspect
from pathlib import Path

try:
    from llama_cpp import Llama
except ImportError:
    print("Error: llama-cpp-python is not installed. Please install it first.")
    sys.exit(1)

class LlamaProfiler:
    def __init__(self):
        self.calls = []
        # Store captured generation config
        self.gen_config = {}
        # Final formatted prompt
        self.final_prompt = None
        self.chat_messages = None

    def profile_hook(self, frame, event, arg):
        if event != 'call':
            return

        func_name = frame.f_code.co_name
        module_name = frame.f_globals.get('__name__', '')

        # We only care about llama_cpp modules
        if not module_name.startswith('llama_cpp'):
            return

        class_name = None
        code = frame.f_code
        if code.co_argcount > 0:
            arg_name = code.co_varnames[0]
            if arg_name in ('self', 'cls'):
                instance = frame.f_locals.get(arg_name)
                if instance is not None:
                    class_name = instance.__class__.__name__

        call_info = {
            "module_name": module_name,
            "class_name": class_name,
            "function_name": func_name,
            "call_order": len(self.calls) + 1,
        }
        self.calls.append(call_info)

        # Special logic to capture generation config and formatted prompt
        # create_completion is the main bottleneck for generation parameters
        if func_name == 'create_completion' and (class_name == 'Llama' or class_name is None):
            # Capture the local variables passed to create_completion
            args, _, _, values = inspect.getargvalues(frame)
            # Create a copy of the arguments safely
            safe_values = {}
            for arg_n in args:
                if arg_n != 'self' and arg_n != 'prompt':
                    val = values.get(arg_n)
                    # Convert to string or basic type if it's not easily serializable
                    if isinstance(val, (int, float, str, bool, type(None), list, dict)):
                        safe_values[arg_n] = val
                    else:
                        safe_values[arg_n] = str(val)
            self.gen_config.update(safe_values)

            # Capture the final prompt string/tokens
            prompt_val = values.get('prompt')
            if isinstance(prompt_val, str):
                self.final_prompt = prompt_val
            elif isinstance(prompt_val, list):
                if len(prompt_val) > 0 and isinstance(prompt_val[0], int):
                    self.final_prompt = prompt_val # It's already tokens
                else:
                    self.final_prompt = "[Tokens or complex object]"

            # Additional params (e.g. kwargs from create_chat_completion)
            kwargs = values.get('kwargs', {})
            for k, v in kwargs.items():
                if isinstance(v, (int, float, str, bool, type(None), list, dict)):
                    self.gen_config[k] = v
                else:
                    self.gen_config[k] = str(v)

    def start(self):
        sys.setprofile(self.profile_hook)

    def stop(self):
        sys.setprofile(None)

def parse_args():
    parser = argparse.ArgumentParser(description="Trace llama-cpp-python execution.")
    parser.add_argument("model", type=str, help="Path to the GGUF model file.")
    parser.add_argument("prompt", type=str, help="Prompt text or JSON string for chat mode.")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum number of tokens to generate.")
    parser.add_argument("--mode", type=str, choices=["completion", "chat"], default="completion", help="Inference mode (completion or chat).")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path (default: trace_<timestamp>.json).")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.output is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(f"trace_{timestamp}.json")
    else:
        output_file = Path(args.output)

    # Initialize model
    print(f"Loading model from {args.model}...", file=sys.stderr)
    try:
        model = Llama(model_path=args.model, verbose=False)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Model loaded successfully.", file=sys.stderr)

    prompt_data = args.prompt
    if args.mode == "chat":
        try:
            prompt_data = json.loads(args.prompt)
        except json.JSONDecodeError:
            print("Error: For chat mode, the prompt must be a valid JSON string of messages.", file=sys.stderr)
            sys.exit(1)

    # The tracing logic
    profiler = LlamaProfiler()

    print("Starting generation and trace...", file=sys.stderr)
    profiler.start()

    output_resp = None
    try:
        if args.mode == "completion":
            output_resp = model.create_completion(prompt_data, max_tokens=args.max_tokens)
        else:
            output_resp = model.create_chat_completion(prompt_data, max_tokens=args.max_tokens)
    finally:
        profiler.stop()

    print("Generation complete.", file=sys.stderr)

    # Extract the final string prompt for tokenization logging.
    final_prompt_val = profiler.final_prompt

    # Process tokenization based on what was passed to create_completion
    if isinstance(final_prompt_val, list) and len(final_prompt_val) > 0 and isinstance(final_prompt_val[0], int):
        input_tokens = final_prompt_val
        final_prompt_text = model.detokenize(input_tokens).decode('utf-8', errors='ignore')
    elif isinstance(final_prompt_val, str):
        final_prompt_text = final_prompt_val
        input_tokens = model.tokenize(final_prompt_text.encode('utf-8'))
    else:
        # Fallback
        if isinstance(prompt_data, str):
            final_prompt_text = prompt_data
        else:
            final_prompt_text = json.dumps(prompt_data)
        input_tokens = model.tokenize(final_prompt_text.encode('utf-8'))

    print("Formatting logs...", file=sys.stderr)

    # Process inputs for logging
    total_input_tokens = len(input_tokens)
    first_50_tokens = input_tokens[:50]
    last_50_tokens = input_tokens[-50:] if total_input_tokens >= 50 else input_tokens

    # Check BOS
    bos_token = model.token_bos()
    bos_inserted = False
    if len(input_tokens) > 0 and input_tokens[0] == bos_token:
         bos_inserted = True

    input_data = {
        "raw_prompt_or_messages": args.prompt if args.mode == "completion" else json.loads(args.prompt),
        "prompt_text_before_formatting": args.prompt if args.mode == "completion" else None,
        "final_prompt_text_after_formatting": final_prompt_text
    }

    tokenization_data = {
        "full_token_id_list": input_tokens,
        "total_token_count": total_input_tokens,
        "first_50_tokens": first_50_tokens,
        "last_50_tokens": last_50_tokens,
        "bos_or_special_inserted_automatically": bos_inserted
    }

    # Determine model output fields
    generated_text = ""
    generated_token_ids = []
    generated_tokens_count = 0

    if args.mode == "completion":
        generated_text = output_resp["choices"][0]["text"]
        generated_tokens_count = output_resp["usage"]["completion_tokens"]
    else:
        generated_text = output_resp["choices"][0]["message"].get("content", "")
        generated_tokens_count = output_resp["usage"]["completion_tokens"]

    generated_token_ids = model.tokenize(generated_text.encode('utf-8'), add_bos=False)

    model_output_data = {
        "generated_text": generated_text,
        "generated_token_ids": generated_token_ids,
        "number_of_generated_tokens": generated_tokens_count
    }

    # Binding components - deduplicate calls by identity
    components_used = []
    seen_components = set()
    for call in profiler.calls:
        identity = (call["module_name"], call["class_name"], call["function_name"])
        if identity not in seen_components:
            seen_components.add(identity)
            components_used.append({
                "module_name": call["module_name"],
                "class_name": call["class_name"],
                "function_name": call["function_name"],
                "first_call_order": call["call_order"]
            })

    def make_json_safe(obj):
        if isinstance(obj, dict):
            return {k: make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_safe(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    # Add generation config and combine
    final_output = {
        "input_data": input_data,
        "tokenization": tokenization_data,
        "generation_configuration": profiler.gen_config,
        "model_output": model_output_data,
        "binding_component_usage": components_used,
        "call_trace": profiler.calls
    }

    final_output = make_json_safe(final_output)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\nTrace log saved to {output_file}", file=sys.stdout)
    print("\n--- Summary ---", file=sys.stdout)
    print(f"Prompt Length (chars): {len(final_prompt_text)}", file=sys.stdout)
    print(f"Tokenized Length (input): {total_input_tokens}", file=sys.stdout)
    print(f"Number of Generated Tokens: {generated_tokens_count}", file=sys.stdout)

    # Print used formatting/handler modules
    formatters = set()
    for comp in components_used:
        mod = comp["module_name"]
        if "format" in mod or "chat" in mod:
            formatters.add(mod)
    print(f"Formatting modules used: {', '.join(formatters) if formatters else 'None'}", file=sys.stdout)

if __name__ == "__main__":
    main()
