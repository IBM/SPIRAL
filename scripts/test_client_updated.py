import time
import os
# Correctly import all necessary components from the single ritz_client.py file
from SPIRAL.scripts.utils.ritz_client import RitsChatClient, MODELMAP, MODEL_ID_MAP

# Ensure we are testing the RITS platform, not Watsonx
os.environ["USE_WATSONX"] = "False"

print("--- Starting RITS Client Full Model Check ---")

# Get the list of all available RITS models from the configuration
models_to_test = list(MODEL_ID_MAP['rits'].keys())
passed_models = []
failed_models = {}  # Using a dict to store model and failure reason

# Loop through each model and perform a sanity check
for model_name in models_to_test:
    print(f"\n" + "="*50)
    print(f"--- 🧪 Testing Model: {model_name} ---")
    print("="*50)

    try:
        # 1. Set the current model to be tested
        # This tells the next RitsChatClient instance which model to use
        MODELMAP.set_model('generate_model', model_name)
        print(f"Active model set to '{model_name}'.")

        # 2. Initialize a new client instance for this specific model
        start_time = time.time()
        client = RitsChatClient(temperature=0.7)
        init_time = time.time() - start_time
        print(f"Client for '{model_name}' initialized in {init_time:.2f} seconds.")

        # 3. Send the test prompt to the specific model endpoint
        print("Sending test prompt...")
        prompt = "Hello! Please respond with just the word 'OK'."
        start_time = time.time()
        response, tokens = client.send(prompt, max_tokens=10)
        send_time = time.time() - start_time

        print(f"Received response in {send_time:.2f} seconds.")
        response_text = response.strip()
        print(f"LLM Response: '{response_text}'")
        print(f"Tokens used: {tokens}")

        # 4. Validate the response
        if response and "ok" in response_text.lower():
            print(f"\n✅ PASSED: Model '{model_name}' is responding correctly.")
            passed_models.append(model_name)
        else:
            error_message = f"Received an unexpected response: '{response_text}'"
            print(f"\n❌ FAILED: {error_message}")
            failed_models[model_name] = error_message

    except Exception as e:
        error_message = f"An exception occurred: {e}"
        print(f"\n❌ FAILED: {error_message}")
        failed_models[model_name] = str(e)
        print("This could indicate a problem with the model endpoint, your API key, or network connection.")

# --- Final Summary ---
print("\n\n" + "#"*60)
print("--- Full Model Check Summary ---")
print("#"*60)

total_models = len(models_to_test)
print(f"\nTested {total_models} models.")

print(f"\n✅ Passed Models ({len(passed_models)}/{total_models}):")
if passed_models:
    for m in sorted(passed_models):
        print(f"- {m}")
else:
    print("None")

print(f"\n❌ Failed Models ({len(failed_models)}/{total_models}):")
if failed_models:
    for model, reason in sorted(failed_models.items()):
        print(f"- {model}: {reason}")
else:
    print("None")

print("\n--- Model Check Complete ---")