import json
import argparse
import shutil
from pathlib import Path
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Filter TaskBench dataset and create a local residual data folder.")
    parser.add_argument('--api_family', required=True, type=str, help="The original API family (e.g., huggingface).")
    parser.add_argument('--results_path', required=True, type=Path, help="Path to the results.json from the CoT run.")
    args = parser.parse_args()

    # 1. Define paths
    original_data_dir = Path("Taskbench") / f"data_{args.api_family}"
    residual_api_family = f"{args.api_family}_residual"
    residual_data_dir = Path("Taskbench") / f"data_{residual_api_family}"

    # Clean up previous residual directory if it exists
    if residual_data_dir.exists():
        shutil.rmtree(residual_data_dir)
    residual_data_dir.mkdir(parents=True)

    # 2. Load results and find failed IDs
    if not args.results_path.exists():
        print(f"Error: Results file not found at {args.results_path}"); return
    with open(args.results_path, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    failed_ids = {res['id'] for res in results_data if res.get('metrics', {}).get('accuracy', 0.0) < 1.0}
    
    if not failed_ids:
        print(f"No failed problems found. Residual directory '{residual_data_dir}' is empty.")
        return

    # 3. Load original dataset and filter
    try:
        full_dataset = load_dataset('microsoft/Taskbench', name=args.api_family, split='test')
    except Exception as e:
        print(f"Error loading dataset '{args.api_family}': {e}"); return

    # 4. Write filtered records to the new residual directory
    output_path = residual_data_dir / "user_requests.jsonl"
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in full_dataset:
            if record['id'] in failed_ids:
                out_record = {'id': record['id'], 'instruction': record['instruction'], 'input': record.get('input', ''), 'tool_steps': record.get('tool_steps', [])}
                f.write(json.dumps(out_record) + '\n')
                count += 1
    
    # 5. Copy tool and graph descriptions to the new directory
    for desc_file in ["tool_desc.json", "graph_desc.json"]:
        if (original_data_dir / desc_file).exists():
            shutil.copy(original_data_dir / desc_file, residual_data_dir / desc_file)

    print(f"✅ Created residual dataset with {count} problems at '{residual_data_dir}'")

if __name__ == '__main__':
    main()