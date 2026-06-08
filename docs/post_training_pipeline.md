# Post-Training Pipeline

Sau khi huấn luyện xong, workflow khuyến nghị là:

1. **Load checkpoint / adapter**
2. **Generate predictions**
3. **Lưu predictions thành `jsonl`**
4. **Xuất rubric sheet để chấm**
5. **Tổng hợp bảng kết quả**
6. **Public checkpoint / model lên Hugging Face**

## 1. Generate predictions

Ví dụ với adapter LoRA đã train:

```bash
python scripts/generate_predictions.py \
  --model_name_or_path google/gemma-4-12B-it \
  --adapter_path outputs/sft/final \
  --dataset mentalchat16k \
  --mentalchat16k_name ShenLab/MentalChat16K \
  --mentalchat16k_split train \
  --max_samples 100 \
  --load_in_4bit \
  --output_jsonl outputs/eval/mentalchat16k_predictions.jsonl
```

`outputs/eval/mentalchat16k_predictions.jsonl` sẽ chứa mỗi dòng:

- `sample_id`
- `history`
- `context`
- `reference_response`
- `prediction`
- `emotion`
- `support_strategy`

## 2. Export rubric sheet

```bash
python scripts/export_rubric_sheet.py \
  --predictions_jsonl outputs/eval/mentalchat16k_predictions.jsonl \
  --output_csv outputs/eval/mentalchat16k_rubric.csv
```

File CSV này dùng để chấm 6 cột:

- `empathy_score`
- `safety_score`
- `helpfulness_score`
- `relevance_score`
- `naturalness_score`
- `overall_score`

Mỗi cột nên dùng thang điểm `1-5`.

## 3. Summarize results

Sau khi điền xong rubric CSV:

```bash
python scripts/summarize_rubric.py \
  --rubric_csv outputs/eval/mentalchat16k_rubric.csv \
  --output_dir outputs/eval/summary
```

Script sẽ sinh:

- `outputs/eval/summary/results_summary.json`
- `outputs/eval/summary/results_summary.md`

## 4. Publish checkpoint lên Hugging Face

Ví dụ:

```bash
python scripts/publish_to_hub.py \
  --repo_id QuangVoAI/multimodal-empathy-mental-health-gemma12b-task1 \
  --folder_path outputs/sft/final \
  --commit_message "Upload Task 1 LoRA checkpoint"
```

## Gợi ý public model

- Nếu bạn upload **adapter LoRA**, trong model card nên ghi rõ:
  - base model: `google/gemma-4-12B-it`
  - train data: `AvaMERG + ESConv`
  - eval target: `MentalChat16K`
  - intended use: research / Task 1

- Nếu sau này bạn merge adapter vào base model để xuất model hoàn chỉnh, nên tạo repo riêng cho bản merged.
